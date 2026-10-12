use super::{prepare::ViewDenoiseResources, Svgf};
use bevy_asset::{load_embedded_asset, AssetServer, Handle};
use bevy_core_pipeline::prepass::{
    PreviousViewData, PreviousViewUniformOffset, PreviousViewUniforms, ViewPrepassTextures,
};
use bevy_ecs::{prelude::*, resource::Resource, system::Commands};
use bevy_render::{
    render_resource::{
        binding_types::{
            sampler, texture_2d, texture_depth_2d, texture_storage_2d, uniform_buffer,
            uniform_buffer_sized,
        },
        BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries,
        CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache,
        SamplerBindingType, SamplerDescriptor, ShaderStages, StorageTextureAccess, TextureFormat,
        TextureSampleType,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
    view::{ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms},
};
use bevy_shader::Shader;
use bevy_utils::default;

/// Strides used for the à-trous filter iterations. Each entry point `atrous_N` in
/// `denoise.wesl` filters at stride N.
const ATROUS_STRIDES: [u32; 5] = [1, 2, 4, 8, 16];

#[derive(Resource)]
pub struct SolariDenoisePipelines {
    temporal_bind_group_layout: BindGroupLayoutDescriptor,
    atrous_bind_group_layout: BindGroupLayoutDescriptor,
    temporal_pipeline: CachedComputePipelineId,
    atrous_pipelines: [CachedComputePipelineId; 5],
    writeback_pipeline: CachedComputePipelineId,
}

/// Denoises [`SolariLighting`](crate::realtime::SolariLighting)'s output.
///
/// Runs after `solari_lighting` (see the `ApplyDeferred` ordering edge in
/// [`SolariDenoisePlugin`](super::SolariDenoisePlugin)) and before the main opaque pass:
/// 1. `temporal`: reprojects the accumulated history through the motion vectors, validates
///    it against the previous frame's G-buffer, and blends in the new noisy frame.
/// 2. `atrous_*`: edge-stopping wavelet passes ping-ponging between the spare history
///    texture and the scratch texture, ending in `view_output`.
pub fn solari_denoise(
    view: ViewQuery<(
        &ViewDenoiseResources,
        &ViewTarget,
        &ViewPrepassTextures,
        &ViewUniformOffset,
        &PreviousViewUniformOffset,
        &Svgf,
    )>,
    pipelines: Option<Res<SolariDenoisePipelines>>,
    pipeline_cache: Res<PipelineCache>,
    view_uniforms: Res<ViewUniforms>,
    previous_view_uniforms: Res<PreviousViewUniforms>,
    render_device: Res<RenderDevice>,
    mut ctx: RenderContext,
) {
    let (
        resources,
        view_target,
        view_prepass_textures,
        view_uniform_offset,
        previous_view_uniform_offset,
        svgf,
    ) = view.into_inner();

    let Some(pipelines) = pipelines else {
        return;
    };
    let iterations = svgf.iterations.clamp(1, 5) as usize;
    let (
        Some(temporal_pipeline),
        Some(gbuffer),
        Some(depth_buffer),
        Some(motion_vectors),
        Some(previous_gbuffer),
        Some(previous_depth_buffer),
        Some(view_uniforms_binding),
        Some(previous_view_uniforms_binding),
    ) = (
        pipeline_cache.get_compute_pipeline(pipelines.temporal_pipeline),
        view_prepass_textures.deferred_view(),
        view_prepass_textures.depth_only_view(),
        view_prepass_textures.motion_vectors_view(),
        view_prepass_textures.previous_deferred_view(),
        view_prepass_textures.previous_depth_only_view(),
        view_uniforms.uniforms.binding(),
        previous_view_uniforms.uniforms.binding(),
    )
    else {
        return;
    };

    let (Some(writeback_pipeline), Some(atrous_pipelines)) = (
        pipeline_cache.get_compute_pipeline(pipelines.writeback_pipeline),
        pipelines
            .atrous_pipelines
            .iter()
            .map(|id| pipeline_cache.get_compute_pipeline(*id))
            .collect::<Option<Vec<_>>>(),
    ) else {
        return;
    };

    let (read, write) = (1 - resources.index, resources.index);
    let view_target_attachment = view_target.get_unsampled_color_attachment();
    let sampler = render_device.create_sampler(&SamplerDescriptor::default());

    let temporal_bind_group = render_device.create_bind_group(
        "solari_denoise_temporal_bind_group",
        &pipeline_cache.get_bind_group_layout(&pipelines.temporal_bind_group_layout),
        &BindGroupEntries::sequential((
            view_target_attachment.view,
            gbuffer,
            depth_buffer,
            motion_vectors,
            previous_gbuffer,
            previous_depth_buffer,
            view_uniforms_binding.clone(),
            previous_view_uniforms_binding.clone(),
            &resources.history[read as usize].default_view,
            &resources.moments[read as usize].default_view,
            &resources.history[write as usize].default_view,
            &resources.moments[write as usize].default_view,
            resources.params.as_entire_binding(),
            &sampler,
        )),
    );

    let (dx, dy) = (
        resources.view_size.x.div_ceil(8),
        resources.view_size.y.div_ceil(8),
    );

    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("solari_denoise"),
            timestamp_writes: None,
        });

    pass.set_bind_group(
        0,
        &temporal_bind_group,
        &[
            view_uniform_offset.offset,
            previous_view_uniform_offset.offset,
        ],
    );
    pass.set_pipeline(temporal_pipeline);
    pass.dispatch_workgroups(dx, dy, 1);

    // À-trous iterations ping-pong between the scratch texture and the now-free "read"
    // history texture, all in demodulated-lighting space. A final writeback pass
    // re-modulates by the surface albedo into the view target.
    let history = &resources.history;
    let filtered = &resources.filtered;
    let spare = &history[read as usize];
    let mut last_dst = &filtered.default_view;
    for i in 0..iterations {
        let (src, dst) = if i == 0 {
            (
                &history[write as usize].default_view,
                &filtered.default_view,
            )
        } else if i % 2 == 1 {
            (&filtered.default_view, &spare.default_view)
        } else {
            (&spare.default_view, &filtered.default_view)
        };
        let bind_group = render_device.create_bind_group(
            "solari_denoise_atrous_bind_group",
            &pipeline_cache.get_bind_group_layout(&pipelines.atrous_bind_group_layout),
            &BindGroupEntries::sequential((
                src,
                dst,
                gbuffer,
                depth_buffer,
                &resources.moments[write as usize].default_view,
                view_uniforms_binding.clone(),
                resources.params.as_entire_binding(),
            )),
        );
        pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
        pass.set_pipeline(atrous_pipelines[i]);
        pass.dispatch_workgroups(dx, dy, 1);
        last_dst = dst;
    }

    let writeback_bind_group = render_device.create_bind_group(
        "solari_denoise_writeback_bind_group",
        &pipeline_cache.get_bind_group_layout(&pipelines.atrous_bind_group_layout),
        &BindGroupEntries::sequential((
            last_dst,
            view_target.main_texture_view(),
            gbuffer,
            depth_buffer,
            &resources.moments[write as usize].default_view,
            view_uniforms_binding.clone(),
            resources.params.as_entire_binding(),
        )),
    );
    pass.set_bind_group(0, &writeback_bind_group, &[view_uniform_offset.offset]);
    pass.set_pipeline(writeback_pipeline);
    pass.dispatch_workgroups(dx, dy, 1);
}

pub fn init_denoise_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
) {
    let temporal_bind_group_layout = BindGroupLayoutDescriptor::new(
        "solari_denoise_temporal_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::ReadWrite),
                texture_2d(TextureSampleType::Uint),
                texture_depth_2d(),
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_2d(TextureSampleType::Uint),
                texture_depth_2d(),
                uniform_buffer::<ViewUniform>(true),
                uniform_buffer::<PreviousViewData>(true),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                uniform_buffer_sized(false, None),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );

    let atrous_bind_group_layout = BindGroupLayoutDescriptor::new(
        "solari_denoise_atrous_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                texture_2d(TextureSampleType::Uint),
                texture_depth_2d(),
                texture_2d(TextureSampleType::Float { filterable: false }),
                uniform_buffer::<ViewUniform>(true),
                uniform_buffer_sized(false, None),
            ),
        ),
    );

    let shader: Handle<Shader> = load_embedded_asset!(asset_server.as_ref(), "denoise.wesl");
    let atrous_shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "denoise_atrous.wesl");

    let temporal_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("solari_denoise_temporal_pipeline".into()),
        layout: vec![temporal_bind_group_layout.clone()],
        shader,
        entry_point: Some("temporal".into()),
        ..default()
    });

    let atrous_pipelines = ATROUS_STRIDES.map(|stride| {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(format!("solari_denoise_atrous_{stride}_pipeline").into()),
            layout: vec![atrous_bind_group_layout.clone()],
            shader: atrous_shader.clone(),
            entry_point: Some(format!("atrous_{stride}").into()),
            ..default()
        })
    });

    let writeback_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("solari_denoise_writeback_pipeline".into()),
        layout: vec![atrous_bind_group_layout.clone()],
        shader: atrous_shader,
        entry_point: Some("writeback".into()),
        ..default()
    });

    commands.insert_resource(SolariDenoisePipelines {
        temporal_bind_group_layout,
        atrous_bind_group_layout,
        temporal_pipeline,
        atrous_pipelines,
        writeback_pipeline,
    });
}
