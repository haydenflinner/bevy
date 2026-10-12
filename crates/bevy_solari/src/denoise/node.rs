use super::{prepare::ViewDenoiseResources, Svgf};
use crate::realtime::prepare::SolariLightingResources;
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
        TextureSampleType, TextureView,
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
    spec_atrous_pipelines: [CachedComputePipelineId; 5],
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
        Option<&SolariLightingResources>,
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
        solari_lighting_resources,
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

    let (Some(writeback_pipeline), Some(atrous_pipelines), Some(spec_atrous_pipelines)) = (
        pipeline_cache.get_compute_pipeline(pipelines.writeback_pipeline),
        pipelines
            .atrous_pipelines
            .iter()
            .map(|id| pipeline_cache.get_compute_pipeline(*id))
            .collect::<Option<Vec<_>>>(),
        pipelines
            .spec_atrous_pipelines
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
            &resources.spec_history[read as usize].default_view,
            &resources.spec_moments[read as usize].default_view,
            &resources.spec_history[write as usize].default_view,
            &resources.spec_moments[write as usize].default_view,
            // Radiance-level specular AOV when `Svgf::true_split` is on; a dummy
            // texture otherwise (params.use_aov == 0 makes the shader ignore it).
            solari_lighting_resources.map_or(&resources.moments[read as usize].default_view, |r| {
                &r.specular_radiance.default_view
            }),
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

    // À-trous iterations ping-pong between each stream's scratch textures, all in
    // demodulated (diffuse) or raw (specular) lighting space. The diffuse stream
    // uses `filtered` and the freed diffuse read-history; the specular stream
    // reuses both freed read-histories. A final writeback pass recombines the
    // streams into the view target.
    let filtered = &resources.filtered;
    let diffuse_spare = &resources.history[read as usize];
    let spec_spare = &resources.spec_history[read as usize];
    let streams = [
        (
            &atrous_pipelines,
            &resources.history[write as usize].default_view,
            &filtered.default_view,
            &diffuse_spare.default_view,
            &resources.moments[write as usize].default_view,
        ),
        (
            &spec_atrous_pipelines,
            &resources.spec_history[write as usize].default_view,
            &spec_spare.default_view,
            &diffuse_spare.default_view,
            &resources.spec_moments[write as usize].default_view,
        ),
    ];

    let mut stream_outputs = Vec::new();
    for (pipelines_set, first_src, ping, pong, moments_view) in streams {
        let mut last_dst: &TextureView = ping;
        for (i, pipeline) in pipelines_set.iter().enumerate().take(iterations) {
            let (src, dst): (&TextureView, &TextureView) = if i == 0 {
                (first_src, ping)
            } else if i % 2 == 1 {
                (ping, pong)
            } else {
                (pong, ping)
            };
            let bind_group = render_device.create_bind_group(
                "solari_denoise_atrous_bind_group",
                &pipeline_cache.get_bind_group_layout(&pipelines.atrous_bind_group_layout),
                &BindGroupEntries::sequential((
                    src,
                    dst,
                    gbuffer,
                    depth_buffer,
                    moments_view,
                    view_uniforms_binding.clone(),
                    resources.params.as_entire_binding(),
                    // spec_src is only used by the writeback entry point; bind a
                    // texture that is never an à-trous dst to avoid usage conflicts.
                    &resources.moments[read as usize].default_view,
                )),
            );
            pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(dx, dy, 1);
            last_dst = dst;
        }
        stream_outputs.push(last_dst);
    }

    let writeback_bind_group = render_device.create_bind_group(
        "solari_denoise_writeback_bind_group",
        &pipeline_cache.get_bind_group_layout(&pipelines.atrous_bind_group_layout),
        &BindGroupEntries::sequential((
            stream_outputs[0],
            view_target.main_texture_view(),
            gbuffer,
            depth_buffer,
            &resources.moments[write as usize].default_view,
            view_uniforms_binding.clone(),
            resources.params.as_entire_binding(),
            stream_outputs[1],
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
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_2d(TextureSampleType::Float { filterable: true }),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                texture_2d(TextureSampleType::Float { filterable: false }),
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
                // Only used by the writeback entry point.
                texture_2d(TextureSampleType::Float { filterable: false }),
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

    let spec_atrous_pipelines = ATROUS_STRIDES.map(|stride| {
        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some(format!("solari_denoise_atrous_spec_{stride}_pipeline").into()),
            layout: vec![atrous_bind_group_layout.clone()],
            shader: atrous_shader.clone(),
            entry_point: Some(format!("atrous_spec_{stride}").into()),
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
        spec_atrous_pipelines,
        writeback_pipeline,
    });
}
