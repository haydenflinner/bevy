use super::Svgf;
use bevy_camera::MainPassResolutionOverride;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    system::{Commands, Query, Res},
};
use bevy_image::ToExtents;
use bevy_math::UVec2;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{Buffer, BufferInitDescriptor, BufferUsages, TextureFormat},
    renderer::{RenderDevice, RenderQueue},
    texture::CachedTexture,
};

/// Per-view denoising state.
///
/// `history` holds the temporally accumulated radiance in rgb and the accumulated sample
/// length in a. `moments` holds the first/second luminance moments used for variance
/// estimation. Both ping-pong each frame: index [`Self::index`] is written this frame while
/// `1 - index` is read as last frame's history. `filtered` is scratch for the à-trous passes.
#[derive(Component)]
pub struct ViewDenoiseResources {
    /// Diffuse (demodulated) history and luma moments.
    pub history: [CachedTexture; 2],
    pub moments: [CachedTexture; 2],
    /// Specular history and luma moments.
    pub spec_history: [CachedTexture; 2],
    pub spec_moments: [CachedTexture; 2],
    pub filtered: CachedTexture,
    pub params: Buffer,
    pub index: u32,
    pub view_size: UVec2,
}

pub fn prepare_svgf(
    mut query: Query<(
        Entity,
        &ExtractedCamera,
        &Svgf,
        Option<&mut ViewDenoiseResources>,
        Option<&MainPassResolutionOverride>,
    )>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut commands: Commands,
) {
    for (entity, camera, svgf, resources, resolution_override) in &mut query {
        let Some(mut view_size) = camera.physical_viewport_size else {
            continue;
        };
        if let Some(MainPassResolutionOverride(resolution)) = resolution_override {
            view_size = *resolution;
        }
        if view_size.min_element() == 0 {
            continue;
        }

        let params = [
            svgf.max_history,
            6.0,   // luma_sigma
            1.0,   // depth_sigma
            128.0, // normal_sigma
        ];

        if let Some(mut resources) = resources.filter(|r| r.view_size == view_size) {
            // Flip the ping-pong index once per rendered frame and refresh the uniforms.
            resources.index = 1 - resources.index;
            render_queue.write_buffer(&resources.params, 0, bytemuck::bytes_of(&params));
            continue;
        }

        let make = |label, format| {
            let texture =
                render_device.create_texture(&bevy_render::render_resource::TextureDescriptor {
                    label: Some(label),
                    size: view_size.to_extents(),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: bevy_render::render_resource::TextureDimension::D2,
                    format,
                    usage: bevy_render::render_resource::TextureUsages::TEXTURE_BINDING
                        | bevy_render::render_resource::TextureUsages::STORAGE_BINDING,
                    view_formats: &[],
                });
            let default_view = texture
                .create_view(&bevy_render::render_resource::TextureViewDescriptor::default());
            CachedTexture {
                texture,
                default_view,
            }
        };

        let params_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("solari_denoise_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        commands.entity(entity).insert(ViewDenoiseResources {
            history: [
                make("solari_denoise_history_a", TextureFormat::Rgba16Float),
                make("solari_denoise_history_b", TextureFormat::Rgba16Float),
            ],
            moments: [
                make("solari_denoise_moments_a", TextureFormat::Rgba16Float),
                make("solari_denoise_moments_b", TextureFormat::Rgba16Float),
            ],
            spec_history: [
                make("solari_denoise_spec_history_a", TextureFormat::Rgba16Float),
                make("solari_denoise_spec_history_b", TextureFormat::Rgba16Float),
            ],
            spec_moments: [
                make("solari_denoise_spec_moments_a", TextureFormat::Rgba16Float),
                make("solari_denoise_spec_moments_b", TextureFormat::Rgba16Float),
            ],
            filtered: make("solari_denoise_filtered", TextureFormat::Rgba16Float),
            params: params_buffer,
            index: 0,
            view_size,
        });
    }
}
