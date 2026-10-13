use super::MetalFxRayReconstruction;
use bevy_camera::{Camera3d, CameraMainTextureUsages};
use bevy_core_pipeline::prepass::{DepthPrepass, MotionVectorPrepass};
use bevy_diagnostic::FrameCount;
use bevy_ecs::prelude::*;
use bevy_math::{vec2, UVec2, Vec4Swizzles};
use bevy_render::{
    camera::TemporalJitter,
    render_resource::{TextureFormat, TextureUsages},
    renderer::RenderDevice,
    view::ExtractedView,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLPixelFormat;
use objc2_metal_fx::{MTLFXTemporalDenoisedScaler, MTLFXTemporalDenoisedScalerDescriptor};
use std::sync::Mutex;
use tracing::warn;
use wgpu::hal;

// Halton sequence (2, 3), minus 0.5. Same jitter pattern as TAA.
const HALTON_SEQUENCE: [bevy_math::Vec2; 8] = [
    vec2(0.0, 0.0),
    vec2(0.0, -0.16666666),
    vec2(-0.25, 0.16666669),
    vec2(0.25, -0.3888889),
    vec2(-0.375, -0.055555552),
    vec2(0.125, 0.2777778),
    vec2(-0.125, -0.2777778),
    vec2(0.375, 0.055555582),
];

/// Newtype wrapper so the scaler can live in an ECS component. A scaler can be configured and
/// encoded from any thread, and the render app touches it only on the render thread anyway.
pub struct MetalFxScaler(pub Retained<ProtocolObject<dyn MTLFXTemporalDenoisedScaler>>);

unsafe impl Send for MetalFxScaler {}
unsafe impl Sync for MetalFxScaler {}

#[derive(Component)]
pub struct MetalFxScalerContext {
    pub scaler: Mutex<MetalFxScaler>,
    pub render_resolution: UVec2,
    pub output_format: TextureFormat,
}

pub fn prepare_metalfx(
    mut query: Query<
        (
            Entity,
            &ExtractedView,
            &mut Camera3d,
            &mut CameraMainTextureUsages,
            &mut TemporalJitter,
            Option<&mut MetalFxScalerContext>,
        ),
        (
            With<Camera3d>,
            With<MetalFxRayReconstruction>,
            With<TemporalJitter>,
            With<DepthPrepass>,
            With<MotionVectorPrepass>,
        ),
    >,
    render_device: Res<RenderDevice>,
    frame_count: Res<FrameCount>,
    mut commands: Commands,
) {
    for (
        entity,
        view,
        mut camera_3d,
        mut camera_main_texture_usages,
        mut temporal_jitter,
        context,
    ) in &mut query
    {
        camera_main_texture_usages.0 |= TextureUsages::STORAGE_BINDING;

        let mut depth_texture_usages = TextureUsages::from(camera_3d.depth_texture_usages);
        depth_texture_usages |= TextureUsages::TEXTURE_BINDING;
        camera_3d.depth_texture_usages = depth_texture_usages.into();

        // MetalFX requires the input textures to exactly match the descriptor's input size.
        // `MainPassResolutionOverride` only shrinks the viewport and leaves the view target
        // textures full-sized, so upscaling cannot use it. Render at native resolution instead.
        let render_resolution = view.viewport.zw();
        let output_format = view.target_format;

        temporal_jitter.offset =
            HALTON_SEQUENCE[frame_count.0 as usize % HALTON_SEQUENCE.len()];

        let recreate = match context.as_deref() {
            Some(context) => {
                context.render_resolution != render_resolution
                    || context.output_format != output_format
            }
            None => true,
        };
        if !recreate {
            continue;
        }

        let Some(scaler) = create_scaler(&render_device, render_resolution, output_format)
        else {
            warn!("Failed to create MetalFX temporal denoised scaler");
            continue;
        };

        commands.entity(entity).insert(MetalFxScalerContext {
            scaler: Mutex::new(scaler),
            render_resolution,
            output_format,
        });
    }
}

fn create_scaler(
    render_device: &RenderDevice,
    resolution: UVec2,
    output_format: TextureFormat,
) -> Option<MetalFxScaler> {
    // SAFETY: the device handle is only read and dropped with its guard. The device belongs to
    // this backend, and the returned scaler is used only for encoding, which MetalFX documents
    // as callable from any thread.
    unsafe {
        let hal_device = render_device
            .wgpu_device()
            .as_hal::<hal::api::Metal>()?;
        let device = hal_device.raw_device();

        let output_pixel_format = mtl_pixel_format(output_format)?;

        let descriptor = MTLFXTemporalDenoisedScalerDescriptor::new();
        descriptor.setColorTextureFormat(output_pixel_format);
        descriptor.setDepthTextureFormat(MTLPixelFormat::R32Float);
        descriptor.setMotionTextureFormat(MTLPixelFormat::RG16Float);
        descriptor.setDiffuseAlbedoTextureFormat(MTLPixelFormat::RGBA8Unorm);
        descriptor.setSpecularAlbedoTextureFormat(MTLPixelFormat::RGBA8Unorm);
        descriptor.setNormalTextureFormat(MTLPixelFormat::RGBA16Float);
        descriptor.setRoughnessTextureFormat(MTLPixelFormat::R8Unorm);
        descriptor.setSpecularHitDistanceTextureFormat(MTLPixelFormat::R32Float);
        descriptor.setSpecularHitDistanceTextureEnabled(true);
        descriptor.setReactiveMaskTextureEnabled(false);
        descriptor.setDenoiseStrengthMaskTextureEnabled(false);
        descriptor.setTransparencyOverlayTextureEnabled(false);
        descriptor.setAutoExposureEnabled(false);
        descriptor.setOutputTextureFormat(output_pixel_format);
        descriptor.setInputWidth(resolution.x as usize);
        descriptor.setInputHeight(resolution.y as usize);
        descriptor.setOutputWidth(resolution.x as usize);
        descriptor.setOutputHeight(resolution.y as usize);
        descriptor.setRequiresSynchronousInitialization(true);

        descriptor
            .newTemporalDenoisedScalerWithDevice(device)
            .map(MetalFxScaler)
    }
}

fn mtl_pixel_format(format: TextureFormat) -> Option<MTLPixelFormat> {
    match format {
        TextureFormat::Rgba16Float => Some(MTLPixelFormat::RGBA16Float),
        TextureFormat::Rgba8Unorm => Some(MTLPixelFormat::RGBA8Unorm),
        TextureFormat::Bgra8Unorm => Some(MTLPixelFormat::BGRA8Unorm),
        _ => None,
    }
}
