use super::{
    prepare::MetalFxScalerContext, MetalFxRayReconstruction,
    ViewMetalFxRayReconstructionTextures,
};
use bevy_core_pipeline::prepass::ViewPrepassTextures;
use bevy_render::{
    camera::TemporalJitter,
    render_resource::Texture,
    renderer::{RenderContext, ViewQuery},
    view::ViewTarget,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLTexture;
use objc2_metal_fx::{MTLFXTemporalDenoisedScaler, MTLFXTemporalDenoisedScalerBase};
use tracing::warn;
use wgpu::hal;

/// Returns the texture's retained `MTLTexture` handle, if this device is a Metal device.
///
/// Each `as_hal` guard holds the device's snatch lock until dropped, and the lock is not
/// re-entrant, so the handle is retained while the guard is briefly alive instead of holding
/// every texture's guard at once.
///
/// SAFETY: the returned [`Retained`] keeps the `MTLTexture` alive independently of the wgpu
/// texture, and Metal command buffers retain resources used by their encoders, so the handle
/// stays valid until the encoded work has finished.
fn mtl_texture(texture: &Texture) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
    unsafe { texture.as_hal::<hal::api::Metal>() }.and_then(|texture| {
        // SAFETY: the handle points to a live `MTLTexture` whose guard is alive for this call
        unsafe { Retained::retain(texture.raw_handle() as *const _ as *mut _) }
    })
}

// NOTE: this node deliberately records *only* through `as_hal_mut`. wgpu-core does not allow
// mixing the wgpu encoding API and the raw hal encoding API on the same command encoder, so no
// wgpu calls (including diagnostic timestamp writes) may touch `ctx.command_encoder()` here.
pub fn metalfx_ray_reconstruction(
    view: ViewQuery<(
        &MetalFxRayReconstruction,
        &MetalFxScalerContext,
        &TemporalJitter,
        &ViewTarget,
        &ViewPrepassTextures,
        &ViewMetalFxRayReconstructionTextures,
    )>,
    mut ctx: RenderContext,
) {
    let (metalfx, context, temporal_jitter, view_target, prepass_textures, rr_textures) =
        view.into_inner();

    let Some(prepass_motion_vectors_texture) = &prepass_textures.motion_vectors else {
        return;
    };

    let view_target = view_target.post_process_write();
    let render_resolution = context.render_resolution;

    let (
        Some(color),
        Some(output),
        Some(motion),
        Some(diffuse_albedo),
        Some(specular_albedo),
        Some(normal),
        Some(roughness),
        Some(depth),
        Some(specular_hit_distance),
    ) = (
        mtl_texture(view_target.source_texture),
        mtl_texture(view_target.destination_texture),
        mtl_texture(&prepass_motion_vectors_texture.texture.texture),
        mtl_texture(&rr_textures.diffuse_albedo.texture),
        mtl_texture(&rr_textures.specular_albedo.texture),
        mtl_texture(&rr_textures.normal.texture),
        mtl_texture(&rr_textures.roughness.texture),
        mtl_texture(&rr_textures.depth.texture),
        mtl_texture(&rr_textures.specular_hit_distance.texture),
    )
    else {
        return;
    };

    // SAFETY: all MetalFX configuration is applied to `scaler` for the duration of this call
    // only, on the render thread, with every texture it reads or writes retained above and alive
    // until the command buffer's work is done. The encoded work runs after prior commands on the
    // same command buffer, so all inputs are resolved by the time MetalFX reads them. No wgpu
    // commands are recorded on this encoder, as required by the raw encoding API.
    unsafe {
        ctx.command_encoder()
            .as_hal_mut::<hal::api::Metal, _, _>(|encoder| {
                let Some(raw_command_buffer) =
                    encoder.and_then(|encoder| encoder.raw_command_buffer())
                else {
                    warn!("Metal command buffer unavailable for MetalFX ray reconstruction");
                    return;
                };

                let scaler = context.scaler.lock().unwrap();
                let scaler = &*scaler.0;
                scaler.setColorTexture(Some(&color));
                scaler.setDepthTexture(Some(&depth));
                scaler.setMotionTexture(Some(&motion));
                scaler.setDiffuseAlbedoTexture(Some(&diffuse_albedo));
                scaler.setSpecularAlbedoTexture(Some(&specular_albedo));
                scaler.setNormalTexture(Some(&normal));
                scaler.setRoughnessTexture(Some(&roughness));
                scaler.setSpecularHitDistanceTexture(Some(&specular_hit_distance));
                scaler.setOutputTexture(Some(&output));
                scaler.setJitterOffsetX(-temporal_jitter.offset.x);
                scaler.setJitterOffsetY(-temporal_jitter.offset.y);
                scaler.setMotionVectorScaleX(-(render_resolution.x as f32));
                scaler.setMotionVectorScaleY(-(render_resolution.y as f32));
                scaler.setDepthReversed(true);
                scaler.setShouldResetHistory(metalfx.reset);
                scaler.setPreExposure(1.0);

                scaler.encodeToCommandBuffer(raw_command_buffer);
            });
    }
}
