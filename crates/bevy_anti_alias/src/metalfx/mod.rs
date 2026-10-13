//! Apple MetalFX ray reconstruction (`MTLFXTemporalDenoisedScaler`).
//!
//! MetalFX ray reconstruction uses machine learning to denoise raytraced lighting in a single
//! pass, intended for use with `bevy_solari`.
//!
//! Requires an Apple silicon GPU, and the Metal rendering backend. Does not work on other platforms.
//!
// Calls into MetalFX are inherently `unsafe` at the FFI boundary. Each unsafe block carries
// its own safety reasoning.
#![allow(unsafe_code)]
//!
//! # Usage
//! 1. Enable Bevy's `metalfx` feature
//! 2. Check for the presence of `Option<Res<MetalFxRayReconstructionSupported>>` at runtime to see if
//!    MetalFX ray reconstruction is supported on the current machine
//! 3. Add the `MetalFxRayReconstruction` component to your camera entity

mod extract;
mod node;
mod prepare;

pub use prepare::MetalFxScalerContext;

use bevy_app::{App, Plugin};
use bevy_camera::Hdr;
use bevy_core_pipeline::{
    prepass::{DepthPrepass, MotionVectorPrepass},
    schedule::{Core3d, Core3dSystems},
};
use bevy_ecs::prelude::*;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    camera::TemporalJitter,
    renderer::RenderDevice,
    texture::CachedTexture,
    view::prepare_view_targets,
    ExtractSchedule, Render, RenderApp, RenderSystems,
};
use objc2_metal_fx::MTLFXTemporalDenoisedScalerDescriptor;
use tracing::info;
use wgpu::hal;

/// Enables MetalFX ray reconstruction support.
#[derive(Default)]
pub struct MetalFxPlugin;

impl Plugin for MetalFxPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<MetalFxRayReconstruction>();
    }

    fn finish(&self, app: &mut App) {
        let render_world = app.sub_app(RenderApp).world();
        let render_device = render_world.resource::<RenderDevice>();

        // SAFETY: the device handle is only read, and dropped with the guard
        let supported = unsafe {
            render_device
                .wgpu_device()
                .as_hal::<hal::api::Metal>()
                .map(|hal_device| {
                    MTLFXTemporalDenoisedScalerDescriptor::supportsDevice(
                        hal_device.raw_device(),
                    )
                })
                .unwrap_or(false)
        };
        if !supported {
            return;
        }
        info!("MetalFX ray reconstruction is supported");

        app.insert_resource(MetalFxRayReconstructionSupported);
        app.sub_app_mut(RenderApp)
            .add_systems(ExtractSchedule, extract::extract_metalfx_rr)
            .add_systems(
                Render,
                prepare::prepare_metalfx
                    .in_set(RenderSystems::PrepareViews)
                    .before(prepare_view_targets),
            )
            .add_systems(
                Core3d,
                node::metalfx_ray_reconstruction.in_set(Core3dSystems::EarlyPostProcess),
            );
    }
}

/// Camera component to enable MetalFX ray reconstruction denoising.
///
/// Only available when the [`MetalFxRayReconstructionSupported`] resource exists.
/// The main pass always renders at native resolution.
#[derive(Component, Reflect, Clone)]
#[reflect(Component, Default, Clone)]
#[require(TemporalJitter, DepthPrepass, MotionVectorPrepass, Hdr)]
pub struct MetalFxRayReconstruction {
    /// Set to true to delete the saved temporal history (past frames).
    ///
    /// Useful for preventing ghosting when the history is no longer
    /// representative of the current frame, such as in sudden camera cuts.
    ///
    /// After setting this to true, it will automatically be toggled
    /// back to false at the end of the frame.
    pub reset: bool,
}

impl Default for MetalFxRayReconstruction {
    fn default() -> Self {
        Self {
            // No temporal history on the first frame
            reset: true,
        }
    }
}

/// When MetalFX ray reconstruction is supported by the current system, this resource will exist in
/// the main world. Otherwise this resource will be absent.
#[derive(Resource, Clone, Copy)]
pub struct MetalFxRayReconstructionSupported;

/// Additional textures needed as inputs for [`MetalFxRayReconstruction`].
#[derive(Component)]
pub struct ViewMetalFxRayReconstructionTextures {
    pub diffuse_albedo: CachedTexture,
    pub specular_albedo: CachedTexture,
    pub normal: CachedTexture,
    pub roughness: CachedTexture,
    pub depth: CachedTexture,
    pub specular_hit_distance: CachedTexture,
}
