//! Temporal accumulation and edge-stopping wavelet denoising for
//! [`SolariLighting`](crate::realtime::SolariLighting).
//!
//! Solari's realtime lighting is ~1 sample-per-pixel Monte Carlo, which is far too noisy to
//! display directly and too noisy for purely spatial denoisers. This module implements a
//! simplified SVGF-style denoiser entirely in WESL compute:
//!
//! 1. **Temporal accumulation**: the noisy radiance is blended into a per-view history buffer,
//!   reprojected through the motion vector prepass and validated against the previous frame's
//!   G-buffer/depth (disocclusion test). Effective sample count grows with history length.
//! 2. **Moments tracking**: first/second luma moments are accumulated alongside color so a
//!   per-pixel variance estimate can guide spatial filtering.
//! 3. **À-trous wavelet filter**: several 3x3 edge-stopping blur passes at doubling strides,
//!   weighted by normal, linear depth and luminance similarity.

mod extract;
mod node;
mod prepare;

pub use node::SolariDenoisePipelines;
pub use prepare::ViewDenoiseResources;

use crate::{realtime::SolariLighting, SolariPlugins};
use bevy_app::{App, Plugin};
use bevy_core_pipeline::{
    core_3d::main_opaque_pass_3d,
    prepass::{DeferredPrepassDoubleBuffer, DepthPrepassDoubleBuffer},
    schedule::{Core3d, Core3dSystems},
};
use bevy_ecs::{
    component::Component,
    reflect::ReflectComponent,
    schedule::{ApplyDeferred, IntoScheduleConfigs},
};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    renderer::RenderDevice, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_shader::load_shader_library;
use tracing::warn;

/// Denoises Solari's noisy output with temporal accumulation and an à-trous wavelet filter.
pub struct SolariDenoisePlugin;

impl Plugin for SolariDenoisePlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "denoise.wesl");
        load_shader_library!(app, "denoise_atrous.wesl");
        app.register_type::<Svgf>();
    }

    fn finish(&self, app: &mut App) {
        let features = app
            .sub_app(RenderApp)
            .world()
            .resource::<RenderDevice>()
            .features();
        if !features.contains(SolariPlugins::required_wgpu_features()) {
            warn!(
                "SolariDenoisePlugin not loaded. GPU lacks support for required features: {:?}.",
                SolariPlugins::required_wgpu_features().difference(features)
            );
            return;
        }

        app.sub_app_mut(RenderApp)
            .add_systems(ExtractSchedule, extract::extract_svgf)
            .add_systems(RenderStartup, node::init_denoise_pipelines)
            .add_systems(
                Render,
                prepare::prepare_svgf.in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                Core3d,
                (
                    // The Core3d schedule disables automatic deferred-buffer application, so
                    // an explicit sync point is needed to queue `solari_lighting`'s command
                    // buffers before this node's.
                    ApplyDeferred.after(crate::realtime::node::solari_lighting),
                    node::solari_denoise
                        .after(ApplyDeferred)
                        .before(main_opaque_pass_3d)
                        .in_set(Core3dSystems::MainPass),
                ),
            );
    }
}

/// A component used to enable temporal accumulation + SVGF-style denoising for a camera
/// rendering with [`SolariLighting`].
///
/// Requires [`SolariDenoisePlugin`] to be loaded (part of [`SolariPlugins`]).
#[derive(Component, Reflect, Clone)]
#[reflect(Component, Default, Clone)]
#[require(SolariLighting, DeferredPrepassDoubleBuffer, DepthPrepassDoubleBuffer)]
pub struct Svgf {
    /// Number of à-trous filter iterations, clamped to `1..=5`. Higher values filter
    /// lower-frequency noise at the cost of some detail and performance.
    pub iterations: u32,
    /// Maximum temporal accumulation length, in frames. Higher values are smoother but
    /// take longer to react to lighting changes (ghosting is mostly suppressed by
    /// neighborhood clamping and the disocclusion test).
    pub max_history: f32,
    /// When true, the diffuse/specular split consumes Solari's radiance-level
    /// specular AOV (first-bounce lobe attribution tracked inside the path
    /// tracer) instead of estimating the specular share per pixel with the
    /// `env_brdf_approx2` heuristic.
    pub true_split: bool,
}

impl Default for Svgf {
    fn default() -> Self {
        Self {
            iterations: 5,
            max_history: 64.0,
            true_split: false,
        }
    }
}
