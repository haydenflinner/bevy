import bevy_pbr::prepass::bindings::PreviousViewUniforms;
import bevy_render::view::View;
import package::realtime::gbuffer_utils::{gpixel_resolve, pixel_dissimilar, ResolvedGPixel};

@group(0) @binding(0) var view_output: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(1) var gbuffer: texture_2d<u32>;
@group(0) @binding(2) var depth_buffer: texture_depth_2d;
@group(0) @binding(3) var motion_vectors: texture_2d<f32>;
@group(0) @binding(4) var previous_gbuffer: texture_2d<u32>;
@group(0) @binding(5) var previous_depth_buffer: texture_depth_2d;
@group(0) @binding(6) var<uniform> view: View;
@group(0) @binding(7) var<uniform> previous_view: PreviousViewUniforms;
@group(0) @binding(8) var history_color_in: texture_2d<f32>;
@group(0) @binding(9) var history_moments_in: texture_2d<f32>;
@group(0) @binding(10) var history_color_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var history_moments_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(12) var<uniform> params: DenoiseParams;
@group(0) @binding(13) var history_sampler: sampler;

struct DenoiseParams {
    max_history: f32,
    luma_sigma: f32,
    depth_sigma: f32,
    normal_sigma: f32,
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

/// Temporal accumulation: reproject last frame's denoised history through the motion
/// vectors, validate it against the previous G-buffer, then blend with the new noisy
/// radiance. Pixels that fail validation restart with a history length of 1.
@compute @workgroup_size(8, 8, 1)
fn temporal(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_id = global_id.xy;
    let size = view.main_pass_viewport.zw;
    if any(vec2f(pixel_id) >= size) { return; }

    var noisy = textureLoad(view_output, pixel_id).rgb;
    let depth = textureLoad(depth_buffer, pixel_id, 0);
    let gpixel = textureLoad(gbuffer, pixel_id, 0);

    // Accumulate and filter lighting demodulated by the surface albedo, so
    // albedo texture detail doesn't enter the noise estimate or block smoothing.
    // Re-modulated by the writeback pass.
    var surface: ResolvedGPixel;
    if depth != 0.0 {
        surface = gpixel_resolve(gpixel, depth, pixel_id, size, view.world_from_clip);
        // Demodulate diffuse surfaces by albedo; on metal, base_color is specular
        // reflectance and demodulating irradiance by it is wrong.
        let demod = mix(
            1.0, max(luma(surface.material.base_color), 0.05), 1.0 - surface.material.metallic);
        noisy /= demod;
    }

    // 5x5 neighborhood statistics of the current noisy frame (more samples →
    // stabler estimate). Used for variance clipping of history, firefly
    // rejection, and as a spatial fallback when reprojected history is rejected.
    var sum = vec3(0.0);
    var sumsq = vec3(0.0);
    var lsum = 0.0;
    var lsumsq = 0.0;
    var count = 0.0;
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let p = vec2i(pixel_id) + vec2i(dx, dy);
            if all(p >= vec2i(0)) && all(vec2f(p) < size) {
                let c = textureLoad(view_output, vec2u(p)).rgb;
                let cl = luma(c);
                sum += c;
                sumsq += c * c;
                lsum += cl;
                lsumsq += cl * cl;
                count += 1.0;
            }
        }
    }
    let nmean = sum / count;
    let nstd = sqrt(max(sumsq / count - nmean * nmean, vec3(0.0)));
    let lmean = lsum / count;
    let lstd = sqrt(max(lsumsq / count - lmean * lmean, 0.0));

    // Firefly rejection: pull lone bright outliers toward the neighborhood range
    // so they don't poison the history for dozens of frames.
    if abs(luma(noisy) - lmean) > 4.0 * lstd + 0.001 {
        noisy = clamp(noisy, nmean - 4.0 * nstd, nmean + 4.0 * nstd + vec3(0.001));
    }
    let l = luma(noisy);

    // Defaults when there is no usable history: a filtered neighborhood estimate
    // rather than raw noise, so disoccluded and marginal pixels still look okay.
    var history_color = nmean;
    var history_len = 1.0;
    var moments = vec2(lmean, lmean * lmean + lstd * lstd);

    // Motion vectors are in UV units, pointing from the previous to the current frame.
    let mv = textureLoad(motion_vectors, pixel_id, 0).xy;
    let prev_uv = (vec2f(pixel_id) + 0.5) / size - mv;
    if all(prev_uv >= vec2(0.0)) && all(prev_uv < vec2(1.0)) {
        let prev_pixel_id = vec2<u32>(prev_uv * size);
        let prev_depth = textureLoad(previous_depth_buffer, prev_pixel_id, 0);

        var valid: bool;
        if depth == 0.0 {
            valid = prev_depth == 0.0;
        } else {
            let prev_surface = gpixel_resolve(
                textureLoad(previous_gbuffer, prev_pixel_id, 0), prev_depth, prev_pixel_id,
                size, previous_view.world_from_clip);
            valid = !pixel_dissimilar(
                depth, surface.world_position, prev_surface.world_position,
                surface.world_normal, prev_surface.world_normal, view);
        }

        if valid {
            // Variance clip: tighter than a noisy min/max box, so less noise leaks
            // through the clamp and less "boiling" remains on flat surfaces.
            let s = textureSampleLevel(history_color_in, history_sampler, prev_uv, 0.0);
            history_color = clamp(
                s.rgb, nmean - 1.5 * nstd, nmean + 1.5 * nstd + vec3(0.001));
            history_len = s.a;
            moments = textureSampleLevel(history_moments_in, history_sampler, prev_uv, 0.0).rg;
        } else {
            // Rejected history: keep a short tail instead of fully resetting, so
            // pixels whose validation flickers don't turn into raw 1spp noise.
            history_len = min(
                textureSampleLevel(history_color_in, history_sampler, prev_uv, 0.0).a, 8.0);
        }
    }

    let new_len = min(history_len + 1.0, params.max_history);
    let alpha = 1.0 / new_len;

    textureStore(history_color_out, pixel_id, vec4(mix(history_color, noisy, alpha), new_len));
    textureStore(
        history_moments_out, pixel_id,
        vec4(mix(moments, vec2(l, l * l), alpha), 0.0, 0.0));
}
