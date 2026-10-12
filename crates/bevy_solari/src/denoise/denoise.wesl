import bevy_pbr::prepass::bindings::PreviousViewUniforms;
import bevy_pbr::render::pbr_functions::{calculate_F0, calculate_diffuse_color};
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
@group(0) @binding(14) var spec_history_in: texture_2d<f32>;
@group(0) @binding(15) var spec_moments_in: texture_2d<f32>;
@group(0) @binding(16) var spec_history_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(17) var spec_moments_out: texture_storage_2d<rgba16float, write>;

struct DenoiseParams {
    max_history: f32,
    luma_sigma: f32,
    depth_sigma: f32,
    normal_sigma: f32,
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

/// Approximate environment BRDF (Lazarov 2013), used to estimate what fraction
/// of a pixel's radiance arrived through the specular lobe of the primary
/// surface. Same approximation the DLSS RR guide-buffer resolve uses.
fn env_brdf_approx2(specular_color: vec3<f32>, alpha: f32, N: vec3<f32>, V: vec3<f32>) -> vec3<f32> {
    let NoV = abs(dot(N, V));

    var X: vec4<f32>;
    X.x = 1.0;
    X.y = NoV;
    X.z = NoV * NoV;
    X.w = NoV * X.z;

    var Y: vec4<f32>;
    Y.x = 1.0;
    Y.y = alpha;
    Y.z = alpha * alpha;
    Y.w = alpha * Y.z;

    let M1 = mat2x2<f32>(0.99044, 1.29678, -1.28514, -0.755907);
    let M2 = mat3x3<f32>(1.0, 20.3225, 121.563, 2.92338, -27.0302, 626.13, 59.4188, 222.592, 316.627);
    let M3 = mat2x2<f32>(0.0365463, 9.0632, 3.32707, -9.04756);
    let M4 = mat3x3<f32>(1.0, 9.04401, 5.56589, 3.59685, -16.3174, 19.7886, -1.36772, 9.22949, -20.2123);

    var bias = dot(M1 * X.xy, Y.xy) / dot(M2 * X.xyw, Y.xyw);
    let scale = dot(M3 * X.xy, Y.xy) / dot(M4 * X.xzw, Y.xyw);

    bias *= saturate(specular_color.g * 50.0);

    return fma(specular_color, vec3(max(0.0, scale)), vec3(max(0.0, bias)));
}

/// Temporal accumulation: reproject last frame's denoised history through the motion
/// vectors, validate it against the previous G-buffer, then blend with the new noisy
/// radiance.
///
/// The input is split into a diffuse stream (demodulated by diffuse albedo, long
/// history, filtered hard) and a specular stream (raw radiance share, shorter
/// history since specular is view-dependent, filtered by a roughness-shaped
/// kernel downstream) — the ReBLUR-style dual-stream approach.
@compute @workgroup_size(8, 8, 1)
fn temporal(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_id = global_id.xy;
    let size = view.main_pass_viewport.zw;
    if any(vec2f(pixel_id) >= size) { return; }

    var noisy = textureLoad(view_output, pixel_id).rgb;
    let depth = textureLoad(depth_buffer, pixel_id, 0);
    let gpixel = textureLoad(gbuffer, pixel_id, 0);

    var surface: ResolvedGPixel;
    var w_spec = 0.0;
    var demod = 1.0;
    if depth != 0.0 {
        surface = gpixel_resolve(gpixel, depth, pixel_id, size, view.world_from_clip);
        let m = surface.material;
        let wo = normalize(view.world_position - surface.world_position);
        let F0 = calculate_F0(m.base_color, m.metallic, vec3(m.reflectance));
        w_spec = saturate(luma(env_brdf_approx2(F0, m.roughness, surface.world_normal, wo)));
        demod = max(luma(calculate_diffuse_color(m.base_color, m.metallic, 0.0, 0.0)), 0.05);
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

    // Split into diffuse (demodulated) and specular shares; they recombine exactly.
    let noisy_diff = noisy * (1.0 - w_spec) / demod;
    let noisy_spec = noisy * w_spec;
    let l_diff = luma(noisy_diff);
    let l_spec = luma(noisy_spec);

    // Defaults when there is no usable history: a filtered neighborhood estimate
    // rather than raw noise, so disoccluded and marginal pixels still look okay.
    let f_diff = (1.0 - w_spec) / demod;
    let f_spec = w_spec;
    var diff_color = nmean * f_diff;
    var spec_color = nmean * f_spec;
    var diff_len = 1.0;
    var spec_len = 1.0;
    var diff_moments = vec2(lmean * f_diff, (lmean * lmean + lstd * lstd) * f_diff * f_diff);
    var spec_moments = vec2(lmean * f_spec, (lmean * lmean + lstd * lstd) * f_spec * f_spec);

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
            let clip_lo_d = (nmean - 1.5 * nstd) * f_diff;
            let clip_hi_d = (nmean + 1.5 * nstd + vec3(0.001)) * f_diff;
            let clip_lo_s = (nmean - 1.5 * nstd) * f_spec;
            let clip_hi_s = (nmean + 1.5 * nstd + vec3(0.001)) * f_spec;

            let sd = textureSampleLevel(history_color_in, history_sampler, prev_uv, 0.0);
            let ss = textureSampleLevel(spec_history_in, history_sampler, prev_uv, 0.0);
            diff_color = clamp(sd.rgb, clip_lo_d, clip_hi_d);
            spec_color = clamp(ss.rgb, clip_lo_s, clip_hi_s);
            diff_len = sd.a;
            spec_len = ss.a;
            diff_moments = textureSampleLevel(history_moments_in, history_sampler, prev_uv, 0.0).rg;
            spec_moments = textureSampleLevel(spec_moments_in, history_sampler, prev_uv, 0.0).rg;
        } else {
            // Rejected history: keep a short tail instead of fully resetting, so
            // pixels whose validation flickers don't turn into raw 1spp noise.
            diff_len = min(
                textureSampleLevel(history_color_in, history_sampler, prev_uv, 0.0).a, 8.0);
            spec_len = min(
                textureSampleLevel(spec_history_in, history_sampler, prev_uv, 0.0).a, 8.0);
        }
    }

    // Specular gets half the accumulation length: it is view-dependent and
    // ghosts faster under motion than diffuse.
    let new_diff_len = min(diff_len + 1.0, params.max_history);
    let new_spec_len = min(spec_len + 1.0, params.max_history * 0.5);
    let diff_alpha = 1.0 / new_diff_len;
    let spec_alpha = 1.0 / new_spec_len;

    textureStore(
        history_color_out, pixel_id,
        vec4(mix(diff_color, noisy_diff, diff_alpha), new_diff_len));
    textureStore(
        history_moments_out, pixel_id,
        vec4(mix(diff_moments, vec2(l_diff, l_diff * l_diff), diff_alpha), 0.0, 0.0));
    textureStore(
        spec_history_out, pixel_id,
        vec4(mix(spec_color, noisy_spec, spec_alpha), new_spec_len));
    textureStore(
        spec_moments_out, pixel_id,
        vec4(mix(spec_moments, vec2(l_spec, l_spec * l_spec), spec_alpha), 0.0, 0.0));
}
