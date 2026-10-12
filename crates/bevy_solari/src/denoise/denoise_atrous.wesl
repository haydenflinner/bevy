import bevy_pbr::deferred::types::unpack_24bit_normal;
import bevy_render::utils::octahedral_decode;
import bevy_render::view::{View, depth_ndc_to_view_z};

@group(0) @binding(0) var atrous_src: texture_2d<f32>;
@group(0) @binding(1) var atrous_dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var gbuffer: texture_2d<u32>;
@group(0) @binding(3) var depth_buffer: texture_depth_2d;
@group(0) @binding(4) var moments: texture_2d<f32>;
@group(0) @binding(5) var<uniform> view: View;
@group(0) @binding(6) var<uniform> params: DenoiseParams;
@group(0) @binding(7) var spec_src: texture_2d<f32>;

struct DenoiseParams {
    max_history: f32,
    luma_sigma: f32,
    depth_sigma: f32,
    normal_sigma: f32,
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

fn linear_depth(depth: f32) -> f32 {
    return -depth_ndc_to_view_z(depth, view.clip_from_view, view.view_from_clip);
}

fn normal_at(p: vec2<u32>) -> vec3<f32> {
    return octahedral_decode(unpack_24bit_normal(textureLoad(gbuffer, p, 0).a));
}

fn perceptual_roughness_at(p: vec2<u32>) -> f32 {
    return unpack4x8unorm(textureLoad(gbuffer, p, 0).r).a;
}

/// One 3x3 edge-stopping à-trous filter pass at `stride` texel spacing. Taps are weighted
/// by surface similarity (normal, linear depth) and luminance similarity scaled by the
/// temporally accumulated variance, so confident history filters harder.
///
/// When `specular` is set, far taps are additionally damped by how smooth the surface
/// is: a mirror's specular radiance changes too fast across space for wide filtering,
/// while a rough surface's specular lobe is almost diffuse and can take the full blur.
fn atrous_impl(pixel_id: vec2<u32>, stride: i32, specular: bool) -> vec4<f32> {
    let size = view.main_pass_viewport.zw;
    if any(vec2f(pixel_id) >= size) { return vec4(0.0); }

    let c0 = textureLoad(atrous_src, pixel_id, 0).rgb;
    let depth0 = textureLoad(depth_buffer, pixel_id, 0);

    // Sky: no surface to edge-stop against, just a small blur of other sky taps.
    if depth0 == 0.0 {
        var sum = c0;
        var wsum = 1.0;
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                if dx == 0 && dy == 0 { continue; }
                let p = vec2i(pixel_id) + vec2i(dx, dy) * stride;
                if all(p >= vec2i(0)) && all(vec2f(p) < size)
                    && textureLoad(depth_buffer, vec2u(p), 0) == 0.0 {
                    sum += textureLoad(atrous_src, vec2u(p), 0).rgb;
                    wsum += 1.0;
                }
            }
        }
        return vec4(sum / wsum, 1.0);
    }

    let n0 = normal_at(pixel_id);
    let z0 = linear_depth(depth0);
    let m = textureLoad(moments, pixel_id, 0).rg;
    let variance = max(m.y - m.x * m.x, 0.0);
    let phi_l = params.luma_sigma * sqrt(variance) + 0.05;
    let l0 = luma(c0);

    // Specular lobe width: how much far taps should still count at this stride.
    // Roughness 1 keeps the full kernel; a perfect mirror keeps only the center.
    var lobe_w = 1.0;
    if specular {
        let r = perceptual_roughness_at(pixel_id);
        lobe_w = exp(-f32(stride * stride) * (1.0 - r) * (1.0 - r) * 0.1);
    }

    // Linear depth gradient from immediate neighbors, used to scale the depth weight.
    let p1 = min(pixel_id + vec2(1u, 0u), vec2u(size) - 1u);
    let p2 = min(pixel_id + vec2(0u, 1u), vec2u(size) - 1u);
    let gz = vec2(
        abs(linear_depth(textureLoad(depth_buffer, p1, 0)) - z0),
        abs(linear_depth(textureLoad(depth_buffer, p2, 0)) - z0),
    );

    var sum = c0;
    var wsum = 1.0;
    // 3x3 binomial kernel, center tap accounted for above.
    let kw = array<f32, 9>(1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if dx == 0 && dy == 0 { continue; }
            let p = vec2i(pixel_id) + vec2i(dx, dy) * stride;
            if any(p < vec2i(0)) || any(vec2f(p) >= size) { continue; }

            let d = textureLoad(depth_buffer, vec2u(p), 0);
            if d == 0.0 { continue; }

            var w = kw[(dy + 1) * 3 + dx + 1] / 16.0;
            w *= pow(max(dot(n0, normal_at(vec2u(p))), 0.0), params.normal_sigma);
            let dz = abs(linear_depth(d) - z0);
            w *= exp(-dz / (params.depth_sigma
                * (gz.x * f32(dx * dx) + gz.y * f32(dy * dy)) * f32(stride) + 1e-2));
            w *= exp(-abs(luma(textureLoad(atrous_src, vec2u(p), 0).rgb) - l0) / phi_l);
            w *= lobe_w;

            sum += textureLoad(atrous_src, vec2u(p), 0).rgb * w;
            wsum += w;
        }
    }
    return vec4(sum / wsum, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn atrous_1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    textureStore(atrous_dst, global_id.xy, atrous_impl(global_id.xy, 1, false));
}

@compute @workgroup_size(8, 8, 1)
fn atrous_2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    textureStore(atrous_dst, global_id.xy, atrous_impl(global_id.xy, 2, false));
}

@compute @workgroup_size(8, 8, 1)
fn atrous_4(@builtin(global_invocation_id) global_id: vec3<u32>) {
    textureStore(atrous_dst, global_id.xy, atrous_impl(global_id.xy, 4, false));
}

@compute @workgroup_size(8, 8, 1)
fn atrous_8(@builtin(global_invocation_id) global_id: vec3<u32>) {
    textureStore(atrous_dst, global_id.xy, atrous_impl(global_id.xy, 8, false));
}

@compute @workgroup_size(8, 8, 1)
fn atrous_16(@builtin(global_invocation_id) global_id: vec3<u32>) {
    textureStore(atrous_dst, global_id.xy, atrous_impl(global_id.xy, 16, false));
}

@compute @workgroup_size(8, 8, 1)
fn atrous_spec_1(@builtin(global_invocation_id) global_id: vec3<u32>) {
    textureStore(atrous_dst, global_id.xy, atrous_impl(global_id.xy, 1, true));
}

@compute @workgroup_size(8, 8, 1)
fn atrous_spec_2(@builtin(global_invocation_id) global_id: vec3<u32>) {
    textureStore(atrous_dst, global_id.xy, atrous_impl(global_id.xy, 2, true));
}

@compute @workgroup_size(8, 8, 1)
fn atrous_spec_4(@builtin(global_invocation_id) global_id: vec3<u32>) {
    textureStore(atrous_dst, global_id.xy, atrous_impl(global_id.xy, 4, true));
}

@compute @workgroup_size(8, 8, 1)
fn atrous_spec_8(@builtin(global_invocation_id) global_id: vec3<u32>) {
    textureStore(atrous_dst, global_id.xy, atrous_impl(global_id.xy, 8, true));
}

@compute @workgroup_size(8, 8, 1)
fn atrous_spec_16(@builtin(global_invocation_id) global_id: vec3<u32>) {
    textureStore(atrous_dst, global_id.xy, atrous_impl(global_id.xy, 16, true));
}

/// Final pass: recombine the streams — re-modulate the filtered diffuse lighting
/// by the surface's diffuse albedo and add the filtered specular radiance.
/// Writes to the view target.
@compute @workgroup_size(8, 8, 1)
fn writeback(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_id = global_id.xy;
    let size = view.main_pass_viewport.zw;
    if any(vec2f(pixel_id) >= size) { return; }

    var rgb = textureLoad(atrous_src, pixel_id, 0).rgb
        + textureLoad(spec_src, pixel_id, 0).rgb;
    if textureLoad(depth_buffer, pixel_id, 0) != 0.0 {
        let gpixel = textureLoad(gbuffer, pixel_id, 0);
        let base_rough = unpack4x8unorm(gpixel.r);
        let base_color = pow(base_rough.rgb, vec3(2.2));
        let metallic = unpack4x8unorm(gpixel.b).g;
        let diffuse_albedo = base_color * (1.0 - metallic);
        rgb = textureLoad(atrous_src, pixel_id, 0).rgb
            * max(luma(diffuse_albedo), 0.05)
            + textureLoad(spec_src, pixel_id, 0).rgb;
    }
    textureStore(atrous_dst, pixel_id, vec4(rgb, 1.0));
}
