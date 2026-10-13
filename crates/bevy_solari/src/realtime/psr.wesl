import bevy_core_pipeline::tonemapping::tonemapping_luminance as luminance;
import bevy_pbr::deferred::types::unpack_24bit_normal;
import bevy_pbr::render::pbr_functions::calculate_F0_dielectric;
import bevy_render::utils::octahedral_decode;
import package::realtime::bindings::{gbuffer, previous_view, view};
import package::scene::brdf::{evaluate_specular_brdf, F_AB, lobe_reflectances};
import package::scene::bindings::{MIRROR_ROUGHNESS_THRESHOLD, ResolvedMaterial};

// Shared helpers for the vendor ray-reconstruction guide passes (DLSS RR and MetalFX).
// These implement primary surface replacement (PSR): the path walk follows a chain of
// mirror reflections and reports the first non-mirror surface it finds. The vendor
// denoiser then sees that reflected surface instead of the mirror.

/// A surface is replaced only when nearly all of its reflectance leaves through the delta lobe.
const FULL_REPLACEMENT_FRACTION = 0.9;

/// The largest shift of a reflector's virtual image between two neighbor pixels, as a fraction
/// of the virtual distance. Above this limit the surface counts as too curved to replace.
const MAX_VIRTUAL_DEPTH_JITTER = 0.002;

/// G-buffer normals use 24-bit octahedral packing, so even a flat surface measures a small
/// nonzero normal change between neighbor texels.
const NORMAL_QUANTIZATION_FLOOR = 0.002;

/// Distance, in meters, that stands in for "infinitely far" when a mirror ray hits nothing.
const ENVIRONMENT_VIRTUAL_DISTANCE = 10000.0;

/// Only mirrors need specular guide tracking; rougher surfaces can reuse the surface's own data.
fn needs_specular_guide(roughness: f32) -> bool {
    return roughness <= 0.0625;
}

/// State that the path walk carries across every bounce. Anything derivable from values the
/// walk already holds is derived at the end instead of stored here.
struct PsrState {
    finished: bool,
    /// True when this pixel's surface is swapped out for the reflected one. False when the
    /// guide only describes the reflection's motion. Keyed on the primary surface; false for
    /// anything glossy.
    replace_fully: bool,
    /// The chain's cumulative transform: the product of one Householder reflection per mirror,
    /// in chain order. Stored as a rotor plus the parity of the reflection count rather than
    /// the 3x3 matrix it denotes.
    mirror_rotor: vec4<f32>,
    mirror_parity_odd: bool,
    /// Distance traveled since the primary surface, counted through every mirror. This is the
    /// reflection's own share of the path, not the total path length.
    reflection_length: f32,
    /// Product of the Fresnel term at each mirror in the chain: the factor that multiplies the
    /// reflected surface's color on its way back to the eye.
    throughput: vec3<f32>,
}

fn psr_init(world_normal: vec3<f32>, material: ResolvedMaterial) -> PsrState {
    var psr: PsrState;
    psr.finished = !needs_specular_guide(material.roughness);
    psr.replace_fully = is_delta_mirror(material);
    // The chain opens with the primary surface's own reflection, so parity starts odd.
    psr.mirror_rotor = mirror_chain_rotor(world_normal, false);
    psr.mirror_parity_odd = true;
    psr.reflection_length = 0.0;
    psr.throughput = vec3(1.0);
    return psr;
}

/// Fold one more mirror into the chain, equivalent to right-multiplying the chain's matrix by this
/// mirror's Householder reflection.
/// https://en.wikipedia.org/wiki/Householder_transformation
///
/// A product of k Householder reflections is orthogonal with determinant (-1)^k, so it is a rotation
/// when k is even, and a rotation composed with any one fixed reflection when k is odd.
fn psr_chain_mirror(psr: PsrState, plane_normal: vec3<f32>) -> PsrState {
    var chained = psr;
    chained.mirror_rotor = quaternion_multiply(psr.mirror_rotor, mirror_chain_rotor(plane_normal, psr.mirror_parity_odd));
    chained.mirror_parity_odd = !psr.mirror_parity_odd;
    return chained;
}

fn primary_surface_glossy(material: ResolvedMaterial) -> bool {
    return material.roughness > MIRROR_ROUGHNESS_THRESHOLD;
}

fn is_delta_mirror(material: ResolvedMaterial) -> bool {
    return material.roughness <= MIRROR_ROUGHNESS_THRESHOLD
        && specular_fraction(material, F_AB(material.perceptual_roughness, 1.0)) >= FULL_REPLACEMENT_FRACTION;
}

/// Share of a surface's total reflectance that leaves through the specular lobe, in 0..1.
fn specular_fraction(material: ResolvedMaterial, F_ab: vec2<f32>) -> f32 {
    let rho = lobe_reflectances(material.base_color, calculate_F0_dielectric(vec3(material.reflectance)), material, F_ab);
    let specular_luminance = luminance(rho.specular);
    return specular_luminance / max(specular_luminance + luminance(rho.diffuse), 0.0001);
}

fn psr_bounce_reflectance(wo: vec3<f32>, wi: vec3<f32>, normal: vec3<f32>, material: ResolvedMaterial, F_ab: vec2<f32>, delta: bool) -> vec3<f32> {
    if delta {
        return evaluate_specular_brdf(wo, wi, normal, material, F_ab);
    }
    return lobe_reflectances(material.base_color, calculate_F0_dielectric(vec3(material.reflectance)), material, F_ab).specular;
}

/// Angle, in radians, that the shading normal changes over one pixel in screen space.
fn reflector_curvature(pixel_id: vec2<u32>, normal: vec3<f32>) -> f32 {
    let max_pixel_id = vec2<u32>(view.main_pass_viewport.zw) - vec2(1u);
    let right = load_gbuffer_normal(min(pixel_id + vec2(1u, 0u), max_pixel_id));
    let down = load_gbuffer_normal(min(pixel_id + vec2(0u, 1u), max_pixel_id));
    let min_cos = min(dot(normal, right), dot(normal, down));
    let angle = sqrt(max(0.0, 2.0 * (1.0 - min_cos)));
    return max(0.0, angle - NORMAL_QUANTIZATION_FLOOR);
}

fn load_gbuffer_normal(pixel_id: vec2<u32>) -> vec3<f32> {
    return octahedral_decode(unpack_24bit_normal(textureLoad(gbuffer, pixel_id, 0).a));
}

fn apply_mirror_chain(psr: PsrState, v: vec3<f32>) -> vec3<f32> {
    var reflected = v;
    if psr.mirror_parity_odd {
        reflected.z = -reflected.z;
    }
    let t = 2.0 * cross(psr.mirror_rotor.xyz, reflected);
    return reflected + psr.mirror_rotor.w * t + cross(psr.mirror_rotor.xyz, t);
}

/// The rotation half of `H(n) * F` at even parity, or of `F * H(n)` at odd, where `H(n)` is the
/// Householder reflection about `plane_normal` and `F` the fixed z-axis one.
fn mirror_chain_rotor(plane_normal: vec3<f32>, parity_odd: bool) -> vec4<f32> {
    if parity_odd {
        return vec4(plane_normal.y, -plane_normal.x, 0.0, plane_normal.z);
    }
    return vec4(-plane_normal.y, plane_normal.x, 0.0, plane_normal.z);
}

fn quaternion_multiply(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

fn calculate_motion_vector(world_position: vec3<f32>, previous_world_position: vec3<f32>) -> vec2<f32> {
    let clip_position_t = view.unjittered_clip_from_world * vec4(world_position, 1.0);
    let clip_position = clip_position_t.xy / clip_position_t.w;
    let previous_clip_position_t = previous_view.unjittered_clip_from_world * vec4(previous_world_position, 1.0);
    let previous_clip_position = previous_clip_position_t.xy / previous_clip_position_t.w;
    // Motion vectors are UV-space offsets in [-1, 1], from one corner to the diagonally-opposite one.
    // A clip-space diagonal difference is in [-2, 2], so scale by 0.5, and flip y since V goes down
    // where clip-space y goes up.
    return (clip_position - previous_clip_position) * vec2(0.5, -0.5);
}

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
