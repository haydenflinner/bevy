import bevy_pbr::render::pbr_functions::{calculate_diffuse_color, calculate_F0};
import package::realtime::gbuffer_utils::gpixel_resolve;
import package::realtime::bindings::{gbuffer, depth_buffer, motion_vectors, view, diffuse_albedo, specular_albedo, metalfx_normal, metalfx_roughness, metalfx_rr_depth, specular_hit_distance};
import package::realtime::psr::{apply_mirror_chain, calculate_motion_vector, env_brdf_approx2, ENVIRONMENT_VIRTUAL_DISTANCE, MAX_VIRTUAL_DEPTH_JITTER, PsrState, reflector_curvature, specular_fraction};
import package::scene::bindings::{ResolvedMaterial, ResolvedRayHitFull};

@compute @workgroup_size(8, 8, 1)
fn resolve_metalfx_rr_textures(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_id = global_id.xy;
    if any(pixel_id >= vec2u(view.main_pass_viewport.zw)) { return; }

    let depth = textureLoad(depth_buffer, global_id.xy, 0);

    textureStore(metalfx_rr_depth, pixel_id, vec4(depth));

    if depth == 0.0 {
        textureStore(specular_hit_distance, pixel_id, vec4(ENVIRONMENT_VIRTUAL_DISTANCE, vec3(0.0)));
        textureStore(diffuse_albedo, pixel_id, vec4(0.0));
        textureStore(specular_albedo, pixel_id, vec4(0.5));
        textureStore(metalfx_normal, pixel_id, vec4(0.0, 0.0, 1.0, 0.0));
        textureStore(metalfx_roughness, pixel_id, vec4(1.0));
        return;
    }

    let surface = gpixel_resolve(textureLoad(gbuffer, pixel_id, 0), depth, pixel_id, view.main_pass_viewport.zw, view.world_from_clip);
    let F0 = calculate_F0(surface.material.base_color, surface.material.metallic, vec3(surface.material.reflectance));
    let wo = normalize(view.world_position - surface.world_position);

    textureStore(diffuse_albedo, pixel_id, vec4(calculate_diffuse_color(surface.material.base_color, surface.material.metallic, 0.0, 0.0), 0.0));
    textureStore(specular_albedo, pixel_id, vec4(env_brdf_approx2(F0, surface.material.roughness, surface.world_normal, wo), 0.0));
    textureStore(metalfx_normal, pixel_id, vec4(surface.world_normal, 0.0));
    textureStore(metalfx_roughness, pixel_id, vec4(surface.material.perceptual_roughness));
    textureStore(specular_hit_distance, pixel_id, vec4(0.0));
}

/// MetalFX reprojects specular history off the specular hit distance, which the path walk
/// writes after this resolve pass. A mirror that misses geometry gets the far distance so the
/// environment map can be reprojected along the reflection ray.
fn psr_environment_miss(pixel_id: vec2<u32>, primary_wo: vec3<f32>) {
    textureStore(specular_hit_distance, pixel_id, vec4(ENVIRONMENT_VIRTUAL_DISTANCE, vec3(0.0)));
}

/// Primary surface replacement for perfect mirrors. The reflection chain has been followed to its
/// first non-mirror hit; write that surface's attributes, reflected into the mirror's virtual space,
/// to the MetalFX guide buffers so the denoiser treats this pixel as directly seeing it.
/// https://developer.nvidia.com/blog/rendering-perfect-reflections-and-refractions-in-path-traced-games/#primary_surface_replacement
fn replace_primary_surface(
    pixel_id: vec2<u32>,
    psr: PsrState,
    ray_hit: ResolvedRayHitFull,
    primary_material: ResolvedMaterial,
    primary_world_position: vec3<f32>,
    primary_normal: vec3<f32>,
    primary_wo: vec3<f32>,
    primary_F_ab: vec2<f32>,
) {
    let primary_distance = length(primary_world_position - view.world_position);
    let path_length = primary_distance + psr.reflection_length;

    // The mirror is too curved to track the reflection. The path walk still writes the first
    // bounce's specular hit distance, which is the best guide available for this pixel.
    if psr.replace_fully && 2.0 * reflector_curvature(pixel_id, primary_normal) * psr.reflection_length > MAX_VIRTUAL_DEPTH_JITTER * path_length {
        return;
    }

    let virtual_position = primary_world_position - (primary_wo * psr.reflection_length);
    let world_motion = ray_hit.previous_frame_world_position - ray_hit.world_position;
    let virtual_previous_position = virtual_position + apply_mirror_chain(psr, world_motion);
    let specular_motion_vector = calculate_motion_vector(virtual_position, virtual_previous_position);

    if !psr.replace_fully {
        return;
    }

    let wo = normalize(view.world_position - virtual_position);
    let virtual_normal = normalize(apply_mirror_chain(psr, ray_hit.world_normal));

    let primary_depth = textureLoad(depth_buffer, pixel_id, 0);
    textureStore(metalfx_rr_depth, pixel_id, vec4(primary_depth * primary_distance / path_length));
    @if(MOTION_VECTORS_READ_WRITE)
    textureStore(motion_vectors, pixel_id, vec4(specular_motion_vector, vec2(0.0)));

    let reflected_diffuse = calculate_diffuse_color(ray_hit.material.base_color, ray_hit.material.metallic, 0.0, 0.0);
    let reflected_F0 = calculate_F0(ray_hit.material.base_color, ray_hit.material.metallic, vec3(ray_hit.material.reflectance));
    let reflected_specular = env_brdf_approx2(reflected_F0, ray_hit.material.roughness, virtual_normal, wo);

    let own_diffuse = calculate_diffuse_color(primary_material.base_color, primary_material.metallic, 0.0, 0.0);
    let own_F0 = calculate_F0(primary_material.base_color, primary_material.metallic, vec3(primary_material.reflectance));
    let own_specular = env_brdf_approx2(own_F0, primary_material.roughness, primary_normal, primary_wo);

    let specular_fraction = specular_fraction(primary_material, primary_F_ab);

    textureStore(diffuse_albedo, pixel_id, vec4(psr.throughput * reflected_diffuse + own_diffuse, 0.0));
    textureStore(specular_albedo, pixel_id, vec4(mix(own_specular, psr.throughput * reflected_specular, specular_fraction), 0.0));
    textureStore(metalfx_normal, pixel_id, vec4(virtual_normal, 0.0));
    textureStore(metalfx_roughness, pixel_id, vec4(ray_hit.material.perceptual_roughness));
}
