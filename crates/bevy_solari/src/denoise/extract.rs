use super::{prepare::ViewDenoiseResources, Svgf};
use bevy_camera::Camera;
use bevy_ecs::system::{Commands, ResMut};
use bevy_render::{sync_world::RenderEntity, MainWorld};

pub fn extract_svgf(mut main_world: ResMut<MainWorld>, mut commands: Commands) {
    let mut cameras_3d = main_world.query::<(RenderEntity, &Camera, Option<&Svgf>)>();

    for (entity, camera, svgf) in cameras_3d.iter_mut(&mut main_world) {
        let mut entity_commands = commands
            .get_entity(entity)
            .expect("Camera entity wasn't synced.");
        if let Some(svgf) = svgf.filter(|_| camera.is_active) {
            entity_commands.insert(svgf.clone());
        } else {
            entity_commands.remove::<(Svgf, ViewDenoiseResources)>();
        }
    }
}
