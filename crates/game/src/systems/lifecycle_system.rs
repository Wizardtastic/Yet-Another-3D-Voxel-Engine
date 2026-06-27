//! Entity lifecycle system: no-op placeholder.
//
// The project doesn't currently expose a `Health` or `Timer` component, so
// this system is intentionally empty. When those land, this is the home for
// death/despawn and timer ticks; until then it just keeps the schedule
// reference resolvable.

use voxel_ecs::World;

pub fn lifecycle_system(_world: &mut World, _dt: f32) {
    // Intentionally empty.
}
