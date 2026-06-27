//! Component and Bundle traits.

use crate::entity::Entity;
use crate::world::World;

/// Marker trait for ECS components. All components must be:
/// - `Send + Sync` (so systems can run in parallel)
/// - `'static` (no borrowed data)
/// - `Sized` (stored in dense columns)
pub trait Component: Send + Sync + 'static {}

impl<T: Send + Sync + 'static> Component for T {}

/// A bundle of components spawned together.
///
/// `Bundle` is implemented for tuples of `Component` types up to length
/// 8. Spawning an entity with `world.spawn((A, B, C))` adds `A`, `B`,
/// and `C` to the new entity in order, possibly moving it across
/// archetypes. To spawn a single component, use a 1-tuple:
/// `world.spawn((A,))`.
pub trait Bundle: 'static {
    /// Add the components in this bundle to the given (typically just
    /// allocated) entity.
    fn add_to(self, world: &mut World, entity: Entity);
}

/// Implement [`Bundle`] for a tuple of components by calling
/// [`World::set`](crate::World::set) on each element in order.
#[macro_export]
macro_rules! impl_bundle_for_tuple {
    ($($t:ident),+) => {
        impl<$($t: $crate::Component),+> $crate::Bundle for ($($t,)+) {
            #[allow(non_snake_case)]
            fn add_to(self, world: &mut $crate::World, entity: $crate::Entity) {
                #[allow(non_snake_case)]
                let ($($t,)+) = self;
                $(world.set(entity, $t);)+
            }
        }
    };
}

impl_bundle_for_tuple!(A);
impl_bundle_for_tuple!(A, B);
impl_bundle_for_tuple!(A, B, C);
impl_bundle_for_tuple!(A, B, C, D);
impl_bundle_for_tuple!(A, B, C, D, E);
impl_bundle_for_tuple!(A, B, C, D, E, F);
impl_bundle_for_tuple!(A, B, C, D, E, F, G);
impl_bundle_for_tuple!(A, B, C, D, E, F, G, H);
