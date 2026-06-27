//! `voxel-ecs` — archetype-based Entity Component System for the
//! voxel engine.
//!
//! Public surface:
//! - [`World`] — owns all entities, archetypes, and resources
//! - [`Entity`] — unique entity identifier
//! - [`Component`] — marker trait for ECS components
//! - [`Bundle`] — spawn an entity with multiple components at once
//! - [`Query`] / [`QueryIter`] — iterate entities matching a component
//!   pattern
//! - [`System`] / [`SystemSchedule`] / [`FnSystem`] — ordered system
//!   execution
//! - [`Resources`] — singleton data accessible to all systems
//! - [`ArchetypeId`] — identifier for an archetype

pub mod archetype;
pub mod component;
pub mod entity;
pub mod query;
pub mod resources;
pub mod schedule;
pub mod world;

pub use archetype::ArchetypeId;
pub use component::{Bundle, Component};
pub use entity::Entity;
pub use query::{Query, QueryIter};
pub use resources::Resources;
pub use schedule::{FnSystem, System, SystemSchedule};
pub use world::World;

// `impl_bundle_for_tuple!` is `#[macro_export]` and is therefore already
// available at the crate root as `voxel_ecs::impl_bundle_for_tuple!`.
// No re-export needed.
