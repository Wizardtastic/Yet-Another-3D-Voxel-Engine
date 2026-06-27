//! Entity identifiers and their internal location.

/// A unique entity identifier.
///
/// 64 bits: 32-bit index + 32-bit generation. The generation is bumped every
/// time an entity slot is recycled, so stale [`Entity`] handles from prior
/// lifetimes can be detected and ignored.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct Entity {
    pub index: u32,
    pub generation: u32,
}

impl Entity {
    /// Sentinel "null" entity. Has the maximum representable index so any
    /// real entity is distinguishable from it.
    pub const NULL: Entity = Entity { index: u32::MAX, generation: 0 };

    pub fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    pub fn is_null(self) -> bool {
        self.index == u32::MAX
    }
}

/// Internal: where an entity currently lives in the world.
#[derive(Copy, Clone, Debug)]
pub(crate) struct EntityLocation {
    pub archetype: u32,
    pub index: u32,
}

impl EntityLocation {
    /// Sentinel archetype used for entities that exist but have no components
    /// yet (i.e. have been allocated but never had a component set on them).
    pub(crate) const EMPTY: u32 = u32::MAX;
}
