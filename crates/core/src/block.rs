//! Block identity. Properties (solidity, transparency, textures) live in the
//! data-driven `BlockRegistry` (`voxel-world`/`voxel-assets`); this type is the
//! lightweight handle stored 4096-per-chunk, so it stays a `u16` wrapper.

/// A block type identifier. `0` is reserved for air.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
#[repr(transparent)]
pub struct BlockId(pub u16);

impl BlockId {
    /// Air / empty space.
    pub const AIR: Self = Self(0);

    #[inline]
    pub const fn new(id: u16) -> Self {
        Self(id)
    }

    #[inline]
    pub const fn raw(self) -> u16 {
        self.0
    }

    #[inline]
    pub const fn is_air(self) -> bool {
        self.0 == 0
    }
}

impl From<u16> for BlockId {
    #[inline]
    fn from(v: u16) -> Self {
        Self(v)
    }
}
impl From<BlockId> for u16 {
    #[inline]
    fn from(v: BlockId) -> Self {
        v.0
    }
}
