//! Hotbar: a 9-slot quick bar with a selected index and the block each slot
//! holds. Simple and data-driven; the inventory/crafting system builds on this
//! later.

use voxel_core::BlockId;
use voxel_world::BlockRegistry;

/// Number of hotbar slots (Minecraft-like).
pub const HOTBAR_SLOTS: usize = 9;

#[derive(Clone, Debug)]
pub struct Hotbar {
    slots: [BlockId; HOTBAR_SLOTS],
    pub selected: usize,
}

impl Default for Hotbar {
    fn default() -> Self {
        // A sensible creative-style starting set.
        let slots = [BlockId::AIR; HOTBAR_SLOTS];
        // Filled by `populate_defaults` once a registry is available.
        Self { slots, selected: 0 }
    }
}

impl Hotbar {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fill the hotbar with a default palette from the registry.
    pub fn populate_defaults(&mut self, reg: &BlockRegistry) {
        let names = [
            "grass",
            "dirt",
            "stone",
            "cobblestone",
            "planks",
            "wood",
            "torch",
            "sand",
            "bucket",
        ];
        for (i, name) in names.iter().enumerate() {
            if let Some(id) = reg.id_of(name) {
                self.slots[i] = id;
            }
        }
    }

    pub fn slot(&self, i: usize) -> BlockId {
        self.slots[i.min(HOTBAR_SLOTS - 1)]
    }

    pub fn set_slot(&mut self, i: usize, id: BlockId) {
        if i < HOTBAR_SLOTS {
            self.slots[i] = id;
        }
    }

    /// Currently selected block (what the player places on right-click).
    pub fn selected_block(&self) -> BlockId {
        self.slots[self.selected.min(HOTBAR_SLOTS - 1)]
    }

    /// Select a slot by index (0..=8). Out-of-range values are ignored.
    pub fn select(&mut self, i: usize) {
        if i < HOTBAR_SLOTS {
            self.selected = i;
        }
    }

    /// Cycle selection by a delta (mouse wheel).
    pub fn cycle(&mut self, delta: i32) {
        let n = HOTBAR_SLOTS as i32;
        let mut s = self.selected as i32 + delta;
        s = ((s % n) + n) % n;
        self.selected = s as usize;
    }

    pub fn slots(&self) -> &[BlockId; HOTBAR_SLOTS] {
        &self.slots
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_world::BlockRegistry;

    #[test]
    fn populate_defaults_fills_all_slots() {
        let mut hb = Hotbar::new();
        let reg = BlockRegistry::with_builtins();
        hb.populate_defaults(&reg);
        // All 9 slots should have non-air blocks (all builtin names exist).
        for i in 0..HOTBAR_SLOTS {
            assert!(!hb.slot(i).is_air(), "slot {i} should be filled");
        }
    }

    #[test]
    fn select_within_range() {
        let mut hb = Hotbar::new();
        hb.select(3);
        assert_eq!(hb.selected, 3);
    }

    #[test]
    fn select_out_of_range_ignored() {
        let mut hb = Hotbar::new();
        hb.select(99);
        assert_eq!(hb.selected, 0);
    }

    #[test]
    fn cycle_wraps_forward() {
        let mut hb = Hotbar::new();
        hb.select(8);
        hb.cycle(1);
        assert_eq!(hb.selected, 0);
    }

    #[test]
    fn cycle_wraps_backward() {
        let mut hb = Hotbar::new();
        hb.select(0);
        hb.cycle(-1);
        assert_eq!(hb.selected, 8);
    }

    #[test]
    fn set_slot_valid() {
        let mut hb = Hotbar::new();
        hb.set_slot(2, BlockId(5));
        assert_eq!(hb.slot(2), BlockId(5));
    }

    #[test]
    fn set_slot_out_of_range_ignored() {
        let mut hb = Hotbar::new();
        let original = hb.slot(0);
        hb.set_slot(99, BlockId(5));
        assert_eq!(hb.slot(0), original);
    }
}
