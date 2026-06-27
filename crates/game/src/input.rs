//! Resolved input state. The engine translates raw `winit` events into this
//! struct each frame; gameplay reads from it. Keeping it data-only makes it
//! trivial to record/replay and to serialize over the network later.

use std::collections::HashSet;

/// Logical actions the player can hold (movement) or trigger (toggles).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    // Movement (held)
    Forward,
    Back,
    Left,
    Right,
    Jump,
    Sprint,
    Sneak,
    // Toggles / one-shot (pressed)
    DebugOverlay,
    Wireframe,
    Fly,
    Chat,
    Screenshot,
    Pause,
    RenderDistanceUp,
    RenderDistanceDown,
    // Hotbar slot selection
    Slot1,
    Slot2,
    Slot3,
    Slot4,
    Slot5,
    Slot6,
    Slot7,
    Slot8,
    Slot9,
    // Undo/Redo
    Undo,
    Redo,
    // Inventory
    BlockPicker,
    // Profiling / Debug
    Profiler,
    ChunkDebug,
}

/// One-shot requests for the current frame (consumed after handling).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Clicks {
    /// Left mouse button pressed this frame (break block / attack).
    pub left: bool,
    /// Right mouse button pressed this frame (place block / use).
    pub right: bool,
}

#[derive(Clone, Debug, Default)]
pub struct InputState {
    pub held: HashSet<Action>,
    pub just_pressed: HashSet<Action>,
    pub clicks: Clicks,
    /// Mouse-motion delta accumulated since last frame (pixels).
    pub mouse_delta: (f32, f32),
    /// Hotbar slot change requested this frame (1..=9 -> index 0..=8), if any.
    pub hotbar_select: Option<usize>,
}

impl InputState {
    pub fn held(&self, a: Action) -> bool {
        self.held.contains(&a)
    }

    pub fn press_action(&mut self, action: Action) {
        self.held.insert(action);
        self.just_pressed.insert(action);
    }

    pub fn take_just_pressed(&mut self) -> HashSet<Action> {
        std::mem::take(&mut self.just_pressed)
    }

    /// Consume and return the accumulated mouse delta, zeroing it.
    pub fn take_mouse_delta(&mut self) -> (f32, f32) {
        let d = self.mouse_delta;
        self.mouse_delta = (0.0, 0.0);
        d
    }

    /// Consume the click flags.
    pub fn take_clicks(&mut self) -> Clicks {
        let c = self.clicks;
        self.clicks = Clicks::default();
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn held_returns_correctly() {
        let mut s = InputState::default();
        assert!(!s.held(Action::Forward));
        s.held.insert(Action::Forward);
        assert!(s.held(Action::Forward));
        assert!(!s.held(Action::Back));
    }

    #[test]
    fn take_mouse_delta_resets() {
        let mut s = InputState::default();
        s.mouse_delta = (1.0, 2.0);
        let d = s.take_mouse_delta();
        assert_eq!(d, (1.0, 2.0));
        assert_eq!(s.mouse_delta, (0.0, 0.0));
    }

    #[test]
    fn take_clicks_resets() {
        let mut s = InputState::default();
        s.clicks.left = true;
        s.clicks.right = true;
        let c = s.take_clicks();
        assert!(c.left);
        assert!(c.right);
        let c2 = s.take_clicks();
        assert!(!c2.left);
        assert!(!c2.right);
    }

    #[test]
    fn default_input_state_is_empty() {
        let s = InputState::default();
        assert!(s.held.is_empty());
        assert_eq!(s.mouse_delta, (0.0, 0.0));
        assert!(s.hotbar_select.is_none());
    }
}
