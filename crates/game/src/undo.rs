//! Undo/redo system for block edits.
//!
//! Records block changes so the player can undo (Ctrl+Z) and redo (Ctrl+Y).
//! Each action stores the list of block positions and their previous values.

use std::collections::VecDeque;

const MAX_UNDO: usize = 100;
const MAX_EDITS_PER_ACTION: usize = 65536;

/// A single block change: position and the block IDs before and after.
#[derive(Clone, Debug)]
pub struct BlockEdit {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub old_block: u16,
    pub new_block: u16,
}

/// A group of block changes that can be undone/redone atomically.
#[derive(Clone, Debug)]
pub struct EditAction {
    pub edits: Vec<BlockEdit>,
}

#[derive(Default)]
pub struct UndoRedoState {
    undo_stack: VecDeque<EditAction>,
    redo_stack: VecDeque<EditAction>,
}

impl UndoRedoState {
    /// Record a new action (pushes to undo stack, clears redo stack).
    pub fn push(&mut self, mut action: EditAction) {
        if action.edits.is_empty() {
            return;
        }
        if action.edits.len() > MAX_EDITS_PER_ACTION {
            action.edits.truncate(MAX_EDITS_PER_ACTION);
        }
        self.undo_stack.push_front(action);
        while self.undo_stack.len() > MAX_UNDO {
            self.undo_stack.pop_back();
        }
        self.redo_stack.clear();
    }

    /// Pop the most recent undo action, if any.
    pub fn pop_undo(&mut self) -> Option<EditAction> {
        let action = self.undo_stack.pop_front()?;
        let returned = action.clone();
        self.redo_stack.push_front(action);
        Some(returned)
    }

    /// Pop the most recent redo action, if any.
    pub fn pop_redo(&mut self) -> Option<EditAction> {
        let action = self.redo_stack.pop_front()?;
        let returned = action.clone();
        self.undo_stack.push_front(action);
        Some(returned)
    }

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }

    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edit(x: i32, y: i32, z: i32, old: u16, new: u16) -> BlockEdit {
        BlockEdit {
            x,
            y,
            z,
            old_block: old,
            new_block: new,
        }
    }

    #[test]
    fn push_and_pop_undo() {
        let mut state = UndoRedoState::default();
        state.push(EditAction {
            edits: vec![edit(0, 0, 0, 1, 0), edit(1, 0, 0, 2, 0)],
        });
        assert!(state.can_undo());
        assert!(!state.can_redo());
        let action = state.pop_undo().unwrap();
        assert_eq!(action.edits.len(), 2);
        assert!(!state.can_undo());
        assert!(state.can_redo());
    }

    #[test]
    fn pop_redo_restores_undo() {
        let mut state = UndoRedoState::default();
        state.push(EditAction {
            edits: vec![edit(0, 0, 0, 1, 0)],
        });
        state.pop_undo();
        let action = state.pop_redo().unwrap();
        assert_eq!(action.edits[0].old_block, 1);
        assert!(state.can_undo());
        assert!(!state.can_redo());
    }

    #[test]
    fn new_push_clears_redo() {
        let mut state = UndoRedoState::default();
        state.push(EditAction {
            edits: vec![edit(0, 0, 0, 1, 0)],
        });
        state.pop_undo();
        assert!(state.can_redo());
        state.push(EditAction {
            edits: vec![edit(1, 0, 0, 2, 0)],
        });
        assert!(!state.can_redo());
    }

    #[test]
    fn empty_action_ignored() {
        let mut state = UndoRedoState::default();
        state.push(EditAction { edits: vec![] });
        assert!(!state.can_undo());
    }

    #[test]
    fn undo_limit() {
        let mut state = UndoRedoState::default();
        for i in 0..150 {
            state.push(EditAction {
                edits: vec![edit(i, 0, 0, 1, 0)],
            });
        }
        assert!(state.undo_count() <= MAX_UNDO);
    }

    #[test]
    fn pop_empty_returns_none() {
        let mut state = UndoRedoState::default();
        assert!(state.pop_undo().is_none());
        assert!(state.pop_redo().is_none());
    }
}
