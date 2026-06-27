//! US QWERTY key-to-character lookup used by the chat input.
//!
//! Kept separate so `lib.rs` doesn't have to scroll through a 50-line
//! const table every time it focuses on the application loop.
//!
//! Returns `Option<char>` so unreachable keys (arrows, modifiers, function
//! keys) cleanly no-op rather than emit a placeholder.

use winit::keyboard::KeyCode;

/// US QWERTY layout: printable physical key → character. Lookup is O(n) but the
/// array is tiny (~40 entries); no need for a perfect-hash or `phf` map.
const PHYSICAL_KEY_CHARS: &[(KeyCode, char)] = &[
    (KeyCode::KeyA, 'a'), (KeyCode::KeyB, 'b'), (KeyCode::KeyC, 'c'),
    (KeyCode::KeyD, 'd'), (KeyCode::KeyE, 'e'), (KeyCode::KeyF, 'f'),
    (KeyCode::KeyG, 'g'), (KeyCode::KeyH, 'h'), (KeyCode::KeyI, 'i'),
    (KeyCode::KeyJ, 'j'), (KeyCode::KeyK, 'k'), (KeyCode::KeyL, 'l'),
    (KeyCode::KeyM, 'm'), (KeyCode::KeyN, 'n'), (KeyCode::KeyO, 'o'),
    (KeyCode::KeyP, 'p'), (KeyCode::KeyQ, 'q'), (KeyCode::KeyR, 'r'),
    (KeyCode::KeyS, 's'), (KeyCode::KeyT, 't'), (KeyCode::KeyU, 'u'),
    (KeyCode::KeyV, 'v'), (KeyCode::KeyW, 'w'), (KeyCode::KeyX, 'x'),
    (KeyCode::KeyY, 'y'), (KeyCode::KeyZ, 'z'),
    (KeyCode::Digit0, '0'), (KeyCode::Digit1, '1'), (KeyCode::Digit2, '2'),
    (KeyCode::Digit3, '3'), (KeyCode::Digit4, '4'), (KeyCode::Digit5, '5'),
    (KeyCode::Digit6, '6'), (KeyCode::Digit7, '7'), (KeyCode::Digit8, '8'),
    (KeyCode::Digit9, '9'),
    (KeyCode::Period, '.'), (KeyCode::Comma, ','), (KeyCode::Semicolon, ';'),
    (KeyCode::Slash, '/'), (KeyCode::Backslash, '\\'), (KeyCode::Minus, '-'),
    (KeyCode::Equal, '='), (KeyCode::BracketLeft, '['), (KeyCode::BracketRight, ']'),
    (KeyCode::Backquote, '`'), (KeyCode::Quote, '\''),
];

/// Map a physical key code to a character (US QWERTY). Returns `None` for
/// keys that don't produce a printable character (arrow keys, function keys,
/// modifiers, etc.).
pub(crate) fn physical_key_to_char(code: KeyCode) -> Option<char> {
    // Linear search — the table is ~45 entries and `KeyCode` is not
    // `Ord`, so a perfect-hash or sorted-table approach would need
    // additional scaffolding. Linear is plenty fast here.
    PHYSICAL_KEY_CHARS
        .iter()
        .find(|(k, _)| *k == code)
        .map(|&(_, c)| c)
}
