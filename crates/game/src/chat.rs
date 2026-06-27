use glam::Vec3;
use std::collections::VecDeque;

const MAX_MESSAGES: usize = 64;
const MAX_HISTORY: usize = 50;

/// All recognized commands and their sub-commands for autocomplete.
const COMMANDS: &[&str] = &[
    "/tp",
    "/time set",
    "/time speed",
    "/give",
    "/setblock",
    "/fill",
    "/gamemode",
    "/pos",
    "/chunk",
    "/fps",
    "/reload",
    "/clear",
    "/save",
    "/load",
    "/copy",
    "/paste",
    "/help",
];

pub enum CommandResult {
    Teleport(Vec3),
    SetTime(f64),
    TimeSpeed(f64),
    Give(String, i32),
    SetBlock(i32, i32, i32, String),
    Fill(i32, i32, i32, i32, i32, i32, String),
    Gamemode(String),
    Position,
    ChunkInfo,
    Fps,
    Reload,
    Clear,
    Save(String),
    Load(String),
    Copy(i32, i32, i32, i32, i32, i32),
    Paste,
    Help,
    Unknown(String),
    Empty,
}

#[derive(Default)]
pub struct ChatState {
    pub open: bool,
    pub input_buf: String,
    pub messages: VecDeque<String>,
    history: VecDeque<String>,
    history_index: Option<usize>,
}

impl ChatState {
    pub fn open(&mut self) {
        self.open = true;
        self.input_buf.clear();
        self.history_index = None;
    }

    pub fn close(&mut self) {
        self.open = false;
        self.input_buf.clear();
        self.history_index = None;
    }

    pub fn submit(&mut self) -> CommandResult {
        let input = self.input_buf.trim().to_string();
        self.input_buf.clear();
        self.open = false;
        self.history_index = None;

        if input.is_empty() {
            return CommandResult::Empty;
        }

        // Record in history (no duplicates of the last entry).
        if self.history.front().map(|s| s.as_str()) != Some(&input) {
            self.history.push_front(input.clone());
            while self.history.len() > MAX_HISTORY {
                self.history.pop_back();
            }
        }

        self.push_message(format!("> {}", input));
        let result = Self::parse_command(&input, Vec3::ZERO);
        if let CommandResult::Unknown(msg) = &result {
            self.push_message(msg.clone());
        }
        result
    }

    pub fn submit_with_pos(&mut self, player_pos: Vec3) -> CommandResult {
        let input = self.input_buf.trim().to_string();
        self.input_buf.clear();
        self.open = false;
        self.history_index = None;

        if input.is_empty() {
            return CommandResult::Empty;
        }

        if self.history.front().map(|s| s.as_str()) != Some(&input) {
            self.history.push_front(input.clone());
            while self.history.len() > MAX_HISTORY {
                self.history.pop_back();
            }
        }

        self.push_message(format!("> {}", input));
        Self::parse_command(&input, player_pos)
    }

    /// Cycle history up (older).
    pub fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let next = match self.history_index {
            None => 0,
            Some(i) if i + 1 < self.history.len() => i + 1,
            _ => return,
        };
        self.history_index = Some(next);
        self.input_buf = self.history[next].clone();
    }

    /// Cycle history down (newer).
    pub fn history_down(&mut self) {
        match self.history_index {
            None => {}
            Some(0) => {
                self.history_index = None;
                self.input_buf.clear();
            }
            Some(i) => {
                let next = i - 1;
                self.history_index = Some(next);
                self.input_buf = self.history[next].clone();
            }
        }
    }

    /// Tab-complete the current input buffer against known commands.
    pub fn tab_complete(&mut self) {
        let input = self.input_buf.trim_start();
        if input.is_empty() {
            return;
        }
        let matches: Vec<&str> = COMMANDS
            .iter()
            .filter(|cmd| cmd.starts_with(input))
            .copied()
            .collect();
        if matches.len() == 1 {
            self.input_buf = format!("{} ", matches[0]);
        } else if matches.len() > 1 {
            // Find the common prefix among matches.
            let prefix = matches.iter().fold(matches[0], |acc, &s| {
                let common_len = acc
                    .chars()
                    .zip(s.chars())
                    .take_while(|(a, b)| a == b)
                    .count();
                &acc[..common_len]
            });
            if prefix.len() > input.len() {
                self.input_buf = prefix.to_string();
            } else {
                // Show all matches as feedback.
                self.push_message(format!("Commands: {}", matches.join(", ")));
            }
        }
    }

    pub fn push_message(&mut self, msg: String) {
        self.messages.push_front(msg);
        while self.messages.len() > MAX_MESSAGES {
            self.messages.pop_back();
        }
    }

    pub fn push_char(&mut self, ch: char) {
        if self.open && !ch.is_control() {
            self.input_buf.push(ch);
            self.history_index = None;
        }
    }

    pub fn backspace(&mut self) {
        if self.open {
            self.input_buf.pop();
            self.history_index = None;
        }
    }

    pub fn parse_tp_args(args: &str, player_pos: Vec3) -> CommandResult {
        let parts: Vec<&str> = args.split_whitespace().collect();
        if parts.len() < 3 {
            return CommandResult::Unknown("/tp requires x y z".into());
        }
        let parse_coord = |s: &str, current: f32| -> Result<f32, String> {
            if let Some(rest) = s.strip_prefix('~') {
                let offset: f32 = if rest.is_empty() {
                    0.0
                } else {
                    rest.parse().map_err(|_| format!("invalid offset: {s}"))?
                };
                Ok(current + offset)
            } else {
                s.parse().map_err(|_| format!("invalid coordinate: {s}"))
            }
        };
        match (
            parse_coord(parts[0], player_pos.x),
            parse_coord(parts[1], player_pos.y),
            parse_coord(parts[2], player_pos.z),
        ) {
            (Ok(x), Ok(y), Ok(z)) => CommandResult::Teleport(Vec3::new(x, y, z)),
            (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => CommandResult::Unknown(e),
        }
    }

    pub fn parse_command(input: &str, player_pos: Vec3) -> CommandResult {
        let parts = parse_args(input);
        if parts.is_empty() {
            return CommandResult::Empty;
        }

        match parts[0].as_str() {
            "/tp" => Self::parse_tp_args(&parts[1..].join(" "), player_pos),
            "/time" => {
                if parts.len() < 2 {
                    return CommandResult::Unknown("/time requires set/speed".into());
                }
                match parts[1].as_str() {
                    "set" => {
                        if parts.len() < 3 {
                            return CommandResult::Unknown("/time set requires a value".into());
                        }
                        match parts[2].as_str() {
                            "day" => CommandResult::SetTime(0.0),
                            "night" => CommandResult::SetTime(0.5),
                            "dawn" => CommandResult::SetTime(0.15),
                            "dusk" => CommandResult::SetTime(0.65),
                            val => match val.parse::<f64>() {
                                Ok(v) => CommandResult::SetTime(v),
                                Err(_) => CommandResult::Unknown(format!("invalid time: {val}")),
                            },
                        }
                    }
                    "speed" => {
                        if parts.len() < 3 {
                            return CommandResult::Unknown(
                                "/time speed requires multiplier".into(),
                            );
                        }
                        match parts[2].parse::<f64>() {
                            Ok(v) if v > 0.0 => CommandResult::TimeSpeed(v),
                            _ => CommandResult::Unknown(format!("invalid speed: {}", parts[2])),
                        }
                    }
                    sub => CommandResult::Unknown(format!("unknown /time subcommand: {sub}")),
                }
            }
            "/give" => {
                if parts.len() < 2 {
                    return CommandResult::Unknown("/give requires <block> [count]".into());
                }
                let block = parts[1].clone();
                let count = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
                CommandResult::Give(block, count)
            }
            "/setblock" => {
                if parts.len() < 5 {
                    return CommandResult::Unknown("/setblock requires x y z <block>".into());
                }
                let x = parts[1].parse::<i32>();
                let y = parts[2].parse::<i32>();
                let z = parts[3].parse::<i32>();
                let block = parts[4].clone();
                match (x, y, z) {
                    (Ok(x), Ok(y), Ok(z)) => CommandResult::SetBlock(x, y, z, block),
                    _ => CommandResult::Unknown("invalid coordinates".into()),
                }
            }
            "/fill" => {
                if parts.len() < 8 {
                    return CommandResult::Unknown(
                        "/fill requires x1 y1 z1 x2 y2 z2 <block>".into(),
                    );
                }
                let x1 = parts[1].parse::<i32>();
                let y1 = parts[2].parse::<i32>();
                let z1 = parts[3].parse::<i32>();
                let x2 = parts[4].parse::<i32>();
                let y2 = parts[5].parse::<i32>();
                let z2 = parts[6].parse::<i32>();
                let block = parts[7].clone();
                match (x1, y1, z1, x2, y2, z2) {
                    (Ok(x1), Ok(y1), Ok(z1), Ok(x2), Ok(y2), Ok(z2)) => {
                        CommandResult::Fill(x1, y1, z1, x2, y2, z2, block)
                    }
                    _ => CommandResult::Unknown("invalid coordinates".into()),
                }
            }
            "/gamemode" => {
                let mode = parts.get(1).map(|s| s.as_str()).unwrap_or("creative").to_string();
                CommandResult::Gamemode(mode)
            }
            "/pos" => CommandResult::Position,
            "/chunk" => CommandResult::ChunkInfo,
            "/fps" => CommandResult::Fps,
            "/reload" => CommandResult::Reload,
            "/clear" => CommandResult::Clear,
            "/save" => {
                let path = parts.get(1).map(|s| s.as_str()).unwrap_or("world_save").to_string();
                CommandResult::Save(path)
            }
            "/load" => {
                let path = parts.get(1).map(|s| s.as_str()).unwrap_or("world_save").to_string();
                CommandResult::Load(path)
            }
            "/copy" => {
                if parts.len() < 7 {
                    return CommandResult::Unknown("/copy requires x1 y1 z1 x2 y2 z2".into());
                }
                let x1 = parts[1].parse::<i32>();
                let y1 = parts[2].parse::<i32>();
                let z1 = parts[3].parse::<i32>();
                let x2 = parts[4].parse::<i32>();
                let y2 = parts[5].parse::<i32>();
                let z2 = parts[6].parse::<i32>();
                match (x1, y1, z1, x2, y2, z2) {
                    (Ok(x1), Ok(y1), Ok(z1), Ok(x2), Ok(y2), Ok(z2)) => {
                        CommandResult::Copy(x1, y1, z1, x2, y2, z2)
                    }
                    _ => CommandResult::Unknown("invalid coordinates".into()),
                }
            }
            "/paste" => CommandResult::Paste,
            "/help" => CommandResult::Help,
            cmd if cmd.starts_with('/') => CommandResult::Unknown(cmd.to_string()),
            _ => CommandResult::Unknown("commands start with /".into()),
        }
    }
}

fn parse_args(input: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    for ch in input.chars() {
        match ch {
            '"' => in_quotes = !in_quotes,
            ' ' if !in_quotes => {
                if !current.is_empty() {
                    args.push(current.clone());
                    current.clear();
                }
            }
            _ => current.push(ch),
        }
    }
    if !current.is_empty() {
        args.push(current);
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty() {
        assert!(matches!(ChatState::parse_command("", Vec3::ZERO), CommandResult::Empty));
    }

    #[test]
    fn parse_help() {
        assert!(matches!(
            ChatState::parse_command("/help", Vec3::ZERO),
            CommandResult::Help
        ));
    }

    #[test]
    fn parse_unknown_command() {
        assert!(matches!(
            ChatState::parse_command("/foo", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_no_slash() {
        assert!(matches!(
            ChatState::parse_command("hello", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_time_set_day() {
        let r = ChatState::parse_command("/time set day", Vec3::ZERO);
        assert!(matches!(r, CommandResult::SetTime(0.0)));
    }

    #[test]
    fn parse_time_set_night() {
        let r = ChatState::parse_command("/time set night", Vec3::ZERO);
        assert!(matches!(r, CommandResult::SetTime(0.5)));
    }

    #[test]
    fn parse_time_set_numeric() {
        let r = ChatState::parse_command("/time set 42.5", Vec3::ZERO);
        assert!(matches!(r, CommandResult::SetTime(v) if (v - 42.5).abs() < f64::EPSILON));
    }

    #[test]
    fn parse_time_set_invalid() {
        assert!(matches!(
            ChatState::parse_command("/time set abc", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_time_set_missing() {
        assert!(matches!(
            ChatState::parse_command("/time set", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_time_speed() {
        let r = ChatState::parse_command("/time speed 2.0", Vec3::ZERO);
        assert!(matches!(r, CommandResult::TimeSpeed(v) if (v - 2.0).abs() < f64::EPSILON));
    }

    #[test]
    fn parse_time_speed_zero() {
        assert!(matches!(
            ChatState::parse_command("/time speed 0", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_time_speed_negative() {
        assert!(matches!(
            ChatState::parse_command("/time speed -1", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_time_unknown_sub() {
        assert!(matches!(
            ChatState::parse_command("/time foo", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_time_missing_args() {
        assert!(matches!(
            ChatState::parse_command("/time", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_tp_missing_args() {
        assert!(matches!(
            ChatState::parse_command("/tp", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_tp_too_few() {
        assert!(matches!(
            ChatState::parse_command("/tp 1 2", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_tp_absolute() {
        let r = ChatState::parse_tp_args("10 20 30", Vec3::ZERO);
        if let CommandResult::Teleport(pos) = r {
            assert!((pos.x - 10.0).abs() < f32::EPSILON);
            assert!((pos.y - 20.0).abs() < f32::EPSILON);
            assert!((pos.z - 30.0).abs() < f32::EPSILON);
        } else {
            panic!("expected Teleport");
        }
    }

    #[test]
    fn parse_tp_relative() {
        let r = ChatState::parse_tp_args("~ ~10 ~-5", Vec3::new(100.0, 50.0, 200.0));
        if let CommandResult::Teleport(pos) = r {
            assert!((pos.x - 100.0).abs() < f32::EPSILON);
            assert!((pos.y - 60.0).abs() < f32::EPSILON);
            assert!((pos.z - 195.0).abs() < f32::EPSILON);
        } else {
            panic!("expected Teleport");
        }
    }

    #[test]
    fn parse_tpbare_tilde() {
        let r = ChatState::parse_tp_args("~ ~ ~", Vec3::new(5.0, 10.0, 15.0));
        if let CommandResult::Teleport(pos) = r {
            assert!((pos.x - 5.0).abs() < f32::EPSILON);
            assert!((pos.y - 10.0).abs() < f32::EPSILON);
            assert!((pos.z - 15.0).abs() < f32::EPSILON);
        } else {
            panic!("expected Teleport");
        }
    }

    #[test]
    fn chat_submit_empty() {
        let mut chat = ChatState::default();
        chat.open();
        chat.input_buf.clear();
        assert!(matches!(chat.submit(), CommandResult::Empty));
        assert!(!chat.open);
    }

    #[test]
    fn chat_submit_adds_message() {
        let mut chat = ChatState::default();
        chat.open();
        chat.input_buf = "/help".into();
        chat.submit();
        assert!(!chat.messages.is_empty());
        assert!(chat.messages[0].contains("/help"));
    }

    #[test]
    fn chat_push_char_ignores_when_closed() {
        let mut chat = ChatState::default();
        chat.push_char('a');
        assert!(chat.input_buf.is_empty());
    }

    #[test]
    fn chat_push_char_adds_when_open() {
        let mut chat = ChatState::default();
        chat.open();
        chat.push_char('h');
        chat.push_char('i');
        assert_eq!(chat.input_buf, "hi");
    }

    #[test]
    fn chat_push_char_ignores_control() {
        let mut chat = ChatState::default();
        chat.open();
        chat.push_char('\n');
        chat.push_char('\x1b');
        assert!(chat.input_buf.is_empty());
    }

    #[test]
    fn chat_backspace() {
        let mut chat = ChatState::default();
        chat.open();
        chat.input_buf = "abc".into();
        chat.backspace();
        assert_eq!(chat.input_buf, "ab");
        chat.backspace();
        assert_eq!(chat.input_buf, "a");
        chat.backspace();
        assert_eq!(chat.input_buf, "");
        chat.backspace();
        assert_eq!(chat.input_buf, "");
    }

    #[test]
    fn chat_message_limit() {
        let mut chat = ChatState::default();
        for i in 0..100 {
            chat.push_message(format!("msg {i}"));
        }
        assert!(chat.messages.len() <= 64);
    }

    // --- History tests ---

    #[test]
    fn history_up_populates_input() {
        let mut chat = ChatState::default();
        chat.open();
        chat.input_buf = "/help".into();
        chat.submit();
        chat.open();
        chat.history_up();
        assert_eq!(chat.input_buf, "/help");
    }

    #[test]
    fn history_down_clears() {
        let mut chat = ChatState::default();
        chat.open();
        chat.input_buf = "/help".into();
        chat.submit();
        chat.open();
        chat.history_up();
        chat.history_down();
        assert!(chat.input_buf.is_empty());
    }

    #[test]
    fn history_no_duplicates() {
        let mut chat = ChatState::default();
        chat.open();
        chat.input_buf = "/help".into();
        chat.submit();
        chat.open();
        chat.input_buf = "/help".into();
        chat.submit();
        assert_eq!(chat.history.len(), 1);
    }

    // --- Tab completion tests ---

    #[test]
    fn tab_complete_unique() {
        let mut chat = ChatState::default();
        chat.open();
        chat.input_buf = "/he".into();
        chat.tab_complete();
        assert_eq!(chat.input_buf, "/help ");
    }

    #[test]
    fn tab_complete_common_prefix() {
        let mut chat = ChatState::default();
        chat.open();
        chat.input_buf = "/ti".into();
        chat.tab_complete();
        // Should extend to "/time " since all /time* commands share that prefix
        assert!(chat.input_buf.len() > 3);
    }

    // --- New command tests ---

    #[test]
    fn parse_give() {
        let r = ChatState::parse_command("/give stone 64", Vec3::ZERO);
        assert!(matches!(r, CommandResult::Give(name, 64) if name == "stone"));
    }

    #[test]
    fn parse_give_default_count() {
        let r = ChatState::parse_command("/give dirt", Vec3::ZERO);
        assert!(matches!(r, CommandResult::Give(name, 1) if name == "dirt"));
    }

    #[test]
    fn parse_give_missing() {
        assert!(matches!(
            ChatState::parse_command("/give", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_setblock() {
        let r = ChatState::parse_command("/setblock 10 20 30 stone", Vec3::ZERO);
        assert!(matches!(r, CommandResult::SetBlock(10, 20, 30, ref b) if b == "stone"));
    }

    #[test]
    fn parse_setblock_missing() {
        assert!(matches!(
            ChatState::parse_command("/setblock 1 2", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_fill() {
        let r = ChatState::parse_command("/fill 0 0 0 5 5 5 stone", Vec3::ZERO);
        assert!(matches!(r, CommandResult::Fill(0, 0, 0, 5, 5, 5, ref b) if b == "stone"));
    }

    #[test]
    fn parse_fill_missing() {
        assert!(matches!(
            ChatState::parse_command("/fill 0 0 0 1 1", Vec3::ZERO),
            CommandResult::Unknown(_)
        ));
    }

    #[test]
    fn parse_gamemode() {
        let r = ChatState::parse_command("/gamemode creative", Vec3::ZERO);
        assert!(matches!(r, CommandResult::Gamemode(ref m) if m == "creative"));
    }

    #[test]
    fn parse_gamemode_default() {
        let r = ChatState::parse_command("/gamemode", Vec3::ZERO);
        assert!(matches!(r, CommandResult::Gamemode(ref m) if m == "creative"));
    }

    #[test]
    fn parse_pos() {
        assert!(matches!(
            ChatState::parse_command("/pos", Vec3::ZERO),
            CommandResult::Position
        ));
    }

    #[test]
    fn parse_chunk() {
        assert!(matches!(
            ChatState::parse_command("/chunk", Vec3::ZERO),
            CommandResult::ChunkInfo
        ));
    }

    #[test]
    fn parse_fps() {
        assert!(matches!(
            ChatState::parse_command("/fps", Vec3::ZERO),
            CommandResult::Fps
        ));
    }

    #[test]
    fn parse_reload() {
        assert!(matches!(
            ChatState::parse_command("/reload", Vec3::ZERO),
            CommandResult::Reload
        ));
    }

    #[test]
    fn parse_clear() {
        assert!(matches!(
            ChatState::parse_command("/clear", Vec3::ZERO),
            CommandResult::Clear
        ));
    }

    #[test]
    fn parse_tp_relative_via_parse_command() {
        let r = ChatState::parse_command("/tp ~ ~10 ~", Vec3::new(100.0, 50.0, 200.0));
        if let CommandResult::Teleport(pos) = r {
            assert!((pos.x - 100.0).abs() < f32::EPSILON);
            assert!((pos.y - 60.0).abs() < f32::EPSILON);
            assert!((pos.z - 200.0).abs() < f32::EPSILON);
        } else {
            panic!("expected Teleport");
        }
    }
}
