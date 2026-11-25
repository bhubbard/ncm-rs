use anyhow::{Result, bail};

use super::PreTokenizer;

#[derive(Debug, Clone, Default)]
pub struct SudokuTokenizer;

impl SudokuTokenizer {
    pub const VOCAB_SIZE: usize = 12;
    pub const BLANK_TOKEN_ID: u32 = 0;
    pub const ROW_SEPARATOR_ID: u32 = 10;
    pub const BOS_TOKEN_ID: u32 = 11;
    pub const SEQ_LEN: usize = 89;
    pub const PROMPT_LEN: usize = 91;

    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl PreTokenizer for SudokuTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let normalized = text.replace("<BOS>", "B");
        let mut out = Vec::with_capacity(normalized.len());
        for c in normalized.chars() {
            if c.is_whitespace() {
                continue;
            }
            let id = match c {
                '_' | '0' => 0,
                '1'..='9' => c as u32 - '0' as u32,
                '|' => Self::ROW_SEPARATOR_ID,
                'B' => Self::BOS_TOKEN_ID,
                other => bail!("Invalid Sudoku character: {other:?}"),
            };
            out.push(id);
        }
        Ok(out)
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let mut out = String::with_capacity(ids.len());
        for &id in ids {
            if skip_special_tokens && id == Self::BOS_TOKEN_ID {
                continue;
            }
            match id {
                0 => out.push('_'),
                1..=9 => out.push(char::from_digit(id, 10).unwrap_or('?')),
                10 => out.push('|'),
                11 => out.push_str("<BOS>"),
                _ => out.push('?'),
            }
        }
        Ok(out)
    }

    fn vocab_size(&self) -> usize {
        Self::VOCAB_SIZE
    }

    fn eos_token_id(&self) -> Option<u32> {
        None
    }

    fn pad_token_id(&self) -> Option<u32> {
        None
    }
}
