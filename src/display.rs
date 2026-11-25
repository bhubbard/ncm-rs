use std::fmt::Write;

use crate::generation::SampleRecord;

#[must_use]
pub fn sudoku_board(ids: &[u32]) -> String {
    let mut trimmed = ids;
    while let Some(&first) = trimmed.first() {
        if first == 11 {
            trimmed = &trimmed[1..];
        } else {
            break;
        }
    }
    while let Some(&last) = trimmed.last() {
        if last == 11 {
            trimmed = &trimmed[..trimmed.len() - 1];
        } else {
            break;
        }
    }

    let (cells, malformed) = if trimmed.len() == 89 {
        let mut c = Vec::with_capacity(81);
        for row in 0..9 {
            for col in 0..9 {
                c.push(trimmed[row * 10 + col]);
            }
        }
        let bad = (0..8).any(|row| trimmed[row * 10 + 9] != 10);
        (c, bad)
    } else if trimmed.len() == 81 {
        (trimmed.to_vec(), false)
    } else {
        return format!("[Malformed board: {} tokens]\n{trimmed:?}", trimmed.len());
    };

    let mut lines = vec!["┌───────┬───────┬───────┐".to_string()];
    for row in 0..9 {
        let values: Vec<String> = cells[row * 9..(row + 1) * 9]
            .iter()
            .map(|&v| match v {
                0 => "·".to_string(),
                1..=9 => v.to_string(),
                _ => "?".to_string(),
            })
            .collect();
        let g0 = values[0..3].join(" ");
        let g1 = values[3..6].join(" ");
        let g2 = values[6..9].join(" ");
        lines.push(format!("│ {g0} │ {g1} │ {g2} │"));
        if row == 2 || row == 5 {
            lines.push("├───────┼───────┼───────┤".to_string());
        }
    }
    lines.push("└───────┴───────┴───────┘".to_string());
    if malformed {
        lines.push("[Invalid row separator]".to_string());
    }
    lines.join("\n")
}

fn wrap_line(line: &str, width: usize) -> Vec<String> {
    if line.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut current = String::new();
    for word in line.split_whitespace() {
        if current.is_empty() {
            current.push_str(word);
        } else if current.len() + 1 + word.len() <= width {
            current.push(' ');
            current.push_str(word);
        } else {
            out.push(std::mem::take(&mut current));
            current.push_str(word);
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn text_block(label: &str, text: &str, width: usize) -> String {
    let mut wrapped = Vec::new();
    for part in text.trim().lines() {
        wrapped.extend(wrap_line(part, width));
    }
    let body = if wrapped.is_empty() {
        "[Empty output]".to_string()
    } else {
        wrapped.join("\n")
    };
    format!("{label}\n{body}")
}

#[must_use]
pub fn format_sample(sample: &SampleRecord, number: usize, width: usize) -> String {
    let mut header = format!("Sample {number} · {} · seed {}", sample.task, sample.seed);
    if let Some(idx) = sample.dataset_index {
        let _ = write!(header, " · input {idx}");
    }
    let rule = "─".repeat(width.min(header.chars().count()));
    let mut sections = vec![header, rule];
    let sudoku = sample.task.starts_with("sudoku");

    if let Some(ref prompt) = sample.prompt {
        if sudoku {
            let pids = sample.prompt_ids.as_deref().unwrap_or(&[]);
            sections.push(format!("Prompt\n{}", sudoku_board(pids)));
        } else {
            sections.push(text_block("Prompt", prompt, width));
        }
    }
    if sudoku {
        sections.push(format!("Generated\n{}", sudoku_board(&sample.raw_ids)));
    } else {
        sections.push(text_block("Generated", &sample.output, width));
    }
    if let Some(ref reference) = sample.reference {
        if sudoku {
            let rids = sample.reference_ids.as_deref().unwrap_or(&[]);
            sections.push(format!("Reference\n{}", sudoku_board(rids)));
        } else {
            sections.push(text_block("Reference", reference, width));
        }
    }
    if sample.prompt_truncated == Some(true) {
        sections
            .push("(Prompt shown is the model input, truncated to its context limit.)".to_string());
    }
    if let Some(ref traj) = sample.trajectory {
        sections
            .push("Trajectory · each hierarchy prefix decoded at full output length".to_string());
        let full_rule = "─".repeat(width);
        for step in traj {
            sections.push(full_rule.clone());
            let label = format!("Level {}/{} · {} codes", step.level, traj.len(), step.codes);
            if sudoku {
                sections.push(format!("{label}\n{}", sudoku_board(&step.raw_ids)));
            } else {
                sections.push(text_block(&label, &step.text, width));
            }
        }
    }
    sections.join("\n\n")
}
