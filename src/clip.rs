use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::NaiveDateTime;
use regex::Regex;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Direction {
    Front,
    Rear,
}

#[derive(Debug, Clone)]
pub struct Clip {
    pub date: String,
    pub time: String,
    pub seq: u32,
    pub direction: Direction,
    pub filename: String,
    pub path: PathBuf,
}

/// A pair of F+R clips sharing the same sequence number
#[derive(Debug, Clone)]
pub struct ClipPair {
    pub seq: u32,
    pub front: Clip,
    pub rear: Clip,
}

/// A group of consecutive clip pairs
#[derive(Debug)]
pub struct ClipGroup {
    pub pairs: Vec<ClipPair>,
    pub name: String,
}

impl ClipGroup {
    pub fn front_paths(&self) -> Vec<&Path> {
        self.pairs.iter().map(|p| p.front.path.as_path()).collect()
    }

    pub fn rear_paths(&self) -> Vec<&Path> {
        self.pairs.iter().map(|p| p.rear.path.as_path()).collect()
    }
}

pub fn parse_filename(path: &Path) -> Option<Clip> {
    let filename = path.file_name()?.to_str()?;
    let re = Regex::new(r"^(\d{6})_(\d{6})_(\d{4})_(F|R)\.MP4$").ok()?;
    let caps = re.captures(filename)?;

    Some(Clip {
        date: caps[1].to_string(),
        time: caps[2].to_string(),
        seq: caps[3].parse().ok()?,
        direction: if &caps[4] == "F" {
            Direction::Front
        } else {
            Direction::Rear
        },
        filename: filename.to_string(),
        path: path.to_path_buf(),
    })
}

/// Scan input directory, pair F+R clips, and group consecutive sequences
pub fn find_groups(input_dir: &Path) -> Result<Vec<ClipGroup>> {
    let mut clips: Vec<Clip> = Vec::new();

    for entry in std::fs::read_dir(input_dir).context("Failed to read input directory")? {
        let entry = entry?;
        let path = entry.path();
        if let Some(clip) = parse_filename(&path) {
            clips.push(clip);
        }
    }

    // Group by seq → (Option<Front>, Option<Rear>)
    let mut by_seq: BTreeMap<u32, (Option<Clip>, Option<Clip>)> = BTreeMap::new();
    for clip in clips {
        let entry = by_seq.entry(clip.seq).or_insert((None, None));
        match clip.direction {
            Direction::Front => entry.0 = Some(clip),
            Direction::Rear => entry.1 = Some(clip),
        }
    }

    // Keep only complete pairs
    let mut pairs: Vec<ClipPair> = Vec::new();
    for (seq, (front, rear)) in by_seq {
        if let (Some(f), Some(r)) = (front, rear) {
            pairs.push(ClipPair {
                seq,
                front: f,
                rear: r,
            });
        }
    }

    // BTreeMap already sorted by key, but sort explicitly for clarity
    pairs.sort_by_key(|p| p.seq);

    // Split into consecutive groups (gap > 60 s → new group)
    let mut groups: Vec<ClipGroup> = Vec::new();
    let mut current: Vec<ClipPair> = Vec::new();

    for pair in pairs {
        if let Some(last) = current.last() {
            let gap = clip_datetime(&pair.front)
                .and_then(|t| clip_datetime(&last.front).map(|p| (t - p).num_seconds()));
            if gap.map_or(true, |s| s > 70) {
                let name = group_name(&current);
                groups.push(ClipGroup {
                    pairs: std::mem::take(&mut current),
                    name,
                });
            }
        }
        current.push(pair);
    }
    if !current.is_empty() {
        let name = group_name(&current);
        groups.push(ClipGroup {
            pairs: current,
            name,
        });
    }

    Ok(groups)
}

fn clip_datetime(clip: &Clip) -> Option<NaiveDateTime> {
    let s = format!("{}{}", clip.date, clip.time);
    NaiveDateTime::parse_from_str(&s, "%y%m%d%H%M%S").ok()
}

fn group_name(pairs: &[ClipPair]) -> String {
    let first = &pairs.first().unwrap().front;
    let last = &pairs.last().unwrap().front;

    // "to" = last clip start + 60 s (actual end), handles cross-day
    if let (Some(from), Some(to)) = (clip_datetime(first), clip_datetime(last)) {
        format!(
            "{}_{}-{}_{}",
            from.format("%y%m%d"),
            from.format("%H%M%S"),
            to.format("%y%m%d"),
            to.format("%H%M%S"),
        )
    } else {
        // fallback (datetime parse failed)
        format!("{}_{}-{}_{}", first.date, first.time, last.date, last.time)
    }
}
