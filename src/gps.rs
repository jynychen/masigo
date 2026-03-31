use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result, bail};
use chrono::{NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Utc};

use crate::ffprobe::probe_video_info;

#[derive(Debug, Clone)]
pub struct GnrmcFix {
    pub lat: f64,
    pub lon: f64,
    /// UTC timestamp of this GPS entry, parsed from the GNRMC sentence.
    /// For invalid entries (no fix / no sentence), derived by ±1 s offset
    /// from the nearest entry that has a parseable time.
    pub time: SystemTime,
    // add speed
}

#[derive(Debug, Clone)]
pub struct GpsPoint {
    /// Path to the source clip this entry came from.
    pub file_path: PathBuf,
    /// 0-based global second index across all clips in the group.
    pub frame_index: usize,

    /// GNRMC-derived position data.  `None` when no valid GNRMC fix was
    /// obtained (no sentence, parse failure, or status != 'A').
    pub gnrmc: Option<GnrmcFix>,
}

/// Extract 1 Hz GPS track from freeGPS boxes embedded in the front clips.
///
/// Returns one `GpsPoint` per second (= total video duration).  Entries with a
/// valid GNRMC/GPRMC fix (status 'A') carry `gnrmc: Some(GnrmcFix)`;
/// entries without a fix have `gnrmc: None`.
pub fn extract_gps_track(front_paths: &[&Path]) -> Result<Vec<GpsPoint>> {
    let mut result: Vec<GpsPoint> = Vec::new();
    let mut global_offset: usize = 0;

    for path in front_paths {
        let entries = read_gps_box_entries(path)
            .with_context(|| format!("failed to read gps box from {}", path.display()))?;

        let entry_count = entries.len();

        let (_, duration) = probe_video_info(path)?;
        let expected = duration.round() as usize;
        if entry_count != expected {
            bail!(
                "{}: gps box has {} entries but video duration is {:.3}s (expected {})",
                path.display(),
                entry_count,
                duration,
                expected
            );
        }

        let mut file =
            File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
        let mut sample = vec![0u8; 300];

        for (i, (file_offset, _)) in entries.iter().enumerate() {
            file.seek(SeekFrom::Start(*file_offset))?;
            file.read_exact(&mut sample)?;

            let frame_index = global_offset + i;
            let gnrmc = find_rmc_sentence(&sample).and_then(|s| {
                let parts: Vec<&str> = s.split(',').collect();
                if parts.get(2).copied() != Some("A") {
                    return None;
                }
                let lat = parts
                    .get(3)
                    .and_then(|v| parts.get(4).map(|h| nmea_to_decimal_degrees(v, h)))
                    .flatten()?;
                let lon = parts
                    .get(5)
                    .and_then(|v| parts.get(6).map(|h| nmea_to_decimal_degrees(v, h)))
                    .flatten()?;
                let time = parse_rmc_timestamp(s)?;
                Some(GnrmcFix { lat, lon, time })
            });

            result.push(GpsPoint {
                file_path: path.to_path_buf(),
                frame_index,
                gnrmc,
            });
        }

        global_offset += entry_count;
    }

    Ok(result)
}

// ── freeGPS box ───────────────────────────────────────────────────────────────

/// Read the `(file_offset, sample_size)` entry table from the `gps ` box.
///
/// Box layout (all big-endian):
///   [size: u32][type "gps ": 4B][version+flags: u32][entry_count: u32]
///   [file_offset: u32][sample_size: u32]  × entry_count
///
/// entry_count equals the clip duration in seconds; each entry covers one second.
/// sample_size is always 0x4000 (16 384 bytes) for this dashcam model.
fn read_gps_box_entries(path: &Path) -> Result<Vec<(u64, u32)>> {
    let mut file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let file_size = file.seek(SeekFrom::End(0))?;

    // moov box is ~24 KiB at the end; scan the last 512 KiB to be safe.
    let scan_len = 524_288_u64.min(file_size);
    file.seek(SeekFrom::Start(file_size - scan_len))?;
    let mut buf = vec![0u8; scan_len as usize];
    file.read_exact(&mut buf)?;

    // Locate the last occurrence of the 'gps ' type field.
    let rel_type = buf
        .windows(4)
        .rposition(|w| w == b"gps ")
        .context("'gps ' box not found")?;
    if rel_type < 4 {
        bail!("gps box type field too close to start of scan buffer");
    }

    let size_start = rel_type - 4;
    let box_size = u32::from_be_bytes(buf[size_start..rel_type].try_into()?) as usize;
    let data_start = rel_type + 4; // skip type field

    if size_start + box_size > buf.len() {
        bail!("gps box extends beyond scan buffer");
    }
    let box_data = &buf[data_start..size_start + box_size];

    if box_data.len() < 8 {
        bail!("gps box body too small");
    }
    let entry_count = u32::from_be_bytes(box_data[4..8].try_into()?) as usize;
    if box_data.len() < 8 + entry_count * 8 {
        bail!("gps box entry table truncated");
    }

    let entries = (0..entry_count)
        .map(|i| {
            let off = 8 + i * 8;
            let file_offset = u32::from_be_bytes(box_data[off..off + 4].try_into()?) as u64;
            let sample_size = u32::from_be_bytes(box_data[off + 4..off + 8].try_into()?);
            Ok((file_offset, sample_size))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(entries)
}

// ── NMEA parsing ──────────────────────────────────────────────────────────────

/// Find the first `$GNRMC` or `$GPRMC` sentence in a raw byte buffer.
fn find_rmc_sentence(buf: &[u8]) -> Option<&str> {
    for sig in [b"$GNRMC".as_ref(), b"$GPRMC".as_ref()] {
        if let Some(start) = buf.windows(sig.len()).position(|w| w == sig) {
            let end = buf[start..]
                .iter()
                .position(|&b| b == b'\n' || b == b'\r')
                .map(|p| start + p)
                .unwrap_or(buf.len());
            return std::str::from_utf8(&buf[start..end]).ok();
        }
    }
    None
}

/// Parse UTC timestamp (fields 1 + 9) from an RMC sentence into `SystemTime`.
fn parse_rmc_timestamp(sentence: &str) -> Option<SystemTime> {
    let parts: Vec<&str> = sentence.split(',').collect();
    let time_str = parts.get(1)?;
    let date_str = parts.get(9)?;

    if time_str.len() < 6 || date_str.len() < 6 {
        return None;
    }

    let hh: u32 = time_str[0..2].parse().ok()?;
    let mm: u32 = time_str[2..4].parse().ok()?;
    let ss: u32 = time_str[4..6].parse().ok()?;
    let dd: u32 = date_str[0..2].parse().ok()?;
    let mo: u32 = date_str[2..4].parse().ok()?;
    let yy: u32 = date_str[4..6].parse().ok()?;

    let date = NaiveDate::from_ymd_opt(2000 + yy as i32, mo, dd)?;
    let time = NaiveTime::from_hms_opt(hh, mm, ss)?;
    let dt = NaiveDateTime::new(date, time);
    Some(SystemTime::from(Utc.from_utc_datetime(&dt)))
}

/// Convert NMEA coordinate (`ddmm.mmmmm` or `dddmm.mmmmm`) and hemisphere
/// letter to decimal degrees.
fn nmea_to_decimal_degrees(coord: &str, hemi: &str) -> Option<f64> {
    if coord.is_empty() {
        return None;
    }
    let dot = coord.find('.')?;
    if dot < 2 {
        return None;
    }
    let deg: f64 = coord[..dot - 2].parse().ok()?;
    let min: f64 = coord[dot - 2..].parse().ok()?;
    let dd = deg + min / 60.0;
    Some(if hemi == "S" || hemi == "W" { -dd } else { dd })
}
