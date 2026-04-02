use core::f64;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result, bail};
use chrono::{NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Utc};

use crate::ffprobe::probe_video_info;

const FREEGPS_MAGIC: &[u8] = b"freeGPS ";
/// Bytes needed to cover all binary header fields (through accel_y at 0x64–0x67).
const HEADER_LEN: usize = 0x68;

/// One second of freeGPS data.  All fields are always populated from the binary
/// header; `fix` indicates whether the GPS position is valid.
#[derive(Debug, Clone)]
pub struct Gnrmc {
    /// `true` when the status byte is `'A'` (valid GPS fix).
    pub fix: bool,
    /// UTC timestamp from the binary header date/time fields.
    /// Valid even when `fix == false`.
    pub time: SystemTime,
    /// Decimal degrees (WGS-84).  `f64::NAN` when `!fix`.
    pub lat: f64,
    pub lon: f64,
    /// Speed over ground in km/h.  `f64::NAN` when `!fix`.
    #[allow(dead_code)]
    pub speed: f64,
    /// True course / heading in degrees.  `f64::NAN` when `!fix`.
    #[allow(dead_code)]
    pub track: f64,
    /// Accelerometer in g
    /// Z ≈ 1.0 at rest (gravity axis).  Always valid.
    #[allow(dead_code)]
    pub accel_z: f64,
    #[allow(dead_code)]
    pub accel_x: f64,
    #[allow(dead_code)]
    pub accel_y: f64,
}

#[derive(Debug, Clone)]
pub struct GpsPoint {
    /// Path to the source clip this entry came from.
    pub file_path: PathBuf,
    /// 0-based global second index across all clips in the group.
    pub frame_index: usize,
    pub gnrmc: Gnrmc,
}

/// Extract 1 Hz GPS track from freeGPS boxes embedded in the front clips.
///
/// Returns one `GpsPoint` per second (= total video duration).
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
        let mut buf = vec![0u8; HEADER_LEN];

        for (i, (file_offset, _)) in entries.iter().enumerate() {
            file.seek(SeekFrom::Start(*file_offset))?;
            file.read_exact(&mut buf)?;

            result.push(GpsPoint {
                file_path: path.to_path_buf(),
                frame_index: global_offset + i,
                gnrmc: parse_freegps_header(&buf),
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

// ── freeGPS sample header ─────────────────────────────────────────────────────
//
// All fields are little-endian.  Layout (offsets in hex):
//   0x00  u32    block size (always 0x4000)
//   0x04  8B     magic "freeGPS "
//   0x0c  u32    (reserved)
//   0x10  u32    UTC hour
//   0x14  u32    UTC minute
//   0x18  u32    UTC second
//   0x1c  u8     GPS status  'A' = valid fix  'V' = void
//   0x20  f64    latitude  in NMEA ddmm.mmmmm
//   0x28  u8     N / S
//   0x30  f64    longitude in NMEA dddmm.mmmmm
//   0x38  u8     E / W
//   0x40  f64    speed over ground (knots)
//   0x48  f64    true course / heading (degrees)
//   0x50  u32    year  (2-digit, e.g. 26 → 2026)
//   0x54  u32    month
//   0x58  u32    day
//   0x5c  i32    accelerometer Z  (÷ 1000 → g; ~1.0 at rest)
//   0x60  i32    accelerometer X  (÷ 1000 → g)
//   0x64  i32    accelerometer Y  (÷ 1000 → g)
//   0x68  …      ' ' + NMEA sentence ($GNRMC / $GPRMC)

fn parse_freegps_header(buf: &[u8]) -> Gnrmc {
    if buf.len() < HEADER_LEN || &buf[4..12] != FREEGPS_MAGIC {
        return Gnrmc {
            fix: false,
            time: SystemTime::UNIX_EPOCH,
            lat: f64::NAN,
            lon: f64::NAN,
            speed: f64::NAN,
            track: f64::NAN,
            accel_z: f64::NAN,
            accel_x: f64::NAN,
            accel_y: f64::NAN,
        };
    }

    let time = build_timestamp(
        le_u32(buf, 0x50),
        le_u32(buf, 0x54),
        le_u32(buf, 0x58),
        le_u32(buf, 0x10),
        le_u32(buf, 0x14),
        le_u32(buf, 0x18),
    )
    .unwrap_or(SystemTime::UNIX_EPOCH);

    Gnrmc {
        fix: buf[0x1c] == b'A',
        time,
        lat: nmea_to_decimal(le_f64(buf, 0x20), buf[0x28] as char),
        lon: nmea_to_decimal(le_f64(buf, 0x30), buf[0x38] as char),
        speed: le_f64(buf, 0x40) * 1.852,
        track: le_f64(buf, 0x48),
        accel_z: le_i32(buf, 0x5c) as f64 / 1000.0,
        accel_x: le_i32(buf, 0x60) as f64 / 1000.0,
        accel_y: le_i32(buf, 0x64) as f64 / 1000.0,
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

#[inline]
fn le_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}
#[inline]
fn le_i32(buf: &[u8], off: usize) -> i32 {
    i32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}
#[inline]
fn le_f64(buf: &[u8], off: usize) -> f64 {
    f64::from_le_bytes(buf[off..off + 8].try_into().unwrap())
}

/// Convert NMEA-format coordinate (ddmm.mmmmm as f64) + hemisphere to decimal degrees.
fn nmea_to_decimal(coord: f64, hemi: char) -> f64 {
    let degrees = (coord / 100.0).floor();
    let minutes = coord - degrees * 100.0;
    let dd = degrees + minutes / 60.0;
    if hemi == 'S' || hemi == 'W' { -dd } else { dd }
}

fn build_timestamp(
    year: u32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
) -> Option<SystemTime> {
    let date = NaiveDate::from_ymd_opt(2000 + year as i32, month, day)?;
    let time = NaiveTime::from_hms_opt(hour, minute, second)?;
    Some(SystemTime::from(
        Utc.from_utc_datetime(&NaiveDateTime::new(date, time)),
    ))
}
