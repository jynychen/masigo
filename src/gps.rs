use core::f64;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result, bail};
use chrono::{NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Utc};

use crate::ffprobe::probe_video_info;

const FREEGPS_MAGIC: &[u8] = b"freeGPS ";
/// Bytes needed to cover all binary header fields (through accel_y at 0x64–0x67).
const HEADER_LEN: usize = 0x68;

// ── freeGPS `free` atom layout ────────────────────────────────────────────────
//
// Each GPS sample occupies one ISO MP4 `free` atom (0x4000 = 16 384 B).
// One atom per second; the atom is null-padded after the NMEA sentence.
//
//   [0x0000 – 0x0067]  binary header  (parsed below)
//   [0x0068 – ~0x00B7] NMEA sentence: ' ' + $GNRMC/GPRMC + "\r\n" + '\0'
//                      duplicates the binary fields; only unique data is the
//                      sub-second timestamp (e.g. "072400.219")
//                      bytes after '\0' may contain stale data from a previous
//                      write cycle (firmware does not zero the remainder)
//   [rest]             mostly 0x00 padding to fill the 0x4000-byte atom
//
// Binary header field offsets (all LE except 0x00):
//   0x00  u32 BE  atom size (0x4000 = 16 384 B on this device)
//   0x04  8B      magic "freeGPS "
//   0x0c  u32     firmware format tag (0x3F0 on NT96xxx; reserved/opaque)
//   0x10  u32     UTC hour
//   0x14  u32     UTC minute
//   0x18  u32     UTC second
//   0x1c  u8      GPS status  'A' = valid fix  'V' = void
//   0x1d  3B      null padding (status stored in 4-byte aligned slot)
//   0x20  f64     latitude  in NMEA ddmm.mmmmm  (0.0 when void)
//   0x28  u8      N / S  ('0' when void)
//   0x29  7B      null padding (hemisphere stored in 8-byte aligned slot)
//   0x30  f64     longitude in NMEA dddmm.mmmmm  (0.0 when void)
//   0x38  u8      E / W  ('0' when void)
//   0x39  7B      null padding (hemisphere stored in 8-byte aligned slot)
//   0x40  f64     speed over ground (knots)
//   0x48  f64     true course / heading (degrees)
//   0x50  u32     year  (2-digit, e.g. 26 → 2026)
//   0x54  u32     month
//   0x58  u32     day
//   0x5c  i32     accelerometer Z  (÷ 1000 → g; ~1.0 at rest)
//   0x60  i32     accelerometer X  (÷ 1000 → g)
//   0x64  i32     accelerometer Y  (÷ 1000 → g)
const HDR_HOUR: usize = 0x10;
const HDR_MINUTE: usize = 0x14;
const HDR_SECOND: usize = 0x18;
const HDR_STATUS: usize = 0x1c;
const HDR_LAT: usize = 0x20;
const HDR_LAT_NS: usize = 0x28;
const HDR_LON: usize = 0x30;
const HDR_LON_EW: usize = 0x38;
const HDR_SPEED: usize = 0x40;
const HDR_COURSE: usize = 0x48;
const HDR_YEAR: usize = 0x50;
const HDR_MONTH: usize = 0x54;
const HDR_DAY: usize = 0x58;
const HDR_ACCEL_Z: usize = 0x5c;
const HDR_ACCEL_X: usize = 0x60;
const HDR_ACCEL_Y: usize = 0x64;

/// How a GPS point's position was determined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionSource {
    /// The floor 1 Hz entry for this frame had a GPS hardware fix.
    /// At exact second boundaries the spline passes through the hardware
    /// fix point; at sub-second offsets the position is still spline-evaluated
    /// between adjacent hardware fixes.
    HardwareFix,
    /// The floor 1 Hz entry had no fix; position is gap-filled by cubic spline.
    Interpolated,
    /// No position available (no fix and no spline possible).
    NoPosition,
}

/// One second of freeGPS sensor data.
#[derive(Debug, Clone)]
pub struct Gnrmc {
    /// UTC timestamp from the binary header date/time fields.
    pub time: SystemTime,
    /// Decimal degrees (WGS-84).  `f64::NAN` when no position.
    pub lat: f64,
    pub lon: f64,
    /// Speed over ground in km/h.  `f64::NAN` when no fix.
    #[allow(dead_code)]
    pub speed: f64,
    /// course over ground in degrees.  `f64::NAN` when no fix.
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
    pub gnrmc: Gnrmc,
    /// How this point's lat/lon were determined.
    pub position: PositionSource,
}

impl GpsPoint {
    /// Returns `true` when this frame has a drawable position —
    /// either a real hardware GPS fix or a gap-filled interpolation.
    pub fn position_valid(&self) -> bool {
        self.position != PositionSource::NoPosition
    }
}

/// Internal 1 Hz record used between parsing and upsampling.
/// Not exposed publicly — the public API is `GpsPoint` (at video fps).
struct RawGpsEntry {
    fix: bool,
    gnrmc: Gnrmc,
}

/// Extract 1 Hz GPS entries from freeGPS boxes embedded in the front clips.
///
/// Returns one `RawGpsEntry` per second (= total video duration).
fn extract_raw_entries(front_paths: &[&Path]) -> Result<Vec<RawGpsEntry>> {
    let mut result: Vec<RawGpsEntry> = Vec::new();

    for path in front_paths {
        let mut file =
            File::open(path).with_context(|| format!("cannot open {}", path.display()))?;

        let entries = read_gps_box_entries(&mut file)
            .with_context(|| format!("failed to read gps box from {}", path.display()))?;

        let entry_count = entries.len();

        let (_, duration) = probe_video_info(path)?;
        // The firmware writes one GPS entry per complete second; the last
        // partial second may or may not produce an entry, so accept both
        // floor and ceil of the duration.
        let floor_secs = duration.floor() as usize;
        let ceil_secs = duration.ceil() as usize;
        if entry_count != floor_secs && entry_count != ceil_secs {
            bail!(
                "{}: gps box has {} entries but video duration is {:.3}s (expected {} or {})",
                path.display(),
                entry_count,
                duration,
                floor_secs,
                ceil_secs,
            );
        }

        let mut buf = vec![0u8; HEADER_LEN];

        for (file_offset, _) in entries.iter() {
            file.seek(SeekFrom::Start(*file_offset))?;
            file.read_exact(&mut buf)?;

            let (gnrmc, fix) = parse_freegps_header(&buf)
                .with_context(|| format!("bad freeGPS header at offset {:#x}", file_offset))?;
            result.push(RawGpsEntry { fix, gnrmc });
        }
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
fn read_gps_box_entries(file: &mut File) -> Result<Vec<(u64, u32)>> {
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

/// Parse a freeGPS sample header from `buf`.
///
/// Returns `(gnrmc, fix)` where `fix` is true when the status byte is `'A'`.
/// Errors when the buffer is too short or the magic bytes are wrong.
fn parse_freegps_header(buf: &[u8]) -> Result<(Gnrmc, bool)> {
    if buf.len() < HEADER_LEN {
        bail!(
            "freeGPS header too short: {} bytes (need {})",
            buf.len(),
            HEADER_LEN
        );
    }
    if &buf[4..12] != FREEGPS_MAGIC {
        bail!("freeGPS magic not found at offset 4");
    }

    let time = build_timestamp(
        le_u32(buf, HDR_YEAR),
        le_u32(buf, HDR_MONTH),
        le_u32(buf, HDR_DAY),
        le_u32(buf, HDR_HOUR),
        le_u32(buf, HDR_MINUTE),
        le_u32(buf, HDR_SECOND),
    )
    .unwrap_or(SystemTime::UNIX_EPOCH);

    let fix = buf[HDR_STATUS] == b'A';
    let gnrmc = Gnrmc {
        time,
        lat: nmea_to_decimal(le_f64(buf, HDR_LAT), buf[HDR_LAT_NS] as char),
        lon: nmea_to_decimal(le_f64(buf, HDR_LON), buf[HDR_LON_EW] as char),
        speed: le_f64(buf, HDR_SPEED) * 1.852,
        track: le_f64(buf, HDR_COURSE),
        accel_z: le_i32(buf, HDR_ACCEL_Z) as f64 / 1000.0,
        accel_x: le_i32(buf, HDR_ACCEL_X) as f64 / 1000.0,
        accel_y: le_i32(buf, HDR_ACCEL_Y) as f64 / 1000.0,
    };
    Ok((gnrmc, fix))
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

// ── Cubic spline interpolation ────────────────────────────────────────────────

/// Natural cubic spline with clamped second derivatives (M₀ = Mₙ = 0).
pub(crate) struct CubicSpline {
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Second derivatives at each knot.
    ms: Vec<f64>,
}

impl CubicSpline {
    /// Build a natural cubic spline from sorted (x, y) pairs.
    ///
    /// `data` must be sorted in strictly increasing x order.
    pub(crate) fn new(data: &[(f64, f64)]) -> Self {
        assert!(data.len() >= 2, "CubicSpline requires at least 2 points");
        debug_assert!(
            data.windows(2).all(|w| w[0].0 < w[1].0),
            "CubicSpline input x values must be strictly increasing"
        );
        let n = data.len() - 1;
        let xs: Vec<f64> = data.iter().map(|&(x, _)| x).collect();
        let ys: Vec<f64> = data.iter().map(|&(_, y)| y).collect();
        let h: Vec<f64> = (0..n).map(|i| xs[i + 1] - xs[i]).collect();

        let ms = if n == 1 {
            // Two-point spline is linear; both boundary second derivatives are zero.
            vec![0.0, 0.0]
        } else {
            // Build the symmetric tridiagonal system for interior second derivatives
            // M_1 .. M_{n-1}  (the boundary values M_0 = M_n = 0 are fixed).
            //
            // For each interior knot `knot` (1 .. n-1), mapped to matrix row `row` (0 .. m-1):
            //   off_diag[row]  = h[knot]          (super-diagonal)
            //   diag[row]      = 2*(h[knot-1] + h[knot])
            //   rhs[row]       = 6 * (divided difference)
            //   sub-diagonal   = h[knot-1] = off_diag[row-1]   (symmetric)
            let m = n - 1;
            let mut diag = vec![0.0f64; m];
            let mut off_diag = vec![0.0f64; m - 1];
            let mut rhs = vec![0.0f64; m];

            for (row, knot) in (1..n).enumerate() {
                diag[row] = 2.0 * (h[knot - 1] + h[knot]);
                rhs[row] = 6.0
                    * ((ys[knot + 1] - ys[knot]) / h[knot]
                        - (ys[knot] - ys[knot - 1]) / h[knot - 1]);
                if row < m - 1 {
                    off_diag[row] = h[knot];
                }
            }

            let sol = solve_tridiagonal(&diag, &off_diag, &rhs);

            let mut ms = vec![0.0f64; n + 1];
            ms[1..=m].copy_from_slice(&sol);
            ms
        };

        CubicSpline { xs, ys, ms }
    }

    /// Evaluate the spline at x.
    ///
    /// Clamps to the endpoint value when `x` is outside `[xs[0], xs[n]]`.
    /// Callers that extend beyond the last GPS entry will therefore get the
    /// final recorded position rather than an extrapolated one.
    pub(crate) fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len() - 1;
        if x <= self.xs[0] {
            return self.ys[0];
        }
        if x >= self.xs[n] {
            return self.ys[n];
        }
        let i = self
            .xs
            .partition_point(|&xi| xi <= x)
            .saturating_sub(1)
            .min(n - 1);
        let h = self.xs[i + 1] - self.xs[i];
        let a = (self.xs[i + 1] - x) / h;
        let b = (x - self.xs[i]) / h;
        a * self.ys[i]
            + b * self.ys[i + 1]
            + (h * h / 6.0) * ((a * a * a - a) * self.ms[i] + (b * b * b - b) * self.ms[i + 1])
    }
}

/// Solve a symmetric tridiagonal linear system Ax = b using the Thomas algorithm.
///
/// `diag`     – main diagonal (length m)
/// `off_diag` – off-diagonal shared by both sub- and super-diagonal (length m-1)
/// `rhs`      – right-hand side (length m)
///
/// The natural cubic spline's tridiagonal system is symmetric, so a single
/// off-diagonal vector suffices.
fn solve_tridiagonal(diag: &[f64], off_diag: &[f64], rhs: &[f64]) -> Vec<f64> {
    let m = diag.len();
    debug_assert_eq!(off_diag.len(), m - 1);
    debug_assert_eq!(rhs.len(), m);

    let mut d = diag.to_vec();
    let mut r = rhs.to_vec();

    // Forward sweep: eliminate sub-diagonal entries.
    for i in 1..m {
        let factor = off_diag[i - 1] / d[i - 1];
        d[i] -= factor * off_diag[i - 1];
        r[i] -= factor * r[i - 1];
    }

    // Back substitution.
    let mut sol = vec![0.0f64; m];
    sol[m - 1] = r[m - 1] / d[m - 1];
    for i in (0..m - 1).rev() {
        sol[i] = (r[i] - off_diag[i] * sol[i + 1]) / d[i];
    }
    sol
}

// ── Shared post-processing helpers ────────────────────────────────────────────

/// Collect valid hardware-fix GPS entries as `(t, lat, lon)` where `t` is
/// the 0-based second index (= array position).
fn collect_position_pts(entries: &[RawGpsEntry]) -> Vec<(f64, f64, f64)> {
    entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.fix)
        .map(|(i, e)| (i as f64, e.gnrmc.lat, e.gnrmc.lon))
        .collect()
}

/// Build independent lat/lon cubic splines from `valid_pts`.
/// Returns `None` when fewer than 2 points are available.
fn build_position_splines(valid_pts: &[(f64, f64, f64)]) -> Option<(CubicSpline, CubicSpline)> {
    if valid_pts.len() < 2 {
        return None;
    }
    let lat_data: Vec<(f64, f64)> = valid_pts.iter().map(|&(t, lat, _)| (t, lat)).collect();
    let lon_data: Vec<(f64, f64)> = valid_pts.iter().map(|&(t, _, lon)| (t, lon)).collect();
    Some((CubicSpline::new(&lat_data), CubicSpline::new(&lon_data)))
}

/// Evaluate the position at time `t` from the available splines or fallback.
///
/// - Two or more hardware fixes: evaluate the cubic spline at `t`.
/// - Exactly one hardware fix: constant fill with that fix's position.
/// - No hardware fixes: return `(NaN, NaN)`.
fn eval_position(
    splines: Option<&(CubicSpline, CubicSpline)>,
    valid_pts: &[(f64, f64, f64)],
    t: f64,
) -> (f64, f64) {
    if let Some((lat_sp, lon_sp)) = splines {
        (lat_sp.eval(t), lon_sp.eval(t))
    } else if let Some(&(_, fill_lat, fill_lon)) = valid_pts.first() {
        (fill_lat, fill_lon)
    } else {
        (f64::NAN, f64::NAN)
    }
}

/// Compute wall-clock time for a frame at offset `t` seconds from `base_time`.
fn eval_time(base_time: Option<SystemTime>, t: f64) -> SystemTime {
    base_time
        .map(|bt| bt + Duration::from_secs_f64(t))
        .unwrap_or(SystemTime::UNIX_EPOCH + Duration::from_secs_f64(t))
}

// ── Heading interpolation helpers ────────────────────────────────────────────

/// Collect track (heading) control points from hardware-fix entries where
/// speed exceeds 5 km/h.  Returns `(t_sec, unwrapped_track_deg)` pairs,
/// where `t_sec` is the 0-based second index and the angles are continuously
/// unwrapped (no 360 → 0 jumps) so a cubic spline can interpolate them.
fn collect_track_pts(entries: &[RawGpsEntry]) -> Vec<(f64, f64)> {
    let mut pts: Vec<(f64, f64)> = Vec::new();
    for (i, e) in entries.iter().enumerate() {
        if !e.fix || e.gnrmc.speed < 5.0 || e.gnrmc.track.is_nan() {
            continue;
        }
        let unwrapped = if let Some(&(_, prev_unwrapped)) = pts.last() {
            let prev_wrapped = prev_unwrapped.rem_euclid(360.0);
            let mut diff = e.gnrmc.track - prev_wrapped;
            if diff > 180.0 {
                diff -= 360.0;
            } else if diff < -180.0 {
                diff += 360.0;
            }
            prev_unwrapped + diff
        } else {
            e.gnrmc.track
        };
        pts.push((i as f64, unwrapped));
    }
    pts
}

/// Build a cubic spline over unwrapped track angles.
/// Returns `None` when fewer than 2 points are available.
fn build_track_spline(track_pts: &[(f64, f64)]) -> Option<CubicSpline> {
    if track_pts.len() < 2 {
        return None;
    }
    Some(CubicSpline::new(track_pts))
}

/// Evaluate heading at time `t` from the track spline.
///
/// - Two or more high-speed fixes: evaluate the cubic spline and re-wrap to [0, 360).
/// - Exactly one: constant fill with that fix's track.
/// - None: return `NaN`.
fn eval_track(spline: Option<&CubicSpline>, track_pts: &[(f64, f64)], t: f64) -> f64 {
    if let Some(sp) = spline {
        sp.eval(t).rem_euclid(360.0)
    } else if let Some(&(_, track)) = track_pts.first() {
        track.rem_euclid(360.0)
    } else {
        f64::NAN
    }
}

// ── Speed spline helpers ──────────────────────────────────────────────────────

/// Collect speed control points from hardware-fix entries.
/// Returns `(t_sec, speed_kmh)` pairs.
fn collect_speed_pts(entries: &[RawGpsEntry]) -> Vec<(f64, f64)> {
    entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.fix && !e.gnrmc.speed.is_nan())
        .map(|(i, e)| (i as f64, e.gnrmc.speed))
        .collect()
}

/// Evaluate speed at time `t` from the spline, clamped to ≥ 0.
///
/// - Two or more fixes: cubic spline, clamped to 0.
/// - Exactly one: constant fill.
/// - None: return `NaN`.
fn eval_speed(spline: Option<&CubicSpline>, speed_pts: &[(f64, f64)], t: f64) -> f64 {
    if let Some(sp) = spline {
        sp.eval(t).max(0.0)
    } else if let Some(&(_, speed)) = speed_pts.first() {
        speed
    } else {
        f64::NAN
    }
}

// ── Per-frame upsampling ──────────────────────────────────────────────────────

/// Pre-built splines and their control-point fallbacks for one recording session.
struct GpsSplines {
    position_splines: Option<(CubicSpline, CubicSpline)>,
    position_pts: Vec<(f64, f64, f64)>,
    track_spline: Option<CubicSpline>,
    track_pts: Vec<(f64, f64)>,
    speed_spline: Option<CubicSpline>,
    speed_pts: Vec<(f64, f64)>,
}

impl GpsSplines {
    fn build(entries: &[RawGpsEntry]) -> Self {
        let position_pts = collect_position_pts(entries);
        let position_splines = build_position_splines(&position_pts);
        let speed_pts = collect_speed_pts(entries);
        let speed_spline = (speed_pts.len() >= 2).then(|| CubicSpline::new(&speed_pts));
        let track_pts = collect_track_pts(entries);
        let track_spline = build_track_spline(&track_pts);
        Self {
            position_splines,
            position_pts,
            track_spline,
            track_pts,
            speed_spline,
            speed_pts,
        }
    }

    fn eval_position(&self, t: f64) -> (f64, f64) {
        eval_position(self.position_splines.as_ref(), &self.position_pts, t)
    }

    fn eval_speed(&self, t: f64) -> f64 {
        eval_speed(self.speed_spline.as_ref(), &self.speed_pts, t)
    }

    fn eval_track(&self, t: f64) -> f64 {
        eval_track(self.track_spline.as_ref(), &self.track_pts, t)
    }
}

/// Compute a single upsampled `GpsPoint` for `frame_idx` at `target_fps`.
fn compute_frame_point(
    raw_entries: &[RawGpsEntry],
    splines: &GpsSplines,
    base_time: Option<SystemTime>,
    frame_idx: usize,
    target_fps: f64,
) -> GpsPoint {
    let record_count = raw_entries.len();
    let last_sec = (record_count - 1) as f64;
    let t = frame_idx as f64 / target_fps;

    // Find the floor 1 Hz index.
    let floor_idx = if t >= last_sec {
        record_count - 1
    } else {
        (t as usize).min(record_count - 1)
    };

    // Determine position source from the floor entry.
    let position = if raw_entries[floor_idx].fix {
        PositionSource::HardwareFix
    } else if !splines.position_pts.is_empty() {
        PositionSource::Interpolated
    } else {
        PositionSource::NoPosition
    };

    let (lat, lon) = splines.eval_position(t);
    let track = splines.eval_track(t);

    GpsPoint {
        //gnrmc: synthetic_gnrmc(lat, lon, track, eval_time(base_time, t)),
        gnrmc: Gnrmc {
            time: eval_time(base_time, t),
            lat,
            lon,
            speed: splines.eval_speed(t),
            track,
            accel_z: f64::NAN,
            accel_x: f64::NAN,
            accel_y: f64::NAN,
        },
        position,
    }
}

// ── GPS parsing + upsampling (single public entry point) ──────────────────────

/// Parse freeGPS data from the front clips and upsample to `target_fps`.
///
/// This is the single public entry point that replaces the former
/// `extract_gps_track` → `fill_gaps` → `upsample_to_fps` chain.
/// Splines are built once and used for both gap-filling and sub-second
/// interpolation.
///
/// Returns one `GpsPoint` per video frame.  `frame_index` is the 0-based
/// frame index at `target_fps`.
pub fn extract_smoothed_to_fps(
    front_paths: &[&Path],
    target_fps: f64,
    total_duration_sec: f64,
) -> Result<(Vec<GpsPoint>, usize)> {
    if front_paths.is_empty() {
        return Ok((Vec::new(), 0));
    }
    let raw_entries = extract_raw_entries(front_paths)?;
    // entries is non-empty: extract_raw_entries validates that every clip has
    // entry_count == duration_in_seconds, and front_paths is non-empty.
    let record_count = raw_entries.len();

    let base_time = Some(raw_entries[0].gnrmc.time);

    // Build splines once over all hardware-fix entries.
    let splines = GpsSplines::build(&raw_entries);

    let total_frames = (total_duration_sec * target_fps).ceil() as usize;
    let mut result = Vec::with_capacity(total_frames);

    for frame_idx in 0..total_frames {
        result.push(compute_frame_point(
            &raw_entries,
            &splines,
            base_time,
            frame_idx,
            target_fps,
        ));
    }

    Ok((result, record_count))
}
