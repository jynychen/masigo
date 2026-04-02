use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime};

use rayon::prelude::*;

use anyhow::{Context, Result, bail};

use crate::gps::{Gnrmc, GpsPoint};

const OVERLAY_SIZE: u32 = 480;
const PADDING_RATIO: f64 = 0.05;

/// Render a transparent GPS track overlay video (RGBA → qtrle .mov).
/// Returns path to the generated overlay video.
pub fn render_overlay_video(
    rmc_raw: &[GpsPoint],
    output_dir: &Path,
    name: &str,
    fps: f64,
    duration_sec: f64,
) -> Result<PathBuf> {
    let output_file = output_dir.join(format!("{}_overlay.mov", name));
    let size = OVERLAY_SIZE;

    // Fill missing GPS entries (gnrmc: None) with cubic spline interpolation.
    let rmc_filled = fill_gaps(rmc_raw);

    // Upsample 1 Hz GPS to match video frame rate for smooth animation.
    let rmc_upsampled = upsample_to_fps(&rmc_filled, fps, duration_sec);
    if rmc_upsampled.is_empty() {
        bail!("No GPS frames to render");
    }

    // Pre-compute viewport (fixed for entire video, fits entire upsampled track)
    let viewport = Viewport::from_track(&rmc_upsampled, size);

    // Validity at 1 Hz from the raw (unfilled) track.
    let rmc_raw_valid: Vec<bool> = rmc_raw.iter().map(|p| p.gnrmc.fix).collect();

    // Fps-resolution pixel track: position from upsampled spline, validity from rmc_raw.
    let track_fps_px: Vec<(f32, f32, bool)> = rmc_upsampled
        .iter()
        .enumerate()
        .map(|(frame_idx, p)| {
            let (x, y) = if p.gnrmc.fix {
                viewport.to_pixel(p.gnrmc.lat, p.gnrmc.lon)
            } else {
                (0.0, 0.0)
            };
            let hz1_idx = ((frame_idx + 1) as f64 / fps).floor() as usize;
            let valid = rmc_raw_valid.get(hz1_idx).copied().unwrap_or(false);
            (x, y, valid)
        })
        .collect();

    // Spawn ffmpeg: raw RGBA in → qtrle .mov out (preserves alpha)
    let mut ffmpeg = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            &format!("{size}x{size}"),
            "-r",
            &format!("{fps}"),
            "-i",
            "pipe:0",
            "-c:v",
            "qtrle",
            "-pix_fmt",
            "argb",
            output_file.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .spawn()
        .context("Failed to spawn ffmpeg for overlay")?;

    let stdin = ffmpeg.stdin.as_mut().context("No ffmpeg stdin")?;
    let total_frames = rmc_upsampled.len();
    let frame_len = (size * size * 4) as usize;

    // Render frames in parallel chunks, write each chunk sequentially to ffmpeg.
    // Chunk size = 2× thread count keeps all cores busy without buffering too much.
    let chunk_size = rayon::current_num_threads() * 2;

    let result = (|| -> Result<()> {
        for chunk_start in (0..total_frames).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(total_frames);
            let frames: Vec<Vec<u8>> = (chunk_start..chunk_end)
                .into_par_iter()
                .map(|frame_idx| {
                    let mut buf = vec![0u8; frame_len];
                    render_frame(&mut buf, &track_fps_px, frame_idx, size);
                    buf
                })
                .collect();

            for frame_buf in frames.iter() {
                stdin.write_all(frame_buf)?;
            }
        }
        Ok(())
    })();

    eprintln!(
        "\r    overlay: {}/{} frames (100%)",
        total_frames, total_frames
    );

    // Always close stdin and reap the child, even on write error.
    drop(ffmpeg.stdin.take());
    let status = ffmpeg.wait()?;

    result?;

    if !status.success() {
        bail!("ffmpeg overlay encoding failed");
    }

    Ok(output_file)
}

// ── GPS interpolation ─────────────────────────────────────────────────────────

/// Unpack lat/lon and validity flag from a `GpsPoint`.
fn unpack_coords(p: &GpsPoint) -> (f64, f64, bool) {
    (p.gnrmc.lat, p.gnrmc.lon, p.gnrmc.fix)
}

/// Upsample 1 Hz GPS points to `target_fps` via cubic spline interpolation.
fn upsample_to_fps(points: &[GpsPoint], target_fps: f64, total_duration_sec: f64) -> Vec<GpsPoint> {
    if points.is_empty() {
        return Vec::new();
    }

    let base_index = points.first().unwrap().frame_index;
    let point_secs: Vec<f64> = points
        .iter()
        .map(|p| (p.frame_index - base_index) as f64)
        .collect();

    let base_time = points.first().map(|p| p.gnrmc.time);

    // Collect valid points for spline construction.
    let valid_pts: Vec<(f64, f64, f64)> = points
        .iter()
        .filter(|p| p.gnrmc.fix)
        .map(|p| {
            let t = (p.frame_index - base_index) as f64;
            (t, p.gnrmc.lat, p.gnrmc.lon)
        })
        .collect();

    // Build cubic splines for lat and lon (need ≥ 2 valid points).
    let splines = if valid_pts.len() >= 2 {
        let lat_data: Vec<(f64, f64)> = valid_pts.iter().map(|&(t, lat, _)| (t, lat)).collect();
        let lon_data: Vec<(f64, f64)> = valid_pts.iter().map(|&(t, _, lon)| (t, lon)).collect();
        Some((CubicSpline::new(&lat_data), CubicSpline::new(&lon_data)))
    } else {
        None
    };

    let last_t = *point_secs.last().unwrap();
    let total_frames = (total_duration_sec * target_fps).ceil() as usize;
    let mut result = Vec::with_capacity(total_frames);

    for frame_idx in 0..total_frames {
        let t = (frame_idx + 1) as f64 / target_fps;

        // Find the floor 1 Hz point for metadata (file_path).
        let floor_pt = if t <= 0.0 {
            points.first().unwrap()
        } else if t >= last_t {
            points.last().unwrap()
        } else {
            let pos = point_secs.partition_point(|&s| s <= t);
            &points[pos.saturating_sub(1)]
        };

        // Evaluate position via spline (or fallback to single-point / unpack).
        let (lat, lon, valid) = if let Some((ref lat_sp, ref lon_sp)) = splines {
            (lat_sp.eval(t), lon_sp.eval(t), true)
        } else {
            unpack_coords(floor_pt)
        };

        let time = base_time
            .map(|bt| bt + Duration::from_secs_f64(t))
            .unwrap_or(SystemTime::UNIX_EPOCH + Duration::from_secs_f64(t));
        let gnrmc = Gnrmc {
            fix: valid,
            time,
            lat: if valid { lat } else { f64::NAN },
            lon: if valid { lon } else { f64::NAN },
            speed: f64::NAN,
            track: f64::NAN,
            accel_z: f64::NAN,
            accel_x: f64::NAN,
            accel_y: f64::NAN,
        };

        result.push(GpsPoint {
            gnrmc,
            frame_index: frame_idx,
            file_path: floor_pt.file_path.clone(),
        });
    }

    result
}

// ── Cubic spline interpolation ────────────────────────────────────────────────

/// Natural cubic spline with clamped second derivatives (M₀ = Mₙ = 0).
struct CubicSpline {
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Second derivatives at each knot.
    ms: Vec<f64>,
}

impl CubicSpline {
    /// Build a natural cubic spline from sorted (x, y) pairs.
    fn new(data: &[(f64, f64)]) -> Self {
        assert!(data.len() >= 2, "CubicSpline requires at least 2 points");
        let n = data.len() - 1;
        let xs: Vec<f64> = data.iter().map(|&(x, _)| x).collect();
        let ys: Vec<f64> = data.iter().map(|&(_, y)| y).collect();
        let h: Vec<f64> = (0..n).map(|i| xs[i + 1] - xs[i]).collect();

        let ms = if n == 1 {
            vec![0.0, 0.0]
        } else {
            // Build tridiagonal system for M_1 .. M_{n-1} (size m = n-1).
            // Boundary: M_0 = M_n = 0 (natural spline).
            let m = n - 1;
            let mut diag = vec![0.0f64; m];
            let mut upper = vec![0.0f64; m - 1];
            let mut rhs = vec![0.0f64; m];

            for i in 0..m {
                let ki = i + 1;
                diag[i] = 2.0 * (h[ki - 1] + h[ki]);
                rhs[i] = 6.0 * ((ys[ki + 1] - ys[ki]) / h[ki] - (ys[ki] - ys[ki - 1]) / h[ki - 1]);
                if i < m - 1 {
                    upper[i] = h[ki];
                }
            }

            // Thomas algorithm: forward sweep
            let mut diag_m = diag;
            let mut rhs_m = rhs;
            for i in 1..m {
                let ki = i + 1;
                let factor = h[ki - 1] / diag_m[i - 1];
                diag_m[i] -= factor * upper[i - 1];
                rhs_m[i] -= factor * rhs_m[i - 1];
            }

            // Back substitution
            let mut sol = vec![0.0f64; m];
            sol[m - 1] = rhs_m[m - 1] / diag_m[m - 1];
            for i in (0..m - 1).rev() {
                sol[i] = (rhs_m[i] - upper[i] * sol[i + 1]) / diag_m[i];
            }

            let mut ms = vec![0.0f64; n + 1];
            ms[1..=m].copy_from_slice(&sol[..m]);
            ms
        };

        CubicSpline { xs, ys, ms }
    }

    /// Evaluate the spline at x. Clamps to endpoint values outside the range.
    fn eval(&self, x: f64) -> f64 {
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

/// Fill gaps (`gnrmc: None`) in the 1 Hz track using cubic spline
/// interpolation over valid entries.  Filled entries receive a synthetic
/// `GnrmcFix` with time derived from the frame-index offset.
fn fill_gaps(track: &[GpsPoint]) -> Vec<GpsPoint> {
    let base_index = track.first().map(|p| p.frame_index).unwrap_or(0);

    // Base time from the first entry (time is always valid).
    let base_time = track.first().map(|p| p.gnrmc.time);

    let valid_pts: Vec<(f64, f64, f64)> = track
        .iter()
        .filter(|p| p.gnrmc.fix)
        .map(|p| {
            let t = (p.frame_index - base_index) as f64;
            (t, p.gnrmc.lat, p.gnrmc.lon)
        })
        .collect();

    if valid_pts.is_empty() {
        return track.to_vec();
    }

    let fill_time = |p: &GpsPoint| -> SystemTime {
        let t = (p.frame_index - base_index) as f64;
        base_time
            .map(|bt| bt + Duration::from_secs_f64(t))
            .unwrap_or(SystemTime::UNIX_EPOCH + Duration::from_secs_f64(t))
    };

    if valid_pts.len() == 1 {
        let (_, fill_lat, fill_lon) = valid_pts[0];
        return track
            .iter()
            .map(|p| {
                if p.gnrmc.fix {
                    p.clone()
                } else {
                    GpsPoint {
                        gnrmc: Gnrmc {
                            fix: false,
                            lat: fill_lat,
                            lon: fill_lon,
                            time: fill_time(p),
                            speed: f64::NAN,
                            track: f64::NAN,
                            accel_z: f64::NAN,
                            accel_x: f64::NAN,
                            accel_y: f64::NAN,
                        },
                        ..p.clone()
                    }
                }
            })
            .collect();
    }

    let lat_data: Vec<(f64, f64)> = valid_pts.iter().map(|&(t, lat, _)| (t, lat)).collect();
    let lon_data: Vec<(f64, f64)> = valid_pts.iter().map(|&(t, _, lon)| (t, lon)).collect();
    let lat_spline = CubicSpline::new(&lat_data);
    let lon_spline = CubicSpline::new(&lon_data);

    track
        .iter()
        .map(|p| {
            if p.gnrmc.fix {
                p.clone()
            } else {
                let t = (p.frame_index - base_index) as f64;
                GpsPoint {
                    gnrmc: Gnrmc {
                        fix: false,
                        lat: lat_spline.eval(t),
                        lon: lon_spline.eval(t),
                        time: fill_time(p),
                        speed: f64::NAN,
                        track: f64::NAN,
                        accel_z: f64::NAN,
                        accel_x: f64::NAN,
                        accel_y: f64::NAN,
                    },
                    ..p.clone()
                }
            }
        })
        .collect()
}

// ── Viewport: GPS → pixel mapping ─────────────────────────────────────────────

struct Viewport {
    center_lat: f64,
    center_lon: f64,
    px_per_deg_lat: f64,
    px_per_deg_lon: f64,
    half: f64,
}

impl Viewport {
    fn from_track(track: &[GpsPoint], size: u32) -> Self {
        let (min_lat, max_lat, min_lon, max_lon) = track
            .iter()
            .filter(|p| p.gnrmc.fix)
            .map(|p| (p.gnrmc.lat, p.gnrmc.lon))
            .fold(
                (
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                ),
                |(mn_la, mx_la, mn_lo, mx_lo), (lat, lon)| {
                    (
                        mn_la.min(lat),
                        mx_la.max(lat),
                        mn_lo.min(lon),
                        mx_lo.max(lon),
                    )
                },
            );

        let center_lat = (min_lat + max_lat) / 2.0;
        let center_lon = (min_lon + max_lon) / 2.0;

        let cos_lat = center_lat.to_radians().cos();
        let deg_span_lat = (max_lat - min_lat).max(1e-6);
        let deg_span_lon = (max_lon - min_lon).max(1e-6);

        // Normalize longitude span to equivalent latitude degrees
        let deg_span_lon_normalized = deg_span_lon * cos_lat;

        // Use the larger span to determine scale, then add padding
        let span = deg_span_lat.max(deg_span_lon_normalized);
        let usable = size as f64 * (1.0 - 2.0 * PADDING_RATIO);
        let px_per_deg_lat = usable / span;
        let px_per_deg_lon = px_per_deg_lat * cos_lat;

        Viewport {
            center_lat,
            center_lon,
            px_per_deg_lat,
            px_per_deg_lon,
            half: size as f64 / 2.0,
        }
    }

    fn to_pixel(&self, lat: f64, lon: f64) -> (f32, f32) {
        let x = self.half + (lon - self.center_lon) * self.px_per_deg_lon;
        let y = self.half - (lat - self.center_lat) * self.px_per_deg_lat;
        (x as f32, y as f32)
    }
}

// ── Frame rendering ───────────────────────────────────────────────────────────

fn render_frame(
    buf: &mut [u8],
    track_fps_px: &[(f32, f32, bool)],
    frame_idx: usize,
    size: u32,
) {
    buf.fill(0); // clear to fully transparent

    // Draw future track (gray, underneath).
    let gray = [128, 128, 128, 180u8];
    for i in frame_idx..track_fps_px.len().saturating_sub(1) {
        let (x0, y0, _) = track_fps_px[i];
        let (x1, y1, _) = track_fps_px[i + 1];
        draw_thick_line(buf, size, (x0, y0), (x1, y1), &gray, 2);
    }

    // Draw past track (white, on top).
    let white = [255, 255, 255, 230u8];
    for i in 0..frame_idx {
        let (x0, y0, _) = track_fps_px[i];
        let (x1, y1, _) = track_fps_px[i + 1];
        draw_thick_line(buf, size, (x0, y0), (x1, y1), &white, 3);
    }

    // Draw current position (red dot) only when the raw fix was valid.
    if let Some(&(cx, cy, true)) = track_fps_px.get(frame_idx) {
        let red = [255, 50, 50, 255u8];
        draw_filled_circle(buf, size, cx as i32, cy as i32, 8, &red);
    }
}

// ── Drawing primitives ────────────────────────────────────────────────────────

fn draw_thick_line(
    buf: &mut [u8],
    size: u32,
    start: (f32, f32),
    end: (f32, f32),
    color: &[u8; 4],
    thickness: i32,
) {
    let w = size as i32;
    let h = size as i32;

    let (x0, y0) = start;
    let (x1, y1) = end;
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 0.5 {
        draw_filled_circle(buf, size, x0 as i32, y0 as i32, thickness, color);
        return;
    }

    // Pre-compute circle offsets once — avoids repeated ox²+oy²≤r² check per step.
    let offsets: Vec<(i32, i32)> = (-thickness..=thickness)
        .flat_map(|oy| {
            (-thickness..=thickness).filter_map(move |ox| {
                if ox * ox + oy * oy <= thickness * thickness {
                    Some((ox, oy))
                } else {
                    None
                }
            })
        })
        .collect();

    let steps = (len * 2.0) as i32;
    for step in 0..=steps {
        let t = step as f32 / steps as f32;
        let px = (x0 + dx * t) as i32;
        let py = (y0 + dy * t) as i32;

        for &(ox, oy) in &offsets {
            let ix = px + ox;
            let iy = py + oy;
            if ix >= 0 && ix < w && iy >= 0 && iy < h {
                alpha_blend_pixel(buf, size, ix as u32, iy as u32, color);
            }
        }
    }
}

fn draw_filled_circle(buf: &mut [u8], size: u32, cx: i32, cy: i32, radius: i32, color: &[u8; 4]) {
    let w = size as i32;
    let h = size as i32;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy <= radius * radius {
                let px = cx + dx;
                let py = cy + dy;
                if px >= 0 && px < w && py >= 0 && py < h {
                    alpha_blend_pixel(buf, size, px as u32, py as u32, color);
                }
            }
        }
    }
}

fn alpha_blend_pixel(buf: &mut [u8], size: u32, x: u32, y: u32, src: &[u8; 4]) {
    let idx = ((y * size + x) * 4) as usize;

    let sa = src[3] as u32;
    if sa == 0 {
        return;
    }

    let da = buf[idx + 3] as u32;

    // Fast path: destination transparent — no blending needed.
    if da == 0 {
        buf[idx] = src[0];
        buf[idx + 1] = src[1];
        buf[idx + 2] = src[2];
        buf[idx + 3] = sa as u8;
        return;
    }

    // Fast path: source fully opaque — just overwrite.
    if sa == 255 {
        buf[idx] = src[0];
        buf[idx + 1] = src[1];
        buf[idx + 2] = src[2];
        buf[idx + 3] = 255;
        return;
    }

    // General case: integer "source over" compositing.
    // out_a   = sa + da*(1 - sa/255)  = sa + da*inv_sa/255
    // out_rgb = (src_rgb*sa + dst_rgb*da_factor) / out_a
    //   where da_factor = da*inv_sa/255
    let inv_sa = 255 - sa;
    let da_factor = (da * inv_sa + 127) / 255;
    let out_a = sa + da_factor;

    let half = out_a / 2;
    buf[idx] = ((src[0] as u32 * sa + buf[idx] as u32 * da_factor + half) / out_a) as u8;
    buf[idx + 1] = ((src[1] as u32 * sa + buf[idx + 1] as u32 * da_factor + half) / out_a) as u8;
    buf[idx + 2] = ((src[2] as u32 * sa + buf[idx + 2] as u32 * da_factor + half) / out_a) as u8;
    buf[idx + 3] = out_a as u8;
}
