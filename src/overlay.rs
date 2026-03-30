use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};

use crate::gps::GpsPoint;

const OVERLAY_SIZE: u32 = 480;
const PADDING_RATIO: f64 = 0.05;

/// Render GPS track overlay video (transparent background, alpha-aware).
/// Returns path to the generated overlay video.
pub fn render_overlay_video(
    track: &[GpsPoint],
    output_dir: &Path,
    name: &str,
    fps: f64,
    duration_sec: f64,
) -> Result<PathBuf> {
    let output_file = output_dir.join(format!("{}_overlay.mov", name));
    let size = OVERLAY_SIZE;

    let interpolated = crate::gps::interpolate_gps(track, fps, duration_sec);
    if interpolated.is_empty() {
        bail!("No GPS frames to render");
    }

    // Pre-compute viewport (fixed for entire video, fits entire track)
    let viewport = Viewport::from_track(track, size);

    // Pre-compute raw track pixel coords (for drawing)
    let track_px: Vec<(f32, f32)> = track
        .iter()
        .map(|p| viewport.to_pixel(p.lat, p.lon))
        .collect();

    // Spawn ffmpeg: raw RGBA in → qtrle .mov out (preserves alpha)
    let mut ffmpeg = Command::new("ffmpeg")
        .args([
            "-y",
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
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("Failed to spawn ffmpeg for overlay")?;

    let stdin = ffmpeg.stdin.as_mut().context("No ffmpeg stdin")?;
    let total_frames = interpolated.len();
    let frame_len = (size * size * 4) as usize;
    let mut buf = vec![0u8; frame_len]; // reused across frames

    let result = (|| -> Result<()> {
        for (frame_idx, current) in interpolated.iter().enumerate() {
            render_frame(&mut buf, &track_px, track, current, &viewport, size);
            stdin.write_all(&buf)?;

            if frame_idx % 300 == 0 {
                eprint!(
                    "\r    overlay: {}/{} frames ({:.0}%)",
                    frame_idx,
                    total_frames,
                    frame_idx as f64 / total_frames as f64 * 100.0
                );
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

// ── Viewport: GPS → pixel mapping ──

struct Viewport {
    center_lat: f64,
    center_lon: f64,
    px_per_deg_lat: f64,
    px_per_deg_lon: f64,
    half: f64,
}

impl Viewport {
    fn from_track(track: &[GpsPoint], size: u32) -> Self {
        let min_lat = track.iter().map(|p| p.lat).fold(f64::INFINITY, f64::min);
        let max_lat = track
            .iter()
            .map(|p| p.lat)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_lon = track.iter().map(|p| p.lon).fold(f64::INFINITY, f64::min);
        let max_lon = track
            .iter()
            .map(|p| p.lon)
            .fold(f64::NEG_INFINITY, f64::max);

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

// ── Frame rendering ──

fn render_frame(
    buf: &mut [u8],
    track_px: &[(f32, f32)],
    track_gps: &[GpsPoint],
    current: &GpsPoint,
    viewport: &Viewport,
    size: u32,
) {
    buf.fill(0); // clear to fully transparent

    let current_time = current.time_sec;
    // Find split: segments 0..split are past, split.. are future.
    let split = track_gps.partition_point(|p| p.time_sec <= current_time);

    // Draw future track (gray, underneath)
    let gray = [128, 128, 128, 180u8];
    for i in split.saturating_sub(1)..track_px.len().saturating_sub(1) {
        draw_thick_line(
            buf,
            size,
            track_px[i].0,
            track_px[i].1,
            track_px[i + 1].0,
            track_px[i + 1].1,
            &gray,
            2,
        );
    }

    // Draw past track (white, on top)
    let white = [255, 255, 255, 230u8];
    for i in 0..split.min(track_px.len().saturating_sub(1)) {
        draw_thick_line(
            buf,
            size,
            track_px[i].0,
            track_px[i].1,
            track_px[i + 1].0,
            track_px[i + 1].1,
            &white,
            3,
        );
    }

    // Draw current position (red dot)
    let red = [255, 50, 50, 255u8];
    let (cx, cy) = viewport.to_pixel(current.lat, current.lon);
    draw_filled_circle(buf, size, cx as i32, cy as i32, 8, &red);
}

// ── Drawing primitives on raw RGBA buffer ──

fn draw_thick_line(
    buf: &mut [u8],
    size: u32,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    color: &[u8; 4],
    thickness: i32,
) {
    let w = size as i32;
    let h = size as i32;

    let dx = x1 - x0;
    let dy = y1 - y0;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 0.5 {
        return;
    }

    let steps = (len * 2.0) as i32;
    for step in 0..=steps {
        let t = step as f32 / steps as f32;
        let px = x0 + dx * t;
        let py = y0 + dy * t;

        for oy in -thickness..=thickness {
            for ox in -thickness..=thickness {
                if ox * ox + oy * oy <= thickness * thickness {
                    let ix = px as i32 + ox;
                    let iy = py as i32 + oy;
                    if ix >= 0 && ix < w && iy >= 0 && iy < h {
                        blend_pixel(buf, size, ix as u32, iy as u32, color);
                    }
                }
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
                    blend_pixel(buf, size, px as u32, py as u32, color);
                }
            }
        }
    }
}

fn blend_pixel(buf: &mut [u8], size: u32, x: u32, y: u32, src: &[u8; 4]) {
    let idx = ((y * size + x) * 4) as usize;

    let sa = src[3] as f32 / 255.0;
    let da = buf[idx + 3] as f32 / 255.0;
    let out_a = sa + da * (1.0 - sa);

    if out_a < 1e-6 {
        buf[idx] = 0;
        buf[idx + 1] = 0;
        buf[idx + 2] = 0;
        buf[idx + 3] = 0;
        return;
    }

    let blend =
        |s: u8, d: u8| -> u8 { ((s as f32 * sa + d as f32 * da * (1.0 - sa)) / out_a) as u8 };

    buf[idx] = blend(src[0], buf[idx]);
    buf[idx + 1] = blend(src[1], buf[idx + 1]);
    buf[idx + 2] = blend(src[2], buf[idx + 2]);
    buf[idx + 3] = (out_a * 255.0) as u8;
}
