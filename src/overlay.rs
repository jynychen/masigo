use std::cell::RefCell;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use rayon::prelude::*;

use anyhow::{Context, Result, bail};

use crate::gps::GpsPoint;
use crate::interp::{fill_gaps, upsample_to_fps};

const PADDING_RATIO: f64 = 0.05;
/// Duration over which a newly-past segment fades from gray → white.
const FADE_SECS: f64 = 0.25;

// ── Visual parameters ─────────────────────────────────────────────────────────
// All colors are specified in linear light; alpha compositing uses
// premultiplied alpha in linear space (Porter-Duff convention).
// Conversion to premultiplied sRGB u8 happens once in the final pass.

/// Track line half-width for future (not-yet-traversed) segments.
/// Used as the SDF radius in `aa_coverage` for the thin track.
const TRACK_RADIUS_THIN: f32 = 2.0;
/// Track line half-width for past (already-traversed) segments.
/// Used as the SDF radius in `aa_coverage` for the thick track.
const TRACK_RADIUS_THICK: f32 = 3.0;
/// Arrow tip protrudes this many pixels forward from the position center.
const ARROW_FRONT: f32 = 10.0;
/// Arrow base sits this many pixels behind the position center.
const ARROW_BACK: f32 = 6.0;
/// Half-width of the arrow base in pixels.
const ARROW_HALF_BASE: f32 = 6.0;
/// White outline thickness around the arrow, in pixels.
const ARROW_OUTLINE: f32 = 1.5;
/// Current-position arrow color in linear light RGB.
const ARROW_COLOR_LIN: [f32; 3] = [1.0, 0.0288, 0.0288]; // sRGB (255,50,50)
/// Opacity of the future (gray) track.
const GRAY_A: f32 = 180.0 / 255.0;
/// Opacity of the fully-past (white) track.
const WHITE_A: f32 = 230.0 / 255.0;
/// Linear-light value of the gray track color (sRGB 192).
const GRAY_LIN: f32 = 0.5271;
/// Linear-light value of the white track color (sRGB 255).
const WHITE_LIN: f32 = 1.0;
/// Sub-pixel sample offset: pixel (px, py) is sampled at its centre (px+0.5, py+0.5).
/// Follows the OpenGL/Vulkan convention where pixel [i] covers [i, i+1).
const PIXEL_CENTER: f32 = 0.5;

thread_local! {
    /// Per-thread scratch buffer reused across frames to avoid per-frame heap allocation.
    static WORK_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Render a transparent GPS track overlay video (RGBA → qtrle .mov).
/// Returns path to the generated overlay video.
pub fn render_overlay_video(
    rmc_raw: &[GpsPoint],
    output_dir: &Path,
    name: &str,
    fps: f64,
    duration_sec: f64,
    size: u32,
) -> Result<PathBuf> {
    if fps <= 0.0 || duration_sec <= 0.0 {
        bail!("fps and duration_sec must be positive (got {fps}, {duration_sec})");
    }
    let output_file = output_dir.join(format!("{}_overlay.mov", name));

    // Fill missing GPS entries (gnrmc: None) with cubic spline interpolation.
    let rmc_filled = fill_gaps(rmc_raw);

    // Upsample 1 Hz GPS to match video frame rate for smooth animation.
    let rmc_upsampled = upsample_to_fps(&rmc_filled, fps, duration_sec);
    if rmc_upsampled.is_empty() {
        bail!("No GPS frames to render");
    }

    // Validity at 1 Hz from the raw (unfilled) track.
    let rmc_raw_valid: Vec<bool> = rmc_raw.iter().map(|p| p.gnrmc.fix).collect();

    // Pre-compute viewport (fixed for entire video, fits entire upsampled track).
    let viewport = Viewport::from_track(&rmc_upsampled, size)
        .context("No valid GPS fixes to compute overlay viewport")?;

    // Fps-resolution pixel track: (x, y, valid, heading_deg).
    // Heading is circularly interpolated from the raw 1 Hz Gnrmc.track field.
    let track_fps_px: Vec<(f32, f32, bool, f32)> = rmc_upsampled
        .iter()
        .enumerate()
        .map(|(frame_idx, p)| {
            let t = frame_idx as f64 / fps;
            let hz1_idx = t.floor() as usize;
            let valid = rmc_raw_valid.get(hz1_idx).copied().unwrap_or(false);
            let (x, y) = if valid {
                viewport.to_pixel(p.gnrmc.lat, p.gnrmc.lon)
            } else {
                (0.0, 0.0)
            };
            let i0 = hz1_idx.min(rmc_raw.len().saturating_sub(1));
            let i1 = (i0 + 1).min(rmc_raw.len().saturating_sub(1));
            let frac = t - t.floor();
            let h0 = rmc_raw[i0].gnrmc.track;
            let h1 = rmc_raw[i1].gnrmc.track;
            let heading = match (h0.is_nan(), h1.is_nan()) {
                (false, false) => {
                    let mut diff = h1 - h0;
                    if diff > 180.0 {
                        diff -= 360.0;
                    } else if diff < -180.0 {
                        diff += 360.0;
                    }
                    ((h0 + frac * diff).rem_euclid(360.0)) as f32
                }
                (false, true) => h0 as f32,
                (true, false) => h1 as f32,
                (true, true) => 0.0,
            };
            (x, y, valid, heading)
        })
        .collect();

    // Pre-compute anti-aliased track raster (once for entire video).
    let raster =
        TrackRaster::precompute(&track_fps_px, size, TRACK_RADIUS_THIN, TRACK_RADIUS_THICK);

    // Spawn ffmpeg: raw RGBA in → qtrle .mov out (preserves alpha)
    let mut ffmpeg = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            // ffmpeg has no premultiplied-rgba pixel format; "rgba" means a byte-only
            // channel reorder to "argb" on output – the premultiplied data passes through
            // unchanged and qtrle/QuickTime receives the correct premultiplied sRGB bytes.
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
            output_file
                .to_str()
                .context("overlay output path is not valid UTF-8")?,
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
    let fade_frames = (FADE_SECS * fps).round().max(1.0) as u32;

    println!("rendering {} frames", total_frames);

    let result = (|| -> Result<()> {
        for chunk_start in (0..total_frames).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(total_frames);
            let frames: Vec<Vec<u8>> = (chunk_start..chunk_end)
                .into_par_iter()
                .map(|frame_idx| {
                    let mut buf = vec![0u8; frame_len];
                    // Reuse per-thread f32 scratch buffer (the u8 output is still per-frame).
                    WORK_BUF.with(|cell| {
                        let mut work = cell.borrow_mut();
                        work.resize(frame_len, 0.0);
                        // cur_seg == frame_idx because track_fps_px has 1 segment per frame.
                        render_frame(
                            &mut buf,
                            &mut work,
                            &raster,
                            &track_fps_px,
                            frame_idx,
                            size,
                            fade_frames,
                        );
                    });
                    buf
                })
                .collect();

            for frame_buf in frames.iter() {
                stdin.write_all(frame_buf)?;
            }
        }
        Ok(())
    })();

    // Always close stdin and reap the child, even on write error.
    drop(ffmpeg.stdin.take());
    let status = ffmpeg.wait()?;

    result?;

    if !status.success() {
        bail!("ffmpeg overlay encoding failed");
    }

    Ok(output_file)
}

// ── Viewport: GPS → pixel mapping ─────────────────────────────────────────────
// Simplified Equirectangular projection with cos(lat) correction.
// Accurate enough for dashcam-scale tracks (< tens of km).

struct Viewport {
    center_lat: f64,
    center_lon: f64,
    px_per_deg_lat: f64,
    px_per_deg_lon: f64,
    half: f64,
}

impl Viewport {
    fn from_track(track: &[GpsPoint], size: u32) -> Option<Self> {
        let mut fixed = track
            .iter()
            .filter(|p| p.gnrmc.fix)
            .map(|p| (p.gnrmc.lat, p.gnrmc.lon));

        let (first_lat, first_lon) = fixed.next()?;
        let (min_lat, max_lat, min_lon, max_lon) = fixed.fold(
            (first_lat, first_lat, first_lon, first_lon),
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

        // cos(lat) compensates for longitude degree shrinking toward poles:
        // 1° lon ≈ 111 km × cos(lat), vs 1° lat ≈ 111 km always.
        let cos_lat = center_lat.to_radians().cos().abs().max(1e-6);
        let deg_span_lat = (max_lat - min_lat).max(1e-6);
        let deg_span_lon = (max_lon - min_lon).max(1e-6);

        // Normalize longitude span to equivalent latitude degrees.
        let deg_span_lon_normalized = deg_span_lon * cos_lat;

        // Uniform scale from the dominant axis; padding prevents track touching edges.
        let span = deg_span_lat.max(deg_span_lon_normalized);
        let usable = size as f64 * (1.0 - 2.0 * PADDING_RATIO);
        let px_per_deg_lat = usable / span;
        let px_per_deg_lon = px_per_deg_lat * cos_lat;

        Some(Viewport {
            center_lat,
            center_lon,
            px_per_deg_lat,
            px_per_deg_lon,
            half: size as f64 / 2.0,
        })
    }

    fn to_pixel(&self, lat: f64, lon: f64) -> (f32, f32) {
        let x = self.half + (lon - self.center_lon) * self.px_per_deg_lon;
        let y = self.half - (lat - self.center_lat) * self.px_per_deg_lat;
        (x as f32, y as f32)
    }
}

// ── Track pre-computation ─────────────────────────────────────────────────────
//
// Rasterise the entire polyline once into per-pixel SDF metadata so that each
// video frame is produced with a single flat O(size²) pixel loop instead of
// re-testing every segment.  Cost model:
//   precompute (once):  O(segments × pixels_per_segment)
//   per frame:          O(size²)  — lookup only, no geometry tests
//
// Storage layout: CSR (Compressed Sparse Row).
//   offsets[i]..offsets[i+1] indexes into `entries` for pixel i.
//   Segments within each pixel span are ordered by seg_idx ascending.

struct TrackRaster {
    /// Flattened `(seg_idx, distance)` entries: the SDF distance from the pixel
    /// centre to segment `seg_idx`.  Only the raw distance is stored; coverage
    /// is computed at render time via `aa_coverage(d, radius)` so the fade
    /// transition can interpolate both radius and color.
    entries: Vec<(u32, f32)>,
    /// Per-pixel start index into `entries` (CSR row-pointer array).
    /// Length = `size×size + 1`; the final sentinel gives the total entry count.
    offsets: Vec<u32>,
}

impl TrackRaster {
    /// Build the raster from the fps-resolution pixel track.
    ///
    /// `radius_thick` is the maximum half-width used for rendering; it determines
    /// the pixel bounding box.  Coverage at any specific radius is computed at
    /// render time via `aa_coverage`.
    /// `radius_thin` and `radius_thick` are the two rendering radii used at draw time;
    /// `radius_thick` (the larger) determines which pixels are recorded.
    fn precompute(
        track: &[(f32, f32, bool, f32)],
        size: u32,
        radius_thin: f32,
        radius_thick: f32,
    ) -> Self {
        debug_assert!(
            radius_thin <= radius_thick,
            "radius_thin must not exceed radius_thick"
        );
        let n = (size * size) as usize;
        let outer = radius_thick + PIXEL_CENTER; // bounding-box expansion

        // Pass 1 (CSR count): count how many segment entries touch each pixel.
        let mut counts = vec![0u32; n];
        for seg_idx in 0..track.len().saturating_sub(1) {
            let (x0, y0, v0, _) = track[seg_idx];
            let (x1, y1, v1, _) = track[seg_idx + 1];
            if !v0 || !v1 {
                continue;
            }
            let (px_min, px_max, py_min, py_max) =
                segment_pixel_bounds(x0, y0, x1, y1, outer, size);
            for py in py_min..=py_max {
                let row_off = (py * size) as usize;
                for px in px_min..=px_max {
                    let d = dist_to_segment(
                        px as f32 + PIXEL_CENTER,
                        py as f32 + PIXEL_CENTER,
                        x0,
                        y0,
                        x1,
                        y1,
                    );
                    if aa_coverage(d, radius_thick) > 0.0 {
                        counts[row_off + px as usize] += 1;
                    }
                }
            }
        }

        // Build CSR prefix-sum offsets from counts.
        let mut offsets = Vec::with_capacity(n + 1);
        let mut total = 0u32;
        for &c in &counts {
            offsets.push(total);
            total += c;
        }
        offsets.push(total);

        // Pass 2 (CSR fill): populate entries. Segments are iterated in index
        // order so each pixel's span is automatically sorted by seg_idx.
        let mut entries = vec![(0u32, 0.0f32); total as usize];
        let mut cursors = vec![0u32; n];

        for seg_idx in 0..track.len().saturating_sub(1) {
            let (x0, y0, v0, _) = track[seg_idx];
            let (x1, y1, v1, _) = track[seg_idx + 1];
            if !v0 || !v1 {
                continue;
            }
            let (px_min, px_max, py_min, py_max) =
                segment_pixel_bounds(x0, y0, x1, y1, outer, size);
            let si = seg_idx as u32;
            for py in py_min..=py_max {
                let row_off = (py * size) as usize;
                for px in px_min..=px_max {
                    let d = dist_to_segment(
                        px as f32 + PIXEL_CENTER,
                        py as f32 + PIXEL_CENTER,
                        x0,
                        y0,
                        x1,
                        y1,
                    );
                    if aa_coverage(d, radius_thick) > 0.0 {
                        let idx = row_off + px as usize;
                        let pos = (offsets[idx] + cursors[idx]) as usize;
                        entries[pos] = (si, d);
                        cursors[idx] += 1;
                    }
                }
            }
        }

        TrackRaster { entries, offsets }
    }
}

// ── Frame rendering ───────────────────────────────────────────────────────────

/// Render one overlay frame into `buf` (RGBA u8, premultiplied sRGB).
///
/// `work` is a f32 scratch buffer of length `size²×4`; it is zeroed on entry.
/// All compositing is performed in linear-light premultiplied-alpha f32; the
/// final pass converts to premultiplied sRGB u8 via:
///   un-premultiply → linear→sRGB → re-premultiply.
///
/// `cur_seg` is the past/future boundary: segment `s` is past iff `s < cur_seg`,
/// future iff `s >= cur_seg`.  The caller maps `frame_idx` to `cur_seg`
/// (1:1 when the track has one segment per frame).
fn render_frame(
    buf: &mut [u8],
    work: &mut [f32],
    raster: &TrackRaster,
    track_fps_px: &[(f32, f32, bool, f32)],
    frame_idx: usize,
    size: u32,
    fade_frames: u32,
) {
    work.fill(0.0);
    let cur_seg = frame_idx as u32;
    let npx = (size * size) as usize;

    // Layer 1: Track – future (bottom) then past (top) via src-over.
    //
    // Past is drawn on top so its brighter/thicker appearance dominates at
    // self-crossing points.  Each past segment is composited individually
    // with its own fade value so that old-vs-recent segments never share a
    // single color.
    for i in 0..npx {
        let start = raster.offsets[i] as usize;
        let end = raster.offsets[i + 1] as usize;
        if start == end {
            continue;
        }

        let off = i * 4;

        // Sub-layer 1a: Future track (thin, gray).
        // All future segments share the same color, so coverage union
        // (P(A∪B) = A + B(1−A)) is used instead of per-segment src-over.
        // This prevents opacity build-up where the track crosses itself.
        let mut future_cov = 0.0f32;
        for &(seg, d) in &raster.entries[start..end] {
            if seg >= cur_seg {
                let cn = aa_coverage(d, TRACK_RADIUS_THIN);
                future_cov = future_cov + cn * (1.0 - future_cov);
            }
        }
        if future_cov > 0.0 {
            composite_gray(work, off, GRAY_LIN, future_cov * GRAY_A);
        }

        // Sub-layer 1b: Past track (fully faded, uniform white).
        // Same coverage-union strategy as future track — all fully-faded
        // segments share the same (white, thick) appearance.
        let mut past_full_cov = 0.0f32;
        for &(seg, d) in &raster.entries[start..end] {
            if seg >= cur_seg {
                continue;
            }
            if (cur_seg - seg) >= fade_frames {
                let ct = aa_coverage(d, TRACK_RADIUS_THICK);
                past_full_cov = past_full_cov + ct * (1.0 - past_full_cov);
            }
        }
        if past_full_cov > 0.0 {
            composite_gray(work, off, WHITE_LIN, past_full_cov * WHITE_A);
        }

        // Sub-layer 1c: Still-fading segments (within FADE_SECS of cursor).
        // Each segment has a unique interpolated color/radius (gray→white,
        // thin→thick), so coverage-union cannot be used; Porter-Duff src-over
        // is applied per segment.  Mild opacity build-up at self-crossings
        // is accepted — the fade window is short (FADE_SECS) and subtle.
        for &(seg, d) in &raster.entries[start..end] {
            if seg >= cur_seg {
                continue;
            }
            let age = cur_seg - seg;
            if age >= fade_frames {
                continue;
            }
            let t = age as f32 / fade_frames as f32;
            let ct = aa_coverage(d, TRACK_RADIUS_THICK);
            let cn = aa_coverage(d, TRACK_RADIUS_THIN);
            let c_lin = GRAY_LIN + t * (WHITE_LIN - GRAY_LIN);
            let cov = cn + t * (ct - cn);
            let a = cov * (GRAY_A + t * (WHITE_A - GRAY_A));
            composite_gray(work, off, c_lin, a);
        }
    }

    // Layer 2: Current-position arrow (Porter-Duff src-over in premultiplied space).
    if let Some(&(cx, cy, true, heading_deg)) = track_fps_px.get(frame_idx) {
        draw_aa_arrow(work, size, cx, cy, heading_deg, ARROW_COLOR_LIN, 1.0);
    }

    // Final pass: premultiplied linear f32 → premultiplied sRGB u8.
    //   step 1: un-premultiply   (÷α to recover straight linear RGB)
    //   step 2: linear → sRGB    (gamma curve; must operate on straight RGB)
    //   step 3: re-premultiply   (×α for qtrle / QuickTime output)
    // Outputting straight alpha here would cause double-premultiplication
    // when the player composites the overlay onto the video.
    for i in 0..npx {
        let si = i * 4;
        let a = work[si + 3];
        if a < 1.0 / 255.0 {
            buf[si] = 0;
            buf[si + 1] = 0;
            buf[si + 2] = 0;
            buf[si + 3] = 0;
        } else {
            let a_out = a.min(1.0);
            let inv_a = 1.0 / a_out;
            // un-premultiply → linear-to-sRGB → re-premultiply
            let r = linear_to_srgb(work[si] * inv_a) * a_out;
            let g = linear_to_srgb(work[si + 1] * inv_a) * a_out;
            let b = linear_to_srgb(work[si + 2] * inv_a) * a_out;
            buf[si] = (r * 255.0 + 0.5) as u8;
            buf[si + 1] = (g * 255.0 + 0.5) as u8;
            buf[si + 2] = (b * 255.0 + 0.5) as u8;
            buf[si + 3] = (a_out * 255.0 + 0.5) as u8;
        }
    }
}

// ── Drawing helpers ───────────────────────────────────────────────────────────

/// Shortest distance from point `(px, py)` to the line segment `(x0,y0)→(x1,y1)`.
/// Used as the SDF (Signed Distance Field) value for anti-aliased rendering.
#[inline]
fn dist_to_segment(px: f32, py: f32, x0: f32, y0: f32, x1: f32, y1: f32) -> f32 {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len_sq = dx * dx + dy * dy;
    // Degenerate segment guard: if endpoints coincide, treat as point distance.
    if len_sq < 1e-6 {
        return ((px - x0).powi(2) + (py - y0).powi(2)).sqrt();
    }
    let t = (((px - x0) * dx + (py - y0) * dy) / len_sq).clamp(0.0, 1.0);
    let proj_x = x0 + t * dx;
    let proj_y = y0 + t * dy;
    ((px - proj_x).powi(2) + (py - proj_y).powi(2)).sqrt()
}

/// Axis-aligned bounding box (pixel coordinates) for a line segment expanded by
/// `outer` pixels and clamped to `[0, size−1]`.
/// Returns `(px_min, px_max, py_min, py_max)`.
#[inline]
fn segment_pixel_bounds(
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    outer: f32,
    size: u32,
) -> (u32, u32, u32, u32) {
    let s = (size - 1) as f32;
    let px_min = (x0.min(x1) - outer).floor().max(0.0) as u32;
    let px_max = (x0.max(x1) + outer).ceil().min(s) as u32;
    let py_min = (y0.min(y1) - outer).floor().max(0.0) as u32;
    let py_max = (y0.max(y1) + outer).ceil().min(s) as u32;
    (px_min, px_max, py_min, py_max)
}

/// Hermite smoothstep: 0 at `edge0`, 1 at `edge1` (GLSL convention, edge0 < edge1).
#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Anti-aliased coverage for a distance-field value `d` against a shape edge at `radius`.
/// Returns 1.0 inside (d ≤ radius − 0.5), 0.0 outside (d ≥ radius + 0.5).
/// The 1 px transition band (smoothstep) replaces multi-sample AA.
#[inline]
fn aa_coverage(d: f32, radius: f32) -> f32 {
    1.0 - smoothstep(radius - 0.5, radius + 0.5, d)
}

/// Linear light → sRGB (clamped to 0..1).
#[inline]
fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Porter-Duff src-over for a grayscale source onto a premultiplied-alpha linear f32 buffer.
/// Formula: C_out = C_src·α + C_dst·(1−α),  α_out = α + α_dst·(1−α).
/// `c_lin` is the straight linear-light luminance (identical for R, G, B), `alpha` is the
/// source opacity.  Premultiplication is performed internally.
#[inline]
fn composite_gray(work: &mut [f32], off: usize, c_lin: f32, alpha: f32) {
    let pm = c_lin * alpha;
    let inv = 1.0 - alpha;
    work[off] = pm + work[off] * inv;
    work[off + 1] = pm + work[off + 1] * inv;
    work[off + 2] = pm + work[off + 2] * inv;
    work[off + 3] = alpha + work[off + 3] * inv;
}

/// Porter-Duff src-over for a colored source onto a premultiplied-alpha linear f32 buffer.
/// `rgb_lin` is the straight linear-light RGB, `alpha` is the source opacity.
#[inline]
fn composite_color(work: &mut [f32], off: usize, rgb_lin: [f32; 3], alpha: f32) {
    let inv = 1.0 - alpha;
    work[off] = rgb_lin[0] * alpha + work[off] * inv;
    work[off + 1] = rgb_lin[1] * alpha + work[off + 1] * inv;
    work[off + 2] = rgb_lin[2] * alpha + work[off + 2] * inv;
    work[off + 3] = alpha + work[off + 3] * inv;
}

/// Draw an anti-aliased filled arrow onto a **premultiplied-alpha linear f32**
/// working buffer via Porter-Duff source-over.
///
/// The arrow is an isosceles triangle pointing in `heading_deg` (clockwise from north).
/// `ARROW_FRONT` pixels protrude forward from `(cx, cy)`; the base is `ARROW_BACK`
/// pixels behind and `2 × ARROW_HALF_BASE` pixels wide.
fn draw_aa_arrow(
    work: &mut [f32],
    size: u32,
    cx: f32,
    cy: f32,
    heading_deg: f32,
    rgb_lin: [f32; 3],
    alpha: f32,
) {
    let theta = heading_deg.to_radians();
    // Forward unit vector in screen space (y-down: north = −y).
    let fx = theta.sin();
    let fy = -theta.cos();
    // Right perpendicular (screen-right when heading north = +x direction).
    let rx = -fy;
    let ry = fx;

    // Triangle vertices in CCW winding order (tip → base-right → base-left).
    // CCW ensures tri_edge_dist returns positive values for interior points.
    let tip = (cx + fx * ARROW_FRONT, cy + fy * ARROW_FRONT);
    let base_r = (
        cx - fx * ARROW_BACK + rx * ARROW_HALF_BASE,
        cy - fy * ARROW_BACK + ry * ARROW_HALF_BASE,
    );
    let base_l = (
        cx - fx * ARROW_BACK - rx * ARROW_HALF_BASE,
        cy - fy * ARROW_BACK - ry * ARROW_HALF_BASE,
    );

    // Bounding box expanded by outline + 0.5 px AA margin.
    let margin = ARROW_OUTLINE + 1.0;
    let s = (size - 1) as f32;
    let px_min = (tip.0.min(base_r.0).min(base_l.0) - margin)
        .floor()
        .max(0.0) as u32;
    let px_max = (tip.0.max(base_r.0).max(base_l.0) + margin).ceil().min(s) as u32;
    let py_min = (tip.1.min(base_r.1).min(base_l.1) - margin)
        .floor()
        .max(0.0) as u32;
    let py_max = (tip.1.max(base_r.1).max(base_l.1) + margin).ceil().min(s) as u32;

    for py in py_min..=py_max {
        for px in px_min..=px_max {
            let p = (px as f32 + PIXEL_CENTER, py as f32 + PIXEL_CENTER);
            // Triangle SDF: positive inside, negative outside.
            let sdf = tri_edge_dist(p, tip, base_r)
                .min(tri_edge_dist(p, base_r, base_l))
                .min(tri_edge_dist(p, base_l, tip));
            let idx = ((py * size + px) * 4) as usize;
            // White outline: expand the shape by ARROW_OUTLINE pixels.
            let cov_out = smoothstep(-0.5, 0.5, sdf + ARROW_OUTLINE);
            if cov_out > 0.0 {
                composite_color(work, idx, [1.0, 1.0, 1.0], cov_out * alpha);
            }
            // Red fill composited on top of the outline.
            let cov_fill = smoothstep(-0.5, 0.5, sdf);
            if cov_fill > 0.0 {
                composite_color(work, idx, rgb_lin, cov_fill * alpha);
            }
        }
    }
}

/// Signed distance from point `p` to the left of directed edge `a → b`.
/// Returns a positive value when `p` is on the left side of the edge,
/// which is the interior side for a CCW-wound triangle.
#[inline]
fn tri_edge_dist(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-6 {
        return 0.0;
    }
    ((p.0 - a.0) * (-dy) + (p.1 - a.1) * dx) / len
}
