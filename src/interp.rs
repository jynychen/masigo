use std::time::{Duration, SystemTime};

use crate::gps::{Gnrmc, GpsPoint};

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
    pub(crate) fn new(data: &[(f64, f64)]) -> Self {
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

// ── GPS gap filling ───────────────────────────────────────────────────────────

/// Fill gaps (`!fix`) in the 1 Hz track using cubic spline interpolation
/// over valid entries.  Filled entries receive a synthetic `Gnrmc` with
/// time derived from the frame-index offset.
pub(crate) fn fill_gaps(track: &[GpsPoint]) -> Vec<GpsPoint> {
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

// ── GPS upsampling ────────────────────────────────────────────────────────────

/// Unpack lat/lon and validity flag from a `GpsPoint`.
fn unpack_coords(p: &GpsPoint) -> (f64, f64, bool) {
    (p.gnrmc.lat, p.gnrmc.lon, p.gnrmc.fix)
}

/// Upsample 1 Hz GPS points to `target_fps` via cubic spline interpolation.
pub(crate) fn upsample_to_fps(
    points: &[GpsPoint],
    target_fps: f64,
    total_duration_sec: f64,
) -> Vec<GpsPoint> {
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
        let t = frame_idx as f64 / target_fps;

        // Find the floor 1 Hz point for metadata (file_path).
        let floor_pt = if t >= last_t {
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
