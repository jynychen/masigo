use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};
use regex::Regex;
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct GpsPoint {
    pub lat: f64,
    pub lon: f64,
    /// Seconds from the start of the clip group
    pub time_sec: f64,
}

/// Run exiftool on front clips, parse embedded GPS into memory
pub fn extract_gps_points(front_paths: &[&Path]) -> Result<Vec<GpsPoint>> {
    let files: Vec<String> = front_paths
        .iter()
        .map(|p| {
            Ok(std::fs::canonicalize(p)
                .with_context(|| format!("Failed to canonicalize {}", p.display()))?
                .to_string_lossy()
                .into_owned())
        })
        .collect::<Result<_>>()?;

    let output = Command::new("exiftool")
        .args(["-ee", "-G3", "-n", "-json"])
        .args(&files)
        .output()
        .context("Failed to run exiftool")?;

    if !output.status.success() {
        bail!(
            "exiftool failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    parse_gps_json(&output.stdout)
}

fn parse_gps_json(data: &[u8]) -> Result<Vec<GpsPoint>> {
    let entries: Vec<Value> = serde_json::from_slice(data)?;
    let re = Regex::new(r"^Doc(\d+):GPS(Latitude|Longitude)$")?;

    let mut all_points: Vec<GpsPoint> = Vec::new();
    let mut global_doc_offset: u32 = 0;

    for entry in &entries {
        let obj = entry.as_object().context("Expected JSON object")?;

        let mut lat_map: BTreeMap<u32, f64> = BTreeMap::new();
        let mut lon_map: BTreeMap<u32, f64> = BTreeMap::new();

        for (key, value) in obj {
            if let Some(caps) = re.captures(key) {
                let doc: u32 = caps[1].parse()?;
                let val = value.as_f64().unwrap_or(0.0);
                match &caps[2] {
                    "Latitude" => {
                        lat_map.insert(doc, val);
                    }
                    "Longitude" => {
                        lon_map.insert(doc, val);
                    }
                    _ => {}
                }
            }
        }

        let mut max_doc = 0u32;
        for (&doc, &lat) in &lat_map {
            if let Some(&lon) = lon_map.get(&doc) {
                if lat.abs() > 0.001 && lon.abs() > 0.001 {
                    let time_sec = (global_doc_offset + doc - 1) as f64;
                    all_points.push(GpsPoint { lat, lon, time_sec });
                }
                if doc > max_doc {
                    max_doc = doc;
                }
            }
        }

        global_doc_offset += max_doc;
    }

    // Sort by time
    all_points.sort_by(|a, b| a.time_sec.partial_cmp(&b.time_sec).unwrap());
    Ok(all_points)
}

/// Interpolate GPS points from ~1Hz to target_fps
pub fn interpolate_gps(
    points: &[GpsPoint],
    target_fps: f64,
    total_duration_sec: f64,
) -> Vec<GpsPoint> {
    if points.is_empty() {
        return Vec::new();
    }

    let total_frames = (total_duration_sec * target_fps).ceil() as usize;
    let mut result = Vec::with_capacity(total_frames);

    for frame_idx in 0..total_frames {
        let t = frame_idx as f64 / target_fps;

        let point = if t <= points.first().unwrap().time_sec {
            points.first().unwrap().clone()
        } else if t >= points.last().unwrap().time_sec {
            points.last().unwrap().clone()
        } else {
            let pos = points.partition_point(|p| p.time_sec <= t);
            let p0 = &points[pos.saturating_sub(1)];
            let p1 = &points[pos.min(points.len() - 1)];

            if (p1.time_sec - p0.time_sec).abs() < 1e-9 {
                p0.clone()
            } else {
                let frac = (t - p0.time_sec) / (p1.time_sec - p0.time_sec);
                GpsPoint {
                    lat: p0.lat + (p1.lat - p0.lat) * frac,
                    lon: p0.lon + (p1.lon - p0.lon) * frac,
                    time_sec: t,
                }
            }
        };

        result.push(GpsPoint {
            time_sec: t,
            ..point
        });
    }

    result
}

/// Get video fps and duration using ffprobe
pub fn get_video_info(path: &Path) -> Result<(f64, f64)> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            path.to_str().context("non-UTF-8 path")?,
        ])
        .output()
        .context("Failed to run ffprobe")?;

    if !output.status.success() {
        bail!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let json: Value = serde_json::from_slice(&output.stdout)?;

    let streams = json["streams"]
        .as_array()
        .context("No streams in ffprobe output")?;
    let video_stream = streams
        .iter()
        .find(|s| s["codec_type"] == "video")
        .context("No video stream found")?;

    let fps_str = video_stream["r_frame_rate"]
        .as_str()
        .context("No r_frame_rate")?;
    let fps = if let Some((n, d)) = fps_str.split_once('/') {
        n.parse::<f64>()? / d.parse::<f64>()?
    } else {
        fps_str.parse::<f64>()?
    };

    let duration = json["format"]["duration"]
        .as_str()
        .context("No duration in ffprobe output")?
        .parse::<f64>()?;

    Ok((fps, duration))
}
