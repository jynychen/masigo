use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde_json::Value;

/// Probe a video file for its frame rate (fps) and duration (seconds).
pub fn probe_video_info(path: &Path) -> Result<(f64, f64)> {
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
