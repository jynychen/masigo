use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::clip::{ClipGroup, Direction};

/// Steps 3 & 4: Concatenate clips of a given direction using ffmpeg
pub fn concat_videos(
    group: &ClipGroup,
    output_dir: &Path,
    direction: Direction,
) -> Result<PathBuf> {
    let (suffix, paths) = match direction {
        Direction::Front => ("F", group.front_paths()),
        Direction::Rear => ("R", group.rear_paths()),
    };

    let concat_file = output_dir.join(format!("concat_{}.txt", suffix.to_lowercase()));
    let output_file = output_dir.join(format!("{}_{}.mp4", group.name, suffix));

    let content: String = paths
        .iter()
        .map(|p| {
            format!(
                "file '{}'",
                std::fs::canonicalize(p).unwrap().to_string_lossy()
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    std::fs::write(&concat_file, &content)?;

    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "warning",
            "-stats",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file.to_str().unwrap(),
            "-c",
            "copy",
            output_file.to_str().unwrap(),
        ])
        .status()
        .context("Failed to run ffmpeg concat")?;

    if !status.success() {
        bail!("ffmpeg concat failed for {} clips", suffix);
    }

    // Clean up concat list file
    let _ = std::fs::remove_file(&concat_file);

    Ok(output_file)
}

/// Step 6: PIP composite – rear camera cropped & scaled on top of front camera
/// If overlay_file is provided, it is composited in the top-right corner.
pub fn pip_composite(
    front_file: &Path,
    rear_file: &Path,
    overlay_file: Option<&Path>,
    output_dir: &Path,
    name: &str,
) -> Result<PathBuf> {
    let output_file = output_dir.join(format!("{}.mp4", name));

    let mut args = vec![
        "-y".to_string(),
        "-loglevel".to_string(),
        "warning".to_string(),
        "-stats".to_string(),
        "-i".to_string(),
        front_file.to_str().unwrap().to_string(),
        "-i".to_string(),
        rear_file.to_str().unwrap().to_string(),
    ];

    let filter = if let Some(ov) = overlay_file {
        args.extend(["-i".to_string(), ov.to_str().unwrap().to_string()]);
        "\
            [1:v]crop=2560:1080:0:360,scale=-2:432[pip];\
            [0:v][pip]overlay=(W-w)/2:0[base];\
            [base][2:v]overlay=W-w:H-h-90[vout];\
            [0:a]volume=4.0,dynaudnorm[aout]"
    } else {
        "\
            [1:v]crop=2560:1080:0:360,scale=-2:432[pip];\
            [0:v][pip]overlay=(W-w)/2:0[vout];\
            [0:a]volume=4.0,dynaudnorm[aout]"
    };

    args.extend([
        "-filter_complex".to_string(),
        filter.to_string(),
        "-map".to_string(),
        "[vout]".to_string(),
        "-map".to_string(),
        "[aout]".to_string(),
        "-c:v".to_string(),
        "hevc_videotoolbox".to_string(),
        "-tag:v".to_string(),
        "hvc1".to_string(),
        "-c:a".to_string(),
        "aac".to_string(),
        "-b:a".to_string(),
        "96k".to_string(),
        output_file.to_str().unwrap().to_string(),
    ]);

    let status = Command::new("ffmpeg")
        .args(&args)
        .status()
        .context("Failed to run ffmpeg PIP composite")?;

    if !status.success() {
        bail!("ffmpeg PIP composite failed for {}", name);
    }

    Ok(output_file)
}
