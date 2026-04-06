mod clip;
mod ffmpeg;
mod ffprobe;
mod gps;
mod interp;
mod overlay;

use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use anyhow::{Result, anyhow};
use clap::Parser;

use clip::Direction;

#[derive(Parser)]
#[command(name = "dashcam", about = "行車記錄器影片處理工具")]
struct Cli {
    /// 輸入目錄（含 *_F.MP4 / *_R.MP4）
    #[arg(short, long, default_value = "vid_src")]
    input: PathBuf,

    /// 輸出目錄
    #[arg(short, long, default_value = "vid_out")]
    output: PathBuf,

    /// 略過確認提示，直接開始處理
    #[arg(short, long)]
    yes: bool,

    /// 使用大尺寸 overlay（1920px，預設 480px）
    #[arg(long)]
    big: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Ctrl-C: 設定旗標，讓目前的 ffmpeg 跑完再結束
    let interrupted = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&interrupted);
    ctrlc::set_handler(move || {
        flag.store(true, Ordering::SeqCst);
        eprintln!("\nwait for existing ffmpeg process to finish...");
    })?;

    std::fs::create_dir_all(&cli.output)?;

    // ── Step 1: 掃描影片，找出連續片段 ──
    println!("=== Step 1: 掃描影片，建立工作清單 ===\n");
    let groups = clip::scan_clip_groups(&cli.input)?;

    if groups.is_empty() {
        println!("no valid clip groups found in {}", cli.input.display());
        return Ok(());
    }

    for (i, group) in groups.iter().enumerate() {
        println!("Group {} [{}]", i + 1, group.name);
        for pair in &group.pairs {
            println!("{} + {}", pair.front.filename, pair.rear.filename);
        }
    }

    if !cli.yes && !wait_for_confirmation_interruptible(interrupted.as_ref())? {
        println!("已取消");
        return Ok(());
    }

    // ── Steps 2–6: 逐組處理 ──
    for (i, group) in groups.iter().enumerate() {
        println!(
            "\n{}\nProcessing Group {} [{}]\n{}",
            "=".repeat(50),
            i + 1,
            group.name,
            "=".repeat(50)
        );

        // Step 2: concat Front
        println!("=== Step 2: 串接前鏡頭 ===");
        let front_file = ffmpeg::concatenate_clips(group, &cli.output, Direction::Front)?;
        if interrupted.load(Ordering::SeqCst) {
            println!("\nUser interrupted, stopping processing.");
            return Ok(());
        }
        println!("F -> {}", front_file.display());

        // Step 3: concat Rear
        println!("=== Step 3: 串接後鏡頭 ===");
        let rear_file = ffmpeg::concatenate_clips(group, &cli.output, Direction::Rear)?;
        if interrupted.load(Ordering::SeqCst) {
            println!("\nUser interrupted, stopping processing.");
            return Ok(());
        }
        println!("R -> {}", rear_file.display());

        // Step 4: GPS
        println!("=== Step 4: 解析 GPS ===");
        let gps_points = match gps::extract_gps_track(&group.front_paths()) {
            Ok(pts) => {
                println!("{} gnrmc records", pts.len());
                Some(pts)
            }
            Err(e) => {
                println!("GPS parsing failed: {e}");
                None
            }
        };

        // Step 5: overlay
        let overlay_file = if let Some(ref pts) = gps_points {
            if interrupted.load(Ordering::SeqCst) {
                println!("\nUser interrupted, stopping processing.");
                return Ok(());
            }
            println!("=== Step 5: 生成 GPS 軌跡 overlay ===");
            let (fps, duration) = ffprobe::probe_video_info(&front_file)?;
            println!("{fps:.2} fps, {duration:.1}s");
            let overlay_size = if cli.big { 1920 } else { 480 };
            match overlay::render_overlay_video(
                pts,
                &cli.output,
                &group.name,
                fps,
                duration,
                overlay_size,
            ) {
                Ok(path) => {
                    println!("overlay -> {}", path.display());
                    Some(path)
                }
                Err(e) => {
                    println!("overlay rendering failed: {e}");
                    None
                }
            }
        } else {
            println!("Step 5: (no GPS data, skipping overlay)");
            None
        };

        // Step 6: PIP
        if interrupted.load(Ordering::SeqCst) {
            println!("\nUser interrupted, stopping processing.");
            return Ok(());
        }
        println!("=== Step 6: PIP 合成 ===");
        let final_file = ffmpeg::compose_pip(
            &front_file,
            &rear_file,
            overlay_file.as_deref(),
            &cli.output,
            &group.name,
        )?;
        println!("-> {}", final_file.display());

        // Clean up intermediate overlay video
        if let Some(ref ov) = overlay_file {
            let _ = std::fs::remove_file(ov);
        }
    }

    println!("\n=== 全部完成！ ===");
    Ok(())
}

fn wait_for_confirmation_interruptible(interrupted: &AtomicBool) -> Result<bool> {
    print!("\n是否開始處理？(y/n): ");
    io::stdout().flush()?;

    let (tx, rx) = mpsc::channel::<io::Result<String>>();
    thread::spawn(move || {
        let mut answer = String::new();
        let result = io::stdin().read_line(&mut answer).map(|_| answer);
        let _ = tx.send(result);
    });

    loop {
        if interrupted.load(Ordering::SeqCst) {
            return Ok(false);
        }
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(Ok(answer)) => return Ok(answer.trim().eq_ignore_ascii_case("y")),
            Ok(Err(e)) => return Err(e.into()),
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(anyhow!("confirmation input channel disconnected"));
            }
        }
    }
}
