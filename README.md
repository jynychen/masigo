# dashcam

行車記錄器影片處理工具。自動掃描前/後鏡頭影片，串接、解析 GPS 軌跡、生成 overlay，並合成 PIP（畫中畫）輸出。

## 需求

- [Rust toolchain](https://rustup.rs/)
- FFmpeg / FFprobe（需在 `PATH` 中）

## 安裝

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

或透過 Homebrew：

```bash
brew install rustup
rustup-init
```

## 建置 & 執行

```bash
cargo run -- -i <輸入目錄> -o <輸出目錄>
```

## 參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `-i, --input <PATH>` | 輸入目錄（含 `*_F.MP4` / `*_R.MP4`） | `vid_src` |
| `-o, --output <PATH>` | 輸出目錄 | `vid_out` |
| `-y, --yes` | 略過確認提示，直接開始處理 | — |

## 處理流程

1. 掃描輸入目錄，將前/後鏡頭影片配對並分組
2. 串接前鏡頭片段
3. 串接後鏡頭片段
4. 解析 GPS 資料（GNRMC）
5. 生成 GPS 軌跡 overlay 影片
6. PIP 合成最終輸出（前鏡頭 + 後鏡頭子畫面 + GPS overlay）

## 範例

```bash
# 使用預設目錄
cargo run

# 指定輸入/輸出
cargo run -- -i ~/Movies/dashcam_clips -o ~/Movies/output

# 跳過確認
cargo run -- -i vid_src -o vid_out -y
```
