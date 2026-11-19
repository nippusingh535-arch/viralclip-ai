# ViralClips AI
Paste a YouTube gaming video (or upload a file) -> Hindi-captioned 9:16 Shorts.
Pipeline: yt-dlp -> faster-whisper (language="hi", transcribe only, never translate) -> audio energy + scene-change + Hindi keyword scoring -> FFmpeg 1080x1920 + animated Devanagari captions (ASS/libass).

## Run (Linux/macOS/Windows, Python 3.10+)
1. Install FFmpeg (with libass) and a Devanagari font: `sudo apt install ffmpeg fonts-noto-core` (Windows: install ffmpeg + "Noto Sans Devanagari", or set `VC_FONT="Nirmala UI"`).
2. `pip install -r requirements.txt`
3. `python main.py` then open http://localhost:8000 (phone on same Wi-Fi: http://<pc-ip>:8000)

## Settings (env vars)
- `WHISPER_MODEL` small (default) | medium (better Hindi, slower) | large-v3 (GPU). CPU: expect roughly 0.3-1x real time on `small`; a 2.5h video takes a while. A GPU is far faster.
- `VC_COOKIES=/path/cookies.txt` for videos YouTube blocks from your server (bot check, age gate).
- `VC_FONT`, `VC_DATA` (project storage; progress is saved there and can be resumed).

## Notes
- Use only videos you own or have permission to clip. YouTube may block downloads from cloud IPs; upload the file instead.
- Projects survive restarts; press Resume to continue from the last finished stage.
