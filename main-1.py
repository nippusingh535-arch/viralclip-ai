"""ViralClips AI backend: yt-dlp -> faster-whisper (Hindi) -> highlight scoring -> FFmpeg 9:16 + captions."""
import os, re, json, time, uuid, shutil, queue, zipfile, threading, subprocess
from pathlib import Path
from typing import Optional, List
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).parent
DATA = Path(os.getenv("VC_DATA", ROOT / "data")); DATA.mkdir(parents=True, exist_ok=True)
MAX_SEC = 43200  # 12 hours
MODEL = os.getenv("WHISPER_MODEL", "small")
FONT = os.getenv("VC_FONT", "Noto Sans Devanagari")
HOT = set("अरे अबे वाह भाई गया मार बचाओ पुलिस हाहा गजब पागल नहीं देखो ये क्या कमाल".split())
app = FastAPI(title="ViralClips AI")
P, jobs = {}, queue.Queue()

def pdir(pid): return DATA / pid
def save(p): (pdir(p["id"]) / "project.json").write_text(json.dumps(p, ensure_ascii=False))
def upd(p, **kw): p.update(kw); save(p)
def get(pid):
    if pid not in P: raise HTTPException(404, "Project not found")
    return P[pid]
def clip(p, cid):
    c = next((x for x in p["clips"] if x["id"] == cid), None)
    if not c: raise HTTPException(404, "Clip not found")
    return c
def words_of(p): return json.loads((pdir(p["id"]) / "transcript.json").read_text())
def find_src(d):
    return next((f for f in d.glob("source.*") if f.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v") and ".f" not in f.stem), None)

for f in DATA.glob("*/project.json"):
    q = json.loads(f.read_text())
    if q["status"] not in ("done", "error"): q.update(status="interrupted", message="Server restarted. Press Resume to continue.")
    for c in q["clips"]:
        if c["status"] in ("rendering", "queued"): c.update(status="error", error="Interrupted")
    P[q["id"]] = q

def explain(m):
    m = str(m)
    for k, v in [("sign in", "YouTube is asking this server to sign in (bot check). Set VC_COOKIES=/path/cookies.txt from a logged-in browser, or upload the video file instead."),
                 ("private video", "This video is private."), ("confirm your age", "This video is age-restricted. Use VC_COOKIES or upload the file."),
                 ("copyright", "This video is blocked for copyright reasons."), ("not available", "This video is unavailable (removed, region-locked or members-only)."),
                 ("live event", "This livestream has not finished yet. Use the recording after it ends."), ("unsupported url", "That is not a supported video link.")]:
        if k in m.lower(): return v
    return m[-400:]

def probe(src):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)], capture_output=True, text=True, check=True).stdout.strip())

def ytopts(**kw):
    o = {"quiet": True, "no_warnings": True, "noplaylist": True, **kw}
    if os.getenv("VC_COOKIES"): o["cookiefile"] = os.getenv("VC_COOKIES")
    return o

def ytinfo(url):
    import yt_dlp
    try:
        with yt_dlp.YoutubeDL(ytopts(skip_download=True)) as y: i = y.extract_info(url, download=False)
    except Exception as e: raise HTTPException(400, explain(e))
    if i.get("is_live"): raise HTTPException(400, explain("live event"))
    d = int(i.get("duration") or 0)
    if d > MAX_SEC: raise HTTPException(400, f"Video is {d // 3600}h {d % 3600 // 60}m long. The limit is 12 hours.")
    return {"title": i.get("title"), "thumb": i.get("thumbnail"), "duration": d}

def download(p):
    import yt_dlp
    d = pdir(p["id"])
    def hook(h):
        t = h.get("total_bytes") or h.get("total_bytes_estimate")
        if h["status"] == "downloading" and t: p.update(progress=2 + 12 * h["downloaded_bytes"] / t, message="Downloading… %d MB" % (h["downloaded_bytes"] // 2**20))
    o = ytopts(format="bv*[height<=1080]+ba/b[height<=1080]/b", merge_output_format="mp4", outtmpl=str(d / "source.%(ext)s"), progress_hooks=[hook])
    with yt_dlp.YoutubeDL(o) as y: y.download([p["url"]])
    return find_src(d)

def transcribe(p, src, dur):
    from faster_whisper import WhisperModel
    m = WhisperModel(MODEL, device="auto", compute_type="auto")
    segs, _ = m.transcribe(str(src), language="hi", task="transcribe", word_timestamps=True, vad_filter=True, beam_size=1,
                           condition_on_previous_text=False, initial_prompt="यह हिंदी गेमिंग कमेंट्री है।")
    out = []
    for s in segs:
        out += [{"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2)} for w in (s.words or [])]
        p.update(progress=15 + 45 * min(s.end / dur, 1), message="Transcribing Hindi… %d/%d min" % (s.end // 60, dur // 60))
    (pdir(p["id"]) / "transcript.json").write_text(json.dumps(out, ensure_ascii=False))

def analyze(p, src, words, dur):
    pr = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], stdout=subprocess.PIPE)
    r = []
    while b := pr.stdout.read(32000 * 60):
        x = np.frombuffer(b[: len(b) // 32000 * 32000], np.int16).astype(np.float32) / 32768
        if len(x): r += np.sqrt((x.reshape(-1, 16000) ** 2).mean(1)).tolist()
    pr.wait()
    rms = np.array(r); n = len(rms)
    if n < 20: raise RuntimeError("Video has no usable audio track or is too short.")
    p.update(progress=65, message="Detecting scene changes…")
    e = subprocess.run(["ffmpeg", "-v", "info", "-i", str(src), "-vf", "fps=2,scale=192:-2,select='gt(scene,0.25)',showinfo", "-an", "-f", "null", "-"], capture_output=True, text=True).stderr
    cuts = [float(m) for m in re.findall(r"pts_time:([\d.]+)", e)]
    z = lambda x: (x - x.mean()) / (x.std() + 1e-6)
    sm = lambda x, k: np.convolve(x, np.ones(k) / k, "same")
    cd, wps, kw = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in cuts: cd[min(int(t), n - 1)] += 1
    for w in words:
        i = min(int(w["s"]), n - 1); wps[i] += 1; kw[i] += w["w"].strip("।,.!?") in HOT
    sc = z(sm(rms, 5)) + .6 * z(sm(cd, 9)) + .5 * z(sm(wps, 7)) + .5 * z(sm(np.maximum(np.diff(rms, prepend=rms[0]), 0), 3)) + 1.5 * sm(kw, 7)
    sc[sm(rms, 5) < .2 * rms.mean()] -= 3  # penalise silence
    L = min(40, n); c = np.concatenate([[0], np.cumsum(sc)]); wm = (c[L:] - c[:-L]) / L
    cands = []
    while len(cands) < p["opts"]["n"] * 2 and wm.max() > -1e8:
        s = int(wm.argmax()); cands.append({"start": s, "end": s + L, "score": round(float(wm[s]), 2)})
        wm[max(0, s - 45): s + 45] = -1e9
    ss, es = [w["s"] for w in words], [w["e"] for w in words]
    near = lambda v, a: (lambda c: c if abs(c - v) <= 5 else v)(min(a, key=lambda x: abs(x - v), default=v))
    for q in cands:  # snap to sentence edges, clamp 15-60s
        a, b = near(q["start"], ss), near(q["end"], es)
        b = min(max(b, a + 15), a + 60, dur); a = max(0, min(a, b - 15))
        q["start"], q["end"] = round(a, 2), round(b, 2)
    return cands

def make_caps(words, a, b):
    ws = [w for w in words if w["s"] >= a - .2 and w["e"] <= b + .2]
    return [{"s": round(max(g[0]["s"] - a, 0), 2), "e": round(min(g[-1]["e"] - a, b - a), 2), "t": " ".join(x["w"] for x in g)}
            for g in (ws[i:i + 4] for i in range(0, len(ws), 4))]

def write_ass(caps, path):
    T = lambda x: "%d:%02d:%05.2f" % (x // 3600, x % 3600 // 60, x % 60)
    ev = ""
    for c in caps:
        ws = c["t"].split()
        if not ws: continue
        e = c["e"] if c["e"] > c["s"] else c["s"] + .5
        k = max(1, int((e - c["s"]) * 100 / len(ws)))
        txt = "".join("{\\k%d}%s " % (k, w) for w in ws).strip()
        ev += "Dialogue: 0,%s,%s,Cap,,0,0,0,,{\\fad(80,60)}%s\n" % (T(c["s"]), T(e), txt)
    hdr = (f"[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nWrapStyle: 0\n\n[V4+ Styles]\n"
           "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
           f"Style: Cap,{FONT},84,&H00FFE500,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,6,2,2,60,60,380,1\n\n"
           "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")
    Path(path).write_text(hdr + ev, encoding="utf-8")

def render(p, c):
    d = pdir(p["id"]); cd = d / "clips"; cd.mkdir(exist_ok=True); w = c["id"]
    write_ass(c["captions"], cd / f"{w}.ass")
    zm = c.get("zoom")
    if p["opts"]["mode"] == "fill":
        fc = f"[0:v]scale=-2:{int(1920 * (1.12 if zm else 1))},crop=1080:1920,subtitles={w}.ass[o]"
    else:
        fc = (f"[0:v]split[a][b];[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:5[bg];"
              f"[b]scale={1242 if zm else 1080}:-2[f];[bg][f]overlay=(W-w)/2:(H-h)/2,subtitles={w}.ass[o]")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(c["start"]), "-t", str(c["end"] - c["start"]), "-i", str(find_src(d)),
                    "-filter_complex", fc, "-map", "[o]", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", f"{w}.mp4"],
                   cwd=cd, check=True, capture_output=True, text=True)

def do_render(p, c):
    c["status"] = "rendering"
    try: render(p, c); c.update(status="ready", v=int(time.time()), error="")
    except Exception as e: c.update(status="error", error=str(getattr(e, "stderr", e))[-300:])
    save(p)

def pipeline(p):
    d = pdir(p["id"]); o = p["opts"]
    try:
        src = find_src(d)
        if not src: upd(p, status="downloading", message="Downloading…", progress=2); src = download(p)
        dur = probe(src)
        if dur > MAX_SEC + 60: raise RuntimeError("Video is longer than 12 hours.")
        if not (d / "transcript.json").exists(): upd(p, status="transcribing", progress=15); transcribe(p, src, dur)
        words = words_of(p)
        if not p["clips"]:
            upd(p, status="analyzing", progress=62, message="Analyzing audio and gameplay…")
            cands = analyze(p, src, words, dur)
            p["spare"] = cands[o["n"]:]
            p["clips"] = [{"id": uuid.uuid4().hex[:8], **q, "captions": make_caps(words, q["start"], q["end"]), "zoom": o["zoom"], "status": "queued"} for q in cands[: o["n"]]]
            save(p)
        todo = [c for c in p["clips"] if c["status"] != "ready"]
        upd(p, status="rendering")
        for i, c in enumerate(todo):
            p.update(progress=70 + 30 * i / max(len(todo), 1), message=f"Rendering clip {i + 1}/{len(todo)}…")
            do_render(p, c)
        upd(p, status="done", progress=100, message="%d clips ready" % sum(c["status"] == "ready" for c in p["clips"]))
    except Exception as e:
        upd(p, status="error", message=explain(e))

def worker():
    while True:
        fn, args = jobs.get()
        try: fn(*args)
        except Exception as e: print("job failed:", e)
threading.Thread(target=worker, daemon=True).start()

@app.get("/api/info")
def info(url: str): return ytinfo(url)

@app.post("/api/projects")
async def create(url: str = Form(""), n: int = Form(8), mode: str = Form("fit"), zoom: bool = Form(False), rights: bool = Form(False), file: Optional[UploadFile] = File(None)):
    if not rights: raise HTTPException(400, "Confirm you own this video or have permission to clip it.")
    if not file and not url.strip(): raise HTTPException(400, "Paste a YouTube link or upload a video.")
    meta = None if file else ytinfo(url.strip())
    pid = uuid.uuid4().hex[:10]; d = pdir(pid); d.mkdir()
    if file:
        src = d / ("source" + (Path(file.filename).suffix or ".mp4"))
        with open(src, "wb") as f:
            while b := await file.read(1 << 23): f.write(b)
        try:
            dur = probe(src)
            if dur > MAX_SEC + 60: raise ValueError("Video is longer than 12 hours.")
        except Exception as e:
            shutil.rmtree(d); raise HTTPException(400, str(e) if isinstance(e, ValueError) else "Could not read that file as a video.")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(min(5, dur / 2)), "-i", str(src), "-frames:v", "1", "-vf", "scale=480:-2", str(d / "thumb.jpg")])
        meta = {"title": file.filename, "thumb": f"/files/{pid}/thumb.jpg", "duration": int(dur)}
    p = {"id": pid, **meta, "url": url.strip(), "status": "queued", "progress": 0, "message": "Queued", "created": time.time(),
         "opts": {"n": max(1, min(n, 20)), "mode": mode if mode in ("fit", "fill") else "fit", "zoom": zoom}, "clips": [], "spare": []}
    P[pid] = p; save(p); jobs.put((pipeline, (p,))); return p

@app.get("/api/projects")
def plist(): return [{"id": p["id"], "title": p["title"], "status": p["status"]} for p in sorted(P.values(), key=lambda p: -p["created"])]

@app.get("/api/projects/{pid}")
def pget(pid: str): return get(pid)

@app.post("/api/projects/{pid}/resume")
def resume(pid: str):
    p = get(pid); upd(p, status="queued", message="Queued"); jobs.put((pipeline, (p,))); return p

class Edit(BaseModel):
    start: Optional[float] = None
    end: Optional[float] = None
    captions: Optional[List[dict]] = None
    zoom: Optional[bool] = None

@app.patch("/api/projects/{pid}/clips/{cid}")
def edit(pid: str, cid: str, e: Edit):
    p = get(pid); c = clip(p, cid)
    a, b = (e.start if e.start is not None else c["start"]), (e.end if e.end is not None else c["end"])
    if not 5 <= b - a <= 60: raise HTTPException(400, "Clip length must be between 5 and 60 seconds.")
    if e.captions is not None: c["captions"] = [{"s": float(x["s"]), "e": float(x["e"]), "t": str(x["t"])} for x in e.captions]
    elif (a, b) != (c["start"], c["end"]): c["captions"] = make_caps(words_of(p), a, b)
    c.update(start=a, end=b, status="queued")
    if e.zoom is not None: c["zoom"] = e.zoom
    save(p); jobs.put((do_render, (p, c))); return p

@app.post("/api/projects/{pid}/clips/{cid}/regenerate")
def regen(pid: str, cid: str):
    p = get(pid); c = clip(p, cid)
    if p.get("spare"):
        q = p["spare"].pop(0); c.update(**q, captions=make_caps(words_of(p), q["start"], q["end"]))
    c["status"] = "queued"; save(p); jobs.put((do_render, (p, c))); return p

@app.delete("/api/projects/{pid}/clips/{cid}")
def rm(pid: str, cid: str):
    p = get(pid); c = clip(p, cid); p["clips"].remove(c)
    for ext in ("mp4", "ass"): (pdir(pid) / "clips" / f"{cid}.{ext}").unlink(missing_ok=True)
    save(p); return p

@app.get("/api/projects/{pid}/zip")
def zipall(pid: str):
    p = get(pid); z = pdir(pid) / "all_clips.zip"; k = 0
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
        for c in p["clips"]:
            f = pdir(pid) / "clips" / f"{c['id']}.mp4"
            if c["status"] == "ready" and f.exists(): k += 1; zf.write(f, f"viralclip_{k}.mp4")
    if not k: raise HTTPException(404, "No finished clips to download yet.")
    return FileResponse(z, filename="viralclips.zip")

@app.get("/files/{pid}/{name:path}")
def files(pid: str, name: str, dl: int = 0):
    get(pid); f = (pdir(pid) / name).resolve()
    if DATA.resolve() not in f.parents or f.suffix not in (".mp4", ".jpg") or not f.exists(): raise HTTPException(404)
    return FileResponse(f, filename=(f"viralclip_{f.stem}.mp4" if dl else None))

@app.get("/")
def index(): return FileResponse(ROOT / "static" / "index.html")

if __name__ == "__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=8000)
