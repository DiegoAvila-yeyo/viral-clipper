"""Clipper: long video (YouTube URL, stream VOD or local file) -> N vertical viral clips.

Usage:
  python tools/clipper.py <url|file> --id <id> [--n 6] [--min 20] [--max 55]
                          [--layout auto|face|stream|fit] [--model small] [--lang es] [--from-step fetch]
                          [--no-render] [--handle @cuenta]

Steps (each one caches its output in content/clips/<id>/, so a 5 h stream can resume):
  fetch      yt-dlp (video <=1080p + info.json with YouTube "most replayed" heatmap + live chat replay)
  audio      16 kHz mono PCM + loudness per 0.5 s
  transcribe faster-whisper in 10-min chunks (resumable), word timestamps
  score      sliding windows scored by loudness peaks, speech rate, heatmap, chat bursts, hype words
  pick       claude -p chooses the best self-contained moments, titles and keywords
  cut        ffmpeg cuts each clip (+ face track for the vertical reframe)
  render     Remotion `ViralClip` (jump cuts, zoom punches, word captions, hook, emojis, SFX)
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
FF = imageio_ffmpeg.get_ffmpeg_exe()
STEPS = ["fetch", "audio", "transcribe", "score", "pick", "cut", "render"]
SR = 16000
HOP = 0.5  # seconds per loudness bin
HYPE = re.compile(r"\b(ja(ja)+|jsjs\w*|no mames|qu[eé] onda|wey|g[uü]ey|neta|alv|chale|dios m[ií]o|no puede ser|incre[ií]ble|brutal|qu[eé] locura|wow|mira|f[ií]jate|secreto|nadie|error|dinero|millon\w*|gratis|truco|nunca|siempre|literal)\b", re.I)


def log(*a):
    print("[clipper]", *a, flush=True)


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


# ---------------------------------------------------------------- fetch
def fetch(src, d):
    if Path(src).exists():
        dst = d / "source.mp4"
        if not dst.exists():
            log("copiando archivo local")
            shutil.copy(src, dst)
        return
    log("descargando con yt-dlp", src)
    sh([sys.executable, "-m", "yt_dlp", src,
        "-f", "bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b",
        "--merge-output-format", "mp4", "--ffmpeg-location", FF, "--js-runtimes", "node",
        "--write-info-json", "--write-subs", "--sub-langs", "live_chat", "--no-playlist",
        "-o", str(d / "source.%(ext)s")])
    info = d / "source.info.json"
    if info.exists():
        j = json.loads(info.read_text("utf-8"))
        meta = {k: j.get(k) for k in ("title", "uploader", "channel", "duration", "webpage_url", "was_live", "heatmap")}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
        info.unlink()


def probe(path):
    """(duration, width, height, fps) via ffmpeg stderr (no ffprobe in imageio-ffmpeg)."""
    err = subprocess.run([FF, "-hide_banner", "-i", str(path)], capture_output=True, text=True, errors="ignore").stderr
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    vid = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", err)
    fps = re.search(r"([\d.]+) fps", err)
    rot = re.search(r"rotate\s*:\s*(-?\d+)|rotation of (-?[\d.]+)", err)
    w, h = int(vid.group(1)), int(vid.group(2))
    if rot and abs(float(rot.group(1) or rot.group(2))) % 180 == 90:
        w, h = h, w
    return int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3)), w, h, float(fps.group(1)) if fps else 30.0


# ---------------------------------------------------------------- audio
def audio(d):
    log("extrayendo audio")
    raw = subprocess.run([FF, "-v", "error", "-i", str(d / "source.mp4"), "-vn", "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    (d / "audio.pcm").write_bytes(raw)
    pcm = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    hop = int(SR * HOP)
    n = len(pcm) // hop
    rms = np.sqrt((pcm[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-9)
    db = 20 * np.log10(rms)
    (d / "loudness.json").write_text(json.dumps([round(float(x), 1) for x in db]), "utf-8")


def load_pcm(d):
    return np.frombuffer((d / "audio.pcm").read_bytes(), np.int16).astype(np.float32) / 32768.0


# ---------------------------------------------------------------- transcribe
def cuda_dlls():
    """Expose pip's nvidia-cublas/cudnn DLLs to CTranslate2 on Windows."""
    import os
    import site
    for sp in [site.getusersitepackages(), *site.getsitepackages()]:
        for b in Path(sp, "nvidia").glob("*/bin"):
            os.add_dll_directory(str(b))
            os.environ["PATH"] = str(b) + os.pathsep + os.environ["PATH"]


def transcribe(d, model_name, lang="es"):
    cuda_dlls()
    if not (d / "audio.pcm").exists():
        audio(d)
    from faster_whisper import WhisperModel
    pcm = load_pcm(d)
    chunk = 600 * SR
    parts = d / "parts"
    parts.mkdir(exist_ok=True)
    model = None
    n = math.ceil(len(pcm) / chunk)
    for i in range(n):
        out = parts / f"{i:03d}.json"
        if out.exists():
            continue
        if model is None:
            try:
                model = WhisperModel(model_name, device="cuda", compute_type="int8")
                list(model.transcribe(np.zeros(SR, np.float32), language=lang)[0])
                log("whisper en GPU")
            except Exception as e:  # missing cuBLAS/cuDNN, low VRAM...
                log("GPU no disponible, CPU:", str(e)[:80])
                model = WhisperModel(model_name, device="cpu", compute_type="int8")
        off = i * 600
        log(f"transcribiendo {i + 1}/{n} ({off // 60}-{min(off + 600, len(pcm) // SR) // 60} min)")
        segs, _ = model.transcribe(pcm[i * chunk:(i + 1) * chunk], language=lang, word_timestamps=True,
                                   vad_filter=True, vad_parameters={"min_silence_duration_ms": 700},
                                   condition_on_previous_text=False)
        words = [{"w": w.word.strip(), "s": round(off + w.start, 2), "e": round(off + w.end, 2)} for s in segs for w in s.words if w.word.strip()]
        out.write_text(json.dumps(words, ensure_ascii=False), "utf-8")
    words = [w for i in range(n) for w in json.loads((parts / f"{i:03d}.json").read_text("utf-8"))]
    (d / "words.json").write_text(json.dumps(words, ensure_ascii=False), "utf-8")
    (d / "audio.pcm").unlink(missing_ok=True)
    log(f"{len(words)} palabras")


def sentences(words):
    """Group words into sentences (punctuation or pause > 0.8 s)."""
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if not nxt or re.search(r"[.?!…]$", w["w"]) or nxt["s"] - w["e"] > 0.8 or len(cur) > 40:
            out.append({"s": cur[0]["s"], "e": cur[-1]["e"], "text": " ".join(x["w"] for x in cur)})
            cur = []
    return out


# ---------------------------------------------------------------- score
def chat_density(d, dur):
    """Messages per 10 s from a YouTube live_chat replay (json lines), if present."""
    f = next(d.glob("source*.live_chat.json"), None)
    if not f:
        return None
    bins = np.zeros(int(dur // 10) + 1)
    for line in f.read_text("utf-8", errors="ignore").splitlines():
        m = re.search(r'"videoOffsetTimeMsec"\s*:\s*"(\d+)"', line)
        if m:
            i = int(m.group(1)) // 10000
            if i < len(bins):
                bins[i] += 1
    return bins


def z(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-6)


def score(d, args):
    words = json.loads((d / "words.json").read_text("utf-8"))
    loud = np.array(json.loads((d / "loudness.json").read_text("utf-8")))
    dur = len(loud) * HOP
    meta = json.loads((d / "meta.json").read_text("utf-8")) if (d / "meta.json").exists() else {}
    heat = meta.get("heatmap") or []
    chat = chat_density(d, dur)
    win, step = args.max, 5.0
    # Per-second features.
    secs = int(dur) + 1
    rate = np.zeros(secs)
    hype = np.zeros(secs)
    for w in words:
        i = min(int(w["s"]), secs - 1)
        rate[i] += 1
        if HYPE.search(w["w"]):
            hype[i] += 1
    ev = len(loud) // 2 * 2
    loud_s = loud[:ev].reshape(-1, 2).max(1) if ev else loud
    loud_s = np.pad(loud_s, (0, max(0, secs - len(loud_s))))[:secs]
    peak = np.maximum(z(loud_s) - 1.0, 0)  # only clearly louder than normal (shouting, laughing)
    heat_s = np.zeros(secs)
    for h in heat:
        heat_s[int(h["start_time"]):int(h["end_time"]) + 1] = h["value"]
    chat_s = np.repeat(chat, 10)[:secs] if chat is not None else np.zeros(secs)
    chat_s = np.pad(chat_s, (0, secs - len(chat_s)))

    def wsum(a, s, e):
        return float(a[int(s):int(e)].sum())

    cands = []
    t = 0.0
    while t + args.min < dur:
        e = min(t + win, dur)
        n = e - t
        spoken = wsum(rate, t, e) / n  # words/s; silence = boring
        f = {
            "peak": wsum(peak, t, e) / n,
            "rate": spoken,
            "hype": wsum(hype, t, e) / n,
            "heat": float(heat_s[int(t):int(e)].mean()) if heat else 0.0,
            "chat": float(chat_s[int(t):int(e)].mean()) if chat is not None else 0.0,
        }
        cands.append((t, e, f))
        t += step
    if not cands:
        return []
    keys = ["peak", "rate", "hype", "heat", "chat"]
    wts = {"peak": 1.0, "rate": 0.6, "hype": 0.8, "heat": 1.5 if heat else 0, "chat": 1.5 if chat is not None else 0}
    cols = {k: z([c[2][k] for c in cands]) if wts[k] else np.zeros(len(cands)) for k in keys}
    scored = []
    for i, (s, e, f) in enumerate(cands):
        if f["rate"] < 1.2:  # mostly silence/music
            continue
        scored.append({"s": s, "e": e, "score": round(float(sum(wts[k] * cols[k][i] for k in keys)), 3), "f": f})
    scored.sort(key=lambda c: -c["score"])
    # Non-overlapping top windows; long sources get more candidates for the model.
    want = min(40, max(args.n * 4, int(dur / 300)))
    picked = []
    for c in scored:
        if all(c["e"] <= p["s"] or c["s"] >= p["e"] for p in picked):
            picked.append(c)
        if len(picked) >= want:
            break
    picked.sort(key=lambda c: c["s"])
    (d / "candidates.json").write_text(json.dumps(picked, ensure_ascii=False, indent=1), "utf-8")
    log(f"{len(picked)} ventanas candidatas (heatmap={'sí' if heat else 'no'}, chat={'sí' if chat is not None else 'no'})")


# ---------------------------------------------------------------- pick (LLM)
PICK_PROMPT = """Eres editor senior de clips virales para TikTok, Reels y Shorts (audiencia Gen Z / millennial hispana).
Fuente: "{title}" ({dur} min). Abajo hay ventanas candidatas preseleccionadas por señales (picos de volumen, risa, ritmo de habla, "lo más visto" de YouTube, ráfagas de chat). Cada línea del transcript es [segundo_inicio-segundo_fin] texto.

Elige los {n} MEJORES clips. Reglas:
- Duración {mn}-{mx} s. Puedes recortar o extender un poco la ventana (usa los tiempos de las frases).
- El clip DEBE arrancar en la frase más fuerte: sin saludos, sin contexto, sin "entonces...". Los primeros 2 s deciden si la gente se queda.
- Debe entenderse solo (sin haber visto el video), y terminar en remate, conclusión o giro, nunca a media idea.
- Prioriza: opiniones polémicas, momentos graciosos, revelaciones/datos sorprendentes, historias con giro, consejos concretos, reacciones fuertes.
- No elijas dos clips del mismo tema.
- title = gancho de pantalla de MÁXIMO 7 palabras que genere curiosidad (no clickbait falso, no spoiler del remate). Puede llevar 1 emoji.
- keywords = 4-10 palabras EXACTAS que aparecen dentro del clip y merecen énfasis visual (números, nombres, palabras emocionales).
- emojis = hasta 6 pares palabra_exacta→emoji para pegar sobre el video cuando se dice esa palabra.
- caption = descripción para publicar (1-2 líneas + 3-5 hashtags).

JSON: {{"clips":[{{"start":seg,"end":seg,"title":"...","keywords":["..."],"emojis":{{"palabra":"😱"}},"caption":"...","score":1-10,"why":"..."}}]}}

CANDIDATAS:
{cands}
"""


def pick(d, args):
    words = json.loads((d / "words.json").read_text("utf-8"))
    cands = json.loads((d / "candidates.json").read_text("utf-8"))
    meta = json.loads((d / "meta.json").read_text("utf-8")) if (d / "meta.json").exists() else {}
    sents = sentences(words)
    blocks = []
    for i, c in enumerate(cands):
        lo, hi = c["s"] - 15, c["e"] + 15  # margin so the model can extend
        lines = [f"[{s['s']:.1f}-{s['e']:.1f}] {s['text']}" for s in sents if s["e"] > lo and s["s"] < hi]
        blocks.append(f"### Ventana {i + 1} (señal {c['score']:+.2f})\n" + "\n".join(lines))
    dur = len(json.loads((d / "loudness.json").read_text("utf-8"))) * HOP
    prompt = PICK_PROMPT.format(title=meta.get("title") or args.id, dur=round(dur / 60), n=args.n, mn=args.min, mx=args.max, cands="\n\n".join(blocks))
    prompt += "\n\nResponde ÚNICAMENTE con el objeto JSON, sin texto extra ni bloques de código."
    log(f"pidiendo a Claude ({len(prompt) // 1000}k chars)")
    r = subprocess.run("claude -p --output-format json --max-turns 2", input=prompt, capture_output=True, text=True, encoding="utf-8", shell=True, timeout=900)
    env = json.loads(r.stdout)
    txt = env.get("result", "")
    m = re.search(r"\{.*\}", txt, re.S)
    clips = json.loads(m.group(0))["clips"]
    # Snap to word boundaries and enforce limits.
    out = []
    for c in sorted(clips, key=lambda c: -c.get("score", 0)):
        ws = [w for w in words if w["s"] >= c["start"] - 0.3 and w["e"] <= c["end"] + 0.5]
        if len(ws) < 8:
            continue
        s, e = ws[0]["s"], ws[-1]["e"]
        if e - s > args.max + 8:
            ws = [w for w in ws if w["e"] <= s + args.max + 3]
            e = ws[-1]["e"]
        out.append({**c, "start": round(s, 2), "end": round(e, 2)})
    # New picks invalidate previous cuts/renders.
    shutil.rmtree(ROOT / "public" / "clips" / args.id, ignore_errors=True)
    shutil.rmtree(ROOT / "out" / "clips" / args.id, ignore_errors=True)
    for f in d.glob("c*.props.json"):
        f.unlink()
    (d / "picks.json").write_text(json.dumps(out[: args.n], ensure_ascii=False, indent=1), "utf-8")
    for i, c in enumerate(out[: args.n]):
        log(f"  {i + 1}. [{c['start']:.0f}-{c['end']:.0f}s] {c['title']}  ({c.get('score')}/10)")


# ---------------------------------------------------------------- cut + reframe
YUNET = ROOT / "tools" / "models" / "face_detection_yunet_2023mar.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def face_track(video, w, h, step=0.2):
    """Main face per sample (YuNet DNN) + scene cuts -> [{t, cx, cy, fh, n, cut}] normalised."""
    import cv2
    if not YUNET.exists():
        import urllib.request
        log("descargando modelo de caras YuNet (opencv_zoo, 230 KB)")
        YUNET.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(YUNET_URL, YUNET)
    det = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), 0.75)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    every = max(1, round(fps * step))
    scale = 640 / max(w, h)
    pts, i, prev_hist, prev_c = [], 0, None, None
    while True:
        if not cap.grab():
            break
        if i % every == 0:
            _, fr = cap.retrieve()
            sm = cv2.resize(fr, None, fx=scale, fy=scale)
            gh, gw = sm.shape[:2]
            hist = cv2.calcHist([cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            cut = prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > 0.45
            prev_hist = hist
            det.setInputSize((gw, gh))
            _, faces = det.detect(sm)
            p = {"t": round(i / fps, 2), "cx": None, "cut": cut}
            if faces is not None and len(faces):
                # Prefer big faces, and the one we were already following (no ping-pong between people).
                def rank(f):
                    cx = (f[0] + f[2] / 2) / gw
                    stay = 0 if prev_c is None or cut else abs(cx - prev_c) * 0.5
                    return f[3] / gh - stay
                x, y, fw, fh = max(faces, key=rank)[:4]
                p.update(cx=(x + fw / 2) / gw, cy=(y + fh / 2) / gh, fh=fh / gh, n=len(faces))
                prev_c = p["cx"]
            pts.append(p)
        i += 1
    cap.release()
    return pts


def smooth_track(pts):
    """Per scene: median-filter + dead-zone so the virtual camera glides; snap on scene cuts."""
    if not pts:
        return []
    scenes, cur = [], []
    for p in pts:
        if p["cut"] and cur:
            scenes.append(cur)
            cur = []
        cur.append(p)
    scenes.append(cur)
    keys = []
    for sc in scenes:
        found = [p for p in sc if p["cx"] is not None]
        if not found:  # b-roll without faces: centre
            keys.append({"t": sc[0]["t"], "cx": 0.5, "cy": 0.45, "fh": 0.25, "cut": True})
            continue
        last = found[0]
        filled = []
        for p in sc:
            if p["cx"] is not None:
                last = p
            filled.append(last)
        hold = None
        for j, p in enumerate(filled):
            win = filled[max(0, j - 4): j + 5]
            cx = float(np.median([q["cx"] for q in win]))
            cy = float(np.median([q["cy"] for q in win]))
            fh = float(np.median([q["fh"] for q in win]))
            if hold is None or abs(cx - hold[0]) > 0.05 or abs(cy - hold[1]) > 0.06:
                hold = (cx, cy, fh)
                keys.append({"t": sc[j]["t"], "cx": round(cx, 3), "cy": round(cy, 3), "fh": round(fh, 3), "cut": j == 0})
    return keys


def burned_subs(video, n=24):
    """Fraction of the frame bottom covered by burned-in subtitles (0 if none).
    Looks for outlined white text concentrated in one low band across many frames."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (k + 0.5) / n))
        ok, fr = cap.read()
        if not ok:
            continue
        fr = cv2.resize(fr, (480, int(480 * fr.shape[0] / fr.shape[1])))
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        txt = (g > 215).astype(np.uint8) & cv2.dilate((g < 60).astype(np.uint8), np.ones((3, 3), np.uint8))
        rows.append(txt.mean(1))
    cap.release()
    if not rows:
        return 0.0
    r = np.mean(rows, 0)
    h = len(r)
    band = lambda lo: float(r[int(h * lo / 100):int(h * (lo + 5) / 100)].mean()) * 1000
    base = np.median([band(lo) for lo in range(50, 70, 5)])
    for lo in range(70, 100, 5):
        if band(lo) > 1.0 and band(lo) > 5 * max(base, 0.05):
            return round(1 - (lo - 2) / 100, 3)
    return 0.0


def choose_layout(pts, w, h, forced):
    if forced != "auto":
        return forced
    found = [p for p in pts if p["cx"] is not None]
    cov = len(found) / max(1, len(pts))
    if w < h:  # already vertical
        return "face"
    if cov < 0.35:
        return "fit"
    med_fh = float(np.median([p["fh"] for p in found]))
    corner = np.mean([(p["cx"] < 0.33 or p["cx"] > 0.67) and (p["cy"] < 0.4 or p["cy"] > 0.6) for p in found])
    if med_fh < 0.16 and corner > 0.6:
        return "stream"  # small facecam in a corner over gameplay/screen
    if np.mean([p.get("n", 1) > 1 for p in found]) > 0.5:
        return "fit"  # podcast with 2+ people on screen
    return "face"


def jump_cuts(words, s, e, gap=0.35, pad=0.12):
    """Keep speech, drop pauses > gap. Returns [{from,to}] in clip-file seconds (file starts at s-0.5)."""
    ws = [w for w in words if w["s"] >= s - 0.05 and w["e"] <= e + 0.05]
    segs = []
    for w in ws:
        a, b = max(s, w["s"] - pad), min(e + 0.25, w["e"] + pad)
        if segs and a - segs[-1][1] < gap:
            segs[-1][1] = max(segs[-1][1], b)
        else:
            segs.append([a, b])
    off = s - 0.5
    return [{"from": round(a - off, 3), "to": round(b - off, 3)} for a, b in segs if b - a > 0.2]


def cut(d, args):
    words = json.loads((d / "words.json").read_text("utf-8"))
    picks = json.loads((d / "picks.json").read_text("utf-8"))
    loud = np.array(json.loads((d / "loudness.json").read_text("utf-8")))
    _, W, H, _ = probe(d / "source.mp4")
    subs = d / "subs.json"
    if not subs.exists():
        subs.write_text(json.dumps({"cropBottom": burned_subs(d / "source.mp4")}), "utf-8")
    crop_bottom = json.loads(subs.read_text("utf-8"))["cropBottom"]
    if crop_bottom:
        log(f"subtítulos quemados detectados: se recorta el {crop_bottom:.0%} inferior")
    pub = ROOT / "public" / "clips" / args.id
    pub.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(picks):
        name = f"c{i + 1:02d}"
        mp4 = pub / f"{name}.mp4"
        s, e = c["start"], c["end"]
        if not mp4.exists():
            log(f"cortando {name} [{s:.1f}-{e:.1f}]")
            sh([FF, "-v", "error", "-y", "-ss", f"{max(0, s - 0.5):.3f}", "-i", str(d / "source.mp4"), "-t", f"{e - s + 1.0:.3f}",
                "-vf", "scale='min(1920,iw)':-2,fps=30,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-g", "15",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", str(mp4)])
        _, w, h, _ = probe(mp4)
        pts = face_track(mp4, w, h)
        layout = choose_layout(pts, w, h, args.layout)
        off = s - 0.5
        segs = jump_cuts(words, s, e)
        cw = [{"w": x["w"], "s": round(x["s"] - off, 2), "e": round(x["e"] - off, 2)} for x in words if x["s"] >= s - 0.05 and x["e"] <= e + 0.05]
        # Loud moments inside the clip -> zoom kicks (clip-file time).
        a, b = int(s / HOP), int(e / HOP)
        seg = loud[a:b]
        thr = np.percentile(seg, 92) if len(seg) else 0
        peaks, lastp = [], -9
        for k, v in enumerate(seg):
            t = s + k * HOP - off
            if v >= thr and t - lastp > 3:
                peaks.append(round(t, 2))
                lastp = t
        props = {
            "src": f"clips/{args.id}/{name}.mp4", "w": w, "h": h, "layout": layout,
            "segments": segs, "words": cw, "track": smooth_track(pts), "peaks": peaks,
            "title": c["title"], "keywords": c.get("keywords", []), "emojis": c.get("emojis", {}),
            "handle": args.handle, "accent": args.accent, "cropBottom": crop_bottom,
        }
        (d / f"{name}.props.json").write_text(json.dumps(props, ensure_ascii=False), "utf-8")
        kept = sum(x["to"] - x["from"] for x in segs)
        log(f"  {name}: layout={layout} · {len(segs)} cortes · {kept:.1f}s · {len(peaks)} golpes")


# ---------------------------------------------------------------- render
def render(d, args):
    out = ROOT / "out" / "clips" / args.id
    out.mkdir(parents=True, exist_ok=True)
    picks = json.loads((d / "picks.json").read_text("utf-8"))
    # Remotion copies the whole public dir into every bundle (reel-engine/public is GBs):
    # stage a minimal one (this source's clips + sfx) and bundle once for all clips.
    stage = out / "_public"
    shutil.rmtree(stage, ignore_errors=True)
    shutil.copytree(ROOT / "public" / "sfx", stage / "sfx")
    (stage / "clips").mkdir()
    shutil.copytree(ROOT / "public" / "clips" / args.id, stage / "clips" / args.id, copy_function=os.link)
    bundle = out / "_bundle"
    shutil.rmtree(bundle, ignore_errors=True)
    log("empaquetando Remotion")
    sh(f'npx remotion bundle --public-dir="{stage}" --out-dir="{bundle}" --log=error', shell=True, cwd=ROOT)
    try:
        for i, c in enumerate(picks):
            name = f"c{i + 1:02d}"
            props = d / f"{name}.props.json"
            raw = out / f"{name}.raw.mp4"
            slug = re.sub(r"[^a-z0-9]+", "-", re.sub(r"[^\w\s-]", "", c["title"].lower()))[:40].strip("-")
            final = out / f"{name}-{slug}.mp4"
            if final.exists():
                continue
            log(f"render {name}: {c['title']}")
            sh(f'npx remotion render "{bundle}" ViralClip "{raw}" --props="{props}" --codec=h264 --crf=18 --concurrency=4 --log=error', shell=True, cwd=ROOT)
            sh([FF, "-v", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-maxrate", "12M", "-bufsize", "24M",
                "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-af", "alimiter=limit=0.89", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(final)])
            raw.unlink()
            note = f"{c['title']}\n\n{c.get('caption', '')}\n\nPor qué: {c.get('why', '')}\n"
            (out / f"{name}-{slug}.txt").write_text(note, "utf-8")
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(bundle, ignore_errors=True)
    log("listo →", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--id", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--min", type=float, default=20)
    ap.add_argument("--max", type=float, default=55)
    ap.add_argument("--layout", default="auto", choices=["auto", "face", "stream", "fit"])
    ap.add_argument("--model", default="small", help="whisper: tiny|base|small|medium|large-v3")
    ap.add_argument("--lang", default="es", help="idioma del audio")
    ap.add_argument("--from-step", default=None, choices=STEPS, help="force re-running from this step")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--handle", default="")
    ap.add_argument("--accent", default="#FFE600")
    args = ap.parse_args()
    d = ROOT / "content" / "clips" / args.id
    d.mkdir(parents=True, exist_ok=True)
    done = {"fetch": d / "source.mp4", "audio": d / "loudness.json", "transcribe": d / "words.json",
            "score": d / "candidates.json", "pick": d / "picks.json", "cut": d / "c01.props.json", "render": None}
    force = STEPS.index(args.from_step) if args.from_step else 99
    run = {"fetch": lambda: fetch(args.src, d), "audio": lambda: audio(d), "transcribe": lambda: transcribe(d, args.model, args.lang),
           "score": lambda: score(d, args), "pick": lambda: pick(d, args), "cut": lambda: cut(d, args), "render": lambda: render(d, args)}
    for i, st in enumerate(STEPS):
        if st == "render" and args.no_render:
            break
        if i < force and done[st] and done[st].exists():
            continue
        run[st]()


if __name__ == "__main__":
    main()
