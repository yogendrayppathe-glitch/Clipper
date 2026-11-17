#!/usr/bin/env python3
"""
Mini OpusClip: long YouTube video -> N Instagram-ready clips (4:5, captions, < 60s).

Pipeline: yt-dlp -> faster-whisper (word timestamps) -> Gemini (picks hook moments)
          -> OpenCV face-centred 4:5 crop -> ffmpeg burns captions.

Usage:
    export GEMINI_API_KEY=...
    python clip.py "https://youtu.be/XXXX" --n 5
    python clip.py --file myvideo.mp4 --n 5
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

OUT_W, OUT_H = 1080, 1350  # Instagram 4:5
MAX_CLIP_SEC = 58.0  # keep safely under 60
MIN_CLIP_SEC = 15.0


# ----------------------------------------------------------------------------
# 1. Download
# ----------------------------------------------------------------------------
def download(url: str, work: Path) -> Path:
    out = work / "source.mp4"
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
        "--merge-output-format", "mp4",
        "-o", str(out),
        # YouTube JS challenge: use deno and fetch the solver script if missing
        "--js-runtimes", "deno",
        "--remote-components", "ejs:github",
    ]
    cookies = Path("cookies.txt")
    if cookies.exists():
        cmd += ["--cookies", str(cookies)]
    cmd.append(url)
    print("Downloading video...")
    subprocess.run(cmd, check=True)
    return out


# ----------------------------------------------------------------------------
# 2. Transcribe (word-level timestamps)
# ----------------------------------------------------------------------------
def transcribe(video: Path, model_size: str) -> list[dict]:
    from faster_whisper import WhisperModel

    print(f"Transcribing with Whisper ({model_size})...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(video), word_timestamps=True, vad_filter=True
    )
    print(f"Detected language: {info.language}")
    words = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append({"w": text, "s": float(w.start), "e": float(w.end)})
    if not words:
        sys.exit("No speech found in the video.")
    return words


def transcript_for_llm(words: list[dict], per_line: int = 14) -> str:
    lines = []
    for i in range(0, len(words), per_line):
        chunk = words[i : i + per_line]
        lines.append(f"[{chunk[0]['s']:.1f}] " + " ".join(x["w"] for x in chunk))
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# 3. Ask Gemini for the best hook moments
# ----------------------------------------------------------------------------
PROMPT = """You are a viral short-form video editor for Instagram Reels.

Below is a timestamped transcript of a long video. Each line starts with the
start time in seconds, like [123.4].

Pick the {k} best moments to turn into Reels. Rules:
- Each clip must be between {mn} and {mx} seconds long (hard limit: under 60s).
- The FIRST 3 seconds must be a strong hook: a bold claim, a surprising fact,
  a question, a strong emotion or a cliffhanger. Start exactly where the hook begins.
- The clip must be a complete thought that makes sense on its own, ending on a
  natural sentence end.
- Clips must not overlap each other.
- start / end are in seconds and must match the transcript timestamps.
- hook_text: on-screen title, max 8 words, punchy, same language as the video.
- caption: Instagram caption, 2-3 short lines, starts with a hook line, ends with
  a call to action. Same language/style as the video (use Hinglish if the video is Hinglish).
- hashtags: 5 to 8 relevant hashtags, each starting with #.
- score: 1-10 how likely this is to stop the scroll.

Return ONLY a JSON array, sorted by score (best first), like:
[{{"start": 12.3, "end": 55.0, "hook_text": "...", "caption": "...",
   "hashtags": ["#a", "#b"], "score": 9, "reason": "..."}}]

TRANSCRIPT:
{transcript}
"""


def ask_gemini(words: list[dict], n: int, model_name: str) -> list[dict]:
    from google import genai
    from google.genai import types

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY first (free key: https://aistudio.google.com/apikey)")
    client = genai.Client(api_key=key)

    prompt = PROMPT.format(
        k=n + 3,  # ask for extras, we drop invalid/overlapping ones
        mn=int(MIN_CLIP_SEC),
        mx=int(MAX_CLIP_SEC),
        transcript=transcript_for_llm(words),
    )
    # Try the chosen model first, then fall back if Google retired it (404)
    candidates = [model_name]
    for fb in ("gemini-3.8-flash", "gemini-flash-latest"):
        if fb not in candidates:
            candidates.append(fb)

    resp = None
    last_err = None
    for m in candidates:
        print(f"Asking Gemini ({m}) for hook moments...")
        try:
            resp = client.models.generate_content(
                model=m,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0.4
                ),
            )
            break
        except Exception as ex:  # noqa: BLE001
            last_err = ex
            msg = str(ex)
            if "404" in msg or "NOT_FOUND" in msg:
                print(f"  {m} not available, trying next model...")
                continue
            raise
    if resp is None:
        sys.exit(f"No Gemini model worked. Last error: {last_err}")
    data = json.loads(resp.text)
    if isinstance(data, dict):
        data = data.get("clips", [])
    return data


def snap_and_validate(cands: list[dict], words: list[dict], n: int) -> list[dict]:
    """Snap times to word boundaries, enforce < 60s, drop overlaps, keep top n."""
    starts = [w["s"] for w in words]
    ends = [w["e"] for w in words]
    total = ends[-1]
    good = []
    for c in cands:
        try:
            s, e = float(c["start"]), float(c["end"])
        except (KeyError, TypeError, ValueError):
            continue
        s = min(starts, key=lambda t: abs(t - s))
        e = min(ends, key=lambda t: abs(t - e))
        s = max(0.0, s - 0.15)
        e = min(total + 0.3, e + 0.25)
        if e - s > MAX_CLIP_SEC:
            e = s + MAX_CLIP_SEC
        if e - s < MIN_CLIP_SEC:
            continue
        c.update(start=s, end=e)
        good.append(c)

    good.sort(key=lambda c: -float(c.get("score", 0)))
    picked = []
    for c in good:
        if all(c["end"] <= p["start"] or c["start"] >= p["end"] for p in picked):
            picked.append(c)
        if len(picked) == n:
            break
    return sorted(picked, key=lambda c: c["start"])


# ----------------------------------------------------------------------------
# 4. Face-centred 4:5 crop
# ----------------------------------------------------------------------------
def video_size(video: Path) -> tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def face_center_x(video: Path, start: float, end: float) -> float | None:
    """Median x-centre (0..1) of the largest face over a few sampled frames."""
    import cv2

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    xs = []
    for i in range(8):
        t = start + (end - start) * (i + 0.5) / 8
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
        if len(faces):
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            xs.append((x + fw / 2) / width)
    cap.release()
    return statistics.median(xs) if xs else None


def crop_filter(video: Path, start: float, end: float) -> str:
    w, h = video_size(video)
    target = OUT_W / OUT_H  # 0.8
    if w / h > target:  # wider than 4:5 -> crop sides
        cw = int(h * target) // 2 * 2
        cx = face_center_x(video, start, end)
        cx = cx if cx is not None else 0.5
        x = int(min(max(cx * w - cw / 2, 0), w - cw))
        return f"crop={cw}:{h}:{x}:0"
    ch = int(w / target) // 2 * 2  # taller than 4:5 -> crop top/bottom
    y = max((h - ch) // 2, 0)
    return f"crop={w}:{ch}:0:{y}"


# ----------------------------------------------------------------------------
# 5. Captions (ASS subtitles, burned in)
# ----------------------------------------------------------------------------
def ass_time(t: float) -> str:
    t = max(t, 0.0)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def make_ass(words: list[dict], start: float, end: float, hook: str, path: Path):
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Arial,78,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,6,2,2,60,60,300,1
Style: Hook,Arial,64,&H0000E5FF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,6,2,8,70,70,150,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    if hook:
        lines.append(
            f"Dialogue: 0,{ass_time(0)},{ass_time(3.0)},Hook,,0,0,0,,{hook.upper()}"
        )
    seg = [w for w in words if w["s"] >= start - 0.05 and w["e"] <= end + 0.3]
    for i in range(0, len(seg), 3):
        chunk = seg[i : i + 3]
        t0 = chunk[0]["s"] - start
        nxt = seg[i + 3]["s"] - start if i + 3 < len(seg) else chunk[-1]["e"] - start + 0.2
        t1 = min(max(nxt, chunk[-1]["e"] - start), end - start)
        text = " ".join(c["w"] for c in chunk).upper().replace("{", "(").replace("}", ")")
        lines.append(f"Dialogue: 0,{ass_time(t0)},{ass_time(t1)},Cap,,0,0,0,,{text}")
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------
# 6. Render with ffmpeg
# ----------------------------------------------------------------------------
def render(video: Path, words: list[dict], clip: dict, idx: int, out_dir: Path) -> Path:
    s, e = clip["start"], clip["end"]
    out = out_dir / f"clip_{idx}.mp4"
    ass_name = f"clip_{idx}.ass"
    make_ass(words, s, e, clip.get("hook_text", ""), out_dir / ass_name)

    vf = f"{crop_filter(video, s, e)},scale={OUT_W}:{OUT_H},ass={ass_name}"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{s:.2f}", "-i", str(video.resolve()), "-t", f"{e - s:.2f}",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
        out.name,
    ]
    # run inside out_dir so the ass filter can use a plain relative filename
    subprocess.run(cmd, check=True, cwd=out_dir)
    return out


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Mini OpusClip")
    ap.add_argument("url", nargs="?", help="YouTube (or other yt-dlp) URL")
    ap.add_argument("--file", help="Local video file instead of URL")
    ap.add_argument("--n", type=int, default=5, help="Number of clips")
    ap.add_argument("--out", default="out", help="Output folder")
    ap.add_argument("--whisper", default="base", help="tiny/base/small/medium")
    ap.add_argument("--gemini-model", default="gemini-3.8-flash")
    args = ap.parse_args()

    if not args.url and not args.file:
        ap.error("give a URL or --file")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path("work")
    work.mkdir(exist_ok=True)

    video = Path(args.file) if args.file else download(args.url, work)
    words = transcribe(video, args.whisper)
    cands = ask_gemini(words, args.n, args.gemini_model)
    clips = snap_and_validate(cands, words, args.n)
    if not clips:
        sys.exit("Gemini did not return usable clips. Try again.")

    notes = ["# Instagram captions\n"]
    for i, c in enumerate(clips, 1):
        print(f"Rendering clip {i}/{len(clips)} ({c['end'] - c['start']:.0f}s)...")
        render(video, words, c, i, out_dir)
        tags = " ".join(c.get("hashtags", []))
        notes.append(
            f"## clip_{i}.mp4  ({c['start']:.0f}s - {c['end']:.0f}s, score {c.get('score', '?')})\n"
            f"**Hook:** {c.get('hook_text', '')}\n\n"
            f"{c.get('caption', '')}\n\n{tags}\n\n_Why:_ {c.get('reason', '')}\n"
        )
    (out_dir / "captions.md").write_text("\n".join(notes), encoding="utf-8")
    for f in out_dir.glob("*.ass"):
        f.unlink()
    print(f"\nDone! {len(clips)} clips + captions.md saved in ./{out_dir}/")


if __name__ == "__main__":
    main()
