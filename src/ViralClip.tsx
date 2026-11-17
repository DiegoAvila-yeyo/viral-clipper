// Vertical viral clip cut from a long horizontal video (YouTube / stream VOD / podcast).
// Built for short attention spans: jump cuts on every pause, a zoom change on almost every cut,
// zoom kicks + shake on loud moments, 1-3 word captions with colored keywords, emoji stickers,
// a hook title from frame 0 and a progress bar. Props come from tools/clipper.py.
import React from "react";
import { AbsoluteFill, Audio, Easing, OffthreadVideo, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { loadFont } from "@remotion/google-fonts/Montserrat";

const { fontFamily } = loadFont("normal", { weights: ["800", "900"], subsets: ["latin", "latin-ext"] });

const W = 1080, H = 1920;

export type Seg = { from: number; to: number };            // clip-file seconds
export type Word = { w: string; s: number; e: number };   // clip-file seconds
export type Key = { t: number; cx: number; cy: number; fh: number; cut?: boolean };
export type ViralClipProps = {
  src: string; w: number; h: number;
  layout: "face" | "stream" | "fit";
  segments: Seg[]; words: Word[]; track: Key[]; peaks: number[];
  title: string; keywords: string[]; emojis: Record<string, string>;
  handle?: string; accent?: string; sfx?: boolean;
  cropBottom?: number;  // fraction of the source bottom with burned-in subtitles, kept out of frame
};

export const clipDuration = (segs: Seg[]) => segs.reduce((a, s) => a + s.to - s.from, 0);

const norm = (s: string) => s.toLowerCase().normalize("NFD").replace(/\p{M}/gu, "").replace(/[^\p{L}\p{N}]/gu, "");
const KEYWORD = "#3DFF6E";

/** Camera target at source time t: hold each keyframe, glide 0.5 s into the next one, snap on scene cuts. */
const camAt = (track: Key[], t: number) => {
  if (!track.length) return { cx: 0.5, cy: 0.4, fh: 0.25 };
  let cur = { ...track[0] };
  for (let i = 1; i < track.length; i++) {
    const k = track[i];
    if (k.cut) { if (t < k.t) break; cur = { ...k }; continue; }
    if (t < k.t - 0.5) break;
    const a = interpolate(t, [k.t - 0.5, k.t], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.inOut(Easing.cubic) });
    cur = { t: k.t, cx: cur.cx + (k.cx - cur.cx) * a, cy: cur.cy + (k.cy - cur.cy) * a, fh: cur.fh + (k.fh - cur.fh) * a };
  }
  return cur;
};

/** Output timeline: where each segment starts, and output<->source time maps. */
const useTimeline = (segs: Seg[]) => React.useMemo(() => {
  const starts: number[] = [];
  let acc = 0;
  segs.forEach((s) => { starts.push(acc); acc += s.to - s.from; });
  const toOut = (src: number) => {
    const i = segs.findIndex((s) => src >= s.from - 0.02 && src <= s.to + 0.02);
    return i < 0 ? null : starts[i] + Math.max(0, src - segs[i].from);
  };
  return { starts, total: acc, toOut };
}, [segs]);

/** Zoom level per segment: change on a cut only if the last change was >= 1.2 s ago (no strobe). */
const segZooms = (segs: Seg[]) => {
  const levels = [1, 1.14, 1.05, 1.22];
  let li = 0, since = 99;
  return segs.map((s) => {
    if (since >= 1.2) { li = (li + 1) % levels.length; since = 0; }
    since += s.to - s.from;
    return levels[li];
  });
};

// ------------------------------------------------------------------ video layer
const Frame: React.FC<{ p: ViralClipProps; seg: Seg; zoom: number; kick: number; shake: [number, number] }> = ({ p, seg, zoom, kick, shake }) => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const dur = (seg.to - seg.from) * fps;
  const t = seg.from + f / fps;
  const cam = camAt(p.track, t);
  const push = interpolate(f, [0, dur], [0, 0.035]);
  const z = zoom + push + kick;
  const fadeFr = 2;
  const vol = (fr: number) => interpolate(fr, [0, fadeFr, dur - fadeFr, dur], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const video = (style: React.CSSProperties, muted = false) => (
    <OffthreadVideo src={staticFile(p.src)} startFrom={Math.round(seg.from * fps)} endAt={Math.round(seg.to * fps) + 1}
      muted={muted} volume={muted ? 0 : vol} style={style} />
  );

  // Usable source height: burned-in subtitles at the bottom are pushed out of every box.
  const ph = p.h * (1 - (p.cropBottom ?? 0));
  // Place a w×h source scaled by S so that (cx,cy) sits at (ax,ay) of a box, clamped to the box edges.
  const place = (S: number, bw: number, bh: number, ax: number, ay: number, cx: number, cy: number): React.CSSProperties => {
    const vw = p.w * S, vh = ph * S;
    const left = Math.min(0, Math.max(bw - vw, ax - cx * vw));
    const top = Math.min(0, Math.max(bh - vh, ay - cy * p.h * S));
    return { position: "absolute", width: vw, height: p.h * S, left: left + shake[0], top: top + shake[1] };
  };

  if (p.layout === "stream") {
    // Top: facecam blown up. Bottom: center of the gameplay/screen.
    const camH = 760, gameH = H - camH;
    const region = Math.min(1, Math.max(0.22, cam.fh * 3.4));     // facecam region height (fraction of source)
    const Sc = (camH / (p.h * region)) * (1 + (z - 1) * 0.6);
    const Sg = Math.max(W / p.w, gameH / ph) * (1 + (z - 1) * 0.5);
    return (
      <AbsoluteFill style={{ background: "#000" }}>
        <div style={{ position: "absolute", top: 0, left: 0, width: W, height: camH, overflow: "hidden" }}>
          {video(place(Sc, W, camH, W / 2, camH * 0.48, cam.cx, cam.cy), true)}
        </div>
        <div style={{ position: "absolute", top: camH, left: 0, width: W, height: gameH, overflow: "hidden" }}>
          {video(place(Sg, W, gameH, W / 2, gameH / 2, 0.5, 0.5))}
        </div>
        <div style={{ position: "absolute", top: camH - 3, left: 0, width: W, height: 6, background: p.accent ?? "#FFE600" }} />
      </AbsoluteFill>
    );
  }
  if (p.layout === "fit") {
    // Wide shot (2+ people / scene) big in the middle, following the action, over a blurred cover of itself.
    const Sb = Math.max(W / p.w, H / ph) * 1.1;
    const Sf = (1000 / ph) * (1 + (z - 1) * 1.2);
    const vw = p.w * Sf, vh = ph * Sf;
    const left = Math.min(0, Math.max(W - vw, W / 2 - cam.cx * vw));
    return (
      <AbsoluteFill style={{ background: "#000" }}>
        <div style={{ position: "absolute", inset: 0, filter: "blur(40px) brightness(0.5) saturate(1.4)", transform: "scale(1.1)" }}>
          {video(place(Sb, W, H, W / 2, H / 2, 0.5, 0.5), true)}
        </div>
        <div style={{ position: "absolute", left: 0, top: H * 0.47 - 500, width: W, height: 1000, overflow: "hidden", boxShadow: "0 0 60px rgba(0,0,0,0.7)" }}>
          {video({ position: "absolute", width: vw, height: p.h * Sf, left: left + shake[0], top: (1000 - vh) / 2 + shake[1] })}
        </div>
      </AbsoluteFill>
    );
  }
  // face: full-height crop that follows the speaker.
  const S = Math.max(W / p.w, H / ph) * z;
  return <AbsoluteFill style={{ background: "#000", overflow: "hidden" }}>{video(place(S, W, H, W / 2, H * 0.36, cam.cx, cam.cy))}</AbsoluteFill>;
};

// ------------------------------------------------------------------ captions
type OW = Word & { key: boolean; seg: number };

const Captions: React.FC<{ words: OW[]; accent: string; top: number }> = ({ words, accent, top }) => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = f / fps;
  const chunks = React.useMemo(() => {
    const out: OW[][] = []; let cur: OW[] = [];
    words.forEach((w, i) => {
      cur.push(w);
      const n = words[i + 1];
      const chars = cur.reduce((a, x) => a + x.w.length, 0);
      if (cur.length >= 3 || chars > 14 || w.key || /[.,:;?!…]$/.test(w.w) || !n || n.key || n.s - w.e > 0.5) { out.push(cur); cur = []; }
    });
    return out;
  }, [words]);
  const idx = chunks.findIndex((c, i) => t >= c[0].s - 0.04 && (i === chunks.length - 1 || t < chunks[i + 1][0].s - 0.04) && t < c[c.length - 1].e + 0.5);
  if (idx < 0) return null;
  const c = chunks[idx];
  const sf = Math.round((c[0].s - 0.04) * fps);
  const sp = spring({ frame: f - sf, fps, config: { damping: 11, stiffness: 260, mass: 0.6 } });
  const tilt = (idx % 2 ? 1 : -1) * 1.5;
  return (
    <div style={{ position: "absolute", top, left: 40, right: 40, textAlign: "center", fontFamily, transform: `scale(${0.6 + 0.4 * sp}) rotate(${tilt * (1 - sp)}deg)` }}>
      {c.map((w, i) => {
        const active = t >= w.s - 0.03 && t < (c[i + 1]?.s ?? w.e + 0.5);
        const color = w.key ? KEYWORD : active ? accent : "#fff";
        const size = w.key ? 112 : 92;
        return (
          <span key={i} style={{ display: "inline-block", margin: "0 12px", fontWeight: 900, fontSize: size, lineHeight: 1.05, letterSpacing: -1, textTransform: "uppercase",
            color, WebkitTextStroke: "14px #000", paintOrder: "stroke fill", textShadow: "0 8px 0 rgba(0,0,0,0.9), 0 14px 30px rgba(0,0,0,0.6)",
            transform: active ? "scale(1.08) translateY(-4px)" : "none" }}>
            {w.w.replace(/[.,;:]+$/, "")}
          </span>
        );
      })}
    </div>
  );
};

const Sticker: React.FC<{ emoji: string; i: number }> = ({ emoji, i }) => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const sp = spring({ frame: f, fps, config: { damping: 8, stiffness: 200 } });
  const out = interpolate(f, [fps * 1.0, fps * 1.3], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const side = i % 2 ? 1 : -1;
  return (
    <div style={{ position: "absolute", top: 930 - f * 1.2, left: W / 2 + side * 300 - 90, fontSize: 180, opacity: out,
      transform: `scale(${sp}) rotate(${side * (12 - 8 * sp)}deg)`, filter: "drop-shadow(0 10px 18px rgba(0,0,0,0.5))" }}>
      {emoji}
    </div>
  );
};

const Title: React.FC<{ text: string; accent: string; top: number }> = ({ text, accent, top }) => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const sp = spring({ frame: f - 1, fps, config: { damping: 10, stiffness: 180 } });
  const big = interpolate(f, [fps * 2.6, fps * 3.2], [1, 0.72], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.inOut(Easing.cubic) });
  const wob = Math.sin(f / 9) * 0.8;
  return (
    <div style={{ position: "absolute", top, left: 0, right: 0, display: "flex", justifyContent: "center", transformOrigin: "50% 0%", transform: `scale(${sp * big}) rotate(${wob}deg)` }}>
      <div style={{ maxWidth: 900, fontFamily, fontWeight: 900, fontSize: 62, lineHeight: 1.12, textAlign: "center", color: "#111", background: "#fff",
        padding: "18px 30px", borderRadius: 22, boxShadow: `0 10px 0 ${accent}, 0 20px 40px rgba(0,0,0,0.45)` }}>
        {text}
      </div>
    </div>
  );
};

// ------------------------------------------------------------------ main
export const ViralClip: React.FC<ViralClipProps> = (p) => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const accent = p.accent ?? "#FFE600";
  const tl = useTimeline(p.segments);
  const zooms = React.useMemo(() => segZooms(p.segments), [p.segments]);
  const keys = React.useMemo(() => new Set(p.keywords.flatMap((k) => k.split(/\s+/)).map(norm).filter((k) => k.length > 1)), [p.keywords]);
  const emo = React.useMemo(() => Object.fromEntries(Object.entries(p.emojis ?? {}).map(([k, v]) => [norm(k), v])), [p.emojis]);

  const words: OW[] = React.useMemo(() => p.words.flatMap((w) => {
    const s = tl.toOut(w.s);
    if (s == null) return [];
    const e = tl.toOut(Math.min(w.e, p.segments.find((g) => w.s >= g.from - 0.02 && w.s <= g.to + 0.02)!.to)) ?? s + 0.2;
    return [{ ...w, s, e: Math.max(e, s + 0.08), key: keys.has(norm(w.w)), seg: 0 }];
  }), [p.words, tl, keys]);

  // Kicks: loud peaks (strong) and keyword hits (soft), in output time.
  const kicks = React.useMemo(() => [
    ...p.peaks.map((t) => ({ t: tl.toOut(t), a: 0.13, shake: true })),
    ...words.filter((w) => w.key).map((w) => ({ t: w.s, a: 0.06, shake: false })),
  ].filter((k): k is { t: number; a: number; shake: boolean } => k.t != null), [p.peaks, words, tl]);
  const t = f / fps;
  let kick = 0, shakeAmt = 0;
  for (const k of kicks) {
    const d = t - k.t;
    if (d >= 0 && d < 0.6) {
      const env = Math.exp(-d * 7) * (1 - Math.exp(-d * 60));
      kick += k.a * env;
      if (k.shake) shakeAmt += 14 * env;
    }
  }
  const shake: [number, number] = [Math.sin(f * 2.7) * shakeAmt, Math.cos(f * 3.3) * shakeAmt];
  const flash = kicks.filter((k) => k.shake).some((k) => t - k.t >= 0 && t - k.t < 2 / fps);

  const stickers = React.useMemo(() => {
    const out: { t: number; e: string }[] = [];
    words.forEach((w) => { const e = emo[norm(w.w)]; if (e && !out.some((o) => Math.abs(o.t - w.s) < 1.5)) out.push({ t: w.s, e }); });
    return out;
  }, [words, emo]);

  const capTop = p.layout === "stream" ? 640 : p.layout === "fit" ? 1230 : 1180;
  const titleTop = p.layout === "stream" ? 60 : 190;
  const sfx = p.sfx !== false;

  return (
    <AbsoluteFill style={{ background: "#000" }}>
      {p.segments.map((s, i) => (
        <Sequence key={i} from={Math.round(tl.starts[i] * fps)} durationInFrames={Math.max(1, Math.round((s.to - s.from) * fps))} name={`seg ${i}`}>
          <Frame p={p} seg={s} zoom={zooms[i]} kick={kick} shake={shake} />
        </Sequence>
      ))}
      <AbsoluteFill style={{ background: "linear-gradient(180deg, rgba(0,0,0,0.35) 0%, rgba(0,0,0,0) 16%, rgba(0,0,0,0) 55%, rgba(0,0,0,0.45) 80%, rgba(0,0,0,0.1) 100%)" }} />
      {flash ? <AbsoluteFill style={{ background: "rgba(255,255,255,0.28)" }} /> : null}
      {stickers.map((s, i) => (
        <Sequence key={`st${i}`} from={Math.round(s.t * fps)} durationInFrames={Math.round(1.4 * fps)}>
          <Sticker emoji={s.e} i={i} />
          {sfx ? <Audio src={staticFile("sfx/pop.mp3")} volume={0.35} /> : null}
        </Sequence>
      ))}
      <Captions words={words} accent={accent} top={capTop} />
      <Title text={p.title} accent={accent} top={titleTop} />
      {p.handle ? <div style={{ position: "absolute", bottom: 150, width: "100%", textAlign: "center", fontFamily, fontWeight: 800, fontSize: 30, color: "rgba(255,255,255,0.7)" }}>{p.handle}</div> : null}
      <div style={{ position: "absolute", top: 0, left: 0, height: 10, width: `${(t / tl.total) * 100}%`, background: accent, boxShadow: `0 0 18px ${accent}` }} />
      {sfx ? <Audio src={staticFile("sfx/whoosh.mp3")} volume={0.4} /> : null}
      {sfx ? p.peaks.map((pk, i) => { const o = tl.toOut(pk); return o == null ? null : (
        <Sequence key={`pk${i}`} from={Math.round(o * fps)} durationInFrames={Math.round(fps)}><Audio src={staticFile("sfx/stamp.mp3")} volume={0.18} /></Sequence>
      ); }) : null}
    </AbsoluteFill>
  );
};
