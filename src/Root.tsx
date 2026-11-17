import React from "react";
import { Composition } from "remotion";
import { ViralClip, ViralClipProps, clipDuration } from "./ViralClip";

const FPS = 30;

export const Root: React.FC = () => (
  // Render with --props=content/clips/<id>/cNN.props.json (written by tools/clipper.py)
  <Composition id="ViralClip" component={ViralClip} width={1080} height={1920} fps={FPS} durationInFrames={300}
    defaultProps={{ src: "clips/demo/c01.mp4", w: 1920, h: 1080, layout: "face", segments: [{ from: 0, to: 10 }], words: [], track: [], peaks: [], title: "Título gancho", keywords: [], emojis: {} } as ViralClipProps}
    calculateMetadata={({ props }) => ({ durationInFrames: Math.max(1, Math.ceil(clipDuration(props.segments) * FPS)) })} />
);
