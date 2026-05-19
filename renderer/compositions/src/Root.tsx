import React from "react";
import { Composition } from "remotion";
import { ShortVideo } from "./compositions/ShortVideo";
import type { ShortVideoProps } from "./lib/types";
import { shortVideoPropsSchema } from "./lib/types";

const DEFAULT_PROPS: ShortVideoProps = {
  videoUrl: "",
  durationInFrames: 900, // 30s at 30fps
  fps: 30,
  width: 1080,
  height: 1920,
  subtitles: {
    captions: [
      { text: "This", startMs: 0, endMs: 400 },
      { text: "is", startMs: 400, endMs: 600 },
      { text: "a", startMs: 600, endMs: 750 },
      { text: "demo", startMs: 750, endMs: 1200 },
      { text: "of", startMs: 1200, endMs: 1400 },
      { text: "animated", startMs: 1400, endMs: 2000 },
      { text: "subtitles", startMs: 2000, endMs: 2800 },
      { text: "in", startMs: 2800, endMs: 3000 },
      { text: "Remotion", startMs: 3000, endMs: 3800 },
      { text: "with", startMs: 4000, endMs: 4300 },
      { text: "word", startMs: 4300, endMs: 4700 },
      { text: "level", startMs: 4700, endMs: 5100 },
      { text: "highlighting", startMs: 5100, endMs: 6000 },
    ],
    position: "bottom",
    style: {
      fontFamily: "Arial",
      fontSize: 52,
      fontColor: "#FFFFFF",
      highlightColor: "#FFDD00",
      borderColor: "#000000",
      borderWidth: 3,
      bgColor: "#000000",
      bgOpacity: 0,
      animation: "pop",
    },
  },
  hook: {
    text: "POV: You just discovered OpenShorts",
    position: "top",
    size: "M",
    entranceAnimation: "spring",
    displayDurationSec: 5,
  },
  effects: {
    segments: [
      {
        startSec: 2,
        endSec: 5,
        zoom: 1.2,
        zoomCenterX: 0.5,
        zoomCenterY: 0.35,
        brightness: 1.05,
        contrast: 1.1,
        saturate: 1.15,
      },
      {
        startSec: 8,
        endSec: 12,
        zoom: 1.15,
        zoomCenterX: 0.5,
        zoomCenterY: 0.4,
        brightness: 1,
        contrast: 1,
        saturate: 1,
      },
    ],
  },
};

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="ShortVideo"
        schema={shortVideoPropsSchema}
        component={ShortVideo}
        durationInFrames={DEFAULT_PROPS.durationInFrames}
        fps={DEFAULT_PROPS.fps}
        width={DEFAULT_PROPS.width}
        height={DEFAULT_PROPS.height}
        defaultProps={DEFAULT_PROPS}
      />
    </>
  );
};
