import { useEffect, useRef, useState } from "react";
import { imageSize, imageSrc } from "../metrics";
import type { Detection, ImageRecord } from "../types";

/** The part of the image that fills a box of `aspect`, centred on `anchor` boxes when given. */
function coverView(width: number, height: number, aspect: number, anchor: Detection[]) {
  const boxes = anchor.map((d) => d.bbox).filter(([x1, y1, x2, y2]) => x2 > x1 && y2 > y1);
  const cx = boxes.length ? (Math.min(...boxes.map((b) => b[0])) + Math.max(...boxes.map((b) => b[2]))) / 2 : width / 2;
  const cy = boxes.length ? (Math.min(...boxes.map((b) => b[1])) + Math.max(...boxes.map((b) => b[3]))) / 2 : height / 2;
  const clamp = (v: number, hi: number) => Math.max(0, Math.min(v, hi));
  if (!aspect || !isFinite(aspect)) return [0, 0, width, height];
  if (aspect > width / height) {
    const h = width / aspect;
    return [0, clamp(cy - h / 2, height - h), width, h];
  }
  const w = height * aspect;
  return [clamp(cx - w / 2, width - w), 0, w, height];
}

const CLASS_COLOR: Record<string, string> = {
  stop: "#e5484d",
  yield: "#f08c00",
  "no-entry": "#d6336c",
  warning: "#c99a06",
  speed: "#12a594",
  "light-red": "#e5484d",
  "light-green": "#1f9d6b",
};

export const classColor = (label: string) => CLASS_COLOR[label.startsWith("speed") ? "speed" : label] ?? "#3e63dd";

export function Frame({
  image,
  detections,
  focus,
  adapted,
  anchor,
}: {
  image: ImageRecord;
  detections: Detection[];
  focus: number;
  adapted: boolean;
  /** Boxes the crop keeps in view; layers stacked for a wipe must share the same anchor. */
  anchor?: Detection[];
}) {
  const ref = useRef<SVGSVGElement>(null);
  const [aspect, setAspect] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setAspect(el.clientWidth / Math.max(1, el.clientHeight));
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  const shown = detections.map((d, i) => ({ ...d, n: i + 1 }));
  const target = shown.filter((d) => d.status !== "missed")[focus];
  const [width, height] = imageSize(image);
  // Fill the whole box: crop the image to the box's shape, keeping the detections in view.
  const view = coverView(width, height, aspect, anchor ?? detections);
  // Overlay sizes are designed for the 160-wide SVG scenes; scale them to the visible width.
  const s = view[2] / 160;
  return (
    <svg ref={ref} className="lv-frame-svg" viewBox={view.join(" ")} preserveAspectRatio="xMidYMid slice">
      <defs>
        <filter id="cam" colorInterpolationFilters="sRGB">
          <feColorMatrix type="saturate" values={adapted ? "1" : "0.45"} />
          <feComponentTransfer>
            <feFuncR type="linear" slope="0.92" intercept="0.07" />
            <feFuncG type="linear" slope="0.92" intercept="0.07" />
            <feFuncB type="linear" slope="0.92" intercept="0.07" />
          </feComponentTransfer>
        </filter>
        <filter id="neon" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation={0.8 * s} result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <image
        key={image.name}
        href={imageSrc(image)}
        width={width}
        height={height}
        filter="url(#cam)"
        className="lv-img"
      />
      {shown.map((d) => {
        const [x1, y1, x2, y2] = d.bbox;
        if (x2 <= x1 || y2 <= y1) return null;
        const pad = 1.6 * s;
        const bad = d.status !== "hit";
        const color = d.status === "hit" ? classColor(d.label) : d.status === "extra" ? "#f08c00" : "#e5484d";
        const conf = d.confidence != null ? ` ${d.confidence.toFixed(2)}` : "";
        const tag =
          d.status === "missed"
            ? `missed ${d.label}`
            : `${d.n} ${d.label}${conf}${d.status === "hit" ? "" : " ✗"}`;
        const w = (tag.length * 2.05 + 3) * s;
        const lift = target && d.n === target.n ? 4.2 * s : 0;
        return (
          <g key={`${d.label}-${x1}-${y1}-${d.status}`} className={`lv-det ${d.status}`}>
            <rect
              x={x1 - pad}
              y={y1 - pad}
              width={x2 - x1 + 2 * pad}
              height={y2 - y1 + 2 * pad}
              rx={1.4 * s}
              fill={bad ? "none" : `${color}22`}
              stroke={color}
              strokeWidth={0.6 * s}
              strokeDasharray={d.status === "missed" || d.status === "extra" ? `${1.6 * s} ${1.2 * s}` : undefined}
            />
            <rect
              x={x1 - pad}
              y={y1 - pad - 4.4 * s - lift}
              width={w}
              height={4.4 * s}
              rx={1.1 * s}
              fill={color}
              opacity={d.status === "missed" ? 0.5 : 0.95}
            />
            <text
              x={x1 - pad + 1.2 * s}
              y={y1 - pad - 1.2 * s - lift}
              className="lv-tag"
              style={{ fontSize: 3 * s }}
            >
              {tag}
            </text>
          </g>
        );
      })}
      {target && <Brackets bbox={target.bbox} bad={target.status !== "hit"} s={s} />}
    </svg>
  );
}

function Brackets({ bbox, bad, s }: { bbox: [number, number, number, number]; bad: boolean; s: number }) {
  const [x1, y1, x2, y2] = bbox;
  const p = 4.5 * s;
  const L = 5 * s;
  const a = [x1 - p, y1 - p, x2 + p, y2 + p];
  const c = bad ? "#e5484d" : "#3e63dd";
  const d = [
    `M${a[0]},${a[1] + L} V${a[1]} H${a[0] + L}`,
    `M${a[2] - L},${a[1]} H${a[2]} V${a[1] + L}`,
    `M${a[0]},${a[3] - L} V${a[3]} H${a[0] + L}`,
    `M${a[2] - L},${a[3]} H${a[2]} V${a[3] - L}`,
  ].join(" ");
  return (
    <g className="lv-focus" filter="url(#neon)" key={bbox.join(",")}>
      <rect
        x={x1 - 2.2 * s}
        y={y1 - 2.2 * s}
        width={x2 - x1 + 4.4 * s}
        height={y2 - y1 + 4.4 * s}
        rx={2.5 * s}
        fill="none"
        stroke={c}
        strokeWidth={0.9 * s}
      />
      <path d={d} fill="none" stroke={c} strokeWidth={1.3 * s} strokeLinecap="square" />
    </g>
  );
}
