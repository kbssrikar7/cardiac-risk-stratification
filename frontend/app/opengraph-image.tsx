import { ImageResponse } from "next/og";

export const alt = "Cardiac Risk Stratification";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const BACKGROUND = "#0a1414";
const SURFACE = "#101d1d";
const BORDER = "#223838";
const FOREGROUND = "#eaf3f2";
const FOREGROUND_MUTED = "#93aeae";
const ACCENT = "#2dd4bf";

export default function Image() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: "72px 80px",
          background: BACKGROUND,
          backgroundImage:
            "radial-gradient(circle at 82% 18%, rgba(45,212,191,0.16), rgba(45,212,191,0) 55%)",
          fontFamily: "sans-serif",
        }}
      >
        <div style={{ display: "flex", alignItems: "center" }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: 56,
              height: 56,
              borderRadius: 14,
              background: SURFACE,
              border: `1px solid ${BORDER}`,
              marginRight: 20,
            }}
          >
            <svg width="30" height="30" viewBox="0 0 24 24" fill="none">
              <path
                d="M3 12h4l2-7 4 14 2-7h6"
                stroke={ACCENT}
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
          <div
            style={{
              display: "flex",
              fontSize: 28,
              fontWeight: 600,
              color: FOREGROUND_MUTED,
              letterSpacing: "0.02em",
            }}
          >
            Cardiac Risk Stratification
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column" }}>
          <div
            style={{
              display: "flex",
              fontSize: 64,
              fontWeight: 700,
              color: FOREGROUND,
              lineHeight: 1.15,
              maxWidth: 920,
            }}
          >
            Explainable multi-modal cardiac risk stratification
          </div>
          <div
            style={{
              display: "flex",
              marginTop: 24,
              fontSize: 28,
              color: FOREGROUND_MUTED,
              maxWidth: 820,
            }}
          >
            Calibrated risk prediction, SHAP explainability, and MRI Grad-CAM
            in one clinical dashboard.
          </div>
        </div>

        <div style={{ display: "flex", gap: 14 }}>
          {["XGBoost", "SHAP", "U-Net Segmentation", "Grad-CAM"].map((tag) => (
            <div
              key={tag}
              style={{
                display: "flex",
                padding: "10px 20px",
                borderRadius: 999,
                background: SURFACE,
                border: `1px solid ${BORDER}`,
                color: ACCENT,
                fontSize: 22,
                fontWeight: 600,
              }}
            >
              {tag}
            </div>
          ))}
        </div>
      </div>
    ),
    { ...size }
  );
}
