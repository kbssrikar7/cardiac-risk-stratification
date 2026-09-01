"use client";

import { ChangeEvent, useState } from "react";
import { fetchGradcam, type GradcamResponse, ApiError } from "@/lib/api";

const MAX_UPLOAD_BYTES = 100 * 1024 * 1024; // 100 MB, matches the backend limit

export default function GradcamViewer() {
  const [fileName, setFileName] = useState<string | null>(null);
  const [result, setResult] = useState<GradcamResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleFile(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setFileName(file.name);
    setResult(null);
    setError(null);

    if (file.size > MAX_UPLOAD_BYTES) {
      setError(`File is too large (max ${MAX_UPLOAD_BYTES / (1024 * 1024)}MB).`);
      e.target.value = "";
      return;
    }

    setLoading(true);
    try {
      const res = await fetchGradcam(file);
      setResult(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to generate Grad-CAM overlay.");
    } finally {
      setLoading(false);
      e.target.value = "";
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <label className="flex cursor-pointer flex-col items-center gap-2 rounded-lg border border-dashed border-border bg-surface-muted px-4 py-8 text-center transition-colors hover:border-accent">
        <span className="text-sm font-medium text-foreground">
          {fileName ?? "Upload a cardiac MRI (.nii / .nii.gz)"}
        </span>
        <span className="text-xs text-foreground-muted">Click to browse, or drop a file here</span>
        <input type="file" accept=".nii,.nii.gz" className="hidden" onChange={handleFile} />
      </label>

      <p className="text-center text-xs text-foreground-muted">
        Don&apos;t have an MRI file handy?{" "}
        <a href="/sample_cardiac_mri.nii.gz" download className="text-accent underline underline-offset-2">
          Download a sample volume
        </a>{" "}
        to try it (synthetic test data, not a real patient scan).
      </p>

      {loading && <p className="text-sm text-foreground-muted">Segmenting myocardium and generating Grad-CAM overlay...</p>}
      {error && <p className="text-sm text-red-500">{error}</p>}

      {result && (
        <div className="flex flex-col gap-3">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`data:image/png;base64,${result.overlay_png_base64}`}
            alt="Grad-CAM myocardium overlay"
            className="w-full max-w-md self-center rounded-lg border border-border bg-black"
          />
          <p className="text-center text-xs text-foreground-muted">
            Slice {result.slice_index + 1} of {result.num_slices}
          </p>
        </div>
      )}
    </div>
  );
}
