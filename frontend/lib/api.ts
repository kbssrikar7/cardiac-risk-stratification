const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function parseErrorDetail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") return body.detail;
  } catch {
    // response wasn't JSON, fall through to statusText
  }
  return res.statusText || `Request failed with status ${res.status}`;
}

export interface ClinicalInput {
  age: number;
  lvef: number;
  troponin: number;
  ntprobnp: number;
}

export interface PredictResponse {
  risk_class: string;
  probabilities: Record<string, number>;
  model_used: string;
  rule_based_risk: string;
  rule_based_reasoning: string;
}

export interface ShapContribution {
  feature: string;
  shap_value: number;
  value: number | null;
}

export interface ShapResponse {
  risk_class: string;
  model_used: string;
  contributions: ShapContribution[];
}

export interface GradcamResponse {
  overlay_png_base64: string;
  slice_index: number;
  num_slices: number;
}

export async function predict(input: ClinicalInput): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) throw new ApiError(res.status, await parseErrorDetail(res));
  return res.json();
}

export async function fetchShap(input: ClinicalInput): Promise<ShapResponse> {
  const params = new URLSearchParams({
    age: String(input.age),
    lvef: String(input.lvef),
    troponin: String(input.troponin),
    ntprobnp: String(input.ntprobnp),
  });
  const res = await fetch(`${API_BASE_URL}/shap?${params.toString()}`);
  if (!res.ok) throw new ApiError(res.status, await parseErrorDetail(res));
  return res.json();
}

export async function fetchGradcam(file: File): Promise<GradcamResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE_URL}/gradcam`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new ApiError(res.status, await parseErrorDetail(res));
  return res.json();
}
