"use client";

import { useState } from "react";
import ClinicalForm from "@/components/ClinicalForm";
import RiskResult from "@/components/RiskResult";
import ShapChart from "@/components/ShapChart";
import GradcamViewer from "@/components/GradcamViewer";
import Card from "@/components/Card";
import { predict, fetchShap, ApiError, type ClinicalInput, type PredictResponse, type ShapResponse } from "@/lib/api";

export default function Home() {
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [shap, setShap] = useState<ShapResponse | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [shapError, setShapError] = useState<string | null>(null);

  async function handleSubmit(input: ClinicalInput) {
    setIsSubmitting(true);
    setError(null);
    setShapError(null);
    try {
      const [predictionResult, shapResult] = await Promise.allSettled([predict(input), fetchShap(input)]);

      if (predictionResult.status === "fulfilled") {
        setPrediction(predictionResult.value);
      } else {
        throw predictionResult.reason;
      }

      if (shapResult.status === "fulfilled") {
        setShap(shapResult.value);
      } else {
        setShap(null);
        setShapError(
          shapResult.reason instanceof ApiError
            ? shapResult.reason.message
            : "Could not load feature contributions."
        );
      }
    } catch (err) {
      setPrediction(null);
      setShap(null);
      setError(err instanceof ApiError ? err.message : "Could not reach the prediction service.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-8 px-6 py-10">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-foreground">Cardiac Risk Stratification</h1>
        <p className="mt-1 text-sm text-foreground-muted">
          Clinical risk assessment with SHAP explainability, plus cardiac MRI segmentation and Grad-CAM visualization
        </p>
      </header>

      <main className="grid flex-1 grid-cols-1 gap-6 lg:grid-cols-2">
        <Card title="Clinical Inputs" subtitle="Enter the patient's clinical biomarkers">
          <ClinicalForm onSubmit={handleSubmit} isSubmitting={isSubmitting} />
          {error && <p className="mt-4 text-sm text-red-500">{error}</p>}
        </Card>

        <Card title="Risk Assessment" subtitle="Calibrated XGBoost prediction with rule-based cross-check">
          {prediction ? (
            <RiskResult result={prediction} />
          ) : (
            <p className="text-sm text-foreground-muted">Submit clinical inputs to see a risk assessment.</p>
          )}
        </Card>

        <Card title="Explainability" subtitle="SHAP feature contributions for this prediction" className="lg:col-span-2">
          {shap ? (
            <ShapChart shap={shap} predictedRiskClass={prediction?.risk_class} />
          ) : shapError ? (
            <p className="text-sm text-red-500">Feature contributions unavailable: {shapError}</p>
          ) : (
            <p className="text-sm text-foreground-muted">Feature contributions will appear here after an assessment.</p>
          )}
        </Card>

        <Card
          title="Cardiac MRI Grad-CAM"
          subtitle="Upload a segmented MRI volume to visualize myocardium attention"
          className="lg:col-span-2"
        >
          <GradcamViewer />
        </Card>
      </main>

      <footer className="text-center text-xs text-foreground-muted">
        Research prototype - not for clinical decision-making.
      </footer>
    </div>
  );
}
