// /predict and /shap normally return a readable risk_class string directly.
// Defensive fallback only: map the raw ordinal code back to a label in case
// a model pickle's label_mapping ever regresses to bare digit strings again
// (this happened once - see training/common.py's risk_score_label_mapping).
const ORDINAL_RISK_LABELS: Record<string, string> = {
  "0": "Low Risk",
  "1": "Moderate Risk",
  "2": "High Risk (Chronic Heart Failure)",
  "3": "Very High Risk (Acute Cardiac Event)",
};

export function displayRiskLabel(cls: string): string {
  return ORDINAL_RISK_LABELS[cls] ?? cls;
}

// /predict (calibrated XGBoost) and /shap (best standalone XGBoost model) can
// legitimately predict different risk classes for the same patient, since
// they run different models. Surfacing shap.risk_class next to the headline
// prediction without flagging a disagreement would show two contradictory
// risk classes for the same patient with no explanation.
export function shapRiskMismatchNote(
  shapRiskClass: string,
  predictedRiskClass: string | undefined,
): string | null {
  if (!predictedRiskClass || shapRiskClass === predictedRiskClass) return null;
  return `Note: the explainer model predicts "${displayRiskLabel(shapRiskClass)}" for this patient, which differs from the "${displayRiskLabel(predictedRiskClass)}" prediction above.`;
}

export function riskAccent(label: string): { bar: string; badge: string } {
  const l = displayRiskLabel(label).toLowerCase();
  if (l.includes("very high") || l.includes("critical")) {
    return { bar: "bg-red-500", badge: "bg-red-500/15 text-red-600 dark:text-red-400 border-red-500/30" };
  }
  if (l.includes("high")) {
    return { bar: "bg-orange-500", badge: "bg-orange-500/15 text-orange-600 dark:text-orange-400 border-orange-500/30" };
  }
  if (l.includes("moderate")) {
    return { bar: "bg-amber-500", badge: "bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/30" };
  }
  if (l.includes("low")) {
    return { bar: "bg-emerald-500", badge: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30" };
  }
  return { bar: "bg-accent", badge: "bg-accent/15 text-accent-strong border-accent/30" };
}
