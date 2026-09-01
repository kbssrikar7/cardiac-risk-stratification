import type { PredictResponse } from "@/lib/api";
import { riskAccent, displayRiskLabel } from "@/lib/risk";

export default function RiskResult({ result }: { result: PredictResponse }) {
  const accent = riskAccent(result.risk_class);
  const sortedProbs = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
  const maxProb = Math.max(...sortedProbs.map(([, p]) => p), 0.0001);

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <span className={`inline-flex items-center rounded-full border px-3 py-1 text-sm font-semibold ${accent.badge}`}>
          {displayRiskLabel(result.risk_class)}
        </span>
        <span className="text-xs uppercase tracking-wide text-foreground-muted">
          model: {result.model_used.replace(/_/g, " ")}
        </span>
      </div>

      <div className="flex flex-col gap-2">
        {sortedProbs.map(([cls, prob]) => {
          const clsAccent = riskAccent(cls);
          return (
            <div key={cls} className="flex items-center gap-3 text-sm">
              <span className="w-52 shrink-0 truncate text-foreground-muted" title={displayRiskLabel(cls)}>
                {displayRiskLabel(cls)}
              </span>
              <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-surface-muted">
                <div
                  className={`h-full rounded-full ${clsAccent.bar}`}
                  style={{ width: `${(prob / maxProb) * 100}%` }}
                />
              </div>
              <span className="w-14 shrink-0 text-right font-mono text-foreground-muted">
                {(prob * 100).toFixed(1)}%
              </span>
            </div>
          );
        })}
      </div>

      <div className="rounded-lg border border-border bg-surface-muted p-4 text-sm">
        <p className="font-medium text-foreground">Rule-based cross-check: {result.rule_based_risk}</p>
        <p className="mt-1 text-foreground-muted">{result.rule_based_reasoning}</p>
      </div>
    </div>
  );
}
