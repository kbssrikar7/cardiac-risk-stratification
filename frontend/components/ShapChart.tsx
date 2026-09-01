import type { ShapResponse } from "@/lib/api";
import { displayRiskLabel, shapRiskMismatchNote } from "@/lib/risk";

export default function ShapChart({
  shap,
  predictedRiskClass,
}: {
  shap: ShapResponse;
  predictedRiskClass?: string;
}) {
  const maxAbs = Math.max(...shap.contributions.map((c) => Math.abs(c.shap_value)), 0.0001);
  const mismatchNote = shapRiskMismatchNote(shap.risk_class, predictedRiskClass);

  return (
    <div className="flex flex-col gap-3">
      <p className="text-xs uppercase tracking-wide text-foreground-muted">
        Feature contributions toward &ldquo;{displayRiskLabel(shap.risk_class)}&rdquo;
      </p>
      {mismatchNote ? (
        <p className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-400">
          {mismatchNote}
        </p>
      ) : null}
      <div className="flex flex-col gap-2.5">
        {shap.contributions.map((c) => {
          const positive = c.shap_value >= 0;
          const widthPct = (Math.abs(c.shap_value) / maxAbs) * 50;
          return (
            <div key={c.feature} className="grid grid-cols-[7rem_1fr_4.5rem] items-center gap-2 text-sm">
              <span className="truncate text-foreground-muted">{c.feature}</span>
              <div className="relative h-3 rounded-sm bg-surface-muted">
                <div className="absolute inset-y-0 left-1/2 w-px bg-border" />
                <div
                  className={`absolute inset-y-0 rounded-sm ${positive ? "bg-red-500" : "bg-emerald-500"}`}
                  style={
                    positive
                      ? { left: "50%", width: `${widthPct}%` }
                      : { right: "50%", width: `${widthPct}%` }
                  }
                />
              </div>
              <span className="text-right font-mono text-xs text-foreground-muted">
                {c.shap_value >= 0 ? "+" : ""}
                {c.shap_value.toFixed(3)}
              </span>
            </div>
          );
        })}
      </div>
      <p className="text-xs text-foreground-muted">
        Red pushes risk higher, green pushes it lower.
      </p>
    </div>
  );
}
