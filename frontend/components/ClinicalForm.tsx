"use client";

import { FormEvent, useState } from "react";
import type { ClinicalInput } from "@/lib/api";

interface Field {
  key: keyof ClinicalInput;
  label: string;
  unit: string;
  placeholder: string;
  step: string;
  min: number;
  max: number;
}

const FIELDS: Field[] = [
  { key: "age", label: "Age", unit: "years", placeholder: "65", step: "1", min: 0, max: 120 },
  { key: "lvef", label: "LVEF", unit: "%", placeholder: "55", step: "0.1", min: 0, max: 80 },
  { key: "troponin", label: "Troponin", unit: "ng/L", placeholder: "12", step: "0.01", min: 0, max: 50000 },
  { key: "ntprobnp", label: "NT-proBNP", unit: "pg/mL", placeholder: "150", step: "1", min: 0, max: 100000 },
];

const DEFAULTS: Record<keyof ClinicalInput, string> = {
  age: "65",
  lvef: "55",
  troponin: "12",
  ntprobnp: "150",
};

export default function ClinicalForm({
  onSubmit,
  isSubmitting,
}: {
  onSubmit: (input: ClinicalInput) => void;
  isSubmitting: boolean;
}) {
  const [values, setValues] = useState<Record<keyof ClinicalInput, string>>(DEFAULTS);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    onSubmit({
      age: Number(values.age),
      lvef: Number(values.lvef),
      troponin: Number(values.troponin),
      ntprobnp: Number(values.ntprobnp),
    });
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-5">
      <div className="grid grid-cols-2 gap-4">
        {FIELDS.map((field) => (
          <label key={field.key} className="flex flex-col gap-1.5 text-sm">
            <span className="font-medium text-foreground">
              {field.label} <span className="font-normal text-foreground-muted">({field.unit})</span>
            </span>
            <input
              required
              type="number"
              inputMode="decimal"
              step={field.step}
              min={field.min}
              max={field.max}
              placeholder={field.placeholder}
              value={values[field.key]}
              onChange={(e) => setValues((v) => ({ ...v, [field.key]: e.target.value }))}
              className="rounded-lg border border-border bg-surface px-3 py-2 text-foreground outline-none transition-colors focus:border-accent focus:ring-2 focus:ring-accent/20"
            />
          </label>
        ))}
      </div>
      <button
        type="submit"
        disabled={isSubmitting}
        className="mt-1 inline-flex items-center justify-center rounded-lg bg-accent px-4 py-2.5 font-medium text-white transition-colors hover:bg-accent-strong disabled:cursor-not-allowed disabled:opacity-60 dark:text-[#04231f]"
      >
        {isSubmitting ? "Analyzing..." : "Assess Risk"}
      </button>
    </form>
  );
}
