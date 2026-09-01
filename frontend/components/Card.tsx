import type { ReactNode } from "react";

export default function Card({
  title,
  subtitle,
  children,
  className = "",
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-2xl border border-border bg-surface p-6 shadow-sm ${className}`}>
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-foreground">{title}</h2>
        {subtitle && <p className="mt-0.5 text-sm text-foreground-muted">{subtitle}</p>}
      </div>
      {children}
    </section>
  );
}
