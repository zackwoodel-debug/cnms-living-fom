import type { ProvenanceTier, ScoreStatus } from "../lib/api";

/** Provenance tier. Measured, calculated, and modeled are never interchangeable. */
export function TierBadge({ tier }: { tier: ProvenanceTier }) {
  const tone =
    tier === "measured" ? "ok" : tier === "calculated" ? "muted" : tier === "modeled" ? "warn" : "bad";
  return <span className={`badge ${tone}`}>{tier}</span>;
}

/**
 * Score status.
 *
 * `not_scored` is shown as prominently as a real score on purpose: a material
 * that could not be scored is a finding about data coverage, not a blank cell
 * to skim past.
 */
export function StatusBadge({ status }: { status: ScoreStatus }) {
  if (status === "scored") return <span className="badge ok">scored</span>;
  if (status === "illustrative") return <span className="badge warn">illustrative</span>;
  return <span className="badge bad">not scored</span>;
}

export function DraftBadge({ approved }: { approved: boolean }) {
  return approved ? (
    <span className="badge ok">approved</span>
  ) : (
    <span className="badge warn">draft weights</span>
  );
}
