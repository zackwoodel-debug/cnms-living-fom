import type { BadgeKind, ProvenanceTier, ScoreStatus } from "../lib/api";

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

const KIND_LABEL: Record<BadgeKind, [string, string]> = {
  reference_literature: ["reference / literature", "muted"],
  user_assumption: ["your assumption", "warn"],
  calculated: ["calculated", "muted"],
  synthetic_prediction: ["synthetic prediction", "warn"],
  experimental_fit: ["experimental fit", "ok"],
  unknown: ["unknown", "bad"],
};

/**
 * What kind of number this is, plus whether a person reviewed it. The two are separate on purpose:
 * "reviewed" never turns a literature value into a measurement.
 */
export function KindBadge({ kind, reviewStatus }: { kind: BadgeKind; reviewStatus?: string }) {
  const [label, tone] = KIND_LABEL[kind];
  return (
    <>
      <span className={`badge ${tone}`}>{label}</span>{" "}
      {reviewStatus !== undefined && (
        <span className={`badge ${reviewStatus === "approved" ? "ok" : reviewStatus === "unreviewed" ? "muted" : "bad"}`}>
          {reviewStatus === "approved" ? "approved by a person" : reviewStatus}
        </span>
      )}
    </>
  );
}
