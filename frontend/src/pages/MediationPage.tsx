import { useState } from "react";

import { Async } from "../components/Async";
import { api, type MediationResponse } from "../lib/api";
import { useAsync } from "../lib/useAsync";

/** Reference point for the elasticity conversion — a mid-range high-k oxide. */
const REFERENCE_PROPERTIES = { k: 25.0, Eg: 5.7, dEc: 1.5, Ebd: 4.0, kappa_th: 1.1, eps_ionic: 20.5 };
const REFERENCE_DESCRIPTORS = { Z_RMS_star: 4.5, omega_TO_min: 140.0, V_fu: 34.0, mu_eff: 12.0 };

function SignedBar({ value, scale }: { value: number; scale: number }) {
  const fraction = Math.min(Math.abs(value) / scale, 1) * 50;
  return (
    <div className="bar">
      <div className="track">
        <div className="zero" />
        <div
          className={`fill ${value >= 0 ? "pos" : "neg"}`}
          style={{ width: `${fraction}%` }}
        />
      </div>
      <span className="mono" style={{ minWidth: 76, textAlign: "right" }}>
        {value >= 0 ? "+" : ""}
        {value.toFixed(4)}
      </span>
    </div>
  );
}

export default function MediationPage() {
  const hypotheses = useAsync(() => api.hypotheses(), []);
  const [application, setApplication] = useState("logic");
  const [result, setResult] = useState<MediationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setResult(
        await api.mediate({
          fom_name: application,
          reference_properties: REFERENCE_PROPERTIES,
          reference_descriptors: REFERENCE_DESCRIPTORS,
        }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const scale = result
    ? Math.max(...Object.values(result.mediated_elasticity).map(Math.abs), 1e-9)
    : 1;

  return (
    <>
      <div className="panel">
        <h2>Mediated structure &rarr; function effects</h2>
        <p className="hint">
          M<sub>ja</sub> = &sum;<sub>q</sub> B<sub>jq</sub> &Gamma;<sub>qa</sub>. B is the estimated
          structure&rarr;property physics; &Gamma; is how this application&rsquo;s weights reward each
          property. This is the scientific result &mdash; a direct descriptor-to-score correlation is
          a summary, because it does not identify the intermediate property pathway.
        </p>
        <div className="row">
          <select value={application} onChange={(event) => setApplication(event.target.value)}>
            <option value="logic">logic</option>
            <option value="power">power</option>
            <option value="rf">rf</option>
          </select>
          <button onClick={run} disabled={busy}>
            {busy ? "Computing…" : "Compute M = B Γ"}
          </button>
        </div>
      </div>

      {error && <div className="notice bad">{error}</div>}

      {result && (
        <div className="panel">
          <h2>
            {result.application} · B from {result.sensitivity_source}
          </h2>
          <p className="hint">
            Bars show d&nbsp;ln&nbsp;F / d&nbsp;ln&nbsp;S &mdash; dimensionless, so descriptors with
            different units compare directly.
          </p>

          {result.notes.map((note) => (
            <div className="notice info" key={note}>
              {note}
            </div>
          ))}

          <table>
            <thead>
              <tr>
                <th>Descriptor</th>
                <th style={{ width: "45%" }}>Elasticity d ln F / d ln S</th>
                <th className="num">M = d ln F / dS</th>
                <th>Dominant channel</th>
              </tr>
            </thead>
            <tbody>
              {result.descriptors
                .slice()
                .sort(
                  (a, b) =>
                    Math.abs(result.mediated_elasticity[b] ?? 0) -
                    Math.abs(result.mediated_elasticity[a] ?? 0),
                )
                .map((descriptor) => (
                  <tr key={descriptor}>
                    <td className="mono">{descriptor}</td>
                    <td>
                      <SignedBar value={result.mediated_elasticity[descriptor] ?? 0} scale={scale} />
                    </td>
                    <td className="num">{result.mediated_effect[descriptor]?.toExponential(3)}</td>
                    <td className="mono muted">{result.dominant_property[descriptor] || "—"}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}

      <Async state={hypotheses}>
        {(registry) => (
          <div className="panel">
            <h2>Pre-registered hypotheses</h2>
            <p className="hint">
              Signs are declared before correlations are computed. A result contradicting its
              prediction is reported as a contradiction, not re-explained. Fingerprint{" "}
              <code>{registry.fingerprint.slice(0, 16)}…</code> is stamped on every analysis run, so
              an edit after the fact is visible.
            </p>
            <table>
              <thead>
                <tr>
                  <th>Relationship</th>
                  <th>Expected sign</th>
                  <th>Mechanism</th>
                </tr>
              </thead>
              <tbody>
                {registry.hypotheses.map((hypothesis) => (
                  <tr key={`${hypothesis.x_key}-${hypothesis.y_key}`}>
                    <td className="mono">
                      {hypothesis.x_key} &rarr; {hypothesis.y_key}
                      {hypothesis.conditional_on.length > 0 && (
                        <div className="muted" style={{ fontSize: 11 }}>
                          conditional on {hypothesis.conditional_on.join(", ")}
                        </div>
                      )}
                    </td>
                    <td>
                      <span
                        className={`badge ${
                          hypothesis.expected_sign === "+"
                            ? "ok"
                            : hypothesis.expected_sign === "-"
                              ? "bad"
                              : "muted"
                        }`}
                      >
                        {hypothesis.expected_sign === "test"
                          ? "test empirically"
                          : hypothesis.expected_sign}
                      </span>
                    </td>
                    <td className="muted">{hypothesis.mechanism}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Async>
    </>
  );
}
