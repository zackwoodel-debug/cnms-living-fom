import { useState } from "react";

import { Async } from "../components/Async";
import { DraftBadge, StatusBadge } from "../components/Badges";
import { api, type ScoreResponse } from "../lib/api";
import { useAsync } from "../lib/useAsync";

export default function FomPage() {
  const definitions = useAsync(() => api.listFomDefinitions(), []);
  const [application, setApplication] = useState("logic");
  const [scores, setScores] = useState<ScoreResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function runScore() {
    setBusy(true);
    setError(null);
    try {
      setScores(await api.score({ fom_name: application }));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="panel">
        <h2>Figures of merit</h2>
        <p className="hint">
          F<sub>a</sub> = &prod;<sub>q</sub> z<sub>q</sub><sup>w</sup>, with weights summing to 1.
          The weights are an application-policy choice, not a measurement, so a definition stays
          <em> draft</em> until someone approves it by name. Bounds, transform, direction, floor,
          and weights are all part of the score&rsquo;s identity: changing one creates a new version.
        </p>
        <div className="row">
          <select value={application} onChange={(event) => setApplication(event.target.value)}>
            <option value="logic">logic</option>
            <option value="power">power</option>
            <option value="rf">rf</option>
          </select>
          <button onClick={runScore} disabled={busy}>
            {busy ? "Scoring…" : "Score all materials"}
          </button>
        </div>
      </div>

      {error && <div className="notice bad">{error}</div>}

      {scores && (
        <div className="panel">
          <h2>
            {scores.fom_name} v{scores.fom_version} <DraftBadge approved={scores.approved} />
          </h2>
          <div className="grid" style={{ margin: "12px 0 18px" }}>
            <div>
              <div className="stat">{scores.n_scored}</div>
              <div className="muted">scored</div>
            </div>
            <div>
              <div className="stat">{scores.n_not_scored}</div>
              <div className="muted">not scored (incomplete inputs)</div>
            </div>
          </div>

          {scores.warnings.map((warning) => (
            <div className="notice" key={warning}>
              {warning}
            </div>
          ))}

          <table>
            <thead>
              <tr>
                <th>Material</th>
                <th>Status</th>
                <th className="num">F</th>
                <th className="num">ln F</th>
                <th>Missing inputs</th>
                <th>Flags</th>
              </tr>
            </thead>
            <tbody>
              {[...scores.results]
                .sort((a, b) => (b.value ?? -Infinity) - (a.value ?? -Infinity))
                .map((result) => (
                  <tr key={result.material_key}>
                    <td className="mono">{result.material_key}</td>
                    <td>
                      <StatusBadge status={result.status} />
                    </td>
                    <td className="num">{result.value === null ? "—" : result.value.toFixed(4)}</td>
                    <td className="num">
                      {result.log_value === null ? "—" : result.log_value.toFixed(3)}
                    </td>
                    <td className="muted mono">
                      {result.missing_inputs.length ? result.missing_inputs.join(", ") : "—"}
                    </td>
                    <td className="muted">
                      {result.out_of_bounds.length > 0 && (
                        <span className="badge warn">out of bounds</span>
                      )}{" "}
                      {result.floored.length > 0 && <span className="badge muted">floored</span>}{" "}
                      {result.uses_modeled_inputs && <span className="badge warn">modeled</span>}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}

      <Async
        state={definitions}
        isEmpty={(rows) => rows.length === 0}
        empty="No stored definitions. The built-in drafts still work; POST /fom/definitions/seed-drafts to persist them."
      >
        {(rows) => (
          <div className="panel">
            <h2>Stored definitions</h2>
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th className="num">Version</th>
                  <th>Weights</th>
                  <th>Status</th>
                  <th>Approved by</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((definition) => (
                  <tr key={definition.id}>
                    <td>{definition.name}</td>
                    <td className="num">{definition.version}</td>
                    <td className="mono muted">
                      {Object.entries(definition.weights)
                        .map(([key, weight]) => `${key}=${Number(weight).toFixed(2)}`)
                        .join("  ")}
                    </td>
                    <td>
                      <DraftBadge approved={definition.approved} />{" "}
                      {definition.frozen && <span className="badge muted">frozen</span>}
                    </td>
                    <td className="muted">{definition.approved_by ?? "—"}</td>
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
