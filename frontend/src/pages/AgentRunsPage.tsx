import { useState } from "react";

import { Async } from "../components/Async";
import { api, type SuggestResponse } from "../lib/api";
import { useAsync } from "../lib/useAsync";

export default function AgentRunsPage() {
  const runs = useAsync(() => api.listBoRuns(), []);
  const corpus = useAsync(() => api.corpus().catch(() => null), []);
  const [suggestions, setSuggestions] = useState<SuggestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyRun, setBusyRun] = useState<number | null>(null);

  async function askForSuggestions(runId: number) {
    setBusyRun(runId);
    setError(null);
    try {
      setSuggestions(await api.suggest(runId, 3));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyRun(null);
    }
  }

  return (
    <>
      <div className="panel">
        <h2>Agent runs</h2>
        <p className="hint">
          Each campaign optimises ln&nbsp;F of its figure of merit. The log of a geometric score is
          the additive quantity, which is the scale a Gaussian process should model on. Below about
          five usable observations the loop returns a Sobol design instead of GP suggestions &mdash;
          a GP fitted on three points is reporting its prior.
        </p>
      </div>

      {error && <div className="notice bad">{error}</div>}

      {suggestions && (
        <div className="panel">
          <h2>
            Suggestions from run {suggestions.run_id}{" "}
            <span className="badge muted">{suggestions.strategy}</span>
          </h2>
          <p className="hint">
            {suggestions.n_observations} observation(s) · objective {suggestions.objective_name}
          </p>
          {suggestions.notes.map((note) => (
            <div className="notice info" key={note}>
              {note}
            </div>
          ))}
          <table>
            <thead>
              <tr>
                <th className="num">#</th>
                <th>Recipe</th>
                <th className="num">Acquisition</th>
                <th className="num">Predicted mean ± sd</th>
              </tr>
            </thead>
            <tbody>
              {suggestions.suggestions.map((suggestion) => (
                <tr key={suggestion.batch_index}>
                  <td className="num">{suggestion.batch_index + 1}</td>
                  <td className="mono">
                    {Object.entries(suggestion.parameters)
                      .map(([key, value]) =>
                        typeof value === "number" ? `${key}=${value.toPrecision(4)}` : `${key}=${value}`,
                      )
                      .join("  ")}
                  </td>
                  <td className="num">
                    {suggestion.acquisition_value === null
                      ? "—"
                      : suggestion.acquisition_value.toExponential(2)}
                  </td>
                  <td className="num">
                    {suggestion.predicted_mean === null
                      ? "—"
                      : `${suggestion.predicted_mean.toFixed(3)} ± ${suggestion.predicted_std?.toFixed(3) ?? "?"}`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <Async
        state={runs}
        isEmpty={(rows) => rows.length === 0}
        empty="No campaigns yet. POST /bo/run with a search space to start one."
      >
        {(rows) => (
          <div className="panel">
            <h2>Campaigns</h2>
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Acquisition</th>
                  <th>Sense</th>
                  <th className="num">Observations</th>
                  <th className="num">Suggestions</th>
                  <th>Status</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {rows.map((run) => (
                  <tr key={run.id}>
                    <td>{run.name}</td>
                    <td className="mono muted">{run.acquisition}</td>
                    <td className="muted">{run.objective_sense}</td>
                    <td className="num">{run.n_observations}</td>
                    <td className="num">{run.n_suggestions}</td>
                    <td>
                      <span className="badge muted">{run.status}</span>
                    </td>
                    <td>
                      <button onClick={() => askForSuggestions(run.id)} disabled={busyRun === run.id}>
                        {busyRun === run.id ? "Thinking…" : "Suggest next"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Async>

      {corpus.data && (
        <div className="panel">
          <h2>Synthesis corpus</h2>
          <p className="hint">
            What the retrieval assistant can answer from. Anything outside this corpus comes back as
            an explicit data gap rather than a guess.
          </p>
          <div className="grid">
            <div>
              <div className="stat">{corpus.data.total_documents}</div>
              <div className="muted">documents</div>
            </div>
            <div>
              <div className="stat">{corpus.data.total_chunks}</div>
              <div className="muted">indexed passages</div>
            </div>
            {Object.entries(corpus.data.documents_by_technique).map(([technique, count]) => (
              <div key={technique}>
                <div className="stat">{count}</div>
                <div className="muted">{technique.replace(/_/g, " ")}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
