import { useState } from "react";

import { KindBadge } from "../components/Badges";
import {
  ApiError,
  api,
  type FitDiagnostics,
  type OpticalSearchResult,
  type FitImported,
  type PromotionResult,
  type SuiteEvidence,
  type SuiteLayer,
  type SuiteMaterial,
  type SuiteReadiness,
} from "../lib/api";
import { useAsync } from "../lib/useAsync";

const TECHNIQUES = ["SE", "SPR", "XRR", "NR", "QCM"] as const;
const OPTICAL = ["SE", "SPR"];

const message = (e: unknown) => (e instanceof ApiError || e instanceof Error ? e.message : String(e));

/** Run an async action and keep its result, its error and a busy flag, so a refusal is shown rather than swallowed. */
function useAction<A extends unknown[], R>(fn: (...a: A) => Promise<R>) {
  const [state, setState] = useState<{ busy: boolean; error: string | null; data: R | null }>({ busy: false, error: null, data: null });
  const run = async (...args: A) => {
    setState({ busy: true, error: null, data: null });
    try {
      const data = await fn(...args);
      setState({ busy: false, error: null, data });
      return data;
    } catch (e) {
      setState({ busy: false, error: message(e), data: null });
      return null;
    }
  };
  return { ...state, run };
}

function Refusal({ text }: { text: string | null }) {
  return text ? (
    <div className="notice bad">
      <strong>Refused.</strong> {text}
    </div>
  ) : null;
}

/**
 * The first integrated workflow: reference data -> conditions -> stack -> readiness -> strict bundle for ModalFit ->
 * import the fit -> diagnostics -> promotion preview -> confirmed promotion by a named operator.
 * CNMS owns the stack, the fit record and the promotion; materials-db owns the reference data; ModalFit does the simulation and refinement.
 */
export default function WorkflowPage() {
  const info = useAsync(() => api.suiteInfo(), []);
  const [query, setQuery] = useState("");
  const [releaseFilter, setReleaseFilter] = useState("");
  const search = useAction((q: string) => api.suiteMaterials(q, releaseFilter || undefined));
  const evidence = useAction((id: number, release?: string) => api.suiteEvidence(id, release));
  const [selected, setSelected] = useState<SuiteMaterial | null>(null);

  const [sWl, setSWl] = useState("633");
  const [nMin, setNMin] = useState("");
  const [nMax, setNMax] = useState("");
  const [kMin, setKMin] = useState("");
  const [kMax, setKMax] = useState("");
  const [sApproved, setSApproved] = useState(false);
  const optSearch = useAction(() =>
    api.opticalSearch({ release: releaseFilter || undefined, wavelength_nm: sWl, n_min: nMin, n_max: nMax, k_min: kMin, k_max: kMax, q: query || undefined, approved_only: sApproved || undefined, limit: 100 }),
  );
  const optical: OpticalSearchResult | null = optSearch.data;

  const [technique, setTechnique] = useState<string>("SE");
  const [lo, setLo] = useState("400");
  const [hi, setHi] = useState("800");
  const [temperature, setTemperature] = useState("");
  const [pickedDataset, setPickedDataset] = useState<{ dataset_id: number; label: string } | null>(null);
  const [densityLabel, setDensityLabel] = useState<string | null>(null);
  const readiness = useAction((mid: number, ds: number | null, release?: string) =>
    api.suiteReadiness({
      release,
      selection: ds !== null ? { dataset_id: ds } : { material_id: mid },
      technique,
      window: OPTICAL.includes(technique) ? { wavelength_nm: [Number(lo), Number(hi)] } : null,
      conditions: temperature.trim() ? { temperature_c: Number(temperature) } : null,
    }),
  );

  const [layers, setLayers] = useState<SuiteLayer[]>([]);
  const [substrate, setSubstrate] = useState("Silicon");
  const [thickness, setThickness] = useState("");
  const [roughness, setRoughness] = useState("");
  const bundleRelease = layers.length ? layers[0].release : undefined;
  const mixedReleases = layers.some((l) => l.release !== bundleRelease);
  const bundle = useAction(() =>
    api.suiteBundle({
      release: bundleRelease,
      layers: layers.map(({ release: _release, ...rest }) => rest),
      substrate,
      techniques: [technique],
      window: OPTICAL.includes(technique) ? { wavelength_nm: [Number(lo), Number(hi)] } : null,
      mode: "strict",
      assumptions: [],
      length_unit: "angstrom",
    }),
  );

  const [fitPath, setFitPath] = useState("");
  const [operator, setOperator] = useState("");
  const importFit = useAction((path: string) => api.importFit({ path, operator: operator.trim() || undefined }));
  const diagnostics = useAction((id: number) => api.fitDiagnostics(id));
  const cnmsMaterials = useAsync(() => api.listMaterials(), []);
  const [cnmsMaterialId, setCnmsMaterialId] = useState<number | null>(null);
  const [layerLabel, setLayerLabel] = useState("");
  const [temperatureK, setTemperatureK] = useState("");
  const preview = useAction((fit: number, mid: number) =>
    api.promote(fit, { material_id: mid, layer_label: layerLabel || undefined, temperature_k: temperatureK ? Number(temperatureK) : undefined, dry_run: true }),
  );
  const confirm = useAction((fit: number, mid: number) =>
    api.promote(fit, { material_id: mid, layer_label: layerLabel || undefined, temperature_k: temperatureK ? Number(temperatureK) : undefined, dry_run: false, operator: operator.trim() }),
  );

  const fit: FitImported | undefined = importFit.data?.imported.find((f) => !f.skipped) ?? importFit.data?.imported[0];
  const ev: SuiteEvidence | null = evidence.data;
  const ready: SuiteReadiness | null = readiness.data;
  const diag: FitDiagnostics | null = diagnostics.data;
  const plan: PromotionResult | null = preview.data;
  const readyForConfirm = !!plan && plan.written === 0 && (plan.eligible?.length ?? 0) > 0 && operator.trim().length > 0 && !diag?.blocks_promotion;

  const download = (blob: Blob) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "stack-bundle.zip";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <div className="panel">
        <h2>Integrated workflow</h2>
        <p className="hint">
          Reference data come from the materials-db service; CNMS owns the stack, the fit record and the promotion; ModalFit does the simulation and refinement.
          Every number below says what kind of number it is, and a review never turns a literature value into a measurement.
        </p>
        {info.data && (
          <p className="hint">
            {info.data.releases
              ? Object.entries(info.data.releases).map(([name, r]) => `${name} (materials-db ${r.version}): ${r.materials.toLocaleString()} materials, ${r.datasets.toLocaleString()} datasets, ${r.reviews} reviews`).join(" · ")
              : `materials-db ${info.data.version}: ${info.data.materials} materials, ${info.data.datasets} datasets, ${info.data.reviews} reviews.`}
          </p>
        )}
        {info.error && <div className="notice bad">{info.error}</div>}
      </div>

      <div className="panel">
        <h2>1. Browse materials</h2>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            search.run(query);
          }}
        >
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="name or formula" />{" "}
          {info.data?.releases && Object.keys(info.data.releases).length > 1 && (
            <select value={releaseFilter} onChange={(e) => setReleaseFilter(e.target.value)}>
              <option value="">all databases</option>
              {Object.keys(info.data.releases).map((r) => <option key={r} value={r}>{r}</option>)}
            </select>
          )}{" "}
          <button disabled={search.busy}>Search</button>
        </form>
        <Refusal text={search.error} />
        {search.data && (
          <table>
            <thead>
              <tr><th>Database</th><th>Material</th><th>Formula</th><th>Datasets (by provenance)</th><th /></tr>
            </thead>
            <tbody>
              {search.data.map((m) => (
                <tr key={`${m.release ?? ""}:${m.material_id}`}>
                  <td>{m.release ?? "—"}</td>
                  <td>{m.name}</td>
                  <td>{m.formula ?? "—"}</td>
                  <td>{Object.entries(m.datasets_by_provenance).map(([k, v]) => `${k} ${v}`).join(", ")}</td>
                  <td>
                    <button
                      onClick={() => {
                        setSelected(m);
                        setPickedDataset(null);
                        setDensityLabel(null);
                        readiness.run(m.material_id, null, m.release);
                        evidence.run(m.material_id, m.release);
                      }}
                    >
                      Inspect
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="panel">
        <h2>1b. Search tabulated n and k at a wavelength</h2>
        <p className="hint">
          Optical constants depend on wavelength, so search where you will use them. Values between table points are interpolated and marked; nothing is extrapolated outside a
          dataset's range. A k limit excludes datasets that report no k (they are counted, never treated as zero). The name box above narrows the search.
        </p>
        <form onSubmit={(e) => { e.preventDefault(); optSearch.run(); }}>
          λ <input style={{ width: 70 }} value={sWl} onChange={(e) => setSWl(e.target.value)} /> nm · n from{" "}
          <input style={{ width: 55 }} value={nMin} onChange={(e) => setNMin(e.target.value)} /> to <input style={{ width: 55 }} value={nMax} onChange={(e) => setNMax(e.target.value)} /> · k from{" "}
          <input style={{ width: 55 }} value={kMin} onChange={(e) => setKMin(e.target.value)} /> to <input style={{ width: 55 }} value={kMax} onChange={(e) => setKMax(e.target.value)} />{" "}
          <label><input type="checkbox" checked={sApproved} onChange={(e) => setSApproved(e.target.checked)} /> approved by a person only</label>{" "}
          <button disabled={optSearch.busy}>Search</button>
        </form>
        <Refusal text={optSearch.error} />
        {optical && (
          <>
            <p className="hint">
              {optical.results.length}{optical.truncated ? "+" : ""} match(es) of {optical.datasets_spanning_wavelength} datasets that span {optical.wavelength_nm} nm
              {optical.skipped_without_k ? `; ${optical.skipped_without_k} skipped because they report no k` : ""}.
            </p>
            <table>
              <thead><tr><th>Kind</th><th>Database</th><th>Material</th><th>Dataset</th><th>n</th><th>k</th><th>At λ</th><th>Source</th></tr></thead>
              <tbody>
                {optical.results.map((h) => (
                  <tr key={`${h.release ?? ""}:${h.dataset_id}`}>
                    <td><KindBadge kind={h.cnms.kind} reviewStatus={h.review} /></td>
                    <td>{h.release ?? "—"}</td><td>{h.material}</td><td>{h.label || "(unlabelled)"}</td>
                    <td>{h.n?.toPrecision(4) ?? "—"}</td><td>{h.k?.toPrecision(3) ?? "not reported"}</td>
                    <td>{h.interpolated ? `interpolated between ${h.between_nm?.[0]} and ${h.between_nm?.[1]} nm` : "tabulated point"}</td>
                    <td>{h.doi ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>

      {selected && (
        <div className="panel">
          <h2>2. Evidence and conditions: {selected.name}{selected.release ? ` (${selected.release})` : ""}</h2>
          <Refusal text={evidence.error} />
          {ev?.structure && (
            <p className="hint">
              Calculated structure: a={ev.structure.a?.toFixed(3)} b={ev.structure.b?.toFixed(3)} c={ev.structure.c?.toFixed(3)} Å, α/β/γ={ev.structure.alpha?.toFixed(1)}/{ev.structure.beta?.toFixed(1)}/{ev.structure.gamma?.toFixed(1)}°,
              space group {ev.structure.spacegroup_number}, {ev.structure.nsites} sites.
              {ev.calculated_properties && ` ${ev.calculated_properties.functional}: band gap ${ev.calculated_properties.band_gap_ind_ev} eV (indirect), energy above hull ${ev.calculated_properties.e_above_hull_ev} eV/atom.`}
            </p>
          )}
          {ev && (
            <table>
              <thead>
                <tr><th>Kind</th><th>Family</th><th>Dataset</th><th>Source</th><th>Range</th><th>Conditions stated</th><th /></tr>
              </thead>
              <tbody>
                {ev.datasets.map((d) => (
                  <tr key={d.dataset_id}>
                    <td><KindBadge kind={d.cnms.kind} reviewStatus={d.review.status} /></td>
                    <td>{d.family}</td>
                    <td>
                      {d.label || "(unlabelled)"}
                      <div className="hint">{d.cnms.label}; basis: {d.basis}</div>
                      {d.cnms.reasons.map((r) => <div key={r} className="badge bad">{r}</div>)}
                      {d.open_findings.map((f) => <div key={f.finding_id} className="badge warn">{f.message ?? f.code}</div>)}
                    </td>
                    <td>{d.source?.doi ?? d.source?.title ?? "—"}</td>
                    <td>
                      {d.optical
                        ? `${d.optical.wavelength_nm[0]}–${d.optical.wavelength_nm[1]} nm, ${d.optical.rows} rows${d.optical.k_missing_rows ? `, k missing in ${d.optical.k_missing_rows}` : ""}`
                        : d.values
                          ? [d.values.density_g_cm3 != null ? `density ${d.values.density_g_cm3.toPrecision(4)} g/cm³` : null, d.values.xray_sld != null ? `x-ray SLD ${d.values.xray_sld.toPrecision(4)}` : null, d.values.neutron_sld != null ? `neutron SLD ${d.values.neutron_sld.toPrecision(4)}` : null].filter(Boolean).join(", ")
                          : "—"}
                    </td>
                    <td>{Object.entries(d.conditions).filter(([, c]) => c.state === "stated").map(([k, c]) => `${k}=${c.value}${c.unit ? ` ${c.unit}` : ""}`).join(", ") || "none stated"}</td>
                    <td>{d.family === "optical" && <button onClick={() => { setPickedDataset(d); readiness.run(selected.material_id, d.dataset_id, selected.release); }}>Use this dataset</button>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p>
            Technique{" "}
            <select value={technique} onChange={(e) => setTechnique(e.target.value)}>
              {TECHNIQUES.map((t) => <option key={t}>{t}</option>)}
            </select>{" "}
            {OPTICAL.includes(technique) && (
              <>
                window <input style={{ width: 70 }} value={lo} onChange={(e) => setLo(e.target.value)} /> to{" "}
                <input style={{ width: 70 }} value={hi} onChange={(e) => setHi(e.target.value)} /> nm{" "}
              </>
            )}
            temperature (°C, optional) <input style={{ width: 70 }} value={temperature} onChange={(e) => setTemperature(e.target.value)} />{" "}
            <button disabled={readiness.busy} onClick={() => readiness.run(selected.material_id, pickedDataset ? pickedDataset.dataset_id : null, selected.release)}>Check readiness</button>
          </p>
          {pickedDataset && <p className="hint">Checking dataset #{pickedDataset.dataset_id} only. <button onClick={() => setPickedDataset(null)}>Use the whole material</button></p>}
          <Refusal text={readiness.error} />
          {ready && (
            <div className={`notice ${ready.status === "ready" ? "info" : "bad"}`}>
              <strong>{technique}: {ready.status.replace("_", " ")}.</strong>
              <ul>
                {ready.reasons.map((r, i) => <li key={i}><code>{r.code}</code> {r.message}</li>)}
                {ready.excluded.map((x, i) => <li key={`x${i}`}>excluded {x.label}: {x.message}</li>)}
                {ready.notes.map((n, i) => <li key={`n${i}`}>{n}</li>)}
              </ul>
              {ready.unchecked.length > 0 && <div className="hint">Not checked (you did not state them): {ready.unchecked.join(", ")}</div>}
            </div>
          )}
          {ready && ready.status === "ambiguous" && ready.candidates.length > 0 && (
            <div className="notice bad">
              <strong>Which one do you want?</strong> {ready.candidates.length} datasets could serve {technique}. Nothing is chosen for you: pick the phase or dataset that matches your sample.
              <table>
                <thead><tr><th>Kind</th><th>Dataset</th><th>Value / range</th><th>Source</th><th /></tr></thead>
                <tbody>
                  {ready.candidates.map((c) => (
                    <tr key={c.dataset_id}>
                      <td>{c.family} · {c.provenance} · {c.review}</td>
                      <td>{c.label || "(unlabelled)"}</td>
                      <td>
                        {c.values ? [c.values.density_g_cm3 != null ? `density ${c.values.density_g_cm3.toPrecision(4)} g/cm³` : null, c.values.xray_sld != null ? `x-ray SLD ${c.values.xray_sld.toPrecision(4)}` : null, c.values.neutron_sld != null ? `neutron SLD ${c.values.neutron_sld.toPrecision(4)}` : null].filter(Boolean).join(", ") : ""}
                        {c.optical ? `${c.optical.wavelength_nm[0]}–${c.optical.wavelength_nm[1]} nm, ${c.optical.rows} rows` : ""}
                      </td>
                      <td>{c.source?.doi ?? c.source?.title ?? "—"}{c.source?.notes ? <div className="hint">{c.source.notes}</div> : null}</td>
                      <td>
                        <button onClick={() => (c.family === "physical" ? setDensityLabel(c.label) : setPickedDataset({ dataset_id: c.dataset_id, label: c.label }))}>
                          Use for my layer
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {densityLabel && <div className="hint">Density dataset chosen: {densityLabel}</div>}
            </div>
          )}
        </div>
      )}

      <div className="panel">
        <h2>3. Build the stack</h2>
        <p className="hint">Top layer first; the substrate is last. Thickness and roughness are your entries (angstrom) and are kept apart from reference data.</p>
        <p>
          {selected ? <>Add <strong>{selected.name}</strong>{pickedDataset ? ` (dataset ${pickedDataset.label})` : ""}: </> : "Pick a material above. "}
          thickness <input style={{ width: 70 }} value={thickness} onChange={(e) => setThickness(e.target.value)} /> Å, roughness{" "}
          <input style={{ width: 70 }} value={roughness} onChange={(e) => setRoughness(e.target.value)} /> Å{" "}
          <button
            disabled={!selected}
            onClick={() =>
              selected &&
              setLayers([...layers, { release: selected.release, material: selected.name, dataset_label: pickedDataset?.label || undefined, density_dataset_label: densityLabel || undefined, thickness: thickness ? Number(thickness) : undefined, roughness: roughness ? Number(roughness) : undefined }])
            }
          >
            Add layer
          </button>
        </p>
        <table>
          <thead><tr><th>#</th><th>Database</th><th>Material</th><th>Dataset</th><th>Density dataset</th><th>Thickness Å</th><th>Roughness Å</th><th>Entered by you</th><th /></tr></thead>
          <tbody>
            {layers.map((l, i) => (
              <tr key={i}>
                <td>{i + 1}</td><td>{l.release ?? "—"}</td><td>{l.material}</td><td>{l.dataset_label ?? "(must be unambiguous)"}</td><td>{l.density_dataset_label ?? "(must be unambiguous)"}</td>
                <td>{l.thickness ?? "—"}</td><td>{l.roughness ?? "—"}</td>
                <td><KindBadge kind="user_assumption" /></td>
                <td><button onClick={() => setLayers(layers.filter((_, j) => j !== i))}>Remove</button></td>
              </tr>
            ))}
            <tr><td>sub</td><td colSpan={7}>substrate <input value={substrate} onChange={(e) => setSubstrate(e.target.value)} />{bundleRelease ? ` (looked up in ${bundleRelease})` : ""}</td><td /></tr>
          </tbody>
        </table>
        <p>
          {mixedReleases && <span className="badge bad">A bundle is built from one database. Remove the layers that come from another.</span>}{" "}
          <button disabled={!layers.length || bundle.busy || mixedReleases} onClick={async () => { const r = await bundle.run(); if (r) download(r.blob); }}>
            Build strict ModalFit bundle ({technique})
          </button>
        </p>
        <Refusal text={bundle.error} />
        {bundle.data && (
          <div className="notice info">
            Bundle built in <strong>{String(bundle.data.summary.mode)}</strong> mode. Assumptions used: {JSON.stringify(bundle.data.summary.assumptions_used ?? [])}. Open it in ModalFit
            to simulate (step 5) and refine against your data (step 6), then export the fit file.
          </div>
        )}
      </div>

      <div className="panel">
        <h2>4. Import the fit from ModalFit and judge it</h2>
        <p className="hint">The path must be visible to the API (under docker-compose, mount it).</p>
        <p>
          Fit file <input style={{ width: 360 }} value={fitPath} onChange={(e) => setFitPath(e.target.value)} placeholder="/path/to/modalfit_export.json" /> operator{" "}
          <input value={operator} onChange={(e) => setOperator(e.target.value)} placeholder="your name" />{" "}
          <button
            disabled={!fitPath.trim() || importFit.busy}
            onClick={async () => { const r = await importFit.run(fitPath.trim()); const f = r?.imported.find((x) => !x.skipped) ?? r?.imported[0]; if (f) diagnostics.run(f.fit_record_id); }}
          >
            Import
          </button>
        </p>
        <Refusal text={importFit.error} />
        {fit && (
          <div className="notice info">
            Fit #{fit.fit_record_id}: {fit.techniques.join("+")}, χ² {fit.chi2_total ?? "not recorded"}, {fit.n_free_parameters ?? "?"} free parameters{fit.skipped ? " (already imported)" : ""}.
            <KindBadge kind="experimental_fit" />
            {[...fit.warnings, ...(importFit.data?.warnings ?? [])].map((w, i) => <div key={i} className="badge warn">{w}</div>)}
          </div>
        )}
        {diag && (
          <>
            <h3>Fit diagnostics and limitations</h3>
            <ul>
              {diag.findings.map((f, i) => (
                <li key={i}><span className={`badge ${f.severity === "blocks_promotion" ? "bad" : f.severity === "caution" ? "warn" : "muted"}`}>{f.severity.replace("_", " ")}</span> {f.message}</li>
              ))}
              {diag.findings.length === 0 && <li>No limitations were detected from the record. That is not a statement that the fit is right.</li>}
            </ul>
          </>
        )}
      </div>

      {fit && (
        <div className="panel">
          <h2>5. Preview and confirm promotion</h2>
          <p className="hint">
            Promotion writes fitted values into CNMS as MEASURED. It is a preview first; a real write needs the material (never inferred) and your name, and is refused without them.
          </p>
          <p>
            CNMS material{" "}
            <select value={cnmsMaterialId ?? ""} onChange={(e) => setCnmsMaterialId(e.target.value ? Number(e.target.value) : null)}>
              <option value="">choose…</option>
              {cnmsMaterials.data?.map((m) => <option key={m.id} value={m.id}>{m.formula} {m.polymorph} ({m.specimen_form})</option>)}
            </select>{" "}
            layer label <input style={{ width: 120 }} value={layerLabel} onChange={(e) => setLayerLabel(e.target.value)} placeholder="if several films" /> temperature (K){" "}
            <input style={{ width: 70 }} value={temperatureK} onChange={(e) => setTemperatureK(e.target.value)} />{" "}
            <button disabled={cnmsMaterialId === null || preview.busy} onClick={() => cnmsMaterialId !== null && preview.run(fit.fit_record_id, cnmsMaterialId)}>Preview (writes nothing)</button>
          </p>
          <Refusal text={preview.error} />
          {plan && (
            <div className="notice info">
              <strong>Would write {plan.eligible?.length ?? 0} value(s):</strong>
              <ul>{plan.eligible?.map((x, i) => <li key={i}>{x.registry_key} = {x.value} {x.units} (layer {x.layer_index})</li>)}</ul>
              {plan.refused.length > 0 && (
                <>
                  <strong>Refused ({plan.refused.length}):</strong>
                  <ul>{plan.refused.map((x, i) => <li key={i}>{x.registry_key ?? x.parameter}: {x.reason}</li>)}</ul>
                </>
              )}
            </div>
          )}
          <p>
            <button disabled={!readyForConfirm || confirm.busy} onClick={() => cnmsMaterialId !== null && confirm.run(fit.fit_record_id, cnmsMaterialId)}>
              Confirm promotion as {operator.trim() || "(enter your name above)"}
            </button>
            {diag?.blocks_promotion && <span className="badge bad"> blocked by diagnostics</span>}
          </p>
          <Refusal text={confirm.error} />
          {confirm.data && (
            <div className="notice info">
              Promoted {confirm.data.written} value(s) by {confirm.data.operator}; tier {confirm.data.provenance_tier}. Scoring happens elsewhere and only when its required inputs are present.
            </div>
          )}
        </div>
      )}
    </>
  );
}
