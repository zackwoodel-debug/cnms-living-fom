/**
 * Typed client for the CNMS Living FOM API.
 *
 * All requests go through the `/api` prefix, which the Vite dev proxy and the
 * production nginx config both forward to the backend. That keeps this file
 * free of environment branching.
 */

const BASE = "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  if (!response.ok) {
    let detail: unknown;
    try {
      detail = (await response.json())?.detail;
    } catch {
      detail = await response.text();
    }
    throw new ApiError(
      typeof detail === "string" ? detail : `Request failed (${response.status})`,
      response.status,
      detail,
    );
  }
  return response.json() as Promise<T>;
}

const get = <T,>(path: string) => request<T>(path);
const post = <T,>(path: string, body: unknown) =>
  request<T>(path, { method: "POST", body: JSON.stringify(body) });

/* ----------------------------------------------------------------------- */
/* Types                                                                    */
/* ----------------------------------------------------------------------- */

export type ProvenanceTier = "measured" | "calculated" | "modeled" | "unavailable";
export type ScoreStatus = "scored" | "not_scored" | "illustrative";

export interface Material {
  id: number;
  formula: string;
  formula_reduced: string;
  polymorph: string;
  specimen_form: string;
  space_group_symbol: string | null;
  space_group_number: number | null;
  source_database: string | null;
  source_identifier: string | null;
  created_at: string;
}

export interface PropertyValue {
  id: number;
  property_key: string;
  value: number | null;
  units: string | null;
  tensor_component: string | null;
  temperature_k: number | null;
  frequency_hz: number | null;
  method: string | null;
  doi: string | null;
  provenance_tier: ProvenanceTier;
}

export interface DescriptorValue {
  descriptor_key: string;
  value: number | null;
  units: string | null;
  method: string | null;
  provenance_tier: ProvenanceTier;
}

export interface MaterialDetail extends Material {
  descriptors: DescriptorValue[];
  properties: PropertyValue[];
}

export interface ScoreComponent {
  raw: number;
  transformed: number;
  z: number;
  floored: boolean;
  out_of_bounds: boolean;
  weight: number;
  ln_z: number;
  weighted_ln_z: number;
}

export interface ScoreResult {
  material_key: string;
  material_id: number | null;
  status: ScoreStatus;
  value: number | null;
  log_value: number | null;
  missing_inputs: string[];
  out_of_bounds: string[];
  floored: string[];
  components: Record<string, ScoreComponent>;
  uses_modeled_inputs: boolean;
  note: string | null;
}

export interface ScoreResponse {
  fom_name: string;
  fom_version: number;
  approved: boolean;
  n_scored: number;
  n_not_scored: number;
  results: ScoreResult[];
  warnings: string[];
}

export interface FomDefinition {
  id: number;
  name: string;
  version: number;
  application: string;
  description: string | null;
  weights: Record<string, number>;
  normalization: Record<string, unknown>;
  approved: boolean;
  approved_by: string | null;
  frozen: boolean;
}

export interface MediationResponse {
  application: string;
  fom_name: string;
  fom_version: number;
  descriptors: string[];
  properties: string[];
  mediated_effect: Record<string, number>;
  mediated_elasticity: Record<string, number>;
  contributions: Record<string, Record<string, number>>;
  dominant_property: Record<string, string>;
  sensitivity_source: string;
  reference_point: Record<string, number>;
  notes: string[];
}

export interface Hypothesis {
  x_key: string;
  y_key: string;
  expected_sign: string;
  mechanism: string;
  conditional_on: string[];
  scope: string;
}

export interface HypothesisRegistry {
  fingerprint: string;
  hypotheses: Hypothesis[];
}

export interface DescriptorSpec {
  key: string;
  symbol: string;
  name: string;
  units: string;
  formula: string;
  interpretation: string;
  default_transform: string;
  missing_policy: string;
  caveat: string;
}

export interface Dictionary {
  structural_descriptors: DescriptorSpec[];
  physical_properties: Array<DescriptorSpec & { direction: string; required_context: string[] }>;
}

export interface BoRun {
  id: number;
  name: string;
  search_space: { parameters: Array<Record<string, unknown>> };
  acquisition: string;
  objective_sense: string;
  status: string;
  n_observations: number;
  n_suggestions: number;
}

export interface Suggestion {
  parameters: Record<string, unknown>;
  acquisition_value: number | null;
  predicted_mean: number | null;
  predicted_std: number | null;
  strategy: string;
  batch_index: number;
}

export interface SuggestResponse {
  run_id: number;
  suggestions: Suggestion[];
  strategy: string;
  n_observations: number;
  objective_name: string;
  notes: string[];
}

export interface RagSource {
  chunk_id: number;
  document_title: string;
  technique: string;
  page: number | null;
  text: string;
  similarity: number;
  doi: string | null;
  citation: string;
}

export interface RagAnswer {
  question: string;
  answer: string;
  sources: RagSource[];
  model: string;
  insufficient_context: boolean;
  disclaimer: string;
}

/* ----------------------------------------------------------------------- */
/* Endpoints                                                                */
/* ----------------------------------------------------------------------- */


/* ----------------------------------------------------------------------- */
/* Suite workflow (docs/specs/suite-workflow/SPEC.md)                       */
/* ----------------------------------------------------------------------- */

export type BadgeKind =
  | "reference_literature"
  | "user_assumption"
  | "calculated"
  | "synthetic_prediction"
  | "experimental_fit"
  | "unknown";

export interface SuiteMaterial {
  release?: string;
  material_id: number;
  name: string;
  formula: string | null;
  datasets_by_provenance: Record<string, number>;
  datasets_by_family: Record<string, number>;
}

export interface SuiteDataset {
  dataset_id: number;
  family: string;
  label: string;
  provenance: string;
  basis: string;
  review: { status: string; conflict: boolean; by: string[] };
  source: { doi: string | null; title: string | null; authors: string | null; year: number | null } | null;
  raw_file: string | null;
  conditions: Record<string, { state: string; value: string | number | null; unit: string | null }>;
  optical: { wavelength_nm: [number, number]; rows: number; k_missing_rows: number } | null;
  open_findings: { finding_id: number; severity: string | null; code: string | null; message: string | null }[];
  cnms: { label: string; tier: string | null; destination: string; reviewed: boolean; reasons: string[]; kind: BadgeKind };
}

export interface SuiteEvidence {
  material: { material_id: number; name: string; formula: string | null };
  datasets: SuiteDataset[];
}

export interface SuiteReadiness {
  status: "ready" | "not_ready" | "ambiguous";
  reasons: { code: string; message: string; dataset_id: number | null }[];
  notes: string[];
  datasets: number[];
  excluded: { dataset_id: number; label: string; code: string; message: string }[];
  unchecked: string[];
}

export interface OpticalHit {
  release?: string;
  material_id: number;
  material: string;
  dataset_id: number;
  label: string;
  provenance: string;
  review: string;
  doi: string | null;
  n: number | null;
  k: number | null;
  interpolated: boolean;
  between_nm: [number, number] | null;
  dataset_span_nm: [number, number];
  cnms: { label: string; kind: BadgeKind; reviewed: boolean };
}

export interface OpticalSearchResult {
  wavelength_nm: number;
  results: OpticalHit[];
  truncated: boolean;
  datasets_spanning_wavelength: number;
  skipped_without_k: number;
  note: string;
}

export interface SuiteLayer {
  release?: string;
  material: string;
  dataset_label?: string;
  thickness?: number;
  roughness?: number;
}

export interface BundleRequest {
  release?: string;
  layers: SuiteLayer[];
  substrate: string;
  techniques: string[];
  window?: { wavelength_nm: [number, number] } | null;
  mode: "strict";
  assumptions: string[];
  length_unit: "angstrom";
}

export interface FitImported {
  fit_record_id: number;
  sample_id: string | null;
  techniques: string[];
  chi2_total: number | null;
  n_free_parameters: number | null;
  skipped: boolean;
  warnings: string[];
}

export interface FitDiagnostics {
  fit_record_id: number;
  blocks_promotion: boolean;
  findings: { code: string; severity: "blocks_promotion" | "caution" | "info"; message: string }[];
}

export interface PromotionResult {
  fit_record_id: number;
  material_id: number;
  written: number;
  operator?: string;
  eligible?: { registry_key: string; parameter: string; value: number; units: string; layer_index: number }[];
  refused: { registry_key?: string; parameter?: string; reason?: string; layer_index?: number }[];
  created?: unknown[];
  provenance_tier?: string;
}

export const api = {
  health: () => get<{ status: string; version: string }>("/health"),
  ready: () => get<{ status: string; checks: Record<string, any> }>("/health/ready"),

  listMaterials: (params: { formula?: string; specimen_form?: string } = {}) => {
    const query = new URLSearchParams(
      Object.entries(params).filter(([, v]) => v) as [string, string][],
    );
    return get<Material[]>(`/materials${query.toString() ? `?${query}` : ""}`);
  },
  getMaterial: (id: number) => get<MaterialDetail>(`/materials/${id}`),
  dictionary: () => get<Dictionary>("/materials/dictionary"),

  listFomDefinitions: () => get<FomDefinition[]>("/fom/definitions"),
  seedDraftFoms: () => post<FomDefinition[]>("/fom/definitions/seed-drafts", {}),
  hypotheses: () => get<HypothesisRegistry>("/fom/hypotheses"),
  score: (body: { fom_name: string; persist?: boolean }) =>
    post<ScoreResponse>("/fom/score", body),
  mediate: (body: {
    fom_name: string;
    reference_properties?: Record<string, number>;
    reference_descriptors?: Record<string, number>;
  }) => post<MediationResponse>("/fom/mediate", body),

  suiteInfo: () =>
    get<{
      version: string;
      materials: number;
      datasets: number;
      reviews: number;
      datasets_by_provenance: Record<string, number>;
      default_release?: string;
      releases?: Record<string, { version: string; materials: number; datasets: number; reviews: number }>;
    }>("/suite/info"),
  suiteMaterials: (q: string, release?: string) =>
    get<SuiteMaterial[]>(`/suite/materials?q=${encodeURIComponent(q)}&limit=50${release ? `&release=${encodeURIComponent(release)}` : ""}`),
  opticalSearch: (p: Record<string, string | number | boolean | undefined>) => {
    const query = new URLSearchParams(Object.entries(p).filter(([, v]) => v !== undefined && v !== "").map(([k, v]) => [k, String(v)]));
    return get<OpticalSearchResult>(`/suite/optical/search?${query}`);
  },
  suiteEvidence: (id: number, release?: string) =>
    get<SuiteEvidence>(`/suite/materials/${id}/evidence${release ? `?release=${encodeURIComponent(release)}` : ""}`),
  suiteReadiness: (body: {
    release?: string;
    selection: { material_id?: number; dataset_id?: number };
    technique: string;
    window?: { wavelength_nm: [number, number] } | null;
    conditions?: Record<string, unknown> | null;
  }) => post<SuiteReadiness>("/suite/readiness", body),
  /** A strict ModalFit bundle. A refusal arrives as an ApiError whose message is the exporter's reason. */
  suiteBundle: async (body: BundleRequest): Promise<{ blob: Blob; summary: Record<string, unknown> }> => {
    const response = await fetch(`${BASE}/suite/bundle`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const detail = (await response.json().catch(() => ({})))?.detail;
      throw new ApiError(typeof detail === "string" ? detail : `Request failed (${response.status})`, response.status, detail);
    }
    const summary = JSON.parse(response.headers.get("x-bundle-summary") ?? "{}");
    return { blob: await response.blob(), summary };
  },
  importFit: (body: { path: string; operator?: string }) =>
    post<{ imported: FitImported[]; warnings: string[] }>("/modalfit/import", body),
  fitDiagnostics: (id: number) => get<FitDiagnostics>(`/suite/fits/${id}/diagnostics`),
  promote: (
    id: number,
    body: { material_id: number; layer_label?: string; temperature_k?: number; dry_run: boolean; operator?: string },
  ) => post<PromotionResult>(`/modalfit/fits/${id}/promote`, body),

  listBoRuns: () => get<BoRun[]>("/bo/runs"),
  suggest: (runId: number, q = 1) =>
    post<SuggestResponse>(`/bo/run/${runId}/suggest`, { q, persist: true }),
  instruments: () => get<Array<Record<string, any>>>("/bo/instruments"),

  ragQuery: (body: { question: string; k?: number }) => post<RagAnswer>("/rag/query", body),
  corpus: () =>
    get<{ total_documents: number; total_chunks: number; documents_by_technique: Record<string, number> }>(
      "/rag/corpus",
    ),
};
