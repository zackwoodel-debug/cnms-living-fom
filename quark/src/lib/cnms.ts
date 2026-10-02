// Quark's one connection: the CNMS backend's research assistant (POST /rag/chat).
//
// The backend runs the local model (Ollama, llama3.1:8b) inside a loop that only answers
// from tools — the corpus, the platform's records and the materials-db reference atlas —
// and replaces an answer with no evidence behind it by an explicit data gap. Talking to
// Ollama directly from the browser skips every one of those checks, so Quark does not.

export const CNMS_API_URL: string =
  (import.meta.env["VITE_CNMS_API_URL"] as string | undefined) ?? "http://localhost:8000";

export type EvidenceCitation = {
  kind: string;
  citation: string;
  material?: string;
  material_id?: number;
  value?: number | null;
  unit?: string | null;
  grade?: string | null;
  value_kind?: string | null;
  citable?: boolean;
  sources?: string[];
  origin?: string;
  title?: string;
  page?: number | null;
  via_tool?: string;
};

export type AssistantTurn = {
  answer: string;
  sessionKey: string;
  toolsUsed: string[];
  citations: EvidenceCitation[];
  insufficientContext: boolean;
  model: string;
  provider: string;
  latencyMs: number;
};

type ChatResponse = {
  session_key: string;
  answer: string;
  tools_used?: string[];
  citations?: EvidenceCitation[];
  insufficient_context?: boolean;
  model?: string;
  provider?: string;
  latency_ms?: number;
};

export async function askAssistant({
  question,
  sessionKey,
  signal,
}: {
  question: string;
  sessionKey?: string | undefined;
  signal: AbortSignal;
}): Promise<AssistantTurn> {
  let response: Response;
  try {
    response = await fetch(`${CNMS_API_URL}/rag/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_key: sessionKey }),
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new Error(
      `The CNMS backend is unavailable at ${CNMS_API_URL}. Start it with \`cnms-fom serve\` (Ollama must be running too). Quark answers only through the backend, where every answer has to rest on retrieved evidence.`,
    );
  }

  if (!response.ok) {
    const detail = await response.text();
    let message = detail;
    try {
      const parsed = JSON.parse(detail) as { detail?: unknown };
      if (typeof parsed.detail === "string") message = parsed.detail;
    } catch {
      // not JSON: show the text as it came
    }
    throw new Error(message || `The CNMS backend returned ${response.status}.`);
  }

  const body = (await response.json()) as ChatResponse;
  return {
    answer: body.answer,
    sessionKey: body.session_key,
    toolsUsed: body.tools_used ?? [],
    citations: body.citations ?? [],
    insufficientContext: Boolean(body.insufficient_context),
    model: body.model ?? "",
    provider: body.provider ?? "",
    latencyMs: body.latency_ms ?? 0,
  };
}

function formatValue(citation: EvidenceCitation): string {
  if (citation.value === null || citation.value === undefined) return "no stored value";
  const unit = citation.unit && citation.unit !== "dimensionless" ? ` ${citation.unit}` : "";
  return `${citation.value}${unit}`;
}

// The evidence behind an answer, as Markdown appended under it: which tools ran, and every
// atlas value or document passage the answer could rest on, with its grade and citability.
export function evidenceMarkdown(turn: AssistantTurn): string {
  const lines: string[] = ["", "---", ""];
  const tools = turn.toolsUsed.length ? turn.toolsUsed.join(", ") : "none";
  const seconds = (turn.latencyMs / 1000).toFixed(1);
  lines.push(`**Evidence** · tools: ${tools} · ${turn.model || "model"} · ${seconds} s`);
  if (turn.insufficientContext) {
    lines.push("", "_Data gap: the retrieved evidence does not answer this question._");
  }
  const atlas = turn.citations.filter((c) => c.kind === "atlas_value");
  const passages = turn.citations.filter((c) => c.kind === "document");
  if (atlas.length) {
    lines.push("");
    for (const c of atlas.slice(0, 12)) {
      const [, quantity = "", stored = ""] = c.citation.split(" :: ");
      const label = stored.startsWith(quantity)
        ? stored
        : [quantity, stored].filter(Boolean).join(" · ");
      const grade = c.grade ? ` · grade ${c.grade}` : "";
      const kind = c.value_kind ? ` · ${c.value_kind}` : "";
      const cite = c.citable ? "citable" : "not citable";
      lines.push(
        `- ${c.material ?? "material"} (id ${c.material_id ?? "?"}): ${label} = ${formatValue(c)}${grade}${kind} · ${cite} · ${c.origin ?? "reference"}`,
      );
    }
  }
  if (passages.length) {
    lines.push("");
    for (const c of passages.slice(0, 8)) {
      lines.push(`- ${c.citation}${c.page ? `, p. ${c.page}` : ""}`);
    }
  }
  return lines.join("\n");
}
