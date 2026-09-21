type OllamaMessage = {
  role: "user" | "assistant";
  content: string;
};

type OllamaChunk = {
  message?: { content?: string };
  error?: string;
};

const OLLAMA_URL = "http://localhost:11434/api/chat";
const OLLAMA_MODEL = "llama3.1:8b";

export async function streamOllamaChat({
  messages,
  signal,
  onStart,
  onToken,
}: {
  messages: OllamaMessage[];
  signal: AbortSignal;
  onStart: () => void;
  onToken: (token: string) => void;
}) {
  let response: Response;

  try {
    response = await fetch(OLLAMA_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        stream: true,
        messages: [
          {
            role: "system",
            content:
              "You are Quark, an evidence-first materials research assistant for CNMS Living FOM. Be precise, distinguish evidence from inference, flag data gaps, and never invent measurements or citations.",
          },
          ...messages,
        ],
      }),
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new Error(
      "Ollama is unavailable. Start Ollama, confirm llama3.1:8b is installed, and allow this app’s origin in OLLAMA_ORIGINS.",
    );
  }

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      detail || `Ollama returned ${response.status}. Check the local runtime.`,
    );
  }

  if (!response.body) {
    throw new Error("Ollama returned an empty response stream.");
  }

  onStart();
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.trim()) continue;
      const chunk = JSON.parse(line) as OllamaChunk;
      if (chunk.error) throw new Error(chunk.error);
      if (chunk.message?.content) onToken(chunk.message.content);
    }

    if (done) break;
  }

  if (buffer.trim()) {
    const chunk = JSON.parse(buffer) as OllamaChunk;
    if (chunk.error) throw new Error(chunk.error);
    if (chunk.message?.content) onToken(chunk.message.content);
  }
}