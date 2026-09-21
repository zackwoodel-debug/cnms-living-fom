import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
} from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
} from "@/components/ai-elements/prompt-input";
import { Button } from "@/components/ui/button";
import { QuarkMark } from "@/components/quark-mark";
import { cn } from "@/lib/utils";
import { streamOllamaChat } from "@/lib/ollama";
import { useNavigate } from "@tanstack/react-router";
import {
  Atom,
  BookOpenText,
  FlaskConical,
  Menu,
  Network,
  PanelLeftClose,
  Plus,
  ShieldCheck,
  X,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

type ChatStatus = "ready" | "submitted" | "streaming" | "error";

type Thread = {
  id: string;
  title: string;
  meta: string;
  messages: ChatMessage[];
};

const initialThreads: Thread[] = [
  {
    id: "hf02-screening",
    title: "HfO₂ logic screening",
    meta: "Pilot workflow",
    messages: [],
  },
  {
    id: "growth-per-cycle",
    title: "Growth-per-cycle evidence",
    meta: "Evidence research",
    messages: [],
  },
  {
    id: "fom-integrity",
    title: "FOM integrity review",
    meta: "Scientific analysis",
    messages: [],
  },
];

const starters = [
  {
    icon: BookOpenText,
    label: "Search evidence",
    prompt: "What does the literature say about growth per cycle for HfO₂?",
  },
  {
    icon: ShieldCheck,
    label: "Audit a FOM",
    prompt: "Check this FOM definition for integrity and unapproved assumptions.",
  },
  {
    icon: Atom,
    label: "Inspect materials",
    prompt: "Compare the available material records and their structural descriptors.",
  },
  {
    icon: FlaskConical,
    label: "Plan a pilot",
    prompt: "Outline the next HfO₂-on-Si pilot iteration and its measurements.",
  },
];

let sessionThreads = initialThreads;

export function QuarkWorkspace({ threadId }: { threadId: string }) {
  const navigate = useNavigate();
  const [threads, setThreads] = useState(sessionThreads);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [status, setStatus] = useState<ChatStatus>("ready");
  const [error, setError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const activeThread =
    threads.find((thread) => thread.id === threadId) ?? threads[0];

  useEffect(() => {
    sessionThreads = threads;
  }, [threads]);

  useEffect(() => {
    textareaRef.current?.focus();
  }, [threadId, status]);

  const selectThread = (id: string) => {
    setSidebarOpen(false);
    navigate({ to: "/chat/$threadId", params: { threadId: id } });
  };

  const createThread = () => {
    const id = `session-${Date.now().toString(36)}`;
    const next: Thread = {
      id,
      title: "New research thread",
      meta: "Session only",
      messages: [],
    };
    setThreads((current) => [...current, next]);
    setSidebarOpen(false);
    navigate({ to: "/chat/$threadId", params: { threadId: id } });
  };

  const updateThreadMessages = (
    id: string,
    update: (messages: ChatMessage[]) => ChatMessage[],
  ) => {
    setThreads((current) =>
      current.map((thread) =>
        thread.id === id
          ? { ...thread, messages: update(thread.messages) }
          : thread,
      ),
    );
  };

  const submitPrompt = async (text: string) => {
    const prompt = text.trim();
    if (
      !prompt ||
      status === "submitted" ||
      status === "streaming" ||
      !activeThread
    )
      return;
    const targetThreadId = activeThread.id;
    const userMessage: ChatMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      content: prompt,
    };
    const assistantMessage: ChatMessage = {
      id: `a-${Date.now()}`,
      role: "assistant",
      content: "",
    };
    const history = [...activeThread.messages, userMessage];
    setThreads((current) =>
      current.map((thread) =>
        thread.id === targetThreadId
          ? {
              ...thread,
              title:
                thread.title === "New research thread"
                  ? prompt.slice(0, 38)
                  : thread.title,
              messages: [...thread.messages, userMessage, assistantMessage],
            }
          : thread,
      ),
    );
    setError(null);
    setStatus("submitted");
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      await streamOllamaChat({
        messages: history.map(({ role, content }) => ({ role, content })),
        signal: controller.signal,
        onStart: () => setStatus("streaming"),
        onToken: (token) => {
          updateThreadMessages(targetThreadId, (messages) =>
            messages.map((message) =>
              message.id === assistantMessage.id
                ? { ...message, content: message.content + token }
                : message,
            ),
          );
        },
      });
      setStatus("ready");
    } catch (streamError) {
      if (controller.signal.aborted) {
        setStatus("ready");
      } else {
        updateThreadMessages(targetThreadId, (messages) =>
          messages.filter((message) => message.id !== assistantMessage.id),
        );
        setError(
          streamError instanceof Error
            ? streamError.message
            : "Quark could not reach the local model.",
        );
        setStatus("error");
      }
    } finally {
      abortControllerRef.current = null;
    }
  };

  const stopGeneration = () => abortControllerRef.current?.abort();

  if (!activeThread) return null;

  return (
    <main className="flex h-dvh overflow-hidden bg-background text-foreground">
      {sidebarOpen && (
        <button
          aria-label="Close conversation list"
          className="fixed inset-0 z-30 bg-foreground/20 md:hidden"
          onClick={() => setSidebarOpen(false)}
          type="button"
        />
      )}

      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex w-[280px] flex-col border-r border-sidebar-border bg-sidebar transition-[transform,width] duration-200 md:relative md:translate-x-0",
          sidebarOpen ? "translate-x-0" : "-translate-x-full",
          sidebarCollapsed && "md:w-0 md:overflow-hidden md:border-r-0",
        )}
      >
        <div className="flex h-20 items-center justify-between border-b border-sidebar-border px-5">
          <div className="flex items-center gap-3">
            <QuarkMark className="text-primary" />
            <div>
              <p className="font-display text-[22px] leading-none">Quark</p>
              <p className="mt-1 text-[10px] font-semibold uppercase text-muted-foreground">CNMS Research Agent</p>
            </div>
          </div>
          <Button
            aria-label="Close conversation list"
            className="md:hidden"
            onClick={() => setSidebarOpen(false)}
            size="icon-sm"
            variant="ghost"
          >
            <X />
          </Button>
        </div>

        <div className="px-4 pt-5">
          <Button className="w-full justify-start shadow-none" onClick={createThread}>
            <Plus /> New research thread
          </Button>
        </div>

        <nav aria-label="Research threads" className="flex-1 overflow-y-auto px-3 py-6">
          <p className="px-2 pb-2 text-[10px] font-bold uppercase text-muted-foreground">Current session</p>
          <div className="space-y-1">
            {threads.map((thread) => (
              <Button
                className={cn(
                  "h-auto w-full justify-start whitespace-normal px-3 py-3 text-left shadow-none",
                  thread.id === activeThread.id
                    ? "bg-sidebar-accent text-sidebar-accent-foreground"
                    : "text-sidebar-foreground hover:bg-sidebar-accent/60",
                )}
                key={thread.id}
                onClick={() => selectThread(thread.id)}
                variant="ghost"
              >
                <span className="min-w-0">
                  <span className="block truncate text-sm font-medium">{thread.title}</span>
                  <span className="mt-1 block text-[11px] font-normal text-muted-foreground">{thread.meta}</span>
                </span>
              </Button>
            ))}
          </div>
        </nav>

        <div className="border-t border-sidebar-border px-5 py-4">
          <div className="flex items-center gap-2 text-xs font-medium">
            <span className="size-2 rounded-full bg-status" />
            Local runtime
            <span className="ml-auto text-muted-foreground">llama3.1:8b</span>
          </div>
          <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground">Threads reset when this page closes.</p>
        </div>
      </aside>

      <section className="relative flex min-w-0 flex-1 flex-col">
        <header className="relative z-20 flex h-20 shrink-0 items-center border-b bg-background/90 px-4 backdrop-blur md:px-7">
          <Button
            aria-label="Open conversation list"
            className={cn("mr-3", !sidebarCollapsed && "md:hidden")}
            onClick={() => {
              if (sidebarCollapsed) setSidebarCollapsed(false);
              else setSidebarOpen(true);
            }}
            size="icon-sm"
            variant="ghost"
          >
            <Menu />
          </Button>
          <div className="min-w-0">
            <h1 className="truncate text-sm font-semibold">{activeThread.title}</h1>
            <p className="mt-1 flex items-center gap-1.5 text-[11px] text-muted-foreground">
              <Network className="size-3" /> CNMS Living FOM · llama3.1:8b
            </p>
          </div>
          <Button
            className="ml-auto hidden md:inline-flex"
            onClick={() => setSidebarCollapsed(true)}
            size="sm"
            variant="ghost"
          >
            <PanelLeftClose /> Hide threads
          </Button>
        </header>

        <div className="relative min-h-0 flex-1">
          <div aria-hidden="true" className="pointer-events-none absolute inset-0 overflow-hidden">
            <div className="absolute left-1/2 top-[43%] w-full max-w-[920px] -translate-x-1/2 -translate-y-1/2 select-none px-8 text-center text-watermark">
              <div className="mx-auto mb-6 flex w-max items-center gap-7 opacity-60">
                <span className="h-px w-20 bg-current" />
                <QuarkMark className="size-16 text-current [&>span]:opacity-80" />
                <span className="h-px w-20 bg-current" />
              </div>
              <p className="font-display text-[clamp(2.2rem,5.5vw,5.6rem)] leading-[0.92]">Oak Ridge National Laboratory</p>
              <p className="mt-5 text-[clamp(.72rem,1.3vw,1rem)] font-bold uppercase leading-relaxed">
                Center for Nanophase Materials Sciences
              </p>
            </div>
          </div>

          <Conversation className="relative z-10 h-full">
            <ConversationContent className="mx-auto min-h-full w-full max-w-3xl justify-end px-4 pb-6 pt-10 md:px-8">
              {activeThread.messages.length === 0 ? (
                <div className="flex min-h-[calc(100dvh-21rem)] flex-col justify-end">
                  <div className="mb-10 max-w-xl">
                    <p className="text-xs font-bold uppercase text-primary">Evidence-first materials research</p>
                    <h2 className="font-display mt-3 text-4xl leading-tight md:text-5xl">What are we investigating?</h2>
                    <p className="mt-4 max-w-lg text-sm leading-6 text-muted-foreground">
                      Ask Quark to trace evidence, inspect a figure of merit, compare material records, or plan the next experiment.
                    </p>
                  </div>
                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                    {starters.map(({ icon: Icon, label, prompt }) => (
                      <Button
                        className="h-auto justify-start border-border/80 bg-background/85 px-4 py-3 text-left shadow-none backdrop-blur hover:border-primary/40 hover:bg-accent"
                        key={label}
                        onClick={() => submitPrompt(prompt)}
                        variant="outline"
                      >
                        <Icon className="size-4 text-primary" />
                        <span className="min-w-0">
                          <span className="block text-xs font-semibold">{label}</span>
                          <span className="mt-1 block truncate text-[11px] font-normal text-muted-foreground">{prompt}</span>
                        </span>
                      </Button>
                    ))}
                  </div>
                </div>
              ) : (
                activeThread.messages.map((message) => (
                  <Message from={message.role} key={message.id}>
                    <MessageContent className="group-[.is-user]:bg-primary group-[.is-user]:text-primary-foreground">
                      <MessageResponse>
                        {message.content || "Connecting to local model…"}
                      </MessageResponse>
                    </MessageContent>
                  </Message>
                ))
              )}
            </ConversationContent>
            <ConversationScrollButton />
          </Conversation>
        </div>

        <div className="relative z-20 shrink-0 bg-background px-4 pb-4 md:px-8 md:pb-6">
          <div className="mx-auto max-w-3xl">
            <PromptInput
              className="rounded-md border-border bg-card shadow-composer"
              onSubmit={({ text }) => submitPrompt(text)}
            >
              <PromptInputTextarea
                autoFocus
                className="min-h-20 px-4 text-sm"
                placeholder="Ask Quark about evidence, materials, FOMs, or experiments…"
                ref={textareaRef}
              />
              <PromptInputFooter className="px-3 pb-3">
                <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
                  <span className="size-1.5 rounded-full bg-status" /> Ollama · localhost:11434
                </div>
                <PromptInputSubmit
                  className="bg-primary text-primary-foreground hover:bg-primary/90"
                  onStop={stopGeneration}
                  status={status}
                />
              </PromptInputFooter>
            </PromptInput>
            {error && (
              <p className="mt-2 text-center text-xs text-destructive" role="alert">
                {error}
              </p>
            )}
            <p className="mt-2 text-center text-[10px] text-muted-foreground">
              Quark can make mistakes. Verify critical measurements and citations.
            </p>
          </div>
        </div>
      </section>
    </main>
  );
}