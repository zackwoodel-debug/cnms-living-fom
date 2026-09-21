import { cn } from "@/lib/utils";

export function QuarkMark({ className }: { className?: string }) {
  return (
    <span
      aria-hidden="true"
      className={cn("relative block size-9 shrink-0", className)}
    >
      <span className="absolute left-1/2 top-1/2 h-px w-8 -translate-x-1/2 -translate-y-1/2 rotate-45 bg-current" />
      <span className="absolute left-1/2 top-1/2 h-px w-8 -translate-x-1/2 -translate-y-1/2 -rotate-45 bg-current" />
      <span className="absolute left-1/2 top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-current bg-background" />
      <span className="absolute left-0 top-0 size-2.5 rounded-full border border-current bg-background" />
      <span className="absolute bottom-0 right-0 size-2.5 rounded-full border border-current bg-background" />
    </span>
  );
}