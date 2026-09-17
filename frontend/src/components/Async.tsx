import type { ReactNode } from "react";

import type { AsyncState } from "../lib/useAsync";

interface Props<T> {
  state: AsyncState<T>;
  children: (data: T) => ReactNode;
  empty?: ReactNode;
  isEmpty?: (data: T) => boolean;
}

/**
 * Render an async result, keeping loading / error / empty visually distinct.
 *
 * "Failed to load" and "nothing here yet" look identical if you collapse them,
 * and on this platform they warrant completely different reactions.
 */
export function Async<T>({ state, children, empty, isEmpty }: Props<T>) {
  if (state.loading && state.data === null) {
    return <div className="panel muted">Loading…</div>;
  }
  if (state.error) {
    return (
      <div className="notice bad">
        <strong>Request failed.</strong> {state.error}{" "}
        <button style={{ marginLeft: 8 }} onClick={state.reload}>
          Retry
        </button>
      </div>
    );
  }
  if (state.data === null) return null;
  if (isEmpty?.(state.data)) {
    return <div className="panel muted">{empty ?? "Nothing here yet."}</div>;
  }
  return <>{children(state.data)}</>;
}
