/** Recovery records belong to one server database and one anonymous identity. */
let namespace = "";
const listeners = new Set<() => void>();
const ACTIVE_KEY = "interview_active_session";
const PENDING_KEY = "interview_pending_answer_v1";

export function browserSessionKey(key: string): string {
  return namespace ? `${key}::${namespace}` : key;
}

export function browserIdentity(): string {
  return namespace;
}

export function onBrowserIdentityChanged(listener: () => void): () => void {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

export function selectBrowserIdentity(serverId: string, identityId: string, migrateLegacy: boolean): void {
  const next = `${serverId}:${identityId}`;
  const previous = namespace;
  // Copy before selecting the new namespace. Failed storage never discards originals.
  if (migrateLegacy) {
    const active = localStorage.getItem(ACTIVE_KEY);
    if (active && !localStorage.getItem(`${ACTIVE_KEY}::${next}`)) {
      localStorage.setItem(`${ACTIVE_KEY}::${next}`, active);
      const draft = localStorage.getItem(`interview_draft_${active}`);
      if (draft !== null) localStorage.setItem(`interview_draft_${active}::${next}`, draft);
    }
    const pending = sessionStorage.getItem(PENDING_KEY);
    if (pending && !sessionStorage.getItem(`${PENDING_KEY}::${next}`)) {
      sessionStorage.setItem(`${PENDING_KEY}::${next}`, pending);
    }
    // Completed migration must not resurrect a deleted session on a later reload.
    localStorage.removeItem(ACTIVE_KEY);
    sessionStorage.removeItem(PENDING_KEY);
    if (active) localStorage.removeItem(`interview_draft_${active}`);
  }
  namespace = next;
  if (previous && previous !== next) listeners.forEach((listener) => listener());
}
