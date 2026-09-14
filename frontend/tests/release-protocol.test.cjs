/* Offline contracts: transpile the real TS sources; mocked transport/hook runtime.
 * Not a DOM/browser end-to-end test. No extra dependency, service or network call.
 */
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const ts = require("typescript");

function storage() {
  const items = new Map();
  return {
    getItem: (key) => items.get(key) ?? null,
    setItem: (key, value) => items.set(key, String(value)),
    removeItem: (key) => items.delete(key),
  };
}

function load(relative, mocks, globals) {
  const filename = path.join(__dirname, "../src", relative);
  const output = ts.transpileModule(fs.readFileSync(filename, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const module = { exports: {} };
  vm.runInNewContext(output, {
    module, exports: module.exports,
    require: (name) => {
      if (name in mocks) return mocks[name];
      if (name === "./browserSession" || name === "../api/browserSession") {
        return { browserSessionKey: (key) => key, browserIdentity: () => "synthetic-identity",
          selectBrowserIdentity: () => {}, onBrowserIdentityChanged: () => () => {} };
      }
      if (!(name in mocks)) throw new Error("Unexpected import: " + name);
      return mocks[name];
    },
    Error, TypeError, Promise, Headers, FormData, ...globals,
  }, { filename });
  return module.exports;
}

const response = (status, payload) => ({ ok: status >= 200 && status < 300, status, json: async () => payload });
const copy = (value) => JSON.parse(JSON.stringify(value));
const flush = async () => { for (let i = 0; i < 50; i++) await Promise.resolve(); };
const pendingKey = "interview_pending_answer_v1";
const credentialKey = "interview_anonymous_bearer_v1";
test("real ApiError provides an actionable OCR setup and text alternative", () => {
  const { ApiError: ActualError } = load("api/client.ts", {}, {});
  const error = new ActualError(503, { code: "ocr_language_missing", missing_languages: ["chi_sim"] });
  assert.equal(error.code, "ocr_language_missing");
  assert.match(error.message, /运行指南.*OCR 初始化/);
  assert.match(error.message, /粘贴文字/);
  assert.doesNotMatch(error.message, /管理员|自动退回英文/);
});
test("real ApiError distinguishes scanned PDF dependency failure", () => {
  const { ApiError: ActualError } = load("api/client.ts", {}, {});
  assert.match(new ActualError(503, { code: "ocr_pdf_renderer_missing" }).message, /PDFium.*Poppler/);
});
test("real ApiError distinguishes OCR timeout from invalid document", () => {
  const { ApiError: ActualError } = load("api/client.ts", {}, {});
  assert.match(new ActualError(504, { code: "ocr_timeout" }).message, /超时/);
  assert.match(new ActualError(422, { code: "ocr_failed" }).message, /识别失败/);
});
const requestId = "00000000-0000-4000-8000-000000000001";
const payload = (version = 1, extra = {}) => ({
  session_id: "synthetic-session", name: "Synthetic", status: "interviewing",
  question: "Synthetic question " + version, question_version: version, done: false,
  messages: [{ role: "assistant", content: "Synthetic question " + version }],
  report: null, usage_total: version, ...extra,
});
const savedPending = () => ({ sessionId: "synthetic-session", body: {
  answer: "Synthetic original answer", answer_request_id: requestId, expected_question_version: 1,
} });

class ApiError extends Error {
  constructor(status, detail) { super(typeof detail === "string" ? detail : detail?.status || "error"); this.status = status; this.detail = detail; }
  get code() { return typeof this.detail === "string" ? this.detail : this.detail?.code || this.detail?.status || ""; }
}

function hook(overrides = {}, options = {}) {
  const localStorage = options.localStorage || storage();
  const sessionStorage = options.sessionStorage || storage();
  localStorage.setItem("interview_active_session", "synthetic-session");
  if (options.pending) sessionStorage.setItem(pendingKey, JSON.stringify(options.pending));
  const slots = []; let cursor = 0; let effects = [];
  const react = {
    useState(initial) {
      const index = cursor++;
      if (!slots[index]) {
        slots[index] = { value: typeof initial === "function" ? initial() : initial };
        slots[index].setter = (value) => { slots[index].value = typeof value === "function" ? value(slots[index].value) : value; };
      }
      return [slots[index].value, slots[index].setter];
    },
    useRef(value) { const index = cursor++; if (!slots[index]) slots[index] = { current: value }; return slots[index]; },
    useCallback(callback, deps) {
      const index = cursor++;
      if (!slots[index] || deps.some((value, i) => !Object.is(value, slots[index].deps[i]))) slots[index] = { callback, deps };
      return slots[index].callback;
    },
    useEffect(callback, deps) {
      const index = cursor++;
      if (!slots[index] || deps.some((value, i) => !Object.is(value, slots[index].deps[i]))) {
        slots[index] = { deps }; effects.push(callback);
      }
    },
  };
  const api = {
    listSessions: async () => [], getSession: async () => payload(),
    answer: async () => payload(2),
    answerStatus: async () => { throw new ApiError(404, "answer_request_not_found"); },
    stop: async () => payload(2, { done: true, report: {} }),
    updateJobTitle: async () => payload(2), deleteSession: async () => ({ deleted: true }),
    ...overrides,
  };
  let ids = 0;
  const { useInterview } = load("hooks/useInterview.ts", { react, "../api/client": {
    api, ApiError, anonymousCredential: options.anonymousCredential || (async () => "synthetic-credential"),
  }, ...(options.browserSession ? { "../api/browserSession": options.browserSession } : {}) }, {
    localStorage, sessionStorage, crypto: { randomUUID: () => { ids++; return requestId; } },
    setTimeout: (callback) => { queueMicrotask(callback); return 0; },
  });
  const render = () => { cursor = 0; return useInterview(); };
  return {
    render, api, localStorage, sessionStorage, ids: () => ids,
    async mount() { render(); const pendingEffects = effects; effects = []; pendingEffects.forEach((effect) => effect()); await flush(); return render(); },
  };
}

test("client bootstraps once and sends only server bearer credentials", async () => {
  const localStorage = storage(); const calls = [];
  const { api } = load("api/client.ts", {}, {
    localStorage, navigator: {}, setTimeout,
    fetch: async (url, init) => {
      calls.push({ url, init });
      if (url === "/api/auth/context") return response(200, { server_id: "1".repeat(32) });
      return url === "/api/auth/anonymous" ? response(201, {
        token: "a".repeat(43), token_type: "Bearer", server_id: "1".repeat(32), identity_id: "2".repeat(32), reused: false,
      }) : response(200, []);
    },
  });
  await Promise.all([api.listSessions(), api.listSessions()]);
  assert.equal(calls.filter((call) => call.url === "/api/auth/anonymous").length, 1);
  for (const call of calls.filter((call) => call.url === "/api/sessions")) {
    assert.ok(call.init.headers.has("Authorization"));
    assert.equal(call.init.headers.has("X-Owner-Id"), false);
    assert.equal(call.init.cache, "no-store");
  }
});

test("bootstrap storage failure preserves the existing credential and HTTP error", async () => {
  const localStorage = storage(); localStorage.setItem(credentialKey, "a".repeat(43)); const calls = [];
  const { api, ApiError } = load("api/client.ts", {}, {
    localStorage, navigator: {}, setTimeout,
    fetch: async (url) => { calls.push(url); return response(503, { detail: "storage_unavailable" }); },
  });
  await assert.rejects(api.listSessions(), (error) => error instanceof ApiError && error.status === 503);
  assert.equal(calls.length, 1);
  assert.equal(calls.includes("/api/auth/anonymous"), false);
  assert.ok(localStorage.getItem(credentialKey));
});

test("answer POST carries all three fields and is not blindly retried", async () => {
  const localStorage = storage(); localStorage.setItem(credentialKey, "a".repeat(43)); const calls = [];
  const { api } = load("api/client.ts", {}, {
    localStorage, navigator: {}, setTimeout,
    fetch: async (url, init) => {
      if (url === "/api/auth/context") return response(200, { server_id: "1".repeat(32) });
      if (url === "/api/auth/anonymous") return response(200, {
        token: "a".repeat(43), token_type: "Bearer", server_id: "1".repeat(32), identity_id: "2".repeat(32), reused: true,
      });
      calls.push(JSON.parse(init.body)); throw new TypeError("synthetic network failure");
    },
  });
  await assert.rejects(api.answer("synthetic-session", savedPending().body));
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0], savedPending().body);
});

test("real ApiError retains string codes while rendering friendly Chinese messages", () => {
  const { ApiError } = load("api/client.ts", {}, { localStorage: storage(), navigator: {}, setTimeout });
  for (const code of ["stale_question_version", "session_busy", "recovery_required", "idempotency_conflict"]) {
    const error = new ApiError(409, code);
    assert.equal(error.code, code);
    assert.ok(/[\u4e00-\u9fff]/.test(error.message));
    assert.notEqual(error.message, code);
  }
  assert.equal(new ApiError(409, { status: "recovery_required", processing: false }).code, "recovery_required");
});

test("synchronous double-click produces one logical request and one optimistic message", async () => {
  const calls = []; let resolve;
  const h = hook({ answer: (sid, body) => { calls.push(copy(body)); return new Promise((done) => { resolve = done; }); } });
  const view = await h.mount();
  const first = view.answer("Synthetic answer"); view.answer("Synthetic answer");
  assert.equal(calls.length, 1); assert.equal(h.ids(), 1);
  assert.equal(h.render().state.messages.filter((message) => message.role === "user").length, 1);
  resolve(payload(2)); await first;
  assert.equal(h.render().state.questionVersion, 2);
  assert.equal(h.sessionStorage.getItem(pendingKey), null);
});

test("network failure preserves payload; retry after 404 resends the same ID, version and answer", async () => {
  const calls = [];
  const h = hook({ answer: async (sid, body) => {
    calls.push(copy(body)); if (calls.length === 1) throw new Error("synthetic network failure"); return payload(2);
  } });
  let view = await h.mount(); await view.answer("Synthetic original answer");
  assert.ok(h.sessionStorage.getItem(pendingKey));
  view = h.render(); await view.answer("This changed text must not become a new request");
  assert.equal(h.ids(), 1); assert.equal(calls.length, 2); assert.deepEqual(calls[0], calls[1]);
  assert.equal(h.sessionStorage.getItem(pendingKey), null);
});

test("202 polls GET only and cannot overwrite the displayed question with pending payload", async () => {
  let posts = 0; let polls = 0;
  const h = hook({
    answer: async () => { posts++; return { processing: true, session_id: "synthetic-session", answer_request_id: requestId }; },
    answerStatus: async () => { polls++; return payload(2); },
  });
  const view = await h.mount(); await view.answer("Synthetic answer");
  assert.equal(posts, 1); assert.equal(polls, 1); assert.equal(h.render().state.questionVersion, 2);
});

test("stale version 409 refreshes current session and clears the obsolete pending request", async () => {
  let reads = 0;
  const h = hook({ getSession: async () => payload(++reads === 1 ? 1 : 2), answer: async () => { throw new ApiError(409, "stale_question_version"); } });
  const view = await h.mount(); await view.answer("Synthetic stale answer");
  assert.equal(h.sessionStorage.getItem(pendingKey), null); assert.equal(h.render().state.questionVersion, 2);
});

test("busy 409 keeps pending and blocks stop/job-title/delete", async () => {
  let stops = 0; let titles = 0; let deletes = 0;
  const h = hook({ answer: async () => { throw new ApiError(409, "session_busy"); },
    stop: async () => { stops++; }, updateJobTitle: async () => { titles++; }, deleteSession: async () => { deletes++; },
  });
  let view = await h.mount(); await view.answer("Synthetic answer"); view = h.render();
  await view.stop(); await view.updateJobTitle("Synthetic role"); await view.deleteSession("synthetic-session");
  assert.equal(stops + titles + deletes, 0); assert.ok(h.sessionStorage.getItem(pendingKey));
});

test("deleting active session removes draft and clears resume/JD without UI notices", async () => {
  const h = hook();
  let view = await h.mount();
  h.localStorage.setItem("interview_draft_synthetic-session", "Synthetic private draft");
  view.patch({ resumeText: "Synthetic private resume", resumeName: "synthetic.txt", jdText: "Synthetic JD", jdOcr: "Synthetic OCR" });
  view = h.render();
  await view.deleteSession("synthetic-session");
  const state = h.render().state;
  assert.equal(h.localStorage.getItem("interview_draft_synthetic-session"), null);
  assert.equal(h.localStorage.getItem("interview_active_session"), null);
  for (const field of ["resumeText", "resumeName", "jdText", "jdOcr", "activeSessionId", "question"]) assert.equal(state[field], "");
  assert.equal(state.error, "");
});

test("deleting historical session removes its draft but preserves active materials", async () => {
  const h = hook();
  let view = await h.mount();
  h.localStorage.setItem("interview_draft_synthetic-history", "Synthetic old draft");
  view.patch({ resumeText: "Synthetic current resume" });
  view = h.render();
  await view.deleteSession("synthetic-history");
  assert.equal(h.localStorage.getItem("interview_draft_synthetic-history"), null);
  assert.equal(h.render().state.resumeText, "Synthetic current resume");
  assert.equal(h.render().state.activeSessionId, "synthetic-session");
});

test("a completed replay fetches the latest projection instead of regressing to cached question", async () => {
  let posts = 0;
  const h = hook({ getSession: async () => payload(3, { done: true, report: {} }),
    answerStatus: async () => payload(2, { replayed: true }), answer: async () => { posts++; return payload(2); },
  }, { pending: savedPending() });
  const view = await h.mount(); await view.retryPendingAnswer();
  assert.equal(posts, 0); assert.equal(h.render().state.questionVersion, 3); assert.ok(h.render().state.report);
  assert.equal(h.sessionStorage.getItem(pendingKey), null);
});

test("recovery_required 409 does not clear pending or display it as a completed response", async () => {
  const h = hook({ answerStatus: async () => { throw new ApiError(409, { status: "recovery_required", processing: false }); } }, { pending: savedPending() });
  const view = await h.mount(); await view.retryPendingAnswer();
  assert.ok(h.sessionStorage.getItem(pendingKey)); assert.equal(h.render().state.questionVersion, 1);
  assert.equal(h.render().state.recoveryRequired, true);
});

test("reload keeps the frozen request and restores it with GET rather than creating another ID", async () => {
  let posts = 0;
  const sharedSession = storage(); sharedSession.setItem(pendingKey, JSON.stringify(savedPending()));
  const h = hook({ answerStatus: async () => payload(2, { replayed: true }), getSession: async () => payload(2),
    answer: async () => { posts++; return payload(2); },
  }, { sessionStorage: sharedSession });
  const view = await h.mount(); await view.retryPendingAnswer();
  assert.equal(posts, 0); assert.equal(h.ids(), 0); assert.equal(sharedSession.getItem(pendingKey), null);
});

test("idempotency-conflict does not discard the original pending request", async () => {
  const h = hook({ answer: async () => { throw new ApiError(409, "idempotency_conflict"); } });
  const view = await h.mount(); await view.answer("Synthetic answer");
  assert.ok(h.sessionStorage.getItem(pendingKey)); assert.equal(h.render().state.pendingAnswer, true);
});

test("out-of-order GET cannot move the active question version backwards", async () => {
  let reads = 0;
  const h = hook({ getSession: async () => payload(++reads === 1 ? 3 : 2),
    answer: async () => { throw new ApiError(409, "stale_question_version"); },
  });
  const view = await h.mount(); await view.answer("Synthetic answer");
  assert.equal(h.render().state.questionVersion, 3);
});

test("browser storage failure prevents POST rather than losing recovery identity", async () => {
  let posts = 0;
  const saved = storage(); saved.setItem = () => { throw new Error("synthetic storage denial"); };
  const h = hook({ answer: async () => { posts++; return payload(2); } }, { sessionStorage: saved });
  const view = await h.mount(); await view.answer("Synthetic answer");
  assert.equal(posts, 0); assert.ok(h.render().state.error);
});

test("confirmed recovery escape preserves the server session and prevents old-session resubmission", async () => {
  let posts = 0; let deletes = 0; let stops = 0;
  const h = hook({ answerStatus: async () => { throw new ApiError(409, "recovery_required"); },
    answer: async () => { posts++; return payload(2); }, deleteSession: async () => { deletes++; }, stop: async () => { stops++; },
  }, { pending: savedPending() });
  let view = await h.mount(); await view.retryPendingAnswer();
  view = h.render(); view.preserveFailedSessionAndReset();
  await view.answer("Must not answer the failed old session"); await view.stop();
  assert.equal(posts + deletes + stops, 0); assert.equal(h.ids(), 0);
  assert.equal(h.sessionStorage.getItem(pendingKey), null);
  assert.equal(h.localStorage.getItem("interview_active_session"), null);
  assert.equal(h.render().state.activeSessionId, "");
});

test("ordinary network uncertainty cannot abandon pending through the recovery-only escape", async () => {
  const h = hook({ answer: async () => { throw new Error("synthetic network uncertainty"); } });
  let view = await h.mount(); await view.answer("Synthetic answer");
  view = h.render(); view.preserveFailedSessionAndReset();
  assert.ok(h.sessionStorage.getItem(pendingKey)); assert.equal(h.render().state.activeSessionId, "synthetic-session");
});

test("JD image upload fills the editable JD field", async () => {
  const h = hook({ uploadJdImage: async () => ({ text: "Synthetic OCR JD", method: "ocr" }) });
  const view = await h.mount();
  await view.uploadJdImage({ name: "synthetic.png" });
  assert.equal(h.render().state.jdText, "Synthetic OCR JD");
  assert.equal(h.render().state.jdOcr, "Synthetic OCR JD");
});

test("starting an interview sends the edited JD once, not hidden stale OCR", async () => {
  let submitted;
  const h = hook({ uploadJdImage: async () => ({ text: "Synthetic old OCR", method: "ocr" }),
    startSession: async (body) => { submitted = body; return payload(); } });
  let view = await h.mount();
  await view.uploadJdImage({ name: "synthetic.png" });
  view = h.render(); view.patch({ jdText: "Synthetic corrected JD", resumeText: "Synthetic resume" });
  await h.render().start();
  assert.equal(submitted.jd_text, "Synthetic corrected JD");
  assert.doesNotMatch(submitted.jd_text, /old OCR/);
});

test("job-title correction after an answer preserves the question plan", async () => {
  let regenerate;
  const h = hook({ getSession: async () => payload(2, { messages: [
    { role: "assistant", content: "Synthetic question" }, { role: "user", content: "Synthetic answer" } ] }),
    updateJobTitle: async (id, title, value) => { regenerate = value; return payload(3); } });
  const view = await h.mount();
  await view.updateJobTitle("Synthetic corrected role");
  assert.equal(regenerate, false);
});

test("job-title correction before answering may regenerate the plan", async () => {
  let regenerate;
  const h = hook({ updateJobTitle: async (id, title, value) => { regenerate = value; return payload(2); } });
  const view = await h.mount();
  await view.updateJobTitle("Synthetic corrected role");
  assert.equal(regenerate, true);
});

test("active status follows the backend instead of inventing interviewing", async () => {
  const h = hook({ getSession: async () => payload(1, { status: "failed", question: "" }) });
  await h.mount();
  assert.equal(h.render().state.status, "failed");
});

test("mount waits for identity preparation before reading recovery data", async () => {
  let ready; let reads = 0;
  const prepared = new Promise((resolve) => { ready = resolve; });
  const h = hook({ getSession: async () => { reads++; return payload(); } }, { anonymousCredential: () => prepared });
  await h.mount();
  assert.equal(reads, 0);
  assert.equal(h.render().state.activeSessionId, "");
  ready("synthetic-credential"); await flush();
  assert.equal(reads, 1);
  assert.equal(h.render().state.activeSessionId, "synthetic-session");
});

test("live environment change preserves materials and isolates old pending state", async () => {
  const localStorage = storage(); const sessionStorage = storage();
  const browserSession = load("api/browserSession.ts", {}, { localStorage, sessionStorage });
  browserSession.selectBrowserIdentity("1".repeat(32), "2".repeat(32), false);
  localStorage.setItem(browserSession.browserSessionKey("interview_active_session"), "synthetic-session");
  sessionStorage.setItem(browserSession.browserSessionKey(pendingKey), JSON.stringify(savedPending()));
  const oldKey = browserSession.browserSessionKey(pendingKey);
  const h = hook({}, { localStorage, sessionStorage, browserSession });
  const view = await h.mount();
  view.patch({ resumeText: "Synthetic edited resume", resumeName: "synthetic.txt", jdText: "Synthetic edited JD" });
  browserSession.selectBrowserIdentity("3".repeat(32), "4".repeat(32), false); await flush();
  const state = h.render().state;
  assert.equal(state.resumeText, "Synthetic edited resume");
  assert.equal(state.jdText, "Synthetic edited JD");
  assert.equal(state.activeSessionId, "");
  assert.equal(state.pendingAnswer, false);
  assert.equal(state.error, "");
  assert.ok(sessionStorage.getItem(oldKey));
});

test("late answer from the old environment cannot clear new pending or display its question", async () => {
  const localStorage = storage(); const sessionStorage = storage();
  const browserSession = load("api/browserSession.ts", {}, { localStorage, sessionStorage });
  browserSession.selectBrowserIdentity("1".repeat(32), "2".repeat(32), false);
  localStorage.setItem(browserSession.browserSessionKey("interview_active_session"), "synthetic-session");
  let complete;
  const h = hook({ answer: () => new Promise((resolve) => { complete = resolve; }) },
    { localStorage, sessionStorage, browserSession });
  const view = await h.mount();
  const submitted = view.answer("Synthetic original answer");
  const oldKey = browserSession.browserSessionKey(pendingKey);
  browserSession.selectBrowserIdentity("3".repeat(32), "4".repeat(32), false); await flush();
  const newKey = browserSession.browserSessionKey(pendingKey);
  sessionStorage.setItem(newKey, "Synthetic new-environment pending sentinel");
  complete(payload(2)); await submitted;
  assert.ok(sessionStorage.getItem(oldKey));
  assert.equal(sessionStorage.getItem(newKey), "Synthetic new-environment pending sentinel");
  assert.equal(h.render().state.questionVersion, 0);
  assert.equal(h.render().state.activeSessionId, "");
});
