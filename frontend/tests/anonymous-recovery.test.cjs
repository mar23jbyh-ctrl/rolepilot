/* Real client/storage sources, synthetic transport. No server or cloud calls. */
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const ts = require("typescript");

const legacyKey = "interview_anonymous_bearer_v1";
const activeKey = "interview_active_session";
const pendingKey = "interview_pending_answer_v1";
const serverA = "1".repeat(32);
const serverB = "2".repeat(32);
const identityA = "3".repeat(32);
const tokenA = "a".repeat(43);
const vaultKey = (id) => `interview_anonymous_bearer_v2::${id}`;
const response = (status, payload) => ({ ok: status >= 200 && status < 300, status, json: async () => payload });

function storage() {
  const values = new Map();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: (key) => values.delete(key),
  };
}

function harness(options = {}) {
  const localStorage = options.localStorage || storage();
  const sessionStorage = options.sessionStorage || storage();
  const environments = options.environments || new Map([
    [serverA, new Map([[tokenA, identityA]])], [serverB, new Map()],
  ]);
  const calls = []; let server = serverA; let issued = 0; let intercept = null;
  const fetch = async (url, init = {}) => {
    const call = { url, init, server };
    calls.push(call);
    if (intercept) {
      const result = await intercept(call);
      if (result) return result;
    }
    if (url === "/api/auth/context") return response(200, { server_id: server });
    const incoming = new Headers(init.headers).get("Authorization")?.replace(/^Bearer /, "");
    const identities = environments.get(server);
    if (url === "/api/auth/anonymous") {
      const reused = identities.has(incoming);
      const token = reused ? incoming : String.fromCharCode(98 + issued++).repeat(43);
      if (!reused) identities.set(token, (10 + issued).toString(16).padStart(32, "0"));
      return response(reused ? 200 : 201, {
        token, token_type: "Bearer", server_id: server, identity_id: identities.get(token), reused,
      });
    }
    if (!identities.has(incoming)) return response(401, { detail: "Anonymous bearer credential required" });
    if (url.startsWith("/api/uploads/")) return response(200, { text: "Synthetic resume", method: "text", filename: "synthetic.txt" });
    if (url === "/api/sessions") return response(200, []);
    return response(200, { session_id: "synthetic-session", question_version: 2 });
  };
  const load = () => {
    const modules = new Map();
    const evaluate = (relative) => {
      if (modules.has(relative)) return modules.get(relative);
      const filename = path.join(__dirname, "../src", relative);
      const output = ts.transpileModule(fs.readFileSync(filename, "utf8"), {
        compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
      }).outputText;
      const module = { exports: {} }; modules.set(relative, module.exports);
      vm.runInNewContext(output, {
        module, exports: module.exports, localStorage, sessionStorage, navigator: {},
        Headers, FormData, Error, TypeError, Promise, fetch,
        setTimeout: (callback) => { queueMicrotask(callback); return 0; },
        require: (name) => {
          if (name === "./browserSession") return evaluate("api/browserSession.ts");
          throw new Error("Unexpected runtime import: " + name);
        },
      }, { filename });
      return module.exports;
    };
    return { ...evaluate("api/client.ts"), storage: evaluate("api/browserSession.ts") };
  };
  return {
    load, localStorage, sessionStorage, environments, calls, issued: () => issued,
    switchTo: (id) => { server = id; }, intercept: (handler) => { intercept = handler; },
  };
}

function saveLegacy(h, token = tokenA) {
  h.localStorage.setItem(legacyKey, token);
  h.localStorage.setItem(activeKey, "synthetic-session");
  h.localStorage.setItem("interview_draft_synthetic-session", "Synthetic draft");
  h.sessionStorage.setItem(pendingKey, JSON.stringify({ sessionId: "synthetic-session", body: { answer: "Synthetic answer" } }));
}

test("stale browser token does not block a real multipart resume request", async () => {
  const h = harness(); saveLegacy(h, "z".repeat(43));
  const client = h.load();
  const file = new Blob(["Synthetic resume"], { type: "text/plain" });
  const result = await client.api.uploadResume(file);
  assert.equal(result.text, "Synthetic resume");
  const uploads = h.calls.filter((c) => c.url === "/api/uploads/resume");
  assert.equal(uploads.length, 1);
  assert.ok(uploads[0].init.body instanceof FormData);
  assert.equal(uploads[0].init.headers.has("Content-Type"), false);
  assert.notEqual(uploads[0].init.headers.get("Authorization"), "Bearer " + "z".repeat(43));
  assert.equal(h.issued(), 1);
  assert.equal(h.localStorage.getItem(activeKey), "synthetic-session");
  assert.equal(h.localStorage.getItem(client.storage.browserSessionKey(activeKey)), null);
});

test("valid legacy identity migrates active session, pending answer and draft", async () => {
  const h = harness(); saveLegacy(h);
  const client = h.load(); await client.api.listSessions();
  assert.equal(h.issued(), 0);
  assert.equal(h.localStorage.getItem(vaultKey(serverA)), tokenA);
  assert.equal(h.localStorage.getItem(client.storage.browserSessionKey(activeKey)), "synthetic-session");
  assert.ok(h.sessionStorage.getItem(client.storage.browserSessionKey(pendingKey)));
  assert.equal(h.localStorage.getItem(client.storage.browserSessionKey("interview_draft_synthetic-session")), "Synthetic draft");
  assert.equal(h.localStorage.getItem(activeKey), null);
  assert.equal(h.sessionStorage.getItem(pendingKey), null);
});

test("malformed legacy token is preserved and never becomes the current identity", async () => {
  const h = harness(); saveLegacy(h, "malformed");
  const client = h.load(); await client.api.listSessions();
  assert.equal(h.issued(), 1);
  assert.equal(h.localStorage.getItem(legacyKey), "malformed");
  assert.equal(h.localStorage.getItem(activeKey), "synthetic-session");
  assert.ok(h.sessionStorage.getItem(pendingKey));
  assert.equal(h.sessionStorage.getItem(client.storage.browserSessionKey(pendingKey)), null);
});

test("switching copies and returning restores each environment's original identity", async () => {
  const h = harness(); saveLegacy(h);
  const a = h.load(); await a.api.listSessions();
  const aPending = a.storage.browserSessionKey(pendingKey);
  h.switchTo(serverB);
  const b = h.load(); await b.api.listSessions();
  assert.equal(h.localStorage.getItem(b.storage.browserSessionKey(activeKey)), null);
  h.localStorage.setItem(b.storage.browserSessionKey(activeKey), "synthetic-b-session");
  h.switchTo(serverA);
  const returned = h.load(); await returned.api.listSessions();
  assert.equal(h.issued(), 1);
  assert.equal(h.localStorage.getItem(vaultKey(serverA)), tokenA);
  assert.equal(h.localStorage.getItem(returned.storage.browserSessionKey(activeKey)), "synthetic-session");
  assert.ok(h.sessionStorage.getItem(aPending));
  h.switchTo(serverB);
  const returnedB = h.load(); await returnedB.api.listSessions();
  assert.equal(h.localStorage.getItem(returnedB.storage.browserSessionKey(activeKey)), "synthetic-b-session");
  assert.equal(h.issued(), 1);
});

test("deleted migrated session does not reappear after a page reload", async () => {
  const h = harness(); saveLegacy(h);
  const first = h.load(); await first.api.listSessions();
  h.localStorage.removeItem(first.storage.browserSessionKey(activeKey));
  h.sessionStorage.removeItem(first.storage.browserSessionKey(pendingKey));
  const reload = h.load(); await reload.api.listSessions();
  assert.equal(h.localStorage.getItem(reload.storage.browserSessionKey(activeKey)), null);
  assert.equal(h.sessionStorage.getItem(reload.storage.browserSessionKey(pendingKey)), null);
});

test("401 on upload recovers once using the same form after a live backend switch", async () => {
  const h = harness(); const client = h.load(); await client.api.listSessions();
  h.switchTo(serverB);
  await client.api.uploadResume(new Blob(["Synthetic resume"]));
  const uploads = h.calls.filter((c) => c.url === "/api/uploads/resume");
  assert.equal(uploads.length, 2);
  assert.equal(uploads[0].init.body, uploads[1].init.body);
  assert.notEqual(uploads[0].init.headers.get("Authorization"), uploads[1].init.headers.get("Authorization"));
});

test("old answer is never replayed to a new identity after an authentication rejection", async () => {
  const h = harness(); saveLegacy(h);
  const client = h.load(); await client.api.listSessions();
  const oldPendingKey = client.storage.browserSessionKey(pendingKey);
  h.switchTo(serverB);
  await assert.rejects(client.api.answer("synthetic-session", { answer: "Synthetic original answer" }),
    (error) => error instanceof client.ApiError && error.code === "interview_environment_changed");
  assert.equal(h.calls.filter((c) => c.url.endsWith("/answer")).length, 1);
  assert.ok(h.sessionStorage.getItem(oldPendingKey));
  assert.equal(h.sessionStorage.getItem(client.storage.browserSessionKey(pendingKey)), null);
});

test("old session GET is not silently moved to the new identity", async () => {
  const h = harness(); const client = h.load(); await client.api.listSessions();
  h.switchTo(serverB);
  await assert.rejects(client.api.getSession("synthetic-session"), (e) => e.code === "interview_environment_changed");
  assert.equal(h.calls.filter((c) => c.url === "/api/sessions/synthetic-session").length, 1);
});

test("concurrent initial uploads and history share one bootstrap", async () => {
  const h = harness(); const client = h.load();
  await Promise.all([client.api.listSessions(), client.api.uploadResume(new Blob(["Synthetic"])), client.api.listSessions()]);
  assert.equal(h.calls.filter((c) => c.url === "/api/auth/context").length, 1);
  assert.equal(h.calls.filter((c) => c.url === "/api/auth/anonymous").length, 1);
  assert.equal(h.issued(), 1);
});

test("concurrent 401 recovery shares one new identity", async () => {
  const h = harness(); const client = h.load(); await client.api.listSessions();
  h.switchTo(serverB);
  await Promise.all([client.api.uploadResume(new Blob(["Synthetic"])), client.api.listSessions()]);
  assert.equal(h.issued(), 2);
  assert.equal(h.calls.filter((c) => c.url === "/api/auth/anonymous").length, 2);
});

test("service 503 preserves the vault and does not issue a replacement", async () => {
  const h = harness(); h.localStorage.setItem(vaultKey(serverA), tokenA);
  h.intercept((c) => c.url === "/api/auth/anonymous" ? response(503, { detail: "unavailable" }) : null);
  const client = h.load();
  await assert.rejects(client.api.uploadResume(new Blob(["Synthetic"])), (e) => e.status === 503);
  assert.equal(h.issued(), 0);
  assert.equal(h.localStorage.getItem(vaultKey(serverA)), tokenA);
  assert.equal(h.calls.some((c) => c.url.startsWith("/api/uploads/")), false);
});

test("offline bootstrap retains all recovery records without starting a write", async () => {
  const h = harness(); saveLegacy(h);
  h.intercept(() => { throw new TypeError("Synthetic offline network"); });
  const client = h.load();
  await assert.rejects(client.api.uploadResume(new Blob(["Synthetic"])));
  assert.ok(h.sessionStorage.getItem(pendingKey));
  assert.equal(h.localStorage.getItem(legacyKey), tokenA);
  assert.equal(h.calls.some((c) => c.url.startsWith("/api/uploads/")), false);
});

test("invalid bootstrap response never persists a token or sends a resume", async () => {
  const h = harness();
  h.intercept((c) => c.url === "/api/auth/anonymous" ? response(201, { token: "invalid", token_type: "Bearer" }) : null);
  const client = h.load(); await assert.rejects(client.api.uploadResume(new Blob(["Synthetic"])));
  assert.equal(h.localStorage.getItem(vaultKey(serverA)), null);
  assert.equal(h.calls.some((c) => c.url.startsWith("/api/uploads/")), false);
});

test("invalid public context is rejected before issuing identity", async () => {
  const h = harness(); h.intercept((c) => c.url === "/api/auth/context" ? response(200, { server_id: "invalid" }) : null);
  const client = h.load(); await assert.rejects(client.api.listSessions());
  assert.equal(h.issued(), 0);
});

test("browser storage denial prevents sending the resume and leaves originals", async () => {
  const h = harness(); saveLegacy(h);
  h.localStorage.setItem = () => { throw new Error("Synthetic storage denial"); };
  const client = h.load(); await assert.rejects(client.api.uploadResume(new Blob(["Synthetic"])));
  assert.equal(h.localStorage.getItem(activeKey), "synthetic-session");
  assert.ok(h.sessionStorage.getItem(pendingKey));
  assert.equal(h.calls.some((c) => c.url.startsWith("/api/uploads/")), false);
});

test("network failure after answer POST never retries the write", async () => {
  const h = harness(); const client = h.load(); await client.api.listSessions();
  h.intercept((c) => { if (c.url.endsWith("/answer")) throw new TypeError("Synthetic uncertain network"); return null; });
  await assert.rejects(client.api.answer("synthetic-session", { answer: "Synthetic original answer" }));
  assert.equal(h.calls.filter((c) => c.url.endsWith("/answer")).length, 1);
});

test("repeated upload 401 is bounded and does not loop", async () => {
  const h = harness(); h.intercept((c) => c.url.startsWith("/api/uploads/") ? response(401, { detail: "invalid" }) : null);
  const client = h.load(); await assert.rejects(client.api.uploadResume(new Blob(["Synthetic"])), (e) => e.status === 401);
  assert.equal(h.calls.filter((c) => c.url.startsWith("/api/uploads/")).length, 2);
  assert.equal(h.issued(), 1);
});

test("backend restart with the same database reuses the identity and recovery scope", async () => {
  const h = harness(); saveLegacy(h);
  const first = h.load(); await first.api.listSessions();
  const scope = first.storage.browserIdentity();
  const restarted = h.load(); await restarted.api.listSessions();
  assert.equal(restarted.storage.browserIdentity(), scope);
  assert.equal(h.issued(), 0);
  assert.ok(h.sessionStorage.getItem(restarted.storage.browserSessionKey(pendingKey)));
});

test("identity-change listeners fire only for an actual environment change", async () => {
  const h = harness(); const client = h.load(); let changed = 0;
  const unsubscribe = client.storage.onBrowserIdentityChanged(() => { changed++; });
  await client.api.listSessions(); await client.anonymousCredential(true);
  assert.equal(changed, 0);
  h.switchTo(serverB); await client.anonymousCredential(true); assert.equal(changed, 1);
  unsubscribe(); h.switchTo(serverA); await client.anonymousCredential(true); assert.equal(changed, 1);
});

test("frontend errors no longer render the removed anonymous-credential warning", () => {
  const h = harness(); const { ApiError } = h.load();
  assert.doesNotMatch(new ApiError(401, "Anonymous bearer credential required").message, /匿名凭证|不会自动更换/);
});

test("backend switch during bootstrap does not overwrite a saved identity in the destination", async () => {
  const h = harness();
  const tokenB = "q".repeat(43); const identityB = "4".repeat(32);
  h.environments.get(serverB).set(tokenB, identityB);
  h.localStorage.setItem(vaultKey(serverB), tokenB);
  let switched = false;
  h.intercept((c) => {
    if (c.url === "/api/auth/anonymous" && !switched) { switched = true; h.switchTo(serverB); }
    return null;
  });
  const client = h.load(); await client.api.listSessions();
  assert.equal(h.localStorage.getItem(vaultKey(serverB)), tokenB);
  assert.equal(client.storage.browserIdentity(), `${serverB}:${identityB}`);
});

test("late history response cannot replace the new environment's history", async () => {
  const h = harness(); const client = h.load(); await client.api.listSessions();
  let finish;
  h.intercept((c) => c.url === "/api/sessions" && c.server === serverA
    ? new Promise((resolve) => { finish = resolve; }) : null);
  const oldList = client.api.listSessions();
  for (let i = 0; i < 30; i++) await Promise.resolve();
  h.switchTo(serverB); await client.anonymousCredential(true);
  finish(response(200, [{ id: "synthetic-old-history" }]));
  const result = await oldList;
  assert.equal(result.length, 0);
});
