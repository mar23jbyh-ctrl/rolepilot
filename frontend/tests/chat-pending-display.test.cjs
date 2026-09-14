const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const ts = require("typescript");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");

const filename = path.join(__dirname, "../src/components/ChatView.tsx");
const source = ts.transpileModule(fs.readFileSync(filename, "utf8"), {
  compilerOptions: {
    module: ts.ModuleKind.CommonJS,
    jsx: ts.JsxEmit.ReactJSX,
    target: ts.ScriptTarget.ES2020,
  },
}).outputText;
const componentModule = { exports: {} };
vm.runInNewContext(source, {
  module: componentModule,
  exports: componentModule.exports,
  localStorage: { getItem: () => "Synthetic unsent draft" },
  require: (name) => {
    if (name === "lucide-react") return new Proxy({}, { get: () => () => null });
    if (name === "react") return { ...React, useState: () => ["Synthetic unsent draft", () => {}] };
    if (name === "./MessageBubble") return { MessageBubble: () => null };
    return require(name);
  },
});
const { ChatView } = componentModule.exports;
const removedNotice = /有一条回答尚未确认|恢复会查询或重发同一个请求/;

function render(props = {}) {
  return renderToStaticMarkup(React.createElement(ChatView, {
    sessionId: "synthetic-session",
    messages: [],
    report: null,
    loading: false,
    error: "",
    onAnswer: () => {},
    onStop: () => {},
    onRetry: () => {},
    onPreserveFailed: () => {},
    ...props,
  }));
}

test("normal pending submission shows thinking without the red recovery notice", () => {
  const html = render({ pendingAnswer: true, loading: true });
  assert.match(html, /面试官正在思考/);
  assert.doesNotMatch(html, removedNotice);
  assert.doesNotMatch(html, /role="status"|恢复原回答请求|保留故障会话/);
});

test("failed pending submission retains its error and recovery button without the removed text", () => {
  const html = render({ pendingAnswer: true, error: "合成网络错误" });
  assert.match(html, /合成网络错误/);
  assert.match(html, /恢复原回答请求/);
  assert.doesNotMatch(html, removedNotice);
  assert.doesNotMatch(html, /输入你的回答/);
});

test("uncertain execution still offers recovery and the existing fault-session escape", () => {
  const html = render({ pendingAnswer: true, recoveryRequired: true });
  assert.match(html, /执行结果尚无法安全确认/);
  assert.match(html, /恢复原回答请求/);
  assert.match(html, /保留故障会话，开始新面试/);
  assert.doesNotMatch(html, removedNotice);
});

test("ordinary answer entry and stop controls remain available", () => {
  const html = render();
  assert.match(html, /输入你的回答/);
  assert.match(html, /发送回答/);
  assert.match(html, /结束并评估/);
  assert.doesNotMatch(html, /role="status"|恢复原回答请求/);
});

test("read-only history does not expose recovery or answer controls", () => {
  const html = render({ readOnly: true, pendingAnswer: true, recoveryRequired: true });
  assert.match(html, /正在查看历史会话/);
  assert.doesNotMatch(html, /role="status"|恢复原回答请求|保留故障会话|输入你的回答/);
});

test("an answer cannot be sent without an active session even with a draft", () => {
  const html = render({ sessionId: "" });
  assert.match(html, /<button[^>]*disabled[^>]*aria-label="发送回答"/);
});
