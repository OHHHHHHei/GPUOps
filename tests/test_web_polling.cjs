const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { resolve } = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const script = readFileSync(resolve(__dirname, "../src/gpuops/web/static/app.js"), "utf8");

class Element {
  constructor() {
    this.children = [];
    this.dataset = {};
    this.classList = { add() {}, remove() {}, toggle() {} };
  }
  appendChild(child) { child.parent = this; this.children.push(child); }
  replaceChildren(...children) {
    this.children.forEach((child) => { child.parent = null; });
    this.children = [];
    children.forEach((child) => this.appendChild(child));
  }
  prepend(child) { child.parent = this; this.children.unshift(child); }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter((child) => child !== this);
    this.parent = null;
  }
  setAttribute() {}
  addEventListener() {}
  querySelector(selector) {
    for (const child of this.children) {
      if (child.className?.split(" ").includes(selector.slice(1))) return child;
      const match = child.querySelector(selector);
      if (match) return match;
    }
    return null;
  }
}

function job(id, content, extra = {}) {
  return { id, name: id, state: "running", raw_state: "running", logs: [content], ...extra };
}

function queueItem(id, extra = {}) {
  return {
    id,
    name: id,
    state: "finished",
    raw_state: "succeeded",
    server: "gpu01",
    gpus: [0],
    project: "project-a",
    priority: "normal",
    finished_at: "2026-09-20T12:00:00Z",
    ...extra,
  };
}

function response(payload) {
  return { ok: true, json: async () => payload };
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function server(id, extra = {}) {
  return {
    id,
    label: id.toUpperCase(),
    status: "online",
    stale: false,
    last_seen: "2026-09-20T10:00:00Z",
    error: null,
    error_code: null,
    gpus: [],
    ...extra,
  };
}

function texts(node) {
  return [node.textContent || "", ...node.children.flatMap(texts)].join(" ");
}

async function dashboard() {
  const elements = new Map();
  const requests = [];
  const routes = new Map([
    ["/api/v1/snapshot", () => response({ summary: {}, servers: [] })],
    ["/api/v1/queue", () => response({ items: [], servers: [] })],
    ["/api/v1/process-etas", () => response({ updated_at: "now", servers: [] })],
    ["/healthz", () => response({ status: "ok", servers: [] })],
  ]);
  let poll;
  const context = vm.createContext({
    Node: Element,
    document: {
      body: { dataset: { pollInterval: "5000" } },
      getElementById(id) {
        if (!elements.has(id)) elements.set(id, new Element());
        return elements.get(id);
      },
      createElement: () => new Element(),
      querySelectorAll: () => [],
    },
    window: {
      setInterval(callback) { poll = callback; },
      confirm: () => true,
    },
    fetch: async (url) => {
      requests.push(url);
      assert.ok(routes.has(url), `unexpected request: ${url}`);
      return routes.get(url)();
    },
  });
  vm.runInContext(script, context, { filename: "app.js" });
  await new Promise(setImmediate);
  return {
    context,
    routes,
    requests,
    poll: () => poll(),
    element: (id) => elements.get(id),
    detail: () => elements.get("job-detail"),
    log: () => elements.get("job-detail").children.find((node) => node.className === "log-output"),
  };
}

test("an empty fleet explains setup and clears the warning after a server is added", async () => {
  const app = await dashboard();
  app.routes.set("/healthz", () => response({
    status: "degraded", reason: "no_servers_registered", servers: [],
  }));
  await app.poll();
  assert.match(texts(app.element("overview-errors")), /gpuops server add SSH_ALIAS/);
  assert.equal(app.element("connection-label").textContent, "未接入服务器");

  app.routes.set("/api/v1/snapshot", () => response({
    summary: {}, servers: [server("new")],
  }));
  app.routes.set("/healthz", () => response({
    status: "ok", servers: [{ id: "new", status: "online" }],
  }));
  await app.poll();
  assert.equal(app.element("overview-errors").children.length, 0);
  assert.equal(app.element("connection-label").textContent, "已连接");
});

test("GPU ETA shows a dated estimate and exposes its log evidence", async () => {
  const app = await dashboard();
  const process = { eta: {
    end_at: "2099-09-21T01:30:00+08:00", scope: "training",
    basis: "已完成 2/3 epoch", source: "/logs/train.log",
    observed_at: "2026-09-21T01:15:00+08:00", reason: "不包含后续任务",
  } };
  assert.match(app.context.formatProcessEta(process), /^约 /);
  const tooltip = app.context.processEtaDetails(process);
  assert.match(tooltip, /2\/3 epoch/);
  assert.match(tooltip, /日志：\/logs\/train.log/);
  assert.match(tooltip, /不包含后续任务/);
  process.eta.scope = "phase";
  assert.match(app.context.formatProcessEta(process), /^阶段约 /);
  process.eta.end_at = null;
  assert.equal(app.context.formatProcessEta(process), "暂无法估计");
});

test("structured progress renders safely and reports phase completion without completing the job", async () => {
  const app = await dashboard();
  const phase = '<img src=x onerror="window.injected=true">';
  const current = job("run-a", "progress logs", { eta: {
    format: "gpuops_progress", source_kind: "logs", scope: "phase", end_at: null,
    progress: { phase, current: 30, total: 100 },
    basis: "structured record", reason: "not enough samples",
  } });
  app.routes.set("/api/v1/jobs/run-a", () => response(current));

  await app.context.selectJob("run-a");

  const definitions = app.detail().children.find((node) => node.className === "job-definitions");
  const lines = definitions.children.map((line) => ({
    label: line.children[0].textContent,
    value: line.children[1].textContent,
    valueNode: line.children[1],
  }));
  const progress = lines.find((line) => line.label === "阶段进度");
  assert.equal(progress.value, `${phase} · 30/100（30%）`);
  assert.equal(progress.valueNode.children.length, 0);
  assert.ok(lines.some((line) => line.label === "估计来源"
    && line.value === "GPUOps 结构化日志"));
  assert.equal(app.context.window?.injected, undefined);

  current.eta.progress.current = 100;
  assert.equal(app.context.formatJobEta(current), "阶段已完成");
  assert.equal(current.state, "running");
  assert.equal(app.context.formatStructuredProgress({
    format: "gpuops_progress",
    progress: { phase: "train", current: 101, total: 100 },
  }), null);
});

test("polling refreshes the selected job's logs without selecting it again", async () => {
  const app = await dashboard();
  let reads = 0;
  app.routes.set("/api/v1/jobs/run-a", () => response(job("run-a", `epoch=${++reads}`)));
  await app.context.selectJob("run-a");
  assert.equal(app.log().textContent, "epoch=1");
  await app.poll();
  assert.equal(app.log().textContent, "epoch=2");
});

test("a delayed response cannot replace a different selected job", async () => {
  const app = await dashboard();
  const first = deferred();
  app.routes.set("/api/v1/jobs/run-a", () => first.promise);
  app.routes.set("/api/v1/jobs/run-b", () => response(job("run-b", "B logs")));
  const selectingA = app.context.selectJob("run-a");
  await app.context.selectJob("run-b");
  first.resolve(response(job("run-a", "A logs")));
  await selectingA;
  assert.equal(app.log().textContent, "B logs");
});

test("a delayed response cannot overwrite a newer selection of the same job", async () => {
  const app = await dashboard();
  const first = deferred();
  let reads = 0;
  app.routes.set("/api/v1/jobs/run-a", () => ++reads === 1
    ? first.promise : response(job("run-a", "new logs")));
  const selecting = app.context.selectJob("run-a");
  await app.context.selectJob("run-a");
  first.resolve(response(job("run-a", "old logs")));
  await selecting;
  assert.equal(app.log().textContent, "new logs");
});

test("slow job details neither block overview polling nor create overlapping detail polls", async () => {
  const app = await dashboard();
  const details = deferred();
  let detailReads = 0;
  let overviewReads = 0;
  app.routes.set("/api/v1/jobs/run-a", () => { detailReads += 1; return details.promise; });
  app.routes.set("/api/v1/snapshot", () => {
    overviewReads += 1;
    return response({ summary: {}, servers: [] });
  });
  const selecting = app.context.selectJob("run-a");
  await app.poll();
  await app.poll();
  assert.equal(overviewReads, 2);
  assert.equal(detailReads, 1);
  details.resolve(response(job("run-a", "logs")));
  await selecting;
});

test("a background detail error preserves the last visible logs", async () => {
  const app = await dashboard();
  app.routes.set("/api/v1/jobs/run-a", () => response(job("run-a", "last known logs")));
  await app.context.selectJob("run-a");
  app.routes.set("/api/v1/jobs/run-a", () => { throw new Error("host unavailable"); });
  await app.poll();
  await app.poll();
  assert.equal(app.log().textContent, "last known logs");
  assert.equal(app.detail().children.filter((node) => node.textContent?.includes("host unavailable")).length, 1);
});

test("unknown GPU telemetry displays missing metrics without implying zero load", async () => {
  const app = await dashboard();
  const row = app.context.renderGpuRow({
    index: 0, status: "unknown", processes: [],
    utilization_percent: null, memory_used_mib: null, memory_total_mib: 24576,
    temperature_c: null,
  });
  assert.equal(row.children[1].children[0].textContent, "未知");
  assert.equal(row.children[2].textContent, "-");
  assert.equal(row.children[3].textContent, "- / 24 GB");
  assert.equal(row.children[4].textContent, "-");
  assert.equal(app.context.formatMemory(0, 24576), "0.0 / 24 GB");
});

test("a detail poll keeps cancellation disabled while the cancel request is pending", async () => {
  const app = await dashboard();
  const cancelling = deferred();
  let cancelRequests = 0;
  let detailReads = 0;
  let currentJob = job("run-a", "logs");
  app.routes.set("/api/v1/jobs/run-a", () => { detailReads += 1; return response(currentJob); });
  app.routes.set("/api/v1/jobs/run-a/cancel", () => {
    cancelRequests += 1;
    return cancelling.promise;
  });
  await app.context.selectJob("run-a");
  const pending = app.context.cancelJob(currentJob, app.detail().querySelector(".danger-button"));
  await app.poll();
  const replacementButton = app.detail().querySelector(".danger-button");
  assert.equal(replacementButton.disabled, true);
  assert.equal(replacementButton.textContent, "正在取消...");
  await app.context.cancelJob(currentJob, replacementButton);
  assert.equal(cancelRequests, 1);
  currentJob = { ...currentJob, raw_state: "cancel_requested" };
  cancelling.resolve(response({ accepted: true }));
  await pending;
  await new Promise(setImmediate);
  assert.equal(app.detail().querySelector(".danger-button"), null);
  assert.equal(detailReads, 3);
});

test("a delayed error cannot replace another job's details", async () => {
  const app = await dashboard();
  const first = deferred();
  app.routes.set("/api/v1/jobs/run-a", () => first.promise);
  app.routes.set("/api/v1/jobs/run-b", () => response(job("run-b", "B logs")));
  const selectingA = app.context.selectJob("run-a");
  await app.context.selectJob("run-b");
  first.reject(new Error("host unavailable"));
  await selectingA;
  assert.equal(app.log().textContent, "B logs");
});

test("accepted cancellation stays disabled until slow job details confirm it", async () => {
  const app = await dashboard();
  const details = deferred();
  let cancelRequests = 0;
  const currentJob = job("run-a", "logs");
  app.routes.set("/api/v1/jobs/run-a", () => response(currentJob));
  app.routes.set("/api/v1/jobs/run-a/cancel", () => {
    cancelRequests += 1;
    return response({ accepted: true });
  });
  await app.context.selectJob("run-a");
  app.routes.set("/api/v1/jobs/run-a", () => details.promise);
  await app.context.cancelJob(currentJob, app.detail().querySelector(".danger-button"));
  const button = app.detail().querySelector(".danger-button");
  assert.equal(button.disabled, true);
  await app.context.cancelJob(currentJob, button);
  assert.equal(cancelRequests, 1);
  details.resolve(response({ ...currentJob, raw_state: "cancel_requested" }));
  await new Promise(setImmediate);
  assert.equal(app.detail().querySelector(".danger-button"), null);
});

test("late detail responses preserve cancellation failures and allow retry", async () => {
  const app = await dashboard();
  const details = deferred();
  const currentJob = job("run-a", "logs");
  app.routes.set("/api/v1/jobs/run-a", () => response(currentJob));
  app.routes.set("/api/v1/jobs/run-a/cancel", () => {
    throw new Error("cancel host unavailable");
  });
  await app.context.selectJob("run-a");
  app.routes.set("/api/v1/jobs/run-a", () => details.promise);
  await app.poll();
  await app.context.cancelJob(currentJob, app.detail().querySelector(".danger-button"));
  details.resolve(response(currentJob));
  await new Promise(setImmediate);
  assert.ok(app.detail().querySelector(".cancel-error").textContent.includes("cancel host unavailable"));
  assert.equal(app.detail().querySelector(".danger-button").disabled, false);
});

test("overview failure does not stop selected job polling, and log errors are visible", async () => {
  const app = await dashboard();
  app.routes.set("/api/v1/jobs/run-a", () => response(job("run-a", "initial")));
  await app.context.selectJob("run-a");
  app.routes.set("/api/v1/snapshot", () => { throw new Error("monitor unavailable"); });
  app.routes.set("/api/v1/jobs/run-a", () => response(job("run-a", "", {
    log_error: "SSH timeout",
  })));
  await app.poll();
  assert.ok(app.detail().children.some((node) => node.textContent?.includes("SSH timeout")));
});

test("job explanations, next actions, and failure reasons render as text", async () => {
  const app = await dashboard();
  const message = '<img src=x onerror="window.injected=true">';
  const nextAction = "<script>custom_action()</script>";
  const failureReason = "<b>process failed</b>";
  app.routes.set("/api/v1/jobs/run-a", () => response(job("run-a", "logs", {
    explanation: { waiting: true, message, next_action: nextAction },
    failure_reason: failureReason,
  })));

  await app.context.selectJob("run-a");

  const definitions = app.detail().children.find((node) => node.className === "job-definitions");
  const values = definitions.children.map((line) => line.children[1]);
  assert.ok(values.some((node) => node.textContent === message && node.children.length === 0));
  assert.ok(values.some((node) => node.textContent === nextAction && node.children.length === 0));
  assert.ok(values.some((node) => node.textContent === failureReason && node.children.length === 0));
  assert.equal(app.context.window?.injected, undefined);
});

test("known next actions are labelled and an explained failure is not repeated", async () => {
  const app = await dashboard();
  const failureReason = "process exited with status 2";
  app.routes.set("/api/v1/jobs/run-a", () => response(job("run-a", "logs", {
    explanation: {
      waiting: false,
      message: `The job failed: ${failureReason}`,
      next_action: "inspect_logs",
      details: { failure_reason: failureReason },
    },
    failure_reason: failureReason,
  })));

  await app.context.selectJob("run-a");

  const definitions = app.detail().children.find((node) => node.className === "job-definitions");
  const lines = definitions.children.map((line) => ({
    label: line.children[0].textContent,
    value: line.children[1].textContent,
  }));
  assert.ok(lines.some((line) => line.label === "建议操作" && line.value === "查看作业日志"));
  assert.equal(lines.filter((line) => line.label === "失败原因").length, 0);
  assert.equal(lines.filter((line) => line.value.includes(failureReason)).length, 1);
});

test("history is loaded on demand and active polling does not replace it", async () => {
  const app = await dashboard();
  const historyUrl = "/api/v1/history?limit=50";
  app.routes.set(historyUrl, () => response({
    updated_at: "2026-09-20T12:00:00Z",
    mode: "history",
    items: [queueItem("historic-run")],
    servers: [],
    projects: ["project-a"],
    next_cursor: null,
  }));
  app.routes.set("/api/v1/queue", () => response({
    updated_at: "2026-09-20T12:01:00Z",
    items: [queueItem("active-run", { state: "queued", raw_state: "queued" })],
    servers: [],
  }));

  assert.equal(app.requests.filter((url) => url.startsWith("/api/v1/history")).length, 0);
  await app.context.setQueueMode("history");
  assert.equal(app.requests.filter((url) => url === historyUrl).length, 1);
  assert.equal(app.element("queue-body").children[0].dataset.jobId, "historic-run");

  await app.poll();

  assert.equal(app.requests.filter((url) => url === historyUrl).length, 1);
  assert.equal(app.element("queue-count").textContent, 1);
  assert.equal(app.element("queue-body").children[0].dataset.jobId, "historic-run");
});

test("history pagination and project filtering reset the page and selection", async () => {
  const app = await dashboard();
  const firstUrl = "/api/v1/history?limit=50";
  const nextUrl = "/api/v1/history?limit=50&cursor=page-2";
  const filteredUrl = "/api/v1/history?limit=50&project=project%20b";
  app.routes.set(firstUrl, () => response({
    updated_at: "2026-09-20T12:00:00Z",
    mode: "history",
    items: [queueItem("old-1")],
    servers: [],
    projects: ["project-a", "project b"],
    next_cursor: "page-2",
  }));
  app.routes.set(nextUrl, () => response({
    updated_at: "2026-09-20T11:00:00Z",
    mode: "history",
    items: [queueItem("old-2")],
    servers: [],
    projects: ["project-a", "project b"],
    next_cursor: null,
  }));
  app.routes.set(filteredUrl, () => response({
    updated_at: "2026-09-20T12:05:00Z",
    mode: "history",
    items: [queueItem("project-b-run", { project: "project b" })],
    servers: [],
    projects: ["project-a", "project b"],
    next_cursor: null,
  }));
  app.routes.set("/api/v1/jobs/old-2", () => response(job("old-2", "complete", {
    state: "finished",
    raw_state: "succeeded",
  })));

  await app.context.setQueueMode("history");
  await app.context.showNextHistoryPage();
  assert.equal(app.element("queue-body").children[0].dataset.jobId, "old-2");
  assert.equal(app.element("history-page-label").textContent, "第 2 页");

  await app.context.selectJob("old-2", { autoRefresh: false });
  app.element("history-project").value = "project b";
  await app.context.changeHistoryProject();

  assert.equal(app.element("queue-body").children[0].dataset.jobId, "project-b-run");
  assert.equal(app.element("history-page-label").textContent, "第 1 页");
  assert.equal(app.detail().children[0].className, "empty-state");
  assert.ok(app.requests.includes(filteredUrl));
});

test("a stale history response cannot overwrite a newer filter", async () => {
  const app = await dashboard();
  const unfiltered = deferred();
  const firstUrl = "/api/v1/history?limit=50";
  const filteredUrl = "/api/v1/history?limit=50&project=project-b";
  app.routes.set(firstUrl, () => unfiltered.promise);
  app.routes.set(filteredUrl, () => response({
    updated_at: "2026-09-20T12:05:00Z",
    mode: "history",
    items: [queueItem("new-filtered-run", { project: "project-b" })],
    servers: [],
    projects: ["project-b"],
    next_cursor: null,
  }));

  const loadingFirstPage = app.context.setQueueMode("history");
  app.element("history-project").value = "project-b";
  await app.context.changeHistoryProject();
  unfiltered.resolve(response({
    updated_at: "2026-09-20T11:00:00Z",
    mode: "history",
    items: [queueItem("stale-run")],
    servers: [],
    projects: ["project-b"],
    next_cursor: null,
  }));
  await loadingFirstPage;

  assert.equal(app.element("queue-body").children[0].dataset.jobId, "new-filtered-run");
});

test("history empty, offline, and error states remain explicit", async () => {
  const app = await dashboard();
  const historyUrl = "/api/v1/history?limit=50";
  app.routes.set(historyUrl, () => response({
    updated_at: "2026-09-20T12:00:00Z",
    mode: "history",
    items: [],
    servers: [{ id: "gpu02", online: false, error: "worker offline" }],
    projects: [],
    next_cursor: null,
  }));

  await app.context.setQueueMode("history");
  assert.equal(app.element("queue-body").children[0].children[0].textContent, "暂无历史作业");
  assert.ok(app.element("queue-errors").children[0].textContent.includes("worker offline"));
  assert.ok(app.element("queue-errors").children[0].textContent.includes("历史记录不完整"));

  app.routes.set(historyUrl, () => { throw new Error("history unavailable"); });
  await app.context.setQueueMode("history");
  assert.ok(app.element("queue-errors").children[0].textContent.includes("history unavailable"));
  assert.equal(app.element("queue-body").children[0].children[0].textContent, "暂无历史作业");
});

test("a failed project filter does not expose the previous project's page or cursor", async () => {
  const app = await dashboard();
  const historyUrl = "/api/v1/history?limit=50";
  const filteredUrl = "/api/v1/history?limit=50&project=project-b";
  app.routes.set(historyUrl, () => response({
    updated_at: "2026-09-20T12:00:00Z",
    mode: "history",
    items: [queueItem("project-a-run")],
    servers: [],
    projects: ["project-a", "project-b"],
    next_cursor: "project-a-next",
  }));
  app.routes.set(filteredUrl, () => { throw new Error("filter unavailable"); });

  await app.context.setQueueMode("history");
  app.element("history-project").value = "project-b";
  await app.context.changeHistoryProject();

  assert.equal(app.element("queue-body").children[0].dataset.jobId, undefined);
  assert.equal(app.element("queue-body").children[0].children[0].textContent, "历史记录暂时不可用");
  assert.equal(app.element("history-page-label").textContent, "第 1 页");
  assert.equal(app.element("history-previous").disabled, true);
  assert.equal(app.element("history-next").disabled, true);
});

test("terminal job details stop polling but remain available for manual refresh", async () => {
  const app = await dashboard();
  let reads = 0;
  app.routes.set("/api/v1/jobs/run-done", () => response(job("run-done", `read=${++reads}`, {
    state: "finished",
    raw_state: "succeeded",
    finished_at: "2026-09-20T12:00:00Z",
  })));

  await app.context.selectJob("run-done", { autoRefresh: false });
  assert.equal(app.log().textContent, "read=1");
  const definitions = app.detail().children.find((node) => node.className === "job-definitions");
  assert.ok(definitions.children.some((line) => line.children[0].textContent === "完成时间"));
  assert.equal(definitions.children.some((line) => line.children[0].textContent === "预计结束"), false);
  await app.poll();
  assert.equal(reads, 1);
  assert.equal(app.log().textContent, "read=1");

  await app.context.refresh({ manual: true });
  await new Promise(setImmediate);
  assert.equal(reads, 2);
  assert.equal(app.log().textContent, "read=2");
});

test("an active selection remains visible after completion and then stops polling", async () => {
  const app = await dashboard();
  let reads = 0;
  app.routes.set("/api/v1/jobs/run-a", () => {
    reads += 1;
    if (reads === 1) return response(job("run-a", "running logs"));
    return response(job("run-a", "final logs", { state: "finished", raw_state: "succeeded" }));
  });

  await app.context.selectJob("run-a");
  await app.poll();
  assert.equal(app.log().textContent, "final logs");
  await app.poll();
  assert.equal(reads, 2);
  assert.equal(app.log().textContent, "final logs");
});


for (const slowPath of ["/api/v1/queue", "/api/v1/process-etas", "/healthz"]) {
  test(`slow ${slowPath} does not delay GPU rendering or the next GPU poll`, async () => {
    const app = await dashboard();
    const slow = deferred();
    let slowReads = 0;
    let gpuReads = 0;
    app.routes.set(slowPath, () => { slowReads++; return slow.promise; });
    app.routes.set("/api/v1/snapshot", () => response({
      summary: { idle: ++gpuReads }, servers: [],
    }));
    const firstPoll = app.poll();
    await new Promise(setImmediate);
    assert.equal(app.element("summary-idle").textContent, 1);
    await app.poll();
    assert.equal(app.element("summary-idle").textContent, 2);
    assert.equal(slowReads, 1);
    slow.resolve(response({ items: [], servers: [], status: "ok" }));
    await firstPoll;
  });
}

test("GPU failure preserves its data while queue refreshes, then clears on recovery", async () => {
  const app = await dashboard();
  app.routes.set("/api/v1/snapshot", () => response({ summary: { idle: 4 }, servers: [] }));
  await app.poll();
  app.routes.set("/api/v1/snapshot", () => { throw new Error("monitor unavailable"); });
  app.routes.set("/api/v1/queue", () => response({ items: [queueItem("new")], servers: [] }));
  await app.poll();
  assert.equal(app.element("summary-idle").textContent, 4);
  assert.equal(app.element("queue-count").textContent, 1);
  assert.match(app.element("snapshot-refresh-error").children[0].textContent, /保留上次数据/);
  assert.equal(app.element("connection-label").textContent, "部分数据异常");
  app.routes.set("/api/v1/snapshot", () => response({ summary: { idle: 5 }, servers: [] }));
  await app.poll();
  assert.equal(app.element("snapshot-refresh-error").children.length, 0);
  assert.equal(app.element("summary-idle").textContent, 5);
  assert.equal(app.element("connection-label").textContent, "已连接");
});

test("queue and ETA failures do not erase the last queue or GPU data", async () => {
  const app = await dashboard();
  app.routes.set("/api/v1/queue", () => response({ items: [queueItem("kept")], servers: [] }));
  await app.poll();
  app.routes.set("/api/v1/queue", () => { throw new Error("queue unavailable"); });
  app.routes.set("/api/v1/process-etas", () => { throw new Error("logs unavailable"); });
  app.routes.set("/api/v1/snapshot", () => response({ summary: { idle: 3 }, servers: [] }));
  await app.poll();
  assert.equal(app.element("queue-body").children[0].dataset.jobId, "kept");
  assert.equal(app.element("summary-idle").textContent, 3);
  assert.match(app.element("queue-refresh-error").children[0].textContent, /queue unavailable/);
  assert.match(app.element("eta-update").textContent, /logs unavailable/);
});

test("late ETA response cannot attach to a reused PID", async () => {
  const app = await dashboard();
  const old = deferred();
  app.routes.set("/api/v1/process-etas", () => old.promise);
  const snapshot = { summary: {}, servers: [{ id: "test", gpus: [{
    index: 0, status: "busy", processes: [{ pid: 123, start_time_ticks: 999 }],
  }] }] };
  app.routes.set("/api/v1/snapshot", () => response(snapshot));
  const pending = app.poll();
  await new Promise(setImmediate);
  old.resolve(response({ servers: [{ id: "test", processes: [{
    pid: 123, start_time_ticks: 456,
    eta: { end_at: "2099-01-01T00:00:00Z", scope: "training" },
  }] }] }));
  await pending;
  function texts(node) { return [node.textContent || "", ...node.children.flatMap(texts)].join(" "); }
  assert.match(texts(app.element("server-list")), /暂无法估计/);
  assert.doesNotMatch(texts(app.element("server-list")), /约 /);
  snapshot.servers[0].gpus[0].processes[0].start_time_ticks = 456;
  await app.poll();
  assert.match(texts(app.element("server-list")), /约 /);
});

test("stale GPU telemetry keeps metrics but never attaches cached process ETAs", async () => {
  const app = await dashboard();
  app.context.renderProcessEtas({
    updated_at: "2026-09-20T10:01:00Z",
    servers: [{ id: "gpu01", processes: [{
      pid: 123,
      start_time_ticks: 456,
      eta: { end_at: "2099-01-01T00:00:00Z", scope: "training" },
    }] }],
  });
  app.context.renderSnapshot({
    generated_at: "2026-09-20T10:02:00Z",
    summary: { idle: 0, busy: 0, residual: 0, unknown: 1, offline: 0 },
    servers: [{
      id: "gpu01",
      label: "GPU 01",
      status: "degraded",
      stale: true,
      last_seen: "2026-09-20T10:00:00Z",
      error: "collection exceeded 15 seconds",
      error_code: "collection_timeout",
      gpus: [{
        index: 0,
        status: "unknown",
        utilization_percent: 91,
        memory_used_mib: 18000,
        memory_total_mib: 24576,
        temperature_c: 70,
        processes: [{
          pid: 123,
          user: "researcher",
          command: "python train.py",
          elapsed_seconds: 600,
          start_time_ticks: 456,
          eta: { end_at: null },
        }],
      }],
    }],
  });

  const rendered = texts(app.element("server-list"));
  assert.match(rendered, /陈旧数据/);
  assert.match(rendered, /上次成功采集/);
  assert.match(rendered, /遥测采集超时/);
  assert.match(rendered, /91%/);
  assert.match(rendered, /暂无法估计/);
  assert.doesNotMatch(rendered, /约 /);
  assert.equal(app.element("summary-idle").textContent, 0);
});

test("per-GPU errors suppress ETAs and render concise partial-host warnings", async () => {
  const app = await dashboard();
  app.context.renderProcessEtas({
    updated_at: "2026-09-20T10:01:00Z",
    servers: [{ id: "gpu01", processes: [{
      pid: 123, start_time_ticks: 456,
      eta: { end_at: "2099-01-01T00:00:00Z", scope: "training" },
    }] }],
  });
  app.context.renderSnapshot({
    generated_at: "2026-09-20T10:02:00Z",
    summary: { idle: 2, unknown: 2 },
    servers: [server("gpu01", {
      status: "degraded",
      gpus: [
        { index: 3, status: "unknown", error_code: "gpu_query_failed",
          error: "NVML query failed", processes: [{
            pid: 123, start_time_ticks: 456, user: "researcher", command: "python train.py",
            eta: { end_at: null },
          }] },
        { index: 4, status: "idle", processes: [] },
        { index: 5, status: "idle", processes: [] },
        { index: 6, status: "unknown", error_code: "telemetry_stale",
          error: "GPU sample is stale", processes: [] },
      ],
    })],
  });

  const rendered = texts(app.element("server-list"));
  assert.match(texts(app.element("overview-errors")), /GPU 3 异常.*GPU 6 延迟/);
  assert.match(rendered, /NVML query failed/);
  assert.match(rendered, /GPU sample is stale/);
  assert.match(rendered, /暂无法估计/);
  assert.doesNotMatch(rendered, /约 /);
  assert.doesNotMatch(app.element("server-list").children[0].className, /is-stale/);

  app.context.renderSnapshot({
    generated_at: "2026-09-20T10:02:10Z", summary: { idle: 2 },
    servers: [server("gpu01", { gpus: [
      { index: 3, status: "idle", processes: [] },
      { index: 6, status: "idle", processes: [] },
    ] })],
  });
  assert.equal(app.element("overview-errors").children.length, 0);
  assert.doesNotMatch(texts(app.element("server-list")), /NVML query failed|GPU sample is stale/);
});

test("device and CUDA failures clear automatically after fresh recovery", async () => {
  const app = await dashboard();
  app.context.renderSnapshot({
    generated_at: "2026-09-28T10:02:00Z",
    summary: { idle: 1, unknown: 2 },
    servers: [server("gpu01", {
      status: "degraded",
      gpus: [
        { index: 4, status: "idle", processes: [] },
        { index: 3, status: "unknown", error_code: "gpu_device_unavailable",
          error: "PCI device is unavailable", processes: [] },
        { index: 6, status: "unknown", error_code: "gpu_cuda_unavailable",
          error: "cuInit failed", processes: [] },
      ],
    })],
  });

  assert.match(texts(app.element("overview-errors")), /GPU 3 异常.*GPU 6 异常/);
  assert.match(texts(app.element("server-list")), /GPU 设备不可用.*PCI device is unavailable/);
  assert.match(texts(app.element("server-list")), /CUDA 初始化失败.*cuInit failed/);
  assert.equal(app.element("summary-idle").textContent, 1);
  assert.equal(app.element("summary-unknown").textContent, 2);

  app.context.renderSnapshot({
    generated_at: "2026-09-28T10:03:00Z", summary: { idle: 3, unknown: 0 },
    servers: [server("gpu01", { gpus: [
      { index: 3, status: "idle", processes: [] },
      { index: 4, status: "idle", processes: [] },
      { index: 6, status: "idle", processes: [] },
    ] })],
  });
  assert.equal(app.element("overview-errors").children.length, 0);
  assert.doesNotMatch(texts(app.element("server-list")), /cuInit failed|PCI device is unavailable/);
  assert.equal(app.element("summary-idle").textContent, 3);
});

test("sampling delay is amber while timeout and connection failures remain red", async () => {
  const app = await dashboard();
  app.context.renderSnapshot({
    generated_at: "2026-09-20T10:02:00Z",
    summary: {},
    servers: [
      server("healthy"),
      server("delayed", {
        status: "degraded", stale: true, error_code: "telemetry_stale",
        error: "Worker sample is stale",
      }),
      server("first-sample", {
        status: "degraded", last_seen: null, error_code: "telemetry_stale",
        error: "Worker sample is stale",
      }),
      server("timeout", {
        status: "degraded", stale: true, error_code: "collection_timeout",
        error: "collection exceeded 15 seconds",
      }),
      server("offline", {
        status: "offline", last_seen: null, error_code: "connection_failed",
        error: "ssh exited 255",
      }),
    ],
  });

  const banners = app.element("overview-errors").children;
  assert.equal(banners.length, 4);
  assert.equal(banners[0].className, "warning-banner");
  assert.match(banners[0].textContent, /GPU 采样延迟/);
  assert.match(banners[0].textContent, /上次成功数据/);
  assert.equal(banners[1].className, "warning-banner");
  assert.match(banners[1].textContent, /暂无可用的成功采样/);
  assert.equal(banners[2].className, "error-banner");
  assert.match(banners[2].textContent, /遥测采集超时.*collection exceeded 15 seconds/);
  assert.equal(banners[3].className, "error-banner");
  assert.match(banners[3].textContent, /SSH 连接失败.*ssh exited 255/);
  assert.doesNotMatch(texts(app.element("overview-errors")), /HEALTHY/);

  const sections = app.element("server-list").children;
  assert.match(sections[1].className, /is-delayed/);
  assert.match(sections[2].className, /is-delayed/);
  assert.doesNotMatch(sections[3].className, /is-delayed/);
  assert.match(texts(sections[2]), /等待首次成功采集/);
});

test("degraded hosts without cached data stay visible and fresh data clears warnings", async () => {
  const app = await dashboard();
  app.context.renderSnapshot({
    generated_at: "2026-09-20T10:02:00Z",
    summary: {},
    servers: [server("gpu02", { status: "degraded", last_seen: null })],
  });
  assert.equal(app.element("overview-errors").children.length, 1);
  assert.match(texts(app.element("overview-errors")), /状态异常.*部分 GPU 遥测不可用/);

  app.context.renderSnapshot({
    generated_at: "2026-09-20T10:02:10Z",
    summary: {},
    servers: [server("gpu02")],
  });
  assert.equal(app.element("overview-errors").children.length, 0);
  assert.doesNotMatch(app.element("server-list").children[0].className, /is-delayed|has-problem/);
  assert.doesNotMatch(texts(app.element("server-list")), /采样延迟|状态异常/);
});

test("managed job ETA displays source, phase and waiting states consistently", async () => {
  const app = await dashboard();
  const current = job("run-a", "epoch=2/5", { eta: {
    source_kind: "logs", scope: "phase", end_at: "2099-01-01T00:00:00Z",
    basis: "200/1000 step", source: "/logs/run/console.log",
  } });
  app.routes.set("/api/v1/jobs/run-a", () => response(current));
  await app.context.selectJob("run-a");
  const definitions = app.detail().children.find((node) => node.className === "job-definitions");
  assert.ok(definitions.children.some((line) => line.children[0].textContent === "阶段预计结束"));
  assert.ok(definitions.children.some((line) => line.children[1].textContent === "日志进度"));
  assert.match(app.context.formatJobEta(current), /^阶段约 /);
  current.eta.source_kind = "submitted";
  current.eta.scope = "job";
  assert.match(app.context.formatJobEta(current), /^提交时估计 /);
  current.raw_state = "queued";
  assert.equal(app.context.formatJobEta(current), "等待调度");
  current.raw_state = "cancel_requested";
  assert.equal(app.context.formatJobEta(current), "取消中");
});
