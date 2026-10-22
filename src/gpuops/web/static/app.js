"use strict";

const STATUS_LABELS = {
  idle: "空闲",
  busy: "占用",
  residual: "残留",
  unknown: "未知",
  offline: "离线",
  error: "异常",
  online: "在线",
  degraded: "部分异常",
  running: "运行中",
  queued: "排队中",
  finished: "已完成",
  failed: "失败",
  cancelled: "已取消",
  cancel_requested: "取消中",
};

const COLLECTION_ERROR_LABELS = {
  telemetry_stale: "GPU 采样延迟",
  telemetry_unavailable: "GPU 遥测不可用",
  gpu_query_timeout: "GPU 查询超时",
  gpu_query_failed: "GPU 查询失败",
  gpu_device_unavailable: "GPU 设备不可用",
  gpu_cuda_unavailable: "CUDA 初始化失败",
  collection_failed: "遥测采集失败",
  connection_failed: "SSH 连接失败",
  collection_timeout: "遥测采集超时",
  collection_error: "遥测采集异常",
};

function serverProblem(server) {
  const delayed = server.error_code === "telemetry_stale";
  const lastGood = server.last_seen
    ? `，显示上次成功数据（${formatDate(server.last_seen)}）`
    : "，暂无可用的成功采样";
  if (delayed) {
    return {
      tone: "warning",
      inline: `GPU 采样延迟：正在等待新的采样结果${lastGood}`,
      overview: `${server.label} GPU 采样延迟${lastGood}`,
    };
  }
  if (server.error) {
    const label = COLLECTION_ERROR_LABELS[server.error_code] || "服务器异常";
    const cached = server.stale && server.last_seen
      ? `；显示上次成功数据（${formatDate(server.last_seen)}）`
      : "";
    return {
      tone: "error",
      inline: `${label}：${server.error}${cached}`,
      overview: `${server.label} ${label}：${server.error}${cached}`,
    };
  }
  if (server.status === "offline") {
    const lastSeen = server.last_seen
      ? `，最后连接 ${formatDate(server.last_seen)}`
      : "，暂无成功连接记录";
    return {
      tone: "error",
      inline: `服务器离线${lastSeen}`,
      overview: `${server.label} 已离线${lastSeen}`,
    };
  }
  const cardProblems = server.gpus
    .filter((gpu) => gpu.error || gpu.error_code)
    .map((gpu) => {
      const delayed = gpu.error_code === "telemetry_stale";
      const state = delayed ? "延迟" : "异常";
      const label = `GPU ${gpu.index} ${state}`;
      return { label, detail: gpu.error ? `${label}：${gpu.error}` : label, delayed };
    });
  if (cardProblems.length) {
    return {
      tone: cardProblems.every((problem) => problem.delayed) ? "warning" : "error",
      inline: cardProblems.map((problem) => problem.detail).join("；"),
      overview: `${server.label} ${cardProblems.map((problem) => problem.label).join("；")}`,
    };
  }
  if (server.status === "degraded") {
    return {
      tone: "error",
      inline: "服务器状态异常：部分 GPU 遥测不可用",
      overview: `${server.label} 状态异常，部分 GPU 遥测不可用`,
    };
  }
  return null;
}

const CANCELLABLE_STATES = new Set(["queued", "leased", "starting", "running"]);
const DETAIL_POLLING_STATES = new Set([
  "queued",
  "leased",
  "starting",
  "running",
  "cancel_requested",
]);
const HISTORY_PAGE_SIZE = 50;
const NEXT_ACTION_LABELS = {
  wait: "继续等待",
  adjust_schedule: "调整调度设置",
  resubmit: "修正后重新提交",
  start_worker: "启动 Worker",
  adjust_resources: "调整资源请求",
  inspect_gpu_status: "检查 GPU 状态",
  inspect_logs: "查看作业日志",
  retry: "重试",
};

const state = {
  selectedJobId: null,
  selectedJobAutoRefresh: false,
  polls: {},
  snapshot: null,
  processEtas: new Map(),
  detailRequestId: 0,
  pendingDetailJobId: null,
  cancellingJobIds: new Set(),
  cancellationErrors: new Map(),
  queueMode: "active",
  activeQueue: null,
  historyPage: null,
  historyProject: "",
  historyCursor: null,
  historyPreviousCursors: [],
  historyRequestId: 0,
  historyLoading: false,
};

const byId = (id) => document.getElementById(id);

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function setConnection(status, label) {
  const container = byId("connection-state");
  container.dataset.status = status;
  byId("connection-label").textContent = label;
}

function formatDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

function formatDuration(totalSeconds) {
  if (!Number.isFinite(totalSeconds)) return "-";
  const seconds = Math.max(0, Math.floor(totalSeconds));
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (days) return `${days}天 ${hours}小时`;
  if (hours) return `${hours}小时 ${minutes}分`;
  return `${minutes}分`;
}

function formatMemory(used, total) {
  const usedGiB = Number.isFinite(used) ? (used / 1024).toFixed(1) : "-";
  const totalGiB = Number.isFinite(total) ? (total / 1024).toFixed(0) : "-";
  return `${usedGiB} / ${totalGiB} GB`;
}

function statusPill(status) {
  return element("span", `status-pill status-${status}`, STATUS_LABELS[status] || status);
}

function valueCell(label, value, className) {
  const cell = element("td", className || "");
  cell.dataset.label = label;
  if (value instanceof Node) cell.appendChild(value);
  else cell.textContent = value ?? "-";
  return cell;
}

function renderSummary(summary) {
  ["idle", "busy", "residual", "unknown", "offline"].forEach((key) => {
    byId(`summary-${key}`).textContent = summary?.[key] ?? "-";
  });
}

function renderProcesses(gpu) {
  const error = gpu.error
    ? `${COLLECTION_ERROR_LABELS[gpu.error_code] || "GPU 异常"}：${gpu.error}`
    : null;
  if (!gpu.processes?.length) {
    return {
      user: "-",
      pid: "-",
      command: error || "-",
      elapsed: "-",
      eta: "-",
    };
  }
  return {
    user: processStack(gpu.processes, (process) => process.user || "unknown"),
    pid: processStack(gpu.processes, (process) => process.pid ?? "-"),
    command: error || processStack(
      gpu.processes,
      (process) => process.command || "unknown",
      (process) => process.command || "unknown",
    ),
    elapsed: processStack(gpu.processes, (process) => formatDuration(process.elapsed_seconds)),
    eta: processStack(gpu.processes, formatProcessEta, processEtaDetails),
  };
}

function formatProcessEta(process) {
  const eta = process.eta;
  const progress = structuredProgress(eta);
  if (progress && progress.total !== null && progress.current === progress.total) {
    return "阶段已完成";
  }
  if (!eta?.end_at || new Date(eta.end_at).getTime() <= Date.now()) return "暂无法估计";
  const prefix = eta.source_kind === "submitted" ? "提交时估计 "
    : eta.scope === "phase" ? "阶段约 " : "约 ";
  return prefix + formatDate(eta.end_at).slice(0, -3);
}

function structuredProgress(eta) {
  const progress = eta?.progress;
  if (eta?.format !== "gpuops_progress" || !progress
      || typeof progress.phase !== "string" || !progress.phase
      || !Number.isInteger(progress.current) || progress.current < 0
      || (progress.total !== null && (!Number.isInteger(progress.total)
        || progress.total <= 0 || progress.current > progress.total))) {
    return null;
  }
  return progress;
}

function formatStructuredProgress(eta) {
  const progress = structuredProgress(eta);
  if (!progress) return null;
  if (progress.total === null) return `${progress.phase} · ${progress.current}/总量未知`;
  const percent = Math.floor((progress.current / progress.total) * 100);
  return `${progress.phase} · ${progress.current}/${progress.total}（${percent}%）`;
}

function processEtaDetails(process) {
  const eta = process.eta;
  return [
    eta?.basis,
    eta?.reason,
    eta?.source && `日志：${eta.source}`,
    eta?.observed_at && `进度更新：${formatDate(eta.observed_at)}`,
    eta?.sampled_at && `日志采样：${formatDate(eta.sampled_at)}`,
  ].filter(Boolean).join("\n") || "未找到足够的日志进度";
}

function processStack(processes, valueFor, titleFor) {
  const stack = element("div", "process-stack");
  processes.forEach((process) => {
    const line = element("span", "process-line", valueFor(process));
    if (titleFor) line.title = titleFor(process);
    stack.appendChild(line);
  });
  return stack;
}

function renderGpuRow(gpu) {
  const row = element("tr", `gpu-row gpu-${gpu.status}`);
  const process = renderProcesses(gpu);
  row.appendChild(valueCell("GPU", String(gpu.index), "gpu-index"));
  row.appendChild(valueCell("状态", statusPill(gpu.status)));
  row.appendChild(valueCell("利用率", Number.isFinite(gpu.utilization_percent)
    ? `${gpu.utilization_percent}%` : "-", "numeric"));
  row.appendChild(
    valueCell("显存", formatMemory(gpu.memory_used_mib, gpu.memory_total_mib), "numeric memory-cell"),
  );
  row.appendChild(valueCell("温度", Number.isFinite(gpu.temperature_c)
    ? `${gpu.temperature_c}°C` : "-", "numeric"));
  row.appendChild(valueCell("用户", process.user));
  row.appendChild(valueCell("PID", process.pid, "numeric mono"));
  row.appendChild(valueCell("命令", process.command, "command-cell mono"));
  row.appendChild(valueCell("已运行", process.elapsed, "duration-cell"));
  row.appendChild(valueCell("训练预计结束", process.eta, "eta-cell"));
  return row;
}

function renderServer(server) {
  const problem = serverProblem(server);
  const delayed = problem?.tone === "warning";
  const section = element(
    "section",
    `server-section${server.stale ? " is-stale" : ""}${delayed ? " is-delayed" : ""}`
      + `${problem && !server.stale ? " has-problem" : ""}`,
  );
  section.dataset.serverId = server.id;

  const heading = element("div", "server-heading");
  const name = element("div", "server-title-line");
  name.appendChild(element("h3", "", server.label));
  name.appendChild(statusPill(server.status));
  heading.appendChild(name);
  heading.appendChild(element(
    "span",
    `server-meta${server.stale || delayed ? " stale-meta" : ""}`,
    delayed
      ? server.last_seen
        ? `采样延迟 · 上次成功采集 ${formatDate(server.last_seen)}`
        : "采样延迟 · 等待首次成功采集"
      : server.stale
        ? `陈旧数据 · 上次成功采集 ${formatDate(server.last_seen)}`
      : `最后更新 ${formatDate(server.last_seen)}`,
  ));
  section.appendChild(heading);

  if (problem) {
    const warning = element(
      "div",
      `inline-warning${delayed ? " is-warning" : ""}`,
      problem.inline,
    );
    warning.setAttribute("role", "status");
    section.appendChild(warning);
  }

  const wrap = element("div", "table-scroll");
  const table = element("table", "data-table gpu-table");
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  ["GPU", "状态", "利用率", "显存", "温度", "用户", "PID", "命令", "已运行", "训练预计结束"].forEach(
    (label) => headRow.appendChild(element("th", "", label)),
  );
  head.appendChild(headRow);
  table.appendChild(head);
  const body = document.createElement("tbody");
  server.gpus.forEach((gpu) => body.appendChild(renderGpuRow(gpu)));
  table.appendChild(body);
  wrap.appendChild(table);
  section.appendChild(wrap);
  return section;
}

function renderGpuServers(snapshot) {
  const servers = snapshot.servers.map((server) => ({
    ...server,
    gpus: server.gpus.map((gpu) => ({
      ...gpu,
      processes: gpu.processes.map((process) => {
        const cached = state.processEtas.get(`${server.id}:${process.pid}`);
        return !server.stale && !gpu.error && !gpu.error_code
          && cached && process.start_time_ticks != null
          && cached.start_time_ticks === process.start_time_ticks
          ? { ...process, eta: cached.eta } : process;
      }),
    })),
  }));
  byId("server-list").replaceChildren(...servers.map(renderServer));
}

function renderProcessEtas(payload) {
  state.processEtas = new Map(payload.servers.flatMap((server) =>
    server.processes.map((process) => [`${server.id}:${process.pid}`, process])));
  byId("eta-update").textContent = `估时检查 ${formatDate(payload.updated_at)} · 日志采样时间见悬停说明`;
  if (state.snapshot) renderGpuServers(state.snapshot);
}

function renderSnapshot(snapshot) {
  state.snapshot = snapshot;
  renderSummary(snapshot.summary);
  byId("snapshot-time").textContent = `采集时间 ${formatDate(snapshot.generated_at)}`;
  renderGpuServers(snapshot);
  const messages = snapshot.servers.length
    ? snapshot.servers.map(serverProblem).filter(Boolean)
    : [{ tone: "warning", overview: "尚未接入服务器。请先运行 gpuops server add SSH_ALIAS。" }];
  const errorArea = byId("overview-errors");
  errorArea.replaceChildren(...messages.map((message) => element(
    "div",
    message.tone === "warning" ? "warning-banner" : "error-banner",
    message.overview,
  )));
}

function queueRow(item, mode) {
  const row = element("tr", "queue-row");
  row.tabIndex = 0;
  row.dataset.jobId = item.id;
  row.setAttribute("role", "button");
  row.setAttribute("aria-label", `查看作业 ${item.name}`);
  if (state.selectedJobId === item.id) row.classList.add("is-selected");
  const visibleState = item.raw_state === "cancel_requested" ? "cancel_requested" : item.state;
  row.appendChild(valueCell("状态", statusPill(visibleState)));

  const jobCell = valueCell("作业", "", "queue-name-cell");
  jobCell.appendChild(element("strong", "", item.name));
  jobCell.appendChild(element("span", "subdued mono", item.id));
  row.appendChild(jobCell);

  const placement = item.gpus?.length
    ? `${item.server} / GPU ${item.gpus.join(", ")}`
    : `${item.server} / 待分配`;
  row.appendChild(valueCell("服务器 / GPU", placement));
  if (mode === "history") {
    row.appendChild(valueCell("项目", item.project, "history-project-cell"));
    row.appendChild(valueCell("优先级", item.priority));
    row.appendChild(valueCell("完成时间", formatDate(item.finished_at)));
  } else {
    row.appendChild(valueCell("优先级", item.priority));
    row.appendChild(valueCell("开始时间", item.estimated_start
      ? formatDate(item.estimated_start) : "等待调度"));
    const end = valueCell("预计结束", formatJobEta(item));
    end.title = processEtaDetails(item);
    row.appendChild(end);
  }

  const select = () => selectJob(item.id, { autoRefresh: mode === "active" });
  row.addEventListener("click", select);
  row.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      select();
    }
  });
  return row;
}

function queueServerErrors(servers, mode) {
  return (servers || [])
    .filter((server) => !server.online)
    .map((server) => {
      const reason = `${server.id} Worker 不可用：${server.error || "未知错误"}`;
      if (mode === "history") {
        return `${reason}；当前历史记录不完整，恢复后请刷新再翻页`;
      }
      return reason;
    });
}

function renderQueueHead(mode) {
  const labels = mode === "history"
    ? ["状态", "作业", "服务器 / GPU", "项目", "优先级", "完成时间"]
    : ["状态", "作业", "服务器 / GPU", "优先级", "开始时间", "预计结束"];
  byId("queue-table").classList.toggle("is-history", mode === "history");
  byId("queue-head-row").replaceChildren(...labels.map((label) => element("th", "", label)));
}

function renderQueuePage(queue, mode) {
  const items = queue.items || [];
  renderQueueHead(mode);
  byId("queue-time").textContent = mode === "history"
    ? `历史更新时间 ${formatDate(queue.updated_at)} · 第 ${state.historyPreviousCursors.length + 1} 页`
    : `更新时间 ${formatDate(queue.updated_at)}`;
  const errors = queueServerErrors(queue.servers, mode);
  byId("queue-errors").replaceChildren(
    ...errors.map((message) => element("div", "error-banner", message)),
  );
  const body = byId("queue-body");
  if (!items.length) {
    const row = document.createElement("tr");
    const emptyText = mode === "history"
      ? (state.historyProject ? "该项目暂无历史作业" : "暂无历史作业")
      : "当前没有排队或运行中的作业";
    const cell = valueCell("", emptyText, "empty-state");
    cell.colSpan = 6;
    row.appendChild(cell);
    body.replaceChildren(row);
    return;
  }
  body.replaceChildren(...items.map((item) => queueRow(item, mode)));
}

function renderQueue(queue) {
  state.activeQueue = queue;
  byId("queue-count").textContent = (queue.items || []).length;
  if (state.queueMode === "active") renderQueuePage(queue, "active");
}

function clearJobSelection() {
  state.selectedJobId = null;
  state.selectedJobAutoRefresh = false;
  state.detailRequestId += 1;
  state.pendingDetailJobId = null;
  byId("job-detail").replaceChildren(
    element("div", "empty-state", "选择一个作业查看命令、进度和日志。"),
  );
}

function historyUrl(project, cursor) {
  let url = `/api/v1/history?limit=${HISTORY_PAGE_SIZE}`;
  if (project) url += `&project=${encodeURIComponent(project)}`;
  if (cursor) url += `&cursor=${encodeURIComponent(cursor)}`;
  return url;
}

function renderHistoryProjectOptions(projects) {
  const select = byId("history-project");
  const values = [...new Set(projects || [])];
  if (state.historyProject && !values.includes(state.historyProject)) {
    values.unshift(state.historyProject);
  }
  const all = element("option", "", "全部项目");
  all.value = "";
  const options = values.map((project) => {
    const option = element("option", "", project);
    option.value = project;
    return option;
  });
  select.replaceChildren(all, ...options);
  select.value = state.historyProject;
}

function updateHistoryControls() {
  const history = state.historyPage;
  byId("history-previous").disabled = state.historyLoading
    || state.historyPreviousCursors.length === 0;
  byId("history-next").disabled = state.historyLoading || !history?.next_cursor;
  byId("history-project").disabled = state.historyLoading;
  byId("history-page-label").textContent = `第 ${state.historyPreviousCursors.length + 1} 页`;
}

function renderHistoryLoading() {
  renderQueueHead("history");
  byId("queue-time").textContent = "正在读取历史记录...";
  byId("queue-errors").replaceChildren();
  const row = document.createElement("tr");
  const cell = valueCell("", "正在读取历史记录...", "loading-row");
  cell.colSpan = 6;
  row.appendChild(cell);
  byId("queue-body").replaceChildren(row);
}

async function loadHistory({
  cursor = null,
  previousCursors = [],
  resetSelection = true,
  preservePageOnError = true,
} = {}) {
  const requestId = ++state.historyRequestId;
  const previousPage = preservePageOnError ? state.historyPage : null;
  state.historyLoading = true;
  if (resetSelection) clearJobSelection();
  renderHistoryLoading();
  updateHistoryControls();
  try {
    const history = await fetchJson(historyUrl(state.historyProject, cursor));
    if (requestId !== state.historyRequestId || state.queueMode !== "history") return;
    state.historyPage = history;
    state.historyCursor = cursor;
    state.historyPreviousCursors = previousCursors;
    renderHistoryProjectOptions(history.projects);
    renderQueuePage(history, "history");
  } catch (error) {
    if (requestId !== state.historyRequestId || state.queueMode !== "history") return;
    if (previousPage) renderQueuePage(previousPage, "history");
    else {
      state.historyPage = null;
      state.historyCursor = null;
      state.historyPreviousCursors = [];
      const row = document.createElement("tr");
      const cell = valueCell("", "历史记录暂时不可用", "empty-state");
      cell.colSpan = 6;
      row.appendChild(cell);
      byId("queue-body").replaceChildren(row);
    }
    byId("queue-errors").replaceChildren(
      element("div", "error-banner", `无法读取历史记录：${error.message}`),
    );
  } finally {
    if (requestId === state.historyRequestId) {
      state.historyLoading = false;
      updateHistoryControls();
    }
  }
}

async function setQueueMode(mode) {
  if (mode !== "active" && mode !== "history") return;
  if (state.queueMode !== mode) clearJobSelection();
  state.queueMode = mode;
  document.querySelectorAll(".queue-mode-button").forEach((button) => {
    const selected = button.dataset.queueMode === mode;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-selected", String(selected));
  });
  byId("history-controls").hidden = mode !== "history";
  if (mode === "history") {
    state.historyProject = byId("history-project").value || "";
    await loadHistory();
  } else {
    state.historyRequestId += 1;
    state.historyLoading = false;
    if (state.activeQueue) renderQueuePage(state.activeQueue, "active");
  }
}

async function changeHistoryProject() {
  state.historyProject = byId("history-project").value || "";
  await loadHistory({ preservePageOnError: false });
}

async function showNextHistoryPage() {
  const nextCursor = state.historyPage?.next_cursor;
  if (!nextCursor || state.historyLoading) return;
  await loadHistory({
    cursor: nextCursor,
    previousCursors: [...state.historyPreviousCursors, state.historyCursor],
  });
}

async function showPreviousHistoryPage() {
  if (!state.historyPreviousCursors.length || state.historyLoading) return;
  const previousCursors = state.historyPreviousCursors.slice(0, -1);
  await loadHistory({
    cursor: state.historyPreviousCursors[state.historyPreviousCursors.length - 1],
    previousCursors,
  });
}

function detailLine(label, value, className) {
  const wrapper = element("div", "detail-line");
  wrapper.appendChild(element("dt", "", label));
  wrapper.appendChild(element("dd", className || "", value ?? "-"));
  return wrapper;
}

function formatJobEta(job) {
  if (["queued", "leased", "starting"].includes(job.raw_state)) return "等待调度";
  if (job.raw_state === "cancel_requested") return "取消中";
  return job.eta ? formatProcessEta(job) : formatDate(job.estimated_end);
}

function renderJob(job) {
  if (state.selectedJobId === job.id) {
    state.selectedJobAutoRefresh = DETAIL_POLLING_STATES.has(job.raw_state);
  }
  if (!CANCELLABLE_STATES.has(job.raw_state)) {
    state.cancellingJobIds.delete(job.id);
    state.cancellationErrors.delete(job.id);
  }
  const detail = byId("job-detail");
  const header = element("div", "job-detail-header");
  const titleBlock = document.createElement("div");
  titleBlock.appendChild(element("h3", "", job.name));
  titleBlock.appendChild(element("span", "subdued mono", job.id));
  header.appendChild(titleBlock);
  const actions = element("div", "job-actions");
  actions.appendChild(statusPill(job.raw_state === "cancel_requested" ? "cancel_requested" : job.state));
  if (CANCELLABLE_STATES.has(job.raw_state)) {
    const cancelling = state.cancellingJobIds.has(job.id);
    const cancelButton = element("button", "danger-button", cancelling ? "正在取消..." : "取消作业");
    cancelButton.type = "button";
    cancelButton.disabled = cancelling;
    cancelButton.addEventListener("click", () => cancelJob(job, cancelButton));
    actions.appendChild(cancelButton);
  }
  header.appendChild(actions);

  const definitions = element("dl", "job-definitions");
  definitions.appendChild(detailLine("项目", job.project));
  definitions.appendChild(detailLine("服务器", job.server));
  definitions.appendChild(
    detailLine("GPU", job.gpus?.length ? job.gpus.join(", ") : "待分配"),
  );
  definitions.appendChild(detailLine("工作目录", job.working_directory, "mono wrap-anywhere"));
  definitions.appendChild(detailLine("命令", job.command, "mono wrap-anywhere"));
  definitions.appendChild(job.finished_at
    ? detailLine("完成时间", formatDate(job.finished_at))
    : detailLine(job.eta?.scope === "phase" ? "阶段预计结束" : "预计结束", formatJobEta(job)));
  if (!job.finished_at && job.eta) {
    const progress = formatStructuredProgress(job.eta);
    if (progress) definitions.appendChild(detailLine("阶段进度", progress));
    const source = job.eta.format === "gpuops_progress"
      ? "GPUOps 结构化日志"
      : { logs: "日志进度", submitted: "提交时长" }[job.eta.source_kind];
    if (source) definitions.appendChild(detailLine("估计来源", source));
    definitions.appendChild(detailLine("估计依据", processEtaDetails(job), "wrap-anywhere"));
  }
  if (job.explanation?.message) {
    definitions.appendChild(detailLine(
      job.explanation.waiting ? "等待原因" : "状态说明",
      job.explanation.message,
    ));
  }
  if (job.explanation?.next_action) {
    const nextAction = NEXT_ACTION_LABELS[job.explanation.next_action] || job.explanation.next_action;
    definitions.appendChild(detailLine("建议操作", nextAction));
  }
  if (job.failure_reason && !job.explanation?.details?.failure_reason) {
    definitions.appendChild(detailLine("失败原因", job.failure_reason));
  }

  const logHeading = element("div", "log-heading");
  logHeading.appendChild(element("h4", "", "最近日志"));
  logHeading.appendChild(element("span", "subdued", `${job.logs?.length || 0} 行`));
  const log = element("pre", "log-output");
  log.textContent = job.logs?.join("\n") || "暂无日志";

  detail.replaceChildren(header, definitions, logHeading, log);
  if (state.cancellationErrors.has(job.id)) {
    detail.prepend(element("div", "error-banner cancel-error", state.cancellationErrors.get(job.id)));
  }
  if (job.log_error) {
    detail.appendChild(element("div", "error-banner", `无法读取日志：${job.log_error}`));
  }
  if (job.detail_error) {
    detail.appendChild(element("div", "error-banner", `无法刷新作业说明：${job.detail_error}`));
  }
}

async function cancelJob(job, button) {
  if (state.cancellingJobIds.has(job.id)) return;
  if (!window.confirm(`确认取消作业 ${job.name}？`)) return;
  state.cancellingJobIds.add(job.id);
  state.cancellationErrors.delete(job.id);
  byId("job-detail").querySelector(".cancel-error")?.remove();
  button.disabled = true;
  button.textContent = "正在取消...";
  let accepted = false;
  try {
    const response = await fetch(`/api/v1/jobs/${encodeURIComponent(job.id)}/cancel`, {
      method: "POST",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `HTTP ${response.status}`);
    }
    accepted = true;
    if (state.selectedJobId === job.id) {
      // Discard any detail response that started before cancellation was accepted.
      state.detailRequestId += 1;
      state.pendingDetailJobId = null;
    }
    await refresh({ forceDetails: true });
  } catch (error) {
    const errorText = `无法取消作业：${error.message}`;
    state.cancellationErrors.set(job.id, errorText);
    const message = element("div", "error-banner cancel-error", errorText);
    if (state.selectedJobId === job.id) byId("job-detail").prepend(message);
  } finally {
    // An accepted request stays disabled until a detail response confirms the
    // state change, even if the overview refresh finishes first.
    if (!accepted) state.cancellingJobIds.delete(job.id);
    if (!accepted && state.selectedJobId === job.id) {
      const currentButton = byId("job-detail").querySelector(".danger-button");
      if (currentButton) {
        currentButton.disabled = false;
        currentButton.textContent = "取消作业";
      }
    }
  }
}

async function selectJob(jobId, { autoRefresh = true } = {}) {
  state.selectedJobId = jobId;
  state.selectedJobAutoRefresh = autoRefresh;
  document.querySelectorAll(".queue-row").forEach((row) => {
    row.classList.toggle("is-selected", row.dataset.jobId === jobId);
  });
  await refreshSelectedJob({ showLoading: true });
}

async function refreshSelectedJob({ showLoading = false } = {}) {
  const jobId = state.selectedJobId;
  if (!jobId) return;
  if (!showLoading && state.pendingDetailJobId === jobId) return;
  const requestId = ++state.detailRequestId;
  state.pendingDetailJobId = jobId;
  const detail = byId("job-detail");
  if (showLoading) {
    detail.replaceChildren(element("div", "loading-row", "正在读取作业详情..."));
  }
  try {
    const job = await fetchJson(`/api/v1/jobs/${encodeURIComponent(jobId)}`);
    if (state.selectedJobId === jobId && requestId === state.detailRequestId) renderJob(job);
  } catch (error) {
    if (state.selectedJobId === jobId && requestId === state.detailRequestId) {
      const warning = element("div", "error-banner detail-error", `无法读取作业详情：${error.message}`);
      if (showLoading) detail.replaceChildren(warning);
      else {
        detail.querySelector(".detail-error")?.remove();
        detail.prepend(warning);
      }
    }
  } finally {
    if (requestId === state.detailRequestId) state.pendingDetailJobId = null;
  }
}

async function fetchJson(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

function updateConnection() {
  const polls = Object.values(state.polls);
  const failed = polls.filter((poll) => poll.error);
  if (failed.length) {
    const offline = ["snapshot", "queue", "health"].every((key) => state.polls[key]?.error);
    setConnection(offline ? "offline" : "degraded", offline ? "连接失败" : "部分数据异常");
  } else if (state.polls.health?.unconfigured) {
    setConnection("degraded", "未接入服务器");
  } else if (polls.some((poll) => poll.degraded)) {
    setConnection("degraded", "部分数据异常");
  } else if (state.polls.snapshot?.succeeded && state.polls.queue?.succeeded
      && state.polls.health?.succeeded) {
    setConnection("online", "已连接");
  }
  byId("refresh-button").classList.toggle("is-spinning", polls.some((poll) => poll.pending));
}

async function refreshStream(key, url, render, errorId, label) {
  const poll = state.polls[key] ||= { pending: false, succeeded: false };
  if (poll.pending) return;
  poll.pending = true;
  updateConnection();
  try {
    const payload = await fetchJson(url);
    render(payload);
    poll.succeeded = true;
    poll.error = null;
    poll.unconfigured = key === "health" && payload.reason === "no_servers_registered";
    poll.degraded = (key === "health" && payload.status !== "ok")
      || payload.servers?.some((server) => server.online === false
        || ["offline", "degraded"].includes(server.status));
    if (errorId) byId(errorId).replaceChildren();
    byId("last-refresh").textContent = `最近收到数据 ${new Date().toLocaleTimeString("zh-CN", { hour12: false })}`;
  } catch (error) {
    poll.error = error.message;
    const message = `${label}刷新失败：${error.message}；${poll.succeeded ? "保留上次数据，等待重试" : "等待重试"}`;
    if (errorId) byId(errorId).replaceChildren(element("div", "error-banner", message));
    if (key === "eta") byId("eta-update").textContent = message;
  } finally {
    poll.pending = false;
    updateConnection();
  }
}

async function refresh({ manual = false, forceDetails = false } = {}) {
  if (manual || forceDetails || state.selectedJobAutoRefresh) refreshSelectedJob();
  if (manual && state.queueMode === "history") {
    loadHistory({
      cursor: state.historyCursor,
      previousCursors: state.historyPreviousCursors,
      resetSelection: false,
    });
  }
  // Each stream renders as soon as it returns and has its own overlap guard.
  await Promise.all([
    refreshStream("snapshot", "/api/v1/snapshot", renderSnapshot, "snapshot-refresh-error", "GPU"),
    refreshStream("queue", "/api/v1/queue", renderQueue, "queue-refresh-error", "当前队列"),
    refreshStream("health", "/healthz", () => {}, null, "健康检查"),
    refreshStream("eta", "/api/v1/process-etas", renderProcessEtas, null, "日志估时"),
  ]);
}

function setView(viewName) {
  document.querySelectorAll(".tab-button").forEach((button) => {
    const selected = button.dataset.view === viewName;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-selected", String(selected));
  });
  document.querySelectorAll(".view").forEach((view) => {
    const selected = view.id === `view-${viewName}`;
    view.classList.toggle("is-active", selected);
    view.hidden = !selected;
  });
}

document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => setView(button.dataset.view));
});
document.querySelectorAll(".queue-mode-button").forEach((button) => {
  button.addEventListener("click", () => setQueueMode(button.dataset.queueMode));
});
byId("history-project").addEventListener("change", changeHistoryProject);
byId("history-previous").addEventListener("click", showPreviousHistoryPage);
byId("history-next").addEventListener("click", showNextHistoryPage);
byId("refresh-button").addEventListener("click", () => refresh({ manual: true }));

const pollInterval = Number(document.body.dataset.pollInterval) || 5000;
refresh();
window.setInterval(refresh, pollInterval);
