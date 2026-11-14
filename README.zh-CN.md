# GPUOps

[English](README.md) | [简体中文](README.zh-CN.md)

GPUOps 是连接人类、终端 Agent 与 GPU 服务器的轻量任务中间层。它通过已有的 SSH 连接管理 Linux NVIDIA 服务器，为实验提供 GPU 监控、持久化队列、自动选卡、任务状态、日志查询和取消能力，并配有可选的 Web 控制台。

人类和 Agent 使用同一套 CLI。每台服务器只需在当前用户下运行一个 Worker，无需开放监听端口；任务被接收后，即使控制端离线，服务器也会继续调度和执行。

GPUOps 负责运行已经准备好的实验命令。实验代码、运行环境、数据集、指标分析和 checkpoint 管理由实验项目负责。目前支持 macOS、Linux 等 POSIX 控制端，以及 Linux NVIDIA 服务器；暂不支持 Windows 控制端和非 NVIDIA 加速器。

## 安装

0.4.0 发布前请使用下方的源码安装方式；下载链接用于正式发布后。

控制端需要 Python 3.10+ 和 OpenSSH（`ssh`、`scp`）。使用 [uv](https://docs.astral.sh/uv/getting-started/installation/) 安装 [v0.4.0](https://github.com/OHHHHHHei/GPUOps/releases/tag/v0.4.0)，包含可选的 Web 控制台：

```bash
uv tool install 'gpuops[web] @ https://github.com/OHHHHHHei/GPUOps/releases/download/v0.4.0/gpuops-0.4.0-py3-none-any.whl'
gpuops version --json
```

安装完成后，可以在任意目录运行 `gpuops`。如果 uv 提示可执行文件目录不在 `PATH` 中，运行 `uv tool update-shell`，然后重新打开终端。

也可以从 Release 下载 wheel，安装到自己的虚拟环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install './gpuops-0.4.0-py3-none-any.whl[web]'
.venv/bin/gpuops version --json
```

GPUOps 目前通过 GitHub Releases 分发，尚未发布到 PyPI。安装好的客户端可以直接部署 Worker，不要求控制端或服务器额外安装 uv。

如需从源码开发：

```bash
git clone https://github.com/OHHHHHHei/GPUOps.git
cd GPUOps
uv sync --all-extras --frozen
uv run gpuops version --json
```

使用源码环境时，在下文的 `gpuops` 命令前加上 `uv run`。

## 接入服务器

服务器需要具备以下条件：

- Linux、NVIDIA 驱动和 `nvidia-smi`。
- Python 3.10+，以及可用的 `venv`。
- 可访问的 Python 依赖源或本地依赖缓存。
- 用户主目录有足够的空间配额，供 GPUOps 在 1 GiB 的自有数据预算内安装和运行。
- 当前账户已经可以通过 SSH 登录。

假设你在 OpenSSH 配置中有一个可用的别名 `gpu-server`：

```bash
ssh gpu-server
```

确认登录正常后，回到控制端执行：

```bash
gpuops server add gpu-server --json
gpuops server list --json
gpuops status --json
```

`server add` 会复用版本相同且已就绪的安装、启动已停止的 Worker；服务器没有安装或版本不同时，从本地 GPUOps 安装包部署。只有确认 Worker 就绪后，才会写入控制端的服务器注册表。

新服务器默认以 SSH 别名作为 ID；已有 GPUOps 安装会保留原有身份。任务清单中应使用返回的 `server.id`。首次安装的注册表为空，不包含预设服务器。接入操作不会复制实验代码、环境或数据，也不会自动安装 Agent skill。

`status` 分别报告 GPU 采样状态和 `worker.ready`。即使 GPU 采样暂时不可用，已就绪的 Worker 仍可接收指定服务器的任务；任务会留在队列中，等待可靠的 GPU 状态后再启动。仅凭监控命令退出码为零，不能判断调度是否就绪。

## 提交实验

先在目标服务器上准备代码、运行环境和数据，再创建任务清单，例如 `job.json`。将命令、工作目录、服务器 ID 和显存需求替换为实验的实际值：

```json
{
  "api_version": "gpuops/v1",
  "id": "baseline-seed-42",
  "project": "example",
  "command": ["python", "train.py", "--seed", "42"],
  "working_directory": "/srv/experiments/example",
  "resources": {
    "host": "gpu-server",
    "gpu_count": 1,
    "memory_mib": 18000
  }
}
```

提交并查询：

```bash
gpuops submit job.json --json
gpuops job RUN_ID --json
gpuops logs RUN_ID --tail-bytes 65536 --json
gpuops queue --json
gpuops history --json
```

保存返回的 `run_id`，并确认 `submission_state` 为 `"accepted"`。后续命令中的 `RUN_ID` 都替换为这个值。提交时会自动验证任务清单；若只想检查而不提交，可以运行 `gpuops validate job.json --json`。

日常使用时省略 `resources.gpus`，Worker 会按指定服务器和 GPU 数量自动选择符合条件的物理显卡。只有明确需要固定卡号时，才设置 `gpus`。`memory_mib` 表示每张卡所需的空闲显存，不会限制实验运行时的实际显存使用量。

`host: "auto"` 还会自动选择服务器。使用前，应确认所有候选服务器都在允许使用的范围内，并已准备好实验所需的代码、环境和数据。服务器选择在提交前确定，恢复提交时会沿用同一目标。

如果提交超时或响应不明确，先恢复已保存的请求：

```bash
gpuops reconcile RUN_ID --json
# 如果 CLI 退出前没有显示 Run ID，先查看待确认的请求：
gpuops reconcile --list --json
```

每次不带 `--run-id` 的新 `submit` 都会创建一次独立实验。对于结果不确定的提交，使用 `reconcile` 保留原请求身份，避免重复运行。取消受管任务时，执行 `gpuops cancel RUN_ID --reason "superseded" --json`，随后继续查询，直到任务进入终态。

## 供 Agent 使用

任何能调用终端并解析 JSON 的 Agent 都可以使用 GPUOps。将 [Agent 使用指南](docs/AGENT_GUIDE.md) 和 [CLI 接口约定](docs/CLI_CONTRACT.md) 提供给它即可，不依赖特定 Agent 厂商、SDK 或 MCP 服务。

Codex 用户可以选择在当前机器上安装随包提供的 skill：

```bash
gpuops skill install --json
```

也可以安装到已注册服务器的 SSH 账户中：

```bash
gpuops skill install --server gpu-server --json
```

默认安装位置为目标账户下的 `~/.codex/skills/gpu-infra`。如果该账户的命令环境设置了 `CODEX_HOME`，则使用 `$CODEX_HOME/skills/gpu-infra`。远端 skill 安装是可选操作，使用前需先注册服务器。

Agent 也可以直接部署在服务器上。只要与 Worker 使用同一个 Unix 账户，就能从任意实验目录调用 `~/.local/bin/gpuops`；访问当前服务器时直接本地执行，无需源码目录，也无需再 SSH 回自己。

旧版服务器 CLI 可能没有 `version` 或 `server list` 命令，但仍可使用原有的 `status`、`queue` 和 `submit`。版本查询及安装维护可从控制端完成。

## Web 控制台

```bash
gpuops web
```

打开 [http://127.0.0.1:8787](http://127.0.0.1:8787)，即可查看物理 GPU 编号、进程归属、当前任务、分页历史、任务状态说明和有长度上限的日志。运行 `gpuops web --demo` 可以查看固定的演示数据。

Web 服务只允许绑定回环地址；如需远程访问，请通过可信的隧道连接。

结束时间估计（ETA）是可选展示能力。可读取的训练日志可以提供阶段或训练进度，也可以用 `schedule.expected_duration` 为受管任务提供手动估计。对于新编写的训练代码，[结构化进度协议](docs/PROGRESS.md) 提供了简短的标准输出格式，以及不依赖额外包的示例输出函数。

日志缺失、过期或信息不足时，ETA 保持未知。估计时间不会预留 GPU，也不参与调度决策。

## 任务结束邮件

每台服务器只需配置一次 SMTP，即可在受管任务成功、失败或取消时收到邮件。邮件由服务器上的 Worker 发送，控制端离线也能收到，已有任务 JSON 无需修改。

```bash
gpuops notify configure smtp.json --server gpu-server --json
gpuops notify test --server gpu-server --json
```

SMTP 配置示例、状态查询和关闭方法见[任务结束邮件文档](docs/NOTIFICATIONS.md)。

如果某个任务只希望在后续分析完成后发人类邮件，可先运行
`gpuops submit job.json --email off --json`，再在后续登记中选择
`analysis_complete`。该单任务覆盖要求远端 Worker 已支持邮件策略能力；旧版
Worker 会在入队前拒绝，需要先升级服务器上的 GPUOps。

## 实验组后续动作（实验性）

原有单任务 Codex 命令仍可使用：

```bash
gpuops followup add RUN_ID --action '读取已保存的指标，更新研究记录，并在当前聊天汇报。' --json
```

跨服务器任务组可编辑
[`examples/followup.example.json`](examples/followup.example.json)，然后运行：

```bash
gpuops followup register examples/followup.example.json --json
```

`per_job` 在每个成员成功、失败或取消时分别产生稳定事件；`all_terminal`
等待全部成员进入终态，再产生一个聚合事件。Harness 接续与人类邮件（`none`、
`terminal`、`analysis_complete`）相互独立。原 Agent 提供动作和上下文，GPUOps
只负责观察与路由，不负责形成研究判断。

单任务 Codex 闭环已通过真实验收；新的任务组和适配器路径目前只完成本地测试，
尚未重新部署做在线验收。Codex 自动接续仍要求原聊天已在桌面中加载。通用外部
适配器只是 JSON/回执接口，不代表已原生支持 Claude Code 或 DeepSeek 会话。
协议、显式接收/完成 ACK、邮件配置和适用边界见[后续动作文档](docs/FOLLOWUPS.md)。

## Mac 上的 Worker 停止提醒

Mac 每分钟检查已注册服务器的 Worker。确认停止后通知一次，恢复就绪后再通知一次；SSH 暂时不通只记为状态未知，不发送停止告警。Web 关闭后仍会检查。

启用桌面通知，并复用任务结束邮件的 SMTP 配置来接收邮件：

```bash
gpuops alerts enable --email-config smtp.json
gpuops alerts test --json
gpuops alerts status --json
```

桌面通知和邮件分别记录投递结果，失败的渠道在后续检查中重试；如果尚未发出停止邮件就已恢复，不再补发过期告警。`alerts test` 的 `email_accepted: true` 表示邮件服务器已接收，还需在邮箱确认到达。

只需桌面通知时运行 `gpuops alerts enable --no-email`；`gpuops alerts disable` 关闭两种提醒，保留邮箱配置供下次启用。Mac 必须保持唤醒并能访问服务器和邮箱服务。该功能不重启远端 Worker；Worker 停止也不代表已启动的实验停止。

## 运行边界与维护

GPUOps 在每台服务器的同一个可信 Unix 账户内协调任务。Worker 会避开可见的外部 GPU 进程，包括其他用户的任务；资源租约只协调该账户下的 GPUOps 作业，无法阻止其他用户或绕过 GPUOps 的程序占用显卡。取消操作只会向身份确认过的受管进程发送信号。

分配前会通过两次采样确认 GPU 空闲，并按设备 UUID 启动实验。故障设备会自动重试检测，恢复后重新参与分配。GPUOps 无法修复 NVIDIA 驱动，也无法隔离驱动内部共享锁造成的阻塞。GPU 状态统一来自 Worker 的共享采样，重复查询不会额外启动探测。

Worker 不需要 Docker、`sudo`、systemd 或服务器端 HTTP 服务。目前服务器重启后仍需显式运行 `gpuops fleet start --json`；停止 Worker 会保留实验监督进程继续运行。

每台服务器上，GPUOps 自有的安装文件、缓存、持久化状态和日志采用合计 1 GiB 的预算。实验输出和数据集由实验项目自行管理。

维护时使用 `gpuops deploy status|install|rollback|uninstall` 和 `gpuops fleet start|stop|status`。例如，升级已注册服务器：

```bash
gpuops deploy install --server gpu-server --json
gpuops deploy status --server gpu-server --json
```

受管任务全部结束后，可移除服务器：

```bash
gpuops server remove gpu-server --json
```

移除操作会停止 Worker 并删除本地注册信息，保留远端安装、历史记录和实验数据。存在活动任务时会拒绝移除，不会等待任务自动结束。部署和回滚的详细说明见 [远端 Worker 文档](docs/REMOTE_WORKER.md)。

## 文档与开发

以下深入文档目前主要使用英文：

- [Agent 使用指南](docs/AGENT_GUIDE.md)与 [CLI 接口约定](docs/CLI_CONTRACT.md)
- [项目架构](docs/ARCHITECTURE.md)与 [远端 Worker 运维](docs/REMOTE_WORKER.md)
- [当前状态、路线图与验收流程](docs/ROADMAP.md)
- [发布验收记录](docs/VALIDATION.md)：CI 覆盖、真实服务器检查及验证范围
- [贡献指南](CONTRIBUTING.md)与[更新日志](CHANGELOG.md)

```bash
uv sync --all-extras
uv run pytest
uv run ruff check .
node --test tests/test_web_polling.cjs
uv build --wheel
```

前端轮询测试需要 Node.js 22+。Linux 进程测试使用模拟 GPU 数据，无需物理 GPU。本地测试通过不等于真实 SSH 部署或 CUDA 验收通过，两者分别验证。

GPUOps 采用 [MIT 许可证](LICENSE)。
