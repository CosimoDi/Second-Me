# 项目复现指南

## 1. 基础复现步骤
1. **下载项目**: 下载并解压，若 `git clone` 失败则手动下载。
2. **环境准备**: 安装 Docker，并配置好 WSL (Windows Subsystem for Linux)。
3. **启动服务**: 在项目根目录下打开终端，运行 `make docker-up`。
4. **访问应用**: 浏览器访问 `localhost:3000`。

如果哪一步搞错了以至于要全重新搞，build这一步就需要Building 2597.4s，后面模型训练差不多也是这个时间，还是要做好版本管理。

## 2. WSL 代理配置 (关键)

由于 WSL 2 默认不支持直接使用宿主机的 localhost 代理，且 IP 地址可能会变动，本项目提供了自动化脚本以便快速配置。

### 脚本位置
脚本位于 `scripts/setup_wsl_proxy.sh`。

### 如何配置代理 (初次使用 & IP/端口变更)
当你连接新的 WiFi 或 IP 地址发生变化时，请按照以下步骤更新 WSL 代理：

**方法 1：使用配置脚本 (推荐)**
在 PowerShell 中运行以下命令（替换为你当前的宿主机 IP 和代理端口）：

```powershell
# 语法: wsl bash scripts/setup_wsl_proxy.sh [宿主机IP] [代理端口]
wsl bash scripts/setup_wsl_proxy.sh 192.168.2.101 16780
```

**方法 2：手动修改**
1. 进入 WSL: `wsl`
2. 编辑代理文件: `nano ~/.bash_proxy`
3. 修改 IP 和端口
4. 生效配置: `source ~/.bashrc`

### Docker 代理问题 (关键：构建报错)
**现象**: 运行 `make docker-up` 时出现 `failed to solve... failed to fetch oauth token... dial tcp ... connectex` 错误。

**原因**: Docker CLI 构建进程没有复用 Docker Desktop 的全局代理设置，或者通过 WSL 调用时环境变量未正确传递。

**✅ 解决方案 (立即生效)**:
不要只修改配置文件，直接在当前 PowerShell 窗口设置环境变量，强制 Docker 走代理：

```powershell
# 1. 设置代理 (关键步骤！替换为你的实际 IP 和端口)
$env:HTTP_PROXY="http://192.168.2.101:16780"
$env:HTTPS_PROXY="http://192.168.2.101:16780"

# 2. 再次尝试构建
make docker-up
```

**❌ 不推荐的做法**:
*   仅修改 `settings-store.json` 或 Docker Desktop GUI 设置：通常需要重启 Docker 进程才能生效，且对命令行构建可能无效。
*   查看 `Local\Docker\log`：效率较低，网络连接问题通常很直接。

## 3. 模型服务配置 (Ollama)

由于无法获取 OpenAI API，使用自定义模型 (Custom Model) 进行训练。

**配置信息:**
*   **Chat**: `qwen3:8b` -> `http://192.168.2.101:11434/v1` (替换为实际IP)
*   **Embedding**: `qwen3-embedding:latest` -> `http://192.168.2.101:11434/v1`
*   **Thinking**: `http://192.168.2.101:11434/v1`

### 启动 Ollama (PowerShell 脚本)
**注意**: 运行过程最好不要改变 IP 地址。建议另存为 `start_ollama.ps1` 运行。

```powershell
# 1. 终止残留进程
taskkill /F /IM ollama.exe 2>&1 | Out-Null
Start-Sleep -Seconds 3

# 2. 清空旧环境变量（先检查是否存在，避免删除报错）
if (Test-Path Env:OLLAMA_CUDA) { Remove-Item Env:OLLAMA_CUDA }
if (Test-Path Env:OLLAMA_MAX_GPU) { Remove-Item Env:OLLAMA_MAX_GPU }

# 3. 适配 RTX 4060 8G 显存的 GPU 配置
$env:OLLAMA_OPENAI_COMPATIBLE=1
$env:OLLAMA_GPU=90
$env:OLLAMA_OFFLOAD_ALL=1
$env:OLLAMA_HOST="0.0.0.0:11434"
$env:OLLAMA_NUM_GPU=1

# 4. 启动 Ollama 并在控制台实时输出日志
Write-Host "🔄 正在启动Ollama服务...（控制台实时输出日志）" -ForegroundColor Cyan
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" serve 2>&1 | ForEach-Object {
    if ($_ -notmatch "failed to get console mode for stderr") {
        if ($_ -match "level=INFO") { Write-Host $_ -ForegroundColor White }
        elseif ($_ -match "level=WARN") { Write-Host $_ -ForegroundColor Yellow }
        elseif ($_ -match "level=ERROR") { Write-Host $_ -ForegroundColor Red }
        else { Write-Host $_ -ForegroundColor Blue }
    }
}
```

### 验证模型服务
修改 endpoint 为本机 IP，新建 PowerShell 窗口测试：
```powershell
Invoke-RestMethod -Uri "http://192.168.2.101:11434/v1/models" -Method Get
```

## 4. 训练流程常见问题 (Troubleshooting)

### API 选择建议
*   **首选**: 硅基流动 (SiliconFlow) - 稳定，省去本地折腾。
*   **次选**: Ollama (本地自定义模型) - 需要较好的硬件配置。

### 训练卡顿与报错解答
| 阶段 | 常见现象 | 解决方案 |
| :--- | :--- | :--- |
| **登录/启动** | 报错 500 | 可能是之前训练中断导致。尝试 `git reset --hard HEAD~n` 回退版本。 |
| **Generate Document Embeddings** | docker 无法访问 ollama | 1. 确信 Ollama 启动时设置了 `$env:OLLAMA_HOST = "0.0.0.0:11434"`。<br>2. 检查 IP 地址是否正确。<br>3. 若 Ollama 卡住，在启动它的 PowerShell 窗口按回车激活。 |
| **Generate Biography** | 报错 500 / 超时 | 模型过大或超时设置过短。建议更换轻量模型或命令 Copilot 调整超时时间。 |
| **Map Your Entity Network** | 速度慢 | 正常现象，风扇转动且日志有更新即为正常。 |
| **Train** | 进度卡在 93% | 1. 检查 CUDA 适配。<br>2. 进程终止失败 PID 错误：代码可能错误识别了主进程 PID 而非子进程 PID。 |
