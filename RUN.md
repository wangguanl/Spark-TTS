# 运行命令

- 项目：Spark-TTS
- 生成时间：2026-09-08
- 运行方式：直接运行（uv + Python 3.12）
- 硬件评估：**满足** — 官方主模型 Spark-TTS-0.5B（约 0.5B 参数，社区经验约 8GB 显存即可）；本机 RTX 4080 16GB，启动前约剩余 15GB，磁盘 E: 空闲约 434GB，内存约 32GB。不走 Triton/TRT-LLM 多服务路径。

## 环境准备

```powershell
# 依赖：清华 PyPI；Windows 上 torch 须装 +cu124 直链轮子（仅用 index-url 易装成 CPU 版）
Set-Location E:\AI\local-voice\Spark-TTS
uv venv --python 3.12
uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --index-strategy unsafe-best-match
uv pip install `
  'https://download.pytorch.org/whl/cu124/torch-2.5.1%2Bcu124-cp312-cp312-win_amd64.whl' `
  'https://download.pytorch.org/whl/cu124/torchaudio-2.5.1%2Bcu124-cp312-cp312-win_amd64.whl'

# 模型（官方 huggingface.co 超时，使用 hf-mirror）
$env:HF_ENDPOINT = 'https://hf-mirror.com'
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('SparkAudio/Spark-TTS-0.5B', local_dir='pretrained_models/Spark-TTS-0.5B')"
```

## 启动

- 推荐：`pwsh -NoProfile -File .\start.ps1`
- 说明：启动后交互选择 Web UI 或 CLI 示例推理；默认选 Web UI。不要默认全开。
- 等价手动命令：

### webui（Gradio）

```powershell
$env:Path = "E:\Programs\ffmpeg-master-latest-win64-gpl\bin;$env:Path"
.\.venv\Scripts\python.exe webui.py --device 0 --server_name 0.0.0.0 --server_port 7860
```

### cli-demo（官方示例文本 + prompt_audio）

```powershell
$env:Path = "E:\Programs\ffmpeg-master-latest-win64-gpl\bin;$env:Path"
.\.venv\Scripts\python.exe -m cli.inference `
  --text "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。" `
  --device 0 `
  --save_dir "example/results" `
  --model_dir "pretrained_models/Spark-TTS-0.5B" `
  --prompt_text "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。" `
  --prompt_speech_path "example/prompt_audio.wav"
```

## 验证

- **webui**：浏览器打开 `http://127.0.0.1:<实际端口>/`，页面可加载；用 Voice Cloning 生成一段音频无 CUDA 报错。
- **cli-demo**：退出码 0，且 `example/results/` 下出现新的 `.wav`。

## 备注

- 端口：Web UI 建议起点 7860，占用则 `start.ps1` 自动顺延。
- 镜像：HF 用 `https://hf-mirror.com`；PyPI 用清华；torch/torchaudio 用 PyTorch cu124 直链轮子。
- ffmpeg：`E:\Programs\ffmpeg-master-latest-win64-gpl\bin`（由 start.ps1 注入 PATH）。
- Python：3.12（官方要求 3.12+）；虚拟环境 `.venv`；已验证 `torch 2.5.1+cu124` + CUDA True。
- 可选 Triton 路径见 `runtime/triton_trtllm/`，本机日常推理不启用。