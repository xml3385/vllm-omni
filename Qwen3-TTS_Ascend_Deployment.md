# Qwen3-TTS 昇腾 910B 部署指南 (vllm-omni Docker)

本文档提供在 Ascend 910B 服务器上使用 `vllm-omni` 的 Docker 镜像，通过 [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base) 自动下载并运行 **Qwen3-TTS-12Hz-1.7B-Base** 文本到语音 (TTS) 模型的详细方案。

## 1. 环境准备

- **硬件要求**: 华为昇腾 (Ascend) 910B 服务器。
- **软件要求**:
  - 已安装 Docker。
  - 已安装 [Ascend Docker Runtime](https://www.hiascend.com/document/detail/zh/mindx-dl/2043/clusterscheduling/clusterschedulingug/dlug_install_015.html)，以支持在 Docker 容器中挂载 NPU 设备。

## 2. 部署方案

我们将使用指定的镜像 `quay.io/ascend/vllm-omni:v0.16.0`。为了实现启动时自动从 ModelScope 下载模型，我们需要配置环境变量 `VLLM_USE_MODELSCOPE=True`。

### 启动命令

运行以下命令拉取镜像并启动 vLLM 容器：

```bash
docker run -d \
  --name qwen3-tts-vllm \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/common \
  -v /usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -p 8000:8000 \
  -e VLLM_USE_MODELSCOPE=True \
  quay.io/ascend/vllm-omni:v0.16.0 \
  --model qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --trust-remote-code
```

*(注意：如果您的 Ascend Docker Runtime 已通过 `--ascend_visible_devices` 配置全局生效，您可以简化上述冗长的 `--device` 和 `-v` 挂载参数，直接使用:)*

```bash
docker run -d \
  --name qwen3-tts-vllm \
  --device /dev/davinci0 \
  -p 8000:8000 \
  -e VLLM_USE_MODELSCOPE=True \
  quay.io/ascend/vllm-omni:v0.16.0 \
  --model qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --trust-remote-code
```

### 启动参数说明 (可根据需求更改)

- `--device /dev/davinci0`: 挂载第 0 张 NPU 卡。如果您想使用其他卡，可以更改为 `/dev/davinci1` 等，或使用多张卡 (如配合 `--tensor-parallel-size 2` 使用多卡推理)。
- `-p 8000:8000`: 端口映射。将容器内的 8000 端口映射到宿主机的 8000 端口。若宿主机端口冲突，可修改为 `-p 8080:8000` 等。
- `-e VLLM_USE_MODELSCOPE=True`: **关键参数**。强制 vLLM 使用 ModelScope 的模型库进行下载，而不是默认的 Hugging Face。
- `--model qwen/Qwen3-TTS-12Hz-1.7B-Base`: 指定在 ModelScope 上的模型 ID。
- `--trust-remote-code`: 允许执行模型仓库中的自定义 Python 代码（运行 Qwen 模型通常需要此参数）。
- `--max-model-len`: (可选) 可以根据显存大小调整最大序列长度，例如 `--max-model-len 4096`。
- `--gpu-memory-utilization`: (可选) 限制 NPU 显存占用比例，默认通常是 0.90，可修改为 `--gpu-memory-utilization 0.8` 防止 OOM。

*(提示: 首次启动时，容器会自动从 ModelScope 下载约数 GB 的模型权重文件，启动时间可能较长。您可以通过 `docker logs -f qwen3-tts-vllm` 查看下载进度和启动日志。)*

## 3. 客户端测试

当 vLLM 成功启动并在 8000 端口监听后，我们可以使用 Python 脚本调用其兼容 OpenAI 的 API 接口，发送文本并接收合成的语音。

### 安装依赖

```bash
pip install openai requests
```

### Python 测试脚本 (`test_tts.py`)

创建一个名为 `test_tts.py` 的文件，填入以下代码：

```python
import os
from openai import OpenAI

# 配置 vLLM 服务的地址和端口 (与 Docker 启动时映射的端口一致)
VLLM_API_BASE = "http://localhost:8000/v1"
# vLLM 不需要真实的 API Key，但 OpenAI 客户端要求该字段不能为空
API_KEY = "EMPTY"
# ModelScope 模型 ID
MODEL_ID = "qwen/Qwen3-TTS-12Hz-1.7B-Base"

client = OpenAI(
    api_key=API_KEY,
    base_url=VLLM_API_BASE,
)

# 待合成的文本
text_to_speak = "你好！这里是基于昇腾 910B 和 vLLM 部署的 Qwen3 语音合成模型。很高兴为您服务。"

print(f"正在请求语音合成...")
print(f"模型: {MODEL_ID}")
print(f"文本: {text_to_speak}")

try:
    # 调用 OpenAI 兼容的 audio/speech 接口
    response = client.audio.speech.create(
        model=MODEL_ID,
        voice="default", # 根据 Qwen3-TTS 的具体支持，可能需要修改特定的 voice ID
        input=text_to_speak,
        response_format="wav" # 请求返回 wav 格式
    )

    output_filename = "output_speech.wav"

    # 将响应内容流式写入文件
    response.stream_to_file(output_filename)

    print(f"语音合成成功！音频已保存至: {os.path.abspath(output_filename)}")

except Exception as e:
    print(f"语音合成失败: {e}")
```

运行测试脚本：

```bash
python test_tts.py
```

执行成功后，当前目录下将生成一个 `output_speech.wav` 文件，您可以使用任何支持 `.wav` 的音频播放器进行试听。
