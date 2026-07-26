# Qwen TTS 本地 API

一个面向 Apple Silicon 的可配置 Qwen3-TTS 服务。它提供 OpenAI 兼容的文字转语音接口，以及 Qwen 专用的音色克隆接口。

## 接口

- `GET /health`：服务与模型状态。
- `GET /v1/models`：可用模型。
- `POST /v1/audio/speech`：OpenAI 兼容的文字转语音接口。
- `POST /v1/qwen/voice-clone`：参考音频的音色克隆接口。

音色克隆只应使用自己拥有或已获得明确授权的声音。

## 安装与启动

建议使用 Python 3.12 的独立虚拟环境。安装依赖后，复制配置模板为 `config.yaml`，再启动服务：

```bash
cd qwen-tts-api
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
cp config.example.yaml config.yaml
qwen-tts-api --config config.yaml
```

首次请求会下载并加载模型；M3 Max 会自动优先使用 MPS。建议先以 `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` 开始。设置 `network.host` 为 `0.0.0.0` 才会允许局域网访问；此时务必设置 `security.api_key` 和受限的 `cors_origins`。

## OpenAI 兼容示例

```bash
curl http://127.0.0.1:9010/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","input":"你好，欢迎使用本地语音服务。","voice":"Vivian","response_format":"wav"}' \
  --output speech.wav
```

`voice` 必须是当前 CustomVoice 模型支持的说话人；可通过服务的模型信息或 Qwen 模型自身的 `get_supported_speakers()` 查询。

## 默认克隆音色

为 `voice_clone` 模型配置 `reference_audio` 和完全匹配的 `reference_text` 后，调用接口时无需每次上传参考音频；请求里上传的音频和 `ref_text` 会优先覆盖默认配置。参考音频必须是你拥有或已获得明确授权使用的声音。

```yaml
voice-clone:
  source: "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
  mode: "voice_clone"
  reference_audio: "./references/default.wav"
  reference_text: "参考音频中实际说出的完整文字。"
```

将参考音频放在 `references/`，它已被默认忽略，不会提交到 GitHub。带默认音色时，请求只需提供待合成的 `text`、模型名和语言。

## 配置原则

- `models.default` 可指向任意已下载的本地目录或 Hugging Face/ModelScope 模型名。
- `runtime.device: auto` 会优先选择 Apple MPS；出现不支持算子时可改为 `cpu` 便于排查。
- `security.api_key` 为空时不鉴权，仅适用于本机监听。
- 可在 `models.entries` 加入 Base 或 VoiceDesign 模型；服务会按请求懒加载并缓存。
