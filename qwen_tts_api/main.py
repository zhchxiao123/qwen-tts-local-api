import argparse
import asyncio
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import yaml
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field


class SpeechRequest(BaseModel):
    model: str = "default"
    input: str = Field(min_length=1)
    voice: str | None = None
    response_format: str = "wav"
    instructions: str | None = None
    language: str = "Chinese"


@dataclass
class Settings:
    raw: dict[str, Any]
    config_dir: Path

    @property
    def api_key(self) -> str:
        return self.raw.get("security", {}).get("api_key", "")

    @property
    def entries(self) -> dict[str, dict[str, str]]:
        return self.raw["models"]["entries"]

    def reference_audio_path(self, entry: dict[str, str]) -> Path | None:
        value = entry.get("reference_audio")
        if not value:
            return None
        path = Path(value)
        return path if path.is_absolute() else self.config_dir / path


def load_settings(path: str) -> Settings:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        return Settings(yaml.safe_load(handle), config_path.parent)


class ModelPool:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.models: dict[str, Any] = {}
        self.lock = asyncio.Lock()

    def device(self) -> str:
        configured = self.settings.raw.get("runtime", {}).get("device", "auto")
        if configured != "auto":
            return configured
        return "mps" if torch.backends.mps.is_available() else "cpu"

    def dtype(self):
        configured = self.settings.raw.get("runtime", {}).get("dtype", "float16")
        return torch.float16 if configured == "float16" and self.device() != "cpu" else torch.float32

    async def get(self, name: str):
        name = self.settings.raw["models"].get("default") if name == "default" else name
        if name not in self.settings.entries:
            raise HTTPException(404, f"Unknown model: {name}")
        if name not in self.models:
            async with self.lock:
                if name not in self.models:
                    from qwen_tts import Qwen3TTSModel
                    source = self.settings.entries[name]["source"]
                    self.models[name] = await asyncio.to_thread(
                        Qwen3TTSModel.from_pretrained,
                        source,
                        device_map=self.device(),
                        dtype=self.dtype(),
                    )
        return name, self.models[name]


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="Local Qwen3-TTS API", version="0.1.0")
    pool = ModelPool(settings)
    semaphore = asyncio.Semaphore(settings.raw.get("runtime", {}).get("max_concurrent_requests", 1))
    network = settings.raw.get("network", {})
    app.add_middleware(CORSMiddleware, allow_origins=network.get("cors_origins", []), allow_methods=["*"], allow_headers=["*"])

    async def authorize(authorization: str | None = Header(default=None)):
        key = settings.api_key
        if key and authorization != f"Bearer {key}":
            raise HTTPException(401, "Missing or invalid bearer token")

    def render(wavs, sample_rate: int, response_format: str) -> Response:
        if response_format not in {"wav", "flac", "ogg"}:
            raise HTTPException(400, "response_format must be wav, flac, or ogg")
        output = io.BytesIO()
        sf.write(output, wavs[0], sample_rate, format=response_format.upper())
        media = {"wav": "audio/wav", "flac": "audio/flac", "ogg": "audio/ogg"}[response_format]
        return Response(output.getvalue(), media_type=media, headers={"Content-Disposition": f"inline; filename=speech.{response_format}"})

    @app.get("/health")
    async def health(_: None = Depends(authorize)):
        return {"status": "ok", "device": pool.device(), "loaded_models": list(pool.models)}

    @app.get("/v1/models")
    async def models(_: None = Depends(authorize)):
        return {"object": "list", "data": [{"id": name, "object": "model", "mode": value["mode"]} for name, value in settings.entries.items()]}

    @app.post("/v1/audio/speech")
    async def speech(request: SpeechRequest, _: None = Depends(authorize)):
        name, model = await pool.get(request.model)
        if settings.entries[name]["mode"] != "custom_voice":
            raise HTTPException(400, "This endpoint requires a custom_voice model")
        if not request.voice:
            raise HTTPException(400, "voice is required for CustomVoice")
        async with semaphore:
            wavs, sample_rate = await asyncio.to_thread(model.generate_custom_voice, text=request.input, language=request.language, speaker=request.voice, instruct=request.instructions)
        return render(wavs, sample_rate, request.response_format)

    @app.post("/v1/qwen/voice-clone")
    async def voice_clone(
        text: str = Form(...), ref_text: str | None = Form(None), reference_audio: UploadFile | None = File(None), model: str = Form("voice-clone"), language: str = Form("Chinese"), response_format: str = Form("wav"), _: None = Depends(authorize)
    ):
        name, loaded = await pool.get(model)
        entry = settings.entries[name]
        if entry["mode"] != "voice_clone":
            raise HTTPException(400, "This endpoint requires a voice_clone model")
        configured_audio = settings.reference_audio_path(entry)
        effective_ref_text = ref_text or entry.get("reference_text")
        if not effective_ref_text:
            raise HTTPException(400, "ref_text is required unless reference_text is configured for this model")

        temp: Path | None = None
        if reference_audio:
            suffix = Path(reference_audio.filename or "reference.wav").suffix or ".wav"
            temp = Path("/tmp") / f"qwen-tts-reference{suffix}"
            temp.write_bytes(await reference_audio.read())
            effective_audio = temp
        elif configured_audio and configured_audio.is_file():
            effective_audio = configured_audio
        else:
            raise HTTPException(400, "reference_audio is required unless a valid reference_audio is configured for this model")
        try:
            async with semaphore:
                wavs, sample_rate = await asyncio.to_thread(loaded.generate_voice_clone, text=text, language=language, ref_audio=str(effective_audio), ref_text=effective_ref_text)
        finally:
            if temp:
                temp.unlink(missing_ok=True)
        return render(wavs, sample_rate, response_format)

    return app


def run():
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    settings = load_settings(args.config)
    network = settings.raw.get("network", {})
    uvicorn.run(create_app(settings), host=network.get("host", "127.0.0.1"), port=network.get("port", 9010))
