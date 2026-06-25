from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    upload_dir: str = "backend/storage/uploads"
    telemetry_dir: str = "backend/storage/telemetry"
    onnx_path: str = "ml/checkpoints/model.fp16.onnx"
    default_fps: float = 10.0
    pid_setpoint_m: float = 10.0

    class Config:
        env_file = ".env"


settings = Settings()
