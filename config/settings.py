import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import cast


@dataclass
class Settings:
    TEXT_EMBEDDING_MODEL_NAME: str = 'BAAI/bge-small-zh-v1.5'
    TEXT_EMBEDDING_MODEL_PATH: str = 'models/embedding'
    VISUAL_TOP_K: int = 3
    TEXT_TOP_K: int = 5
    CHROMA_TEXT_PATH: str = 'data/vectorstore'
    IOT_DOCUMENTS_DIR: str = 'data/iot_knowledge'
    LEGACY_DOCUMENTS_DIR: str = 'data/documents'
    AIGTEXT_ROLE: str = 'host_a'
    HOST_A_BIND: str = '0.0.0.0'
    HOST_A_PORT: int = 8080
    VISION_BACKEND_URL: str = 'http://127.0.0.1:9101'
    VISION_BACKEND_API_KEY: str = ''
    VISION_BACKEND_TIMEOUT_SECONDS: float = 180.0
    VISION_FALLBACK_MODE: str = 'error'
    VISUAL_TOP_K_MAX: int = 10
    MODEL_OUTPUT_DIR: str = 'data/models'
    MODEL_TEMP_DIR: str = 'data/temp/models'
    DEPTH_MODEL_PATH: str = 'models/depth-anything-v2'

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> 'Settings':
        env = os.environ if environ is None else environ
        return cls(
            AIGTEXT_ROLE=_env_str(env, 'AIGTEXT_ROLE', cls.AIGTEXT_ROLE),
            HOST_A_BIND=_env_str(env, 'HOST_A_BIND', cls.HOST_A_BIND),
            HOST_A_PORT=_env_int(env, 'HOST_A_PORT', cls.HOST_A_PORT),
            VISION_BACKEND_URL=_env_str(
                env, 'VISION_BACKEND_URL', cls.VISION_BACKEND_URL
            ),
            VISION_BACKEND_API_KEY=_env_str(
                env, 'VISION_BACKEND_API_KEY', cls.VISION_BACKEND_API_KEY
            ),
            VISION_BACKEND_TIMEOUT_SECONDS=_env_float(
                env,
                'VISION_BACKEND_TIMEOUT_SECONDS',
                cls.VISION_BACKEND_TIMEOUT_SECONDS,
            ),
            VISION_FALLBACK_MODE=_env_str(
                env, 'VISION_FALLBACK_MODE', cls.VISION_FALLBACK_MODE
            ),
            VISUAL_TOP_K_MAX=_env_int(
                env, 'VISUAL_TOP_K_MAX', cls.VISUAL_TOP_K_MAX
            ),
        )

    def validate(self) -> None:
        optional_secret_fields = {'VISION_BACKEND_API_KEY'}
        for settings_field in fields(self):
            name = settings_field.name
            value = cast(object, getattr(self, name))
            if name in optional_secret_fields:
                continue
            if value is None or (isinstance(value, str) and not value):
                raise ValueError(f'{name} must not be empty or None')
        if self.VISUAL_TOP_K <= 0:
            raise ValueError(
                f'VISUAL_TOP_K must be > 0, got {self.VISUAL_TOP_K}'
            )
        if self.TEXT_TOP_K <= 0:
            raise ValueError(
                f'TEXT_TOP_K must be > 0, got {self.TEXT_TOP_K}'
            )
        if self.VISUAL_TOP_K_MAX <= 0:
            raise ValueError(
                f'VISUAL_TOP_K_MAX must be > 0, got {self.VISUAL_TOP_K_MAX}'
            )
        if self.VISUAL_TOP_K > self.VISUAL_TOP_K_MAX:
            top_k_error = (
                'VISUAL_TOP_K must be <= VISUAL_TOP_K_MAX, got '
                f'{self.VISUAL_TOP_K} > {self.VISUAL_TOP_K_MAX}'
            )
            raise ValueError(top_k_error)
        if self.HOST_A_PORT <= 0 or self.HOST_A_PORT > 65535:
            raise ValueError(
                f'HOST_A_PORT must be between 1 and 65535, got {self.HOST_A_PORT}'
            )
        if self.VISION_BACKEND_TIMEOUT_SECONDS <= 0:
            timeout_error = (
                'VISION_BACKEND_TIMEOUT_SECONDS must be > 0, got '
                f'{self.VISION_BACKEND_TIMEOUT_SECONDS}'
            )
            raise ValueError(
                timeout_error
            )


def _env_str(env: Mapping[str, str], name: str, default: str) -> str:
    return env[name].strip() if name in env else default


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    return int(env[name]) if name in env else default


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    return float(env[name]) if name in env else default


settings = Settings.from_env()
