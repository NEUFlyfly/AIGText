#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings


class VisionServerMain(Protocol):
    def __call__(
        self,
        *,
        host: str,
        port: int,
        backend_mode: str,
        fallback_mode: str,
        api_key: str,
        top_k_max: int,
    ) -> None:
        ...


@dataclass(frozen=True)
class HostBConfig:
    host: str
    port: int
    backend_mode: str
    fallback_mode: str
    api_key: str
    top_k_max: int


def build_parser(defaults: Settings | None = None) -> argparse.ArgumentParser:
    default_settings = defaults or Settings.from_env()
    parser = argparse.ArgumentParser(
        description='Start AIGText Host B vision server.'
    )
    _ = parser.add_argument(
        '--host',
        default=default_settings.VISION_BIND,
        help=f'bind address (default: {default_settings.VISION_BIND})',
    )
    _ = parser.add_argument(
        '--port',
        type=int,
        default=default_settings.VISION_PORT,
        help=f'bind port (default: {default_settings.VISION_PORT})',
    )
    _ = parser.add_argument(
        '--backend-mode',
        choices=('model', 'stub'),
        default=default_settings.VISION_BACKEND_MODE,
        help='vision backend mode; stub is only used when explicitly selected',
    )
    _ = parser.add_argument(
        '--fallback-mode',
        choices=('error', 'empty'),
        default=default_settings.VISION_FALLBACK_MODE,
        help='response mode when the visual backend is unavailable',
    )
    _ = parser.add_argument(
        '--api-key',
        default=default_settings.VISION_API_KEY,
        help='optional API key required by Host B',
    )
    _ = parser.add_argument(
        '--top-k-max',
        type=int,
        default=default_settings.VISUAL_TOP_K_MAX,
        help='maximum visual Top-K allowed by Host B',
    )
    return parser


def parse_args(
    argv: Sequence[str] | None = None,
    environ: dict[str, str] | None = None,
) -> HostBConfig:
    defaults = Settings.from_env(environ)
    args = build_parser(defaults).parse_args(argv)
    return HostBConfig(
        host=cast(str, args.host),
        port=cast(int, args.port),
        backend_mode=cast(str, args.backend_mode),
        fallback_mode=cast(str, args.fallback_mode),
        api_key=cast(str, args.api_key),
        top_k_max=cast(int, args.top_k_max),
    )


def run(config: HostBConfig) -> int:
    os.environ['AIGTEXT_ROLE'] = 'host_b'
    os.environ['VISION_BACKEND_MODE'] = config.backend_mode
    os.environ['VISION_FALLBACK_MODE'] = config.fallback_mode
    os.environ['VISION_API_KEY'] = config.api_key
    os.environ['VISUAL_TOP_K_MAX'] = str(config.top_k_max)

    vision_server = importlib.import_module('src.vision.vision_server')
    server_main = cast(VisionServerMain, getattr(vision_server, 'main'))

    server_main(
        host=config.host,
        port=config.port,
        backend_mode=config.backend_mode,
        fallback_mode=config.fallback_mode,
        api_key=config.api_key,
        top_k_max=config.top_k_max,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
