#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Settings


@dataclass(frozen=True)
class HostAConfig:
    host: str
    port: int
    backend: str
    static: str
    vision_backend_url: str
    vision_backend_api_key: str
    vision_backend_timeout_seconds: float


def build_parser(defaults: Settings | None = None) -> argparse.ArgumentParser:
    default_settings = defaults or Settings.from_env()
    parser = argparse.ArgumentParser(
        description='Start AIGText Host A frontend and language proxy.'
    )
    _ = parser.add_argument(
        '--host',
        default=default_settings.HOST_A_BIND,
        help=f'bind address (default: {default_settings.HOST_A_BIND})',
    )
    _ = parser.add_argument(
        '--port',
        type=int,
        default=default_settings.HOST_A_PORT,
        help=f'bind port (default: {default_settings.HOST_A_PORT})',
    )
    _ = parser.add_argument(
        '--backend',
        default='http://127.0.0.1:18080',
        help='llama.cpp backend URL (default: http://127.0.0.1:18080)',
    )
    _ = parser.add_argument(
        '--static',
        default=str(PROJECT_ROOT / 'frontend'),
        help='frontend static directory',
    )
    _ = parser.add_argument(
        '--vision-backend-url',
        default=default_settings.VISION_BACKEND_URL,
        help='Host B vision backend URL',
    )
    _ = parser.add_argument(
        '--vision-backend-api-key',
        default=default_settings.VISION_BACKEND_API_KEY,
        help='optional Host B API key',
    )
    _ = parser.add_argument(
        '--vision-backend-timeout-seconds',
        type=float,
        default=default_settings.VISION_BACKEND_TIMEOUT_SECONDS,
        help='Host B request timeout in seconds',
    )
    return parser


def parse_args(
    argv: Sequence[str] | None = None,
    environ: dict[str, str] | None = None,
) -> HostAConfig:
    defaults = Settings.from_env(environ)
    args = build_parser(defaults).parse_args(argv)
    return HostAConfig(
        host=cast(str, args.host),
        port=cast(int, args.port),
        backend=cast(str, args.backend).rstrip('/'),
        static=cast(str, args.static),
        vision_backend_url=cast(str, args.vision_backend_url).rstrip('/'),
        vision_backend_api_key=cast(str, args.vision_backend_api_key),
        vision_backend_timeout_seconds=cast(
            float, args.vision_backend_timeout_seconds
        ),
    )


def run(config: HostAConfig) -> int:
    os.environ['AIGTEXT_ROLE'] = 'host_a'
    os.environ['VISION_BACKEND_URL'] = config.vision_backend_url
    os.environ['VISION_BACKEND_API_KEY'] = config.vision_backend_api_key
    os.environ['VISION_BACKEND_TIMEOUT_SECONDS'] = str(
        config.vision_backend_timeout_seconds
    )
    return _run_front_server([
        '--host',
        config.host,
        '--port',
        str(config.port),
        '--backend',
        config.backend,
        '--static',
        config.static,
    ])


def _run_front_server(argv: list[str]) -> int:
    from src import front_server

    original_argv = sys.argv[:]
    sys.argv = ['src.front_server', *argv]
    try:
        front_server.main()
    finally:
        sys.argv = original_argv
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
