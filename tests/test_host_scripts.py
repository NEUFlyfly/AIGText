import subprocess
import shutil
import sys
from pathlib import Path

import pytest

from src import run_host_a, run_host_b


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_host_a_parse_args_uses_env_defaults() -> None:
    config = run_host_a.parse_args([], {
        'HOST_A_BIND': '127.0.0.1',
        'HOST_A_PORT': '8088',
        'VISION_BACKEND_URL': 'http://host-b.example:9101',
        'VISION_BACKEND_API_KEY': 'secret-for-test',
        'VISION_BACKEND_TIMEOUT_SECONDS': '15',
    })

    assert config.host == '127.0.0.1'
    assert config.port == 8088
    assert config.backend == 'http://127.0.0.1:18080'
    assert config.vision_backend_url == 'http://host-b.example:9101'
    assert config.vision_backend_api_key == 'secret-for-test'
    assert config.vision_backend_timeout_seconds == 15


def test_host_a_cli_args_override_env_defaults() -> None:
    config = run_host_a.parse_args([
        '--host',
        '0.0.0.0',
        '--port',
        '8090',
        '--backend',
        'http://127.0.0.1:18081/',
        '--vision-backend-url',
        'http://192.0.2.10:9101/',
    ], {
        'HOST_A_BIND': '127.0.0.1',
        'HOST_A_PORT': '8088',
    })

    assert config.host == '0.0.0.0'
    assert config.port == 8090
    assert config.backend == 'http://127.0.0.1:18081'
    assert config.vision_backend_url == 'http://192.0.2.10:9101'


def test_host_a_run_delegates_without_importing_visual_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run_front_server(argv: list[str]) -> int:
        calls.append(argv)
        return 0

    monkeypatch.setattr(run_host_a, '_run_front_server', fake_run_front_server)

    result = run_host_a.run(run_host_a.HostAConfig(
        host='127.0.0.1',
        port=8080,
        backend='http://127.0.0.1:18080',
        static='frontend',
        vision_backend_url='http://127.0.0.1:9101',
        vision_backend_api_key='',
        vision_backend_timeout_seconds=30,
    ))

    assert result == 0
    assert calls == [[
        '--host',
        '127.0.0.1',
        '--port',
        '8080',
        '--backend',
        'http://127.0.0.1:18080',
        '--static',
        'frontend',
    ]]


def test_host_b_parse_args_uses_env_defaults() -> None:
    config = run_host_b.parse_args([], {
        'VISION_BIND': '127.0.0.1',
        'VISION_PORT': '9102',
        'VISION_BACKEND_MODE': 'stub',
        'VISION_FALLBACK_MODE': 'empty',
        'VISION_API_KEY': 'server-secret',
        'VISUAL_TOP_K_MAX': '8',
    })

    assert config.host == '127.0.0.1'
    assert config.port == 9102
    assert config.backend_mode == 'stub'
    assert config.fallback_mode == 'empty'
    assert config.api_key == 'server-secret'
    assert config.top_k_max == 8


def test_host_b_cli_args_override_env_defaults() -> None:
    config = run_host_b.parse_args([
        '--host',
        '0.0.0.0',
        '--port',
        '9103',
        '--backend-mode',
        'model',
        '--fallback-mode',
        'error',
        '--top-k-max',
        '9',
    ], {
        'VISION_BIND': '127.0.0.1',
        'VISION_PORT': '9102',
        'VISION_BACKEND_MODE': 'stub',
        'VISION_FALLBACK_MODE': 'empty',
    })

    assert config.host == '0.0.0.0'
    assert config.port == 9103
    assert config.backend_mode == 'model'
    assert config.fallback_mode == 'error'
    assert config.top_k_max == 9


def test_host_b_run_delegates_to_vision_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeVisionServer:
        @staticmethod
        def main(**kwargs: object) -> None:
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, 'src.vision.vision_server', FakeVisionServer)

    result = run_host_b.run(run_host_b.HostBConfig(
        host='127.0.0.1',
        port=9101,
        backend_mode='stub',
        fallback_mode='empty',
        api_key='secret',
        top_k_max=6,
    ))

    assert result == 0
    assert calls == [{
        'host': '127.0.0.1',
        'port': 9101,
        'backend_mode': 'stub',
        'fallback_mode': 'empty',
        'api_key': 'secret',
        'top_k_max': 6,
    }]


def _assert_help_output(result: subprocess.CompletedProcess[str]) -> None:
    assert '--host' in result.stdout
    assert '--port' in result.stdout
    assert 'usage:' in result.stdout


def _bash_executable() -> str:
    bash = shutil.which('bash')
    if bash is None:
        pytest.fail('bash executable is required for host shell launcher tests')
    return bash


def test_host_python_scripts_help_is_timeout_safe() -> None:
    for script_name in ('run_host_a.py', 'run_host_b.py'):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / 'src' / script_name), '--help'],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=5,
            check=True,
        )

        _assert_help_output(result)


def test_host_shell_scripts_help_is_timeout_safe() -> None:
    bash = _bash_executable()
    for script_name in ('run_host_a.sh', 'run_host_b.sh'):
        result = subprocess.run(
            [bash, f'scripts/{script_name}', '--help'],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=5,
            check=True,
        )

        _assert_help_output(result)
