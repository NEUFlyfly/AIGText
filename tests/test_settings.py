from dataclasses import fields
from typing import cast

import pytest

from config.settings import Settings, settings


class TestSettingsSingleton:
    def test_singleton_has_all_fields(self):
        field_names = [
            'TEXT_EMBEDDING_MODEL_NAME',
            'TEXT_EMBEDDING_MODEL_PATH',
            'VISUAL_EMBEDDING_MODEL_NAME',
            'VISUAL_EMBEDDING_MODEL_PATH',
            'VISUAL_TOP_K',
            'TEXT_TOP_K',
            'VISUAL_MIN_SCORE',
            'COARSE_CONFIDENCE_THRESHOLD',
            'CHROMA_TEXT_PATH',
            'CHROMA_VISUAL_PATH',
            'IOT_DOCUMENTS_DIR',
            'LEGACY_DOCUMENTS_DIR',
            'AIGTEXT_ROLE',
            'HOST_A_BIND',
            'HOST_A_PORT',
            'VISION_BIND',
            'VISION_PORT',
            'VISION_BACKEND_URL',
            'VISION_BACKEND_API_KEY',
            'VISION_BACKEND_TIMEOUT_SECONDS',
            'VISION_FALLBACK_MODE',
            'VISION_BACKEND_MODE',
            'VISION_API_KEY',
            'VISUAL_TOP_K_MAX',
        ]
        for name in field_names:
            assert hasattr(settings, name), f'{name} missing'

    def test_no_field_is_none_or_empty(self):
        optional_empty_fields = {'VISION_BACKEND_API_KEY', 'VISION_API_KEY'}
        for settings_field in fields(settings):
            name = settings_field.name
            value = cast(object, getattr(settings, name))
            assert value is not None, f'{name} is None'
            if isinstance(value, str) and name not in optional_empty_fields:
                assert value != '', f'{name} is empty string'

    def test_top_k_positive(self):
        assert settings.VISUAL_TOP_K > 0
        assert settings.TEXT_TOP_K > 0


class TestSettingsValidation:
    def test_validate_passes_with_defaults(self):
        settings.validate()

    def test_validate_raises_on_zero_visual_top_k(self):
        with pytest.raises(ValueError, match='VISUAL_TOP_K must be > 0'):
            s = Settings(VISUAL_TOP_K=0)
            s.validate()

    def test_validate_raises_on_negative_text_top_k(self):
        with pytest.raises(ValueError, match='TEXT_TOP_K must be > 0'):
            s = Settings(TEXT_TOP_K=-1)
            s.validate()

    def test_validate_raises_on_empty_field(self):
        with pytest.raises(ValueError, match='must not be empty or None'):
            s = Settings(TEXT_EMBEDDING_MODEL_NAME='')
            s.validate()

    def test_validate_raises_on_none_field(self):
        with pytest.raises(ValueError, match='must not be empty or None'):
            s = Settings(TEXT_EMBEDDING_MODEL_NAME=cast(str, cast(object, None)))
            s.validate()

    def test_from_env_reads_dual_host_runtime_values(self):
        s = Settings.from_env({
            'AIGTEXT_ROLE': 'host_b',
            'HOST_A_BIND': '127.0.0.1',
            'HOST_A_PORT': '8088',
            'VISION_BIND': '127.0.0.2',
            'VISION_PORT': '9102',
            'VISION_BACKEND_URL': 'http://host-b.local:9101',
            'VISION_BACKEND_API_KEY': 'test-key',
            'VISION_BACKEND_TIMEOUT_SECONDS': '12.5',
            'VISION_FALLBACK_MODE': 'empty',
            'VISION_BACKEND_MODE': 'stub',
            'VISION_API_KEY': 'server-key',
            'VISUAL_TOP_K_MAX': '7',
        })

        assert s.AIGTEXT_ROLE == 'host_b'
        assert s.HOST_A_BIND == '127.0.0.1'
        assert s.HOST_A_PORT == 8088
        assert s.VISION_BIND == '127.0.0.2'
        assert s.VISION_PORT == 9102
        assert s.VISION_BACKEND_URL == 'http://host-b.local:9101'
        assert s.VISION_BACKEND_API_KEY == 'test-key'
        assert s.VISION_BACKEND_TIMEOUT_SECONDS == 12.5
        assert s.VISION_FALLBACK_MODE == 'empty'
        assert s.VISION_BACKEND_MODE == 'stub'
        assert s.VISION_API_KEY == 'server-key'
        assert s.VISUAL_TOP_K_MAX == 7
