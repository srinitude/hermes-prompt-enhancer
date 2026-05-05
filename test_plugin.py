"""
Regression tests for prompt-enhancer-plugin heuristic priority.

Run with: python3 test_plugin.py

These tests validate that the provider resolution logic correctly prioritizes:
1. Explicit provider name match (highest priority)
2. Explicit base_url from model config (second priority)
3. API key heuristic (fallback only, never overwrites explicit config)

This prevents the bug where a kimi-coding config with a Venice API key
gets its base_url overwritten to Venice's endpoint, causing 404 errors.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add plugin to path
PLUGIN_DIR = Path.home() / ".hermes" / "plugins" / "prompt-enhancer-plugin"
sys.path.insert(0, str(PLUGIN_DIR))

# Import after path setup
import __init__ as pe


class TestResolveEnhancerConfig(unittest.TestCase):
    """Test _resolve_enhancer_config() provider resolution priority."""

    # The plugin normalizes base_url with .rstrip("/") then later adds "/"
    # via httpx client construction. We accept either form.
    def _assert_base_url(self, resolved, expected):
        actual = resolved["base_url"]
        # Normalize both for comparison
        actual_norm = actual.rstrip("/")
        expected_norm = expected.rstrip("/")
        self.assertEqual(actual_norm, expected_norm,
                         f"base_url mismatch: {actual!r} != {expected!r}")

    def _make_cfg(self, model_provider, model_base_url, model_api_key,
                  model_default="test-model", custom_providers=None):
        """Build a config dict matching ~/.hermes/config.yaml shape."""
        cfg = {
            "model": {
                "provider": model_provider,
                "base_url": model_base_url,
                "api_key": model_api_key,
                "default": model_default,
            }
        }
        if custom_providers is not None:
            cfg["custom_providers"] = custom_providers
        return cfg

    def test_name_match_over_api_key(self):
        """
        CRITICAL: When provider name matches a custom_provider, use that
        custom_provider's config EVEN if the API key matches a different
        custom_provider.

        This is the core regression test for the kimi-coding + Venice key bug.
        """
        cfg = self._make_cfg(
            model_provider="kimi-coding",
            model_base_url="https://api.kimi.com/coding",
            model_api_key="VENICE_ADMIN_KEY_xxx",
            custom_providers=[
                {
                    "name": "venice",
                    "base_url": "https://api.venice.ai/api/v1",
                    "api_key": "VENICE_ADMIN_KEY_xxx",
                },
                {
                    "name": "kimi-coding",
                    "base_url": "https://api.kimi.com/coding",
                    "api_key": "KIMI_API_KEY_xxx",
                },
            ]
        )
        resolved = pe._resolve_enhancer_config(cfg)
        self.assertEqual(resolved["provider"], "kimi-coding")
        self._assert_base_url(resolved, "https://api.kimi.com/coding")
        self.assertEqual(resolved["api_key"], "KIMI_API_KEY_xxx")

    def test_explicit_base_url_blocks_api_key_heuristic(self):
        """
        When model config has an explicit base_url, the API key heuristic
        must NOT overwrite it, even if the API key matches a custom_provider.
        """
        cfg = self._make_cfg(
            model_provider="kimi-coding",
            model_base_url="https://api.kimi.com/coding",
            model_api_key="VENICE_ADMIN_KEY_xxx",
            custom_providers=[
                {
                    "name": "venice",
                    "base_url": "https://api.venice.ai/api/v1",
                    "api_key": "VENICE_ADMIN_KEY_xxx",
                }
            ]
        )
        resolved = pe._resolve_enhancer_config(cfg)
        self._assert_base_url(resolved, "https://api.kimi.com/coding")
        self.assertEqual(resolved["provider"], "kimi-coding")

    def test_api_key_fallback_when_no_base_url(self):
        """
        API key heuristic IS allowed when there is NO explicit base_url
        and NO provider name match. This is the legitimate fallback case.
        """
        cfg = self._make_cfg(
            model_provider="some-unknown-provider",
            model_base_url="",
            model_api_key="VENICE_ADMIN_KEY_xxx",
            custom_providers=[
                {
                    "name": "venice",
                    "base_url": "https://api.venice.ai/api/v1",
                    "api_key": "VENICE_ADMIN_KEY_xxx",
                }
            ]
        )
        resolved = pe._resolve_enhancer_config(cfg)
        self._assert_base_url(resolved, "https://api.venice.ai/api/v1")
        self.assertEqual(resolved["provider"], "venice")

    def test_no_custom_providers_uses_model_config_directly(self):
        """
        Without custom_providers, model config values pass through unchanged.
        """
        cfg = self._make_cfg(
            model_provider="anthropic",
            model_base_url="https://api.anthropic.com/v1",
            model_api_key="sk-ant-xxx",
        )
        resolved = pe._resolve_enhancer_config(cfg)
        resolved = pe._resolve_enhancer_config(cfg)
        self._assert_base_url(resolved, "https://api.anthropic.com/v1")
        self.assertEqual(resolved["api_key"], "sk-ant-xxx")

    def test_empty_base_url_falls_back_to_inference(self):
        """
        When base_url is empty and no custom_providers match, infer from
        provider name.
        """
        cfg = self._make_cfg(
            model_provider="openrouter",
            model_base_url="",
            model_api_key="sk-or-xxx",
        )
        resolved = pe._resolve_enhancer_config(cfg)
        self.assertEqual(resolved["provider"], "openrouter")
        self._assert_base_url(resolved, "https://openrouter.ai/api/v1")

    def test_provider_name_match_sets_all_fields(self):
        """
        When provider name matches a custom_provider, all fields
        (base_url, api_key, model) are pulled from the custom_provider.
        """
        cfg = self._make_cfg(
            model_provider="venice",
            model_base_url="https://old.wrong.url",
            model_api_key="old-wrong-key",
            model_default="old-model",
            custom_providers=[
                {
                    "name": "venice",
                    "provider": "venice",
                    "base_url": "https://api.venice.ai/api/v1",
                    "api_key": "VENICE_ADMIN_KEY_xxx",
                    "model": "llama-3.3-70b",
                }
            ]
        )
        resolved = pe._resolve_enhancer_config(cfg)
        self.assertEqual(resolved["provider"], "venice")
        self._assert_base_url(resolved, "https://api.venice.ai/api/v1")
        self.assertEqual(resolved["api_key"], "VENICE_ADMIN_KEY_xxx")
        self.assertEqual(resolved["model"], "llama-3.3-70b")

    def test_multiple_custom_providers_first_match_wins(self):
        """
        When multiple custom_providers could match, the first one in the
        list wins (provider name match takes priority over api_key match).
        """
        cfg = self._make_cfg(
            model_provider="kimi-coding",
            model_base_url="",
            model_api_key="VENICE_ADMIN_KEY_xxx",
            custom_providers=[
                {
                    "name": "venice",
                    "base_url": "https://api.venice.ai/api/v1",
                    "api_key": "VENICE_ADMIN_KEY_xxx",
                },
                {
                    "name": "kimi-coding",
                    "base_url": "https://api.kimi.com/coding",
                    "api_key": "KIMI_API_KEY_xxx",
                },
            ]
        )
        resolved = pe._resolve_enhancer_config(cfg)
        # Should match kimi-coding by name, not venice by api_key
        self.assertEqual(resolved["provider"], "kimi-coding")
        self._assert_base_url(resolved, "https://api.kimi.com/coding")

    def test_env_var_fallback_when_no_api_key_in_config(self):
        """
        When no api_key in model config or custom_providers, fall back to
        provider-specific environment variable.
        """
        cfg = self._make_cfg(
            model_provider="anthropic",
            model_base_url="",
            model_api_key="",
        )
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-ant-xxx"}):
            resolved = pe._resolve_enhancer_config(cfg)
        self.assertEqual(resolved["api_key"], "env-ant-xxx")

    def test_model_default_passes_through_when_no_custom_override(self):
        """
        Model default from config passes through when no custom_provider
        overrides it.
        """
        cfg = self._make_cfg(
            model_provider="openai",
            model_base_url="https://api.openai.com/v1",
            model_api_key="sk-openai-xxx",
            model_default="gpt-4o",
        )
        resolved = pe._resolve_enhancer_config(cfg)
        self.assertEqual(resolved["model"], "gpt-4o")

    def test_kimi_coding_with_venice_key_real_world_scenario(self):
        """
        THE EXACT BUG: User has kimi-coding provider with Venice API key
        in custom_providers. Before the fix, this would 404 because the
        base_url got overwritten to Venice. After the fix, it stays on Kimi.
        """
        cfg = self._make_cfg(
            model_provider="kimi-coding",
            model_base_url="https://api.kimi.com/coding",
            model_api_key="VENICE_ADMIN_KEY_6aidYKxVR8UikmbDAQLkpO6mCmaYlbxD_HsCjgqkMF",
            model_default="kimi-k2.6",
            custom_providers=[
                {
                    "name": "venice",
                    "base_url": "https://api.venice.ai/api/v1",
                    "api_key": "VENICE_ADMIN_KEY_6aidYKxVR8UikmbDAQLkpO6mCmaYlbxD_HsCjgqkMF",
                }
            ]
        )
        resolved = pe._resolve_enhancer_config(cfg)
        self._assert_base_url(resolved, "https://api.kimi.com/coding")
        self.assertEqual(
            resolved["provider"], "kimi-coding",
            "CRITICAL: provider must stay as kimi-coding"
        )


class TestHookRegistration(unittest.TestCase):
    """Test that register() correctly wires the pre_llm_call hook."""

    def test_register_adds_hook(self):
        """register() must call ctx.register_hook('pre_llm_call', ...)."""
        ctx = MagicMock()
        ctx.register_hook = MagicMock()
        ctx._hooks = {"pre_llm_call": []}
        pe.register(ctx)
        ctx.register_hook.assert_called_once()
        args = ctx.register_hook.call_args[0]
        self.assertEqual(args[0], "pre_llm_call")
        self.assertEqual(args[1], pe.pre_llm_call_hook)

    def test_register_is_callable(self):
        """The module must export a callable register function."""
        self.assertTrue(hasattr(pe, "register"))
        self.assertTrue(callable(pe.register))

    def test_pre_llm_call_hook_is_callable(self):
        """The module must export a callable pre_llm_call_hook."""
        self.assertTrue(hasattr(pe, "pre_llm_call_hook"))
        self.assertTrue(callable(pe.pre_llm_call_hook))

    def test_pre_llm_call_hook_accepts_hermes_kwargs(self):
        """pre_llm_call_hook must accept the kwargs Hermes actually passes."""
        # Hermes invokes with: session_id, user_message, conversation_history,
        # is_first_turn, model, platform, sender_id
        with patch.object(pe, '_is_enabled', return_value=False):
            result = pe.pre_llm_call_hook(
                session_id="test-sess",
                user_message="hello world",
                conversation_history=[],
                is_first_turn=True,
                model="gpt-4",
                platform="cli",
                sender_id="user-123",
            )
        # When disabled, returns empty dict
        self.assertEqual(result, {})

    def test_pre_llm_call_hook_skips_slash_commands(self):
        """pre_llm_call_hook should skip messages starting with /."""
        with patch.object(pe, '_is_enabled', return_value=True):
            result = pe.pre_llm_call_hook(
                session_id="test-sess",
                user_message="/reset",
                conversation_history=[],
                is_first_turn=True,
                model="gpt-4",
                platform="cli",
                sender_id="user-123",
            )
        self.assertEqual(result, {})

    def test_pre_llm_call_hook_skips_empty_messages(self):
        """pre_llm_call_hook should skip empty messages."""
        with patch.object(pe, '_is_enabled', return_value=True):
            result = pe.pre_llm_call_hook(
                session_id="test-sess",
                user_message="",
                conversation_history=[],
                is_first_turn=True,
                model="gpt-4",
                platform="cli",
                sender_id="user-123",
            )
        self.assertEqual(result, {})


class TestEnabledCheck(unittest.TestCase):
    """Test the PROMPT_ENHANCER_ENABLED environment variable handling."""

    def test_enabled_by_default(self):
        """When env var is not set, plugin should be enabled."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove PROMPT_ENHANCER_ENABLED if present
            os.environ.pop("PROMPT_ENHANCER_ENABLED", None)
            self.assertTrue(pe._is_enabled())

    def test_disabled_when_set_to_zero(self):
        """When env var is '0', plugin should be disabled."""
        with patch.dict(os.environ, {"PROMPT_ENHANCER_ENABLED": "0"}):
            self.assertFalse(pe._is_enabled())

    def test_enabled_when_set_to_one(self):
        """When env var is '1', plugin should be enabled."""
        with patch.dict(os.environ, {"PROMPT_ENHANCER_ENABLED": "1"}):
            self.assertTrue(pe._is_enabled())


class TestInferBaseUrl(unittest.TestCase):
    """Test the _infer_base_url fallback function."""

    def test_known_providers(self):
        """Known providers return correct URLs."""
        self.assertEqual(
            pe._infer_base_url("anthropic"),
            "https://api.anthropic.com/v1"
        )
        self.assertEqual(
            pe._infer_base_url("openrouter"),
            "https://openrouter.ai/api/v1"
        )
        self.assertEqual(
            pe._infer_base_url("kimi-coding"),
            "https://api.kimi.com/coding"
        )

    def test_unknown_provider_defaults_to_openrouter(self):
        """Unknown providers fall back to OpenRouter."""
        self.assertEqual(
            pe._infer_base_url("unknown-provider"),
            "https://openrouter.ai/api/v1"
        )


class TestEnvKeyForProvider(unittest.TestCase):
    """Test the _env_key_for_provider fallback function."""

    def test_known_providers(self):
        """Known providers return correct env var names."""
        # _env_key_for_provider returns the KEY VALUE, not the key name
        # When env var is not set, it returns empty string
        # We test by mocking the env var
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-ant-key"}):
            self.assertEqual(
                pe._env_key_for_provider("anthropic"),
                "test-ant-key"
            )
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-or-key"}):
            self.assertEqual(
                pe._env_key_for_provider("openrouter"),
                "test-or-key"
            )
        with patch.dict(os.environ, {"KIMI_API_KEY": "test-kimi-key"}):
            self.assertEqual(
                pe._env_key_for_provider("kimi-coding"),
                "test-kimi-key"
            )

    def test_unknown_provider_returns_empty(self):
        """Unknown providers return empty string (not None)."""
        self.assertEqual(pe._env_key_for_provider("unknown"), "")


if __name__ == "__main__":
    # Run all tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestResolveEnhancerConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestHookRegistration))
    suite.addTests(loader.loadTestsFromTestCase(TestEnabledCheck))
    suite.addTests(loader.loadTestsFromTestCase(TestInferBaseUrl))
    suite.addTests(loader.loadTestsFromTestCase(TestEnvKeyForProvider))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with non-zero if any tests failed
    sys.exit(0 if result.wasSuccessful() else 1)
