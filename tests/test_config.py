"""Tests for configuration management."""


from universal_context.config import UCConfig


def test_config_defaults():
    config = UCConfig()
    assert config.use_llm is True
    assert config.llm_provider == "openrouter"
    assert config.auto_detect_claude is True
    assert config.auto_detect_gemini is True
    assert config.watcher_poll_interval == 2.0
    assert config.max_job_attempts == 10
    assert config.memory_enabled is True
    assert config.memory_update_threshold == 5
    assert config.memory_max_summaries == 30


def test_config_save_load_roundtrip(tmp_path):
    config_path = tmp_path / "config.yaml"

    original = UCConfig(llm_provider="claude", summary_max_chars=300)
    original.save(config_path)

    loaded = UCConfig.load(config_path)
    assert loaded.llm_provider == "claude"
    assert loaded.summary_max_chars == 300
    assert loaded.use_llm is True  # default preserved


def test_config_env_override(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    UCConfig().save(config_path)

    monkeypatch.setenv("UC_LLM_PROVIDER", "openai")
    monkeypatch.setenv("UC_USE_LLM", "false")
    monkeypatch.setenv("UC_WATCHER_POLL_INTERVAL", "5.0")

    loaded = UCConfig.load(config_path)
    assert loaded.llm_provider == "openai"
    assert loaded.use_llm is False
    assert loaded.watcher_poll_interval == 5.0


def test_config_ignores_unknown_fields(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("unknown_field: 42\nuse_llm: false\n")

    loaded = UCConfig.load(config_path)
    assert loaded.use_llm is False
    assert not hasattr(loaded, "unknown_field")


def test_resolved_db_path():
    config = UCConfig(db_path="~/.uc/data/surreal.db")
    resolved = config.resolved_db_path
    assert resolved.is_absolute()
    assert "~" not in str(resolved)
