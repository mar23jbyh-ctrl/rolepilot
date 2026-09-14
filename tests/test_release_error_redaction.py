"""Synthetic credential-like text must not escape a handled quality-audit error."""
import importlib


def test_self_check_failure_never_reflects_exception_payload(monkeypatch):
    module = importlib.import_module("app.nodes.self_check")

    def fail(*args, **kwargs):
        raise RuntimeError("SYNTHETIC_SECRET_NOT_A_REAL_KEY private synthetic resume")

    monkeypatch.setattr(module, "chat_with_usage", fail)
    result = module.self_check({})
    assert result["self_check_report"]["status"] == "error"
    assert "RuntimeError" in str(result["self_check_report"]["findings"])
    assert "SYNTHETIC_SECRET" not in str(result)
    assert "private synthetic resume" not in str(result)
    assert result['usage_records'][0]['usage_status'] == 'missing'
    assert result['usage_records'][0]['total_tokens'] is None
