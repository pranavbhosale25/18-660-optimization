"""Tests for the environment checker's machine-readable success status."""

import json
import sys
from types import SimpleNamespace

from scripts import check_environment


def test_optional_cyipopt_is_not_required(monkeypatch):
    def installed_version(module, distribution):
        return "MISSING" if module == "cyipopt" else "1.0"

    warp = SimpleNamespace(init=lambda: None, get_devices=lambda: ["cpu"])
    monkeypatch.setattr(check_environment, "_installed_version", installed_version)
    monkeypatch.setitem(sys.modules, "warp", warp)

    report = check_environment.collect_report()

    assert report["cyipopt"] == "MISSING"
    assert report["ok"] is True
    assert report["errors"] == []


def test_main_returns_failure_and_preserves_json_report(monkeypatch, capsys):
    expected = {
        "python": "test",
        "ok": False,
        "errors": ["required dependency warp-lang: MISSING"],
    }
    monkeypatch.setattr(
        check_environment,
        "collect_report",
        lambda **kwargs: expected,
    )

    assert check_environment.main([]) == 1
    assert json.loads(capsys.readouterr().out) == expected
