import os
from datetime import timedelta

import pytest

from stockpred.cli import load_dotenv, parse_interval


def test_load_dotenv_sets_missing_vars_only(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nSP_A=1\nexport SP_B="two"\nSP_C=from-file\n\n', encoding="utf-8")
    monkeypatch.delenv("SP_A", raising=False)
    monkeypatch.delenv("SP_B", raising=False)
    monkeypatch.setenv("SP_C", "from-env")

    load_dotenv(env)

    assert os.environ["SP_A"] == "1"
    assert os.environ["SP_B"] == "two"
    assert os.environ["SP_C"] == "from-env"
    for key in ("SP_A", "SP_B"):
        monkeypatch.delenv(key)


def test_load_dotenv_missing_file_is_noop(tmp_path):
    load_dotenv(tmp_path / "absent.env")


@pytest.mark.parametrize(
    ("text", "expected"),
    [("30s", timedelta(seconds=30)), ("5m", timedelta(minutes=5)), ("1h", timedelta(hours=1))],
)
def test_parse_interval(text, expected):
    assert parse_interval(text) == expected
