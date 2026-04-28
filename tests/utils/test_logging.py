"""Tests for wem.utils.logging."""

import re

from wem.utils.logging import log


def test_log_message_appears(capsys):
    log("hello world")
    err = capsys.readouterr().err
    assert "hello world" in err


def test_log_timestamp_format(capsys):
    log("test")
    err = capsys.readouterr().err
    assert re.search(r"\[\d{2}:\d{2}:\d{2}\]", err)


def test_log_custom_message(capsys):
    log("[INFO] custom message 123")
    err = capsys.readouterr().err
    assert "[INFO] custom message 123" in err
