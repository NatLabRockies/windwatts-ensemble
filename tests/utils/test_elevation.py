"""Tests for wem.utils.elevation."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from wem.utils.elevation import identify_point_3857, identify_with_fallbacks


class TestIdentifyPoint3857:
    def _mock_session(self, json_response, status_code=200):
        session = MagicMock()
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_response
        resp.raise_for_status.return_value = None
        session.post.return_value = resp
        return session

    def test_success(self):
        session = self._mock_session({"value": "1234.5"})
        result = identify_point_3857(0, 0, session=session)
        assert result == 1234.5

    def test_error_response(self):
        session = self._mock_session({"error": {"code": 500}})
        result = identify_point_3857(0, 0, session=session)
        assert result is None

    def test_null_value(self):
        session = self._mock_session({"value": None})
        result = identify_point_3857(0, 0, session=session)
        assert result is None

    def test_non_finite(self):
        session = self._mock_session({"value": "inf"})
        result = identify_point_3857(0, 0, session=session)
        assert result is None

    def test_network_error(self):
        session = MagicMock()
        session.post.side_effect = ConnectionError("timeout")
        result = identify_point_3857(0, 0, session=session)
        assert result is None

    def test_nan_value(self):
        session = self._mock_session({"value": "nan"})
        result = identify_point_3857(0, 0, session=session)
        assert result is None


class TestIdentifyWithFallbacks:
    def test_fallback_cascade(self):
        session = MagicMock()
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            if call_count[0] <= 2:
                resp.json.return_value = {"error": {"code": 500}}
            else:
                resp.json.return_value = {"value": "42.0"}
            return resp

        session.post.side_effect = side_effect
        result = identify_with_fallbacks(0, 0, None, session, timeout=5.0)
        assert result == 42.0
        assert call_count[0] == 3  # tried 10m, 30m, 90m

    def test_all_fail(self):
        session = MagicMock()
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"error": {"code": 500}}
        session.post.return_value = resp
        result = identify_with_fallbacks(0, 0, None, session, timeout=5.0)
        assert result is None

    def test_first_succeeds(self):
        session = MagicMock()
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"value": "100.0"}
        session.post.return_value = resp
        result = identify_with_fallbacks(0, 0, None, session, timeout=5.0)
        assert result == 100.0
        assert session.post.call_count == 1
