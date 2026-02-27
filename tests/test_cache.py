"""Tests for router/cache.py."""

from unittest.mock import patch

import pytest

from orkestra.router.cache import ROUTER_MANIFEST, get_router_path


class TestGetRouterPath:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="No router model available"):
            get_router_path("nonexistent")

    def test_returns_cached_path_if_exists(self, tmp_path):
        pkl = tmp_path / "router-google.pkl"
        pkl.write_bytes(b"fake")
        with patch("orkestra.router.cache.CACHE_DIR", tmp_path):
            path = get_router_path("google")
            assert path == pkl

    def test_copies_local_fallback(self, tmp_path):
        # Create a fake fallback file
        fallback = tmp_path / "fallback" / "knnrouter.pkl"
        fallback.parent.mkdir()
        fallback.write_bytes(b"fake-pkl-data")

        cache_dir = tmp_path / "cache"

        with (
            patch("orkestra.router.cache.CACHE_DIR", cache_dir),
            patch("orkestra.router.cache._LOCAL_FALLBACK", {"google": fallback}),
            patch.dict(ROUTER_MANIFEST, {"google": {"url": "", "sha256": "", "filename": "router-google.pkl", "version": "0.2.0"}}),
        ):
            path = get_router_path("google")
            assert path.exists()
            assert path.read_bytes() == b"fake-pkl-data"

    def test_raises_when_no_source(self, tmp_path):
        with (
            patch("orkestra.router.cache.CACHE_DIR", tmp_path / "empty"),
            patch("orkestra.router.cache._LOCAL_FALLBACK", {}),
            patch.dict(ROUTER_MANIFEST, {"google": {"url": "", "sha256": "", "filename": "google_knn.pkl", "version": "0.2.0"}}),
        ):
            with pytest.raises(FileNotFoundError, match="not found"):
                get_router_path("google")


class TestManifest:
    def test_all_providers_in_manifest(self):
        assert "google" in ROUTER_MANIFEST
        assert "anthropic" in ROUTER_MANIFEST
        assert "openai" in ROUTER_MANIFEST

    def test_manifest_has_required_fields(self):
        for provider, info in ROUTER_MANIFEST.items():
            assert "filename" in info
            assert "version" in info
