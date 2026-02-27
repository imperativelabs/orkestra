"""Tests for router/knn.py."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from orkestra.router.knn import KNNRouter


class TestKNNRouter:
    def test_route_calls_embedding_and_predict(self):
        fake_knn = MagicMock()
        fake_knn.predict.return_value = ["gemini-3-flash-preview"]

        with (
            patch("orkestra.router.knn.get_router_path") as mock_cache,
            patch("orkestra.router.knn.get_longformer_embedding") as mock_embed,
            patch("builtins.open", MagicMock()),
            patch("orkestra.router.knn.pickle") as mock_pickle,
        ):
            mock_cache.return_value = "/fake/path.pkl"
            mock_pickle.load.return_value = fake_knn
            mock_embed.return_value = np.zeros(768)

            router = KNNRouter("google")
            result = router.route("Hello world")

            mock_embed.assert_called_once_with("Hello world")
            fake_knn.predict.assert_called_once()
            assert result == "gemini-3-flash-preview"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            KNNRouter("nonexistent")
