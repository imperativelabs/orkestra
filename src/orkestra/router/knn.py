"""Provider-agnostic KNN router."""

import pickle

from orkestra.router.cache import get_router_path
from orkestra.router.embedder import get_longformer_embedding


class KNNRouter:
    """Routes queries to the optimal model for a given provider using KNN.

    The KNN classifier is trained on Longformer embeddings of benchmark queries.
    Each provider has its own trained model that maps queries to the provider's
    model tiers (budget, balanced, premium).
    """

    def __init__(self, provider: str):
        pkl_path = get_router_path(provider)
        with open(pkl_path, "rb") as f:
            self._knn = pickle.load(f)
        self._provider = provider

    def route(self, query: str) -> str:
        """Route a query to the best model for this provider.

        Args:
            query: The prompt text to route.

        Returns:
            Model name string.
        """
        embedding = get_longformer_embedding(query).reshape(1, -1)
        return self._knn.predict(embedding)[0]
