"""Provider backend registry."""

from orkestra.providers.base import ProviderBackend

_BACKEND_CLASSES: dict[str, str] = {
    "google": "orkestra.providers.google.GoogleBackend",
    "anthropic": "orkestra.providers.anthropic.AnthropicBackend",
    "openai": "orkestra.providers.openai.OpenAIBackend",
}

SUPPORTED_PROVIDERS = set(_BACKEND_CLASSES.keys())


def create_backend(name: str, api_key: str) -> ProviderBackend:
    """Create a provider backend by name.

    Args:
        name: Provider name (e.g. "google", "anthropic", "openai").
        api_key: API key for the provider.

    Returns:
        A ProviderBackend instance.

    Raises:
        ValueError: If provider is not supported.
        ImportError: If the provider's SDK is not installed.
    """
    if name not in _BACKEND_CLASSES:
        raise ValueError(
            f"Unknown provider '{name}'. "
            f"Supported: {sorted(SUPPORTED_PROVIDERS)}"
        )

    module_path, class_name = _BACKEND_CLASSES[name].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(api_key)


__all__ = ["ProviderBackend", "create_backend", "SUPPORTED_PROVIDERS"]
