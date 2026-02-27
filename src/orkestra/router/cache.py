"""Router model download and caching.

Router pkl files are cached in ~/.orkestra/routers/router-<provider>.pkl.
On first use, the model is downloaded from cloud storage and verified via SHA256.
"""

import hashlib
import shutil
import urllib.request
from pathlib import Path

CACHE_DIR = Path.home() / ".orkestra" / "routers"

# Repo root for local fallback (4 levels up from this file: router/cache.py → orkestra → src → repo)
_REPO_ROOT = Path(__file__).resolve().parents[3]

ROUTER_MANIFEST: dict[str, dict] = {
    "google": {
        "url": "http://imperativemachines.com/routers/router-google.pkl",
        "sha256": "11358e8d3417f33f2d8c3d8487990c86c6e5017310dd2840cd52305597b03ccb",
        "filename": "router-google.pkl",
        "version": "0.2.0",
    },
    "anthropic": {
        "url": "http://imperativemachines.com/routers/router-anthropic.pkl",
        "sha256": "0a7a58c43245fcd30600e629cee4dbd848ade65e74dedf18582a1f451bdf1d18",
        "filename": "router-anthropic.pkl",
        "version": "0.2.0",
    },
    "openai": {
        "url": "http://imperativemachines.com/routers/router-openai.pkl",
        "sha256": "916a8241c320e9dfa4917f9b0684debc721f8dd07dc5a9a6b0ad36287458baed",
        "filename": "router-openai.pkl",
        "version": "0.2.0",
    },
}

# Local fallback: routers/ directory at repo root (for development)
_LOCAL_FALLBACK: dict[str, Path] = {
    name: _REPO_ROOT / "routers" / info["filename"]
    for name, info in ROUTER_MANIFEST.items()
}


def get_router_path(provider: str) -> Path:
    """Return path to cached router model, downloading if needed.

    Checks in order:
    1. Cache directory (~/.orkestra/routers/)
    2. Cloud download (if URL configured)
    3. Local fallback (routers/ in repo root, for development)

    Args:
        provider: Provider name (e.g. "google", "anthropic", "openai").

    Returns:
        Path to the pkl file.

    Raises:
        ValueError: If provider is not supported.
        FileNotFoundError: If no router model is available.
    """
    if provider not in ROUTER_MANIFEST:
        available = sorted(ROUTER_MANIFEST.keys())
        raise ValueError(
            f"No router model available for '{provider}'. "
            f"Available providers: {available}"
        )

    manifest = ROUTER_MANIFEST[provider]
    cached_path = CACHE_DIR / manifest["filename"]

    if cached_path.exists():
        return cached_path

    # Try cloud download
    if manifest["url"]:
        try:
            return _download_router(provider)
        except Exception:
            pass  # Fall through to local fallback

    # Try local fallback
    fallback = _LOCAL_FALLBACK.get(provider)
    if fallback and fallback.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fallback, cached_path)
        return cached_path

    raise FileNotFoundError(
        f"Router model for '{provider}' not found. "
        f"Expected at {cached_path} or {fallback}."
    )


def _download_router(provider: str) -> Path:
    """Download router model from cloud and verify checksum."""
    manifest = ROUTER_MANIFEST[provider]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    dest = CACHE_DIR / manifest["filename"]
    tmp = dest.with_suffix(".tmp")

    try:
        urllib.request.urlretrieve(manifest["url"], tmp)

        if manifest["sha256"]:
            actual = hashlib.sha256(tmp.read_bytes()).hexdigest()
            if actual != manifest["sha256"]:
                tmp.unlink()
                raise RuntimeError(
                    f"Checksum mismatch for {provider} router. "
                    f"Expected {manifest['sha256']}, got {actual}."
                )

        tmp.rename(dest)
        return dest
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
