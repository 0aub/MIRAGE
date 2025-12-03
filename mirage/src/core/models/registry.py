"""
MIRAGE V2 Model Registry
Handles HuggingFace model downloads, caching, and version management.

Features:
- Automatic model download from HuggingFace Hub
- Local caching in configurable directory
- Model version tracking
- GPU/CPU device management
- Lazy loading to minimize memory usage
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from loguru import logger

try:
    from huggingface_hub import snapshot_download, hf_hub_download, HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("huggingface_hub not installed - model downloads will be limited")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch not installed - GPU detection limited")


@dataclass
class ModelInfo:
    """Information about a registered model"""
    model_id: str
    model_type: str  # "embedding" | "llm" | "reranker" | "ner"
    local_path: Optional[str] = None
    revision: str = "main"
    downloaded: bool = False
    download_date: Optional[str] = None
    size_bytes: Optional[int] = None
    device: str = "auto"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        return cls(**data)


class ModelRegistry:
    """
    Central registry for managing ML models.

    Responsibilities:
    - Track registered models and their locations
    - Download models from HuggingFace Hub
    - Cache models locally for offline use
    - Manage model versions and updates
    - Handle GPU/CPU device selection
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        offline_mode: bool = False
    ):
        """
        Initialize model registry.

        Args:
            cache_dir: Directory for model cache (default: /data/models_cache)
            offline_mode: If True, only use locally cached models
        """
        self.cache_dir = Path(cache_dir or os.environ.get(
            "MODELS_CACHE_DIR",
            "/data/models_cache"
        ))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.offline_mode = offline_mode
        self._registry: Dict[str, ModelInfo] = {}
        self._loaded_models: Dict[str, Any] = {}

        # Registry persistence file
        self._registry_file = self.cache_dir / "registry.json"

        # Load existing registry
        self._load_registry()

        # Detect available device
        self._default_device = self._detect_device()

        logger.info(
            f"ModelRegistry initialized: cache={self.cache_dir}, "
            f"device={self._default_device}, offline={offline_mode}"
        )

    def _detect_device(self) -> str:
        """Detect best available device for inference"""
        if not TORCH_AVAILABLE:
            return "cpu"

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
            return "cuda"

        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Apple MPS detected")
            return "mps"

        return "cpu"

    def _load_registry(self) -> None:
        """Load registry from disk"""
        if self._registry_file.exists():
            try:
                with open(self._registry_file, 'r') as f:
                    data = json.load(f)
                    self._registry = {
                        k: ModelInfo.from_dict(v)
                        for k, v in data.items()
                    }
                logger.debug(f"Loaded {len(self._registry)} models from registry")
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
                self._registry = {}

    def _save_registry(self) -> None:
        """Persist registry to disk"""
        try:
            with open(self._registry_file, 'w') as f:
                data = {k: v.to_dict() for k, v in self._registry.items()}
                json.dump(data, f, indent=2)
            logger.debug("Registry saved to disk")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def _get_model_cache_path(self, model_id: str) -> Path:
        """Get local cache path for a model"""
        # Convert model_id to safe directory name
        safe_name = model_id.replace("/", "--")
        return self.cache_dir / safe_name

    def register(
        self,
        model_id: str,
        model_type: str,
        device: str = "auto",
        revision: str = "main",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModelInfo:
        """
        Register a model in the registry.

        Args:
            model_id: HuggingFace model ID (e.g., "sentence-transformers/all-MiniLM-L6-v2")
            model_type: Type of model ("embedding", "llm", "reranker", "ner")
            device: Target device ("cuda", "cpu", "auto")
            revision: Model revision/branch
            metadata: Additional metadata

        Returns:
            ModelInfo object
        """
        # Check if already registered
        if model_id in self._registry:
            info = self._registry[model_id]
            logger.debug(f"Model already registered: {model_id}")
            return info

        # Resolve device
        if device == "auto":
            device = self._default_device

        # Create model info
        local_path = self._get_model_cache_path(model_id)
        downloaded = local_path.exists() and any(local_path.iterdir())

        info = ModelInfo(
            model_id=model_id,
            model_type=model_type,
            local_path=str(local_path),
            revision=revision,
            downloaded=downloaded,
            device=device,
            metadata=metadata or {}
        )

        if downloaded:
            info.download_date = datetime.now().isoformat()
            info.size_bytes = sum(
                f.stat().st_size for f in local_path.rglob("*") if f.is_file()
            )

        self._registry[model_id] = info
        self._save_registry()

        logger.info(f"Registered model: {model_id} (type={model_type}, downloaded={downloaded})")
        return info

    def download(
        self,
        model_id: str,
        force: bool = False,
        revision: str = "main"
    ) -> Path:
        """
        Download a model from HuggingFace Hub.

        Args:
            model_id: HuggingFace model ID
            force: Force re-download even if cached
            revision: Model revision/branch

        Returns:
            Local path to downloaded model
        """
        if not HF_AVAILABLE:
            raise RuntimeError("huggingface_hub not installed. Install with: pip install huggingface-hub")

        if self.offline_mode:
            local_path = self._get_model_cache_path(model_id)
            if local_path.exists():
                return local_path
            raise RuntimeError(f"Model not cached and offline mode enabled: {model_id}")

        local_path = self._get_model_cache_path(model_id)

        # Skip if already downloaded and not forcing
        if local_path.exists() and any(local_path.iterdir()) and not force:
            logger.info(f"Model already cached: {model_id}")
            return local_path

        logger.info(f"Downloading model: {model_id} (revision={revision})")

        try:
            # Download using snapshot_download for full model
            downloaded_path = snapshot_download(
                repo_id=model_id,
                revision=revision,
                local_dir=str(local_path),
                local_dir_use_symlinks=False
            )

            # Update registry
            if model_id in self._registry:
                info = self._registry[model_id]
                info.downloaded = True
                info.download_date = datetime.now().isoformat()
                info.size_bytes = sum(
                    f.stat().st_size for f in local_path.rglob("*") if f.is_file()
                )
                self._save_registry()

            logger.info(f"Model downloaded: {model_id} -> {local_path}")
            return Path(downloaded_path)

        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            raise

    def get_local_path(self, model_id: str, auto_download: bool = True) -> Optional[Path]:
        """
        Get local path for a model, optionally downloading if needed.

        Args:
            model_id: Model ID
            auto_download: Download if not cached

        Returns:
            Local path or None if not available
        """
        local_path = self._get_model_cache_path(model_id)

        if local_path.exists() and any(local_path.iterdir()):
            return local_path

        if auto_download and not self.offline_mode:
            return self.download(model_id)

        return None

    def is_downloaded(self, model_id: str) -> bool:
        """Check if a model is downloaded"""
        local_path = self._get_model_cache_path(model_id)
        return local_path.exists() and any(local_path.iterdir())

    def get_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get model info from registry"""
        return self._registry.get(model_id)

    def list_models(self, model_type: Optional[str] = None) -> List[ModelInfo]:
        """
        List registered models.

        Args:
            model_type: Filter by type (optional)

        Returns:
            List of ModelInfo objects
        """
        models = list(self._registry.values())

        if model_type:
            models = [m for m in models if m.model_type == model_type]

        return models

    def list_downloaded(self) -> List[ModelInfo]:
        """List only downloaded models"""
        return [m for m in self._registry.values() if m.downloaded]

    def delete(self, model_id: str, remove_files: bool = True) -> bool:
        """
        Remove a model from registry and optionally delete files.

        Args:
            model_id: Model ID
            remove_files: Also delete downloaded files

        Returns:
            True if removed
        """
        if model_id not in self._registry:
            return False

        if remove_files:
            local_path = self._get_model_cache_path(model_id)
            if local_path.exists():
                import shutil
                shutil.rmtree(local_path)
                logger.info(f"Deleted model files: {local_path}")

        # Remove from loaded models
        if model_id in self._loaded_models:
            del self._loaded_models[model_id]

        del self._registry[model_id]
        self._save_registry()

        logger.info(f"Removed model from registry: {model_id}")
        return True

    def get_cache_size(self) -> Tuple[int, int]:
        """
        Get cache statistics.

        Returns:
            Tuple of (total_bytes, num_models)
        """
        total_bytes = 0
        num_models = 0

        for path in self.cache_dir.iterdir():
            if path.is_dir() and path.name != "registry.json":
                num_models += 1
                total_bytes += sum(
                    f.stat().st_size for f in path.rglob("*") if f.is_file()
                )

        return total_bytes, num_models

    def clear_cache(self, keep_registered: bool = True) -> int:
        """
        Clear model cache.

        Args:
            keep_registered: Keep models that are registered

        Returns:
            Number of bytes freed
        """
        freed_bytes = 0
        import shutil

        for path in self.cache_dir.iterdir():
            if path.is_dir() and path.name != "registry.json":
                model_id = path.name.replace("--", "/")

                if keep_registered and model_id in self._registry:
                    continue

                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                shutil.rmtree(path)
                freed_bytes += size
                logger.info(f"Deleted: {path}")

        logger.info(f"Freed {freed_bytes / (1024**2):.1f} MB from cache")
        return freed_bytes

    @property
    def default_device(self) -> str:
        """Get default device for model loading"""
        return self._default_device


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_model_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get or create global ModelRegistry instance"""
    global _model_registry

    if _model_registry is None:
        _model_registry = ModelRegistry()

    return _model_registry


def register_model(
    model_id: str,
    model_type: str,
    **kwargs
) -> ModelInfo:
    """Convenience function to register a model"""
    return get_model_registry().register(model_id, model_type, **kwargs)


def download_model(model_id: str, **kwargs) -> Path:
    """Convenience function to download a model"""
    return get_model_registry().download(model_id, **kwargs)
