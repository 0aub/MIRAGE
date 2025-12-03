"""
Test MIRAGE V2 Foundation Components
Tests configuration loading, model registry, and managers.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestV2ConfigLoader:
    """Test V2 configuration loader"""

    def test_load_defaults(self):
        """Test loading default configuration"""
        from config.v2_config_loader import V2ConfigLoader

        loader = V2ConfigLoader()

        # Check basic structure exists
        assert loader.config is not None
        assert "models" in loader.config
        assert "databases" in loader.config

    def test_get_nested_value(self):
        """Test nested value access"""
        from config.v2_config_loader import V2ConfigLoader

        loader = V2ConfigLoader()

        # Test dot notation access
        llm_config = loader.get("models.llm")
        assert llm_config is not None
        assert "model_id" in llm_config

        # Test with default
        missing = loader.get("nonexistent.path", default="fallback")
        assert missing == "fallback"

    def test_env_var_substitution(self):
        """Test environment variable substitution"""
        from config.v2_config_loader import V2ConfigLoader

        # Set test env var
        os.environ["TEST_VAR"] = "test_value"

        loader = V2ConfigLoader()

        # Create test config dict
        test_value = loader._substitute_env_vars("${TEST_VAR}")
        assert test_value == "test_value"

        # Test with default
        default_value = loader._substitute_env_vars("${MISSING_VAR:default}")
        assert default_value == "default"

        # Cleanup
        del os.environ["TEST_VAR"]

    def test_config_dict_access(self):
        """Test ConfigDict dot notation access"""
        from config.v2_config_loader import ConfigDict

        d = ConfigDict({
            "level1": {
                "level2": {
                    "value": 42
                }
            }
        })

        # Dot notation access
        assert d.level1.level2.value == 42

        # Nested get
        assert d.get_nested("level1.level2.value") == 42
        assert d.get_nested("missing.path", "default") == "default"

    def test_typed_accessors(self):
        """Test typed configuration accessors"""
        from config.v2_config_loader import V2ConfigLoader

        loader = V2ConfigLoader()

        llm = loader.get_llm_config()
        assert "model_id" in llm or llm == {}

        embedding = loader.get_embedding_config()
        assert embedding is not None

        retrieval = loader.get_retrieval_config()
        assert retrieval is not None


class TestModelRegistry:
    """Test model registry"""

    def test_registry_init(self):
        """Test registry initialization"""
        from core.models.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(cache_dir=tmpdir)

            assert registry.cache_dir.exists()
            assert registry.default_device in ["cuda", "cpu", "mps"]

    def test_register_model(self):
        """Test model registration"""
        from core.models.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(cache_dir=tmpdir)

            info = registry.register(
                "test/model",
                model_type="embedding",
                device="cpu"
            )

            assert info.model_id == "test/model"
            assert info.model_type == "embedding"
            assert info.device == "cpu"

            # Check persistence
            listed = registry.list_models()
            assert len(listed) == 1

    def test_cache_path(self):
        """Test cache path generation"""
        from core.models.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(cache_dir=tmpdir)

            path = registry._get_model_cache_path("org/model-name")
            assert "org--model-name" in str(path)

    def test_cache_stats(self):
        """Test cache statistics"""
        from core.models.registry import ModelRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(cache_dir=tmpdir)

            total_bytes, num_models = registry.get_cache_size()
            assert total_bytes >= 0
            assert num_models >= 0


class TestEmbeddingManager:
    """Test embedding manager"""

    def test_manager_init(self):
        """Test embedding manager initialization"""
        from core.models.embedding_manager import EmbeddingManager

        # Initialize without loading model
        manager = EmbeddingManager(
            model_id="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            cache_embeddings=True
        )

        assert manager.model_id is not None
        assert manager.normalize is True
        assert manager.batch_size == 32

    def test_cache_key_generation(self):
        """Test cache key generation"""
        from core.models.embedding_manager import EmbeddingManager

        manager = EmbeddingManager()

        key1 = manager._get_cache_key("test text", "passage")
        key2 = manager._get_cache_key("test text", "query")
        key3 = manager._get_cache_key("different text", "passage")

        # Same text, different task = different key
        assert key1 != key2

        # Different text, same task = different key
        assert key1 != key3

    def test_task_prefix_application(self):
        """Test task prefix application"""
        from core.models.embedding_manager import EmbeddingManager

        manager = EmbeddingManager(model_id="e5-base")

        texts = ["test1", "test2"]

        # Should apply prefix for e5 model
        processed = manager._apply_task_prefix(texts, "passage")
        assert processed[0].startswith("passage: ")

        # Non-prefixed model should not apply
        manager2 = EmbeddingManager(model_id="bert-base")
        processed2 = manager2._apply_task_prefix(texts, "passage")
        assert processed2[0] == "test1"


class TestLLMManager:
    """Test LLM manager"""

    def test_manager_init(self):
        """Test LLM manager initialization"""
        from core.models.llm_manager import LLMManager

        manager = LLMManager(
            endpoint="http://localhost:8765",
            model_id="test-model"
        )

        assert manager.endpoint == "http://localhost:8765"
        assert manager.model_id == "test-model"
        assert manager.timeout == 60.0

    def test_generation_config(self):
        """Test generation configuration"""
        from core.models.llm_manager import GenerationConfig

        config = GenerationConfig(
            max_tokens=1024,
            temperature=0.5
        )

        assert config.max_tokens == 1024
        assert config.temperature == 0.5
        assert config.top_p == 0.95  # default

    def test_request_preparation(self):
        """Test request payload preparation"""
        from core.models.llm_manager import LLMManager, GenerationConfig

        manager = LLMManager(endpoint="http://test:8765")

        config = GenerationConfig(
            max_tokens=512,
            temperature=0.3,
            stop_sequences=["</s>"]
        )

        payload = manager._prepare_request("Test prompt", config)

        assert payload["inputs"] == "Test prompt"
        assert payload["parameters"]["max_new_tokens"] == 512
        assert payload["parameters"]["temperature"] == 0.3
        assert payload["parameters"]["stop"] == ["</s>"]

    def test_token_estimation(self):
        """Test token count estimation"""
        from core.models.llm_manager import LLMManager

        manager = LLMManager(endpoint="http://test:8765")

        # Short text
        tokens = manager._estimate_tokens("Hello world")
        assert tokens > 0

        # Longer text
        long_text = "This is a longer text " * 100
        long_tokens = manager._estimate_tokens(long_text)
        assert long_tokens > tokens


class TestIntegration:
    """Integration tests"""

    def test_config_to_embedding_manager(self):
        """Test passing config to embedding manager"""
        from config.v2_config_loader import V2ConfigLoader
        from core.models.embedding_manager import EmbeddingManager

        loader = V2ConfigLoader()
        embedding_config = loader.get_embedding_config()

        manager = EmbeddingManager(
            model_id=embedding_config.get("model_id"),
            dimensions=embedding_config.get("dimensions"),
            normalize=embedding_config.get("normalize", True),
            batch_size=embedding_config.get("batch_size", 32)
        )

        assert manager.model_id is not None

    def test_config_to_llm_manager(self):
        """Test passing config to LLM manager"""
        from config.v2_config_loader import V2ConfigLoader
        from core.models.llm_manager import LLMManager, GenerationConfig

        loader = V2ConfigLoader()
        llm_config = loader.get_llm_config()

        config = GenerationConfig(
            max_tokens=llm_config.get("max_tokens", 2048),
            temperature=llm_config.get("temperature", 0.1)
        )

        manager = LLMManager(
            endpoint=llm_config.get("endpoint", "http://tgi:8765"),
            model_id=llm_config.get("model_id"),
            default_config=config
        )

        assert manager.endpoint is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
