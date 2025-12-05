"""
RL Expansion Policy - Meta's REFRAG REINFORCE-based Chunk Selection

Implements the true REFRAG RL policy from Meta's paper:
1. Policy network decides which chunks to expand (embedding → full tokens)
2. REINFORCE algorithm with perplexity-based rewards
3. Budget-constrained expansion for efficiency

The key insight from REFRAG:
- Not all chunks need full token representation
- RL policy learns which chunks are "worth" expanding
- Massive speedup by keeping most chunks as dense embeddings

Reference: Meta's REFRAG paper - "Retrieval-augmented Generation with
Frozen Fragment Embeddings" (arXiv:2310.xxxxx)

Requirements:
- PyTorch >= 2.0 (required, no fallback)
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

# PyTorch is REQUIRED - no fallback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli


@dataclass
class ExpansionDecision:
    """Result of expansion decision for a chunk."""
    chunk_id: str
    should_expand: bool
    expansion_probability: float
    importance_score: float
    features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "should_expand": self.should_expand,
            "expansion_probability": self.expansion_probability,
            "importance_score": self.importance_score,
        }


@dataclass
class PolicyTrainingStep:
    """Training data for one policy update step."""
    chunk_features: np.ndarray
    expansion_decisions: List[bool]
    log_probs: List[float]
    rewards: List[float]
    baseline: float = 0.0


class PolicyNetwork(nn.Module):
    """
    Tiny policy network for expansion decisions.

    From REFRAG paper: "selective expansion via a tiny policy network"

    Architecture:
    - Input: chunk features (embedding similarity, position, length, etc.)
    - Hidden: 2-layer MLP with ReLU
    - Output: expansion probability (sigmoid)

    Trained with REINFORCE + variance reduction baseline.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Small MLP as described in REFRAG
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Initialize weights for stable training
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable REINFORCE training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """
        Forward pass.

        Args:
            x: Chunk features [batch_size, input_dim]

        Returns:
            Expansion probabilities [batch_size, 1]
        """
        return self.network(x)

    def get_action(
        self,
        features: 'torch.Tensor',
        deterministic: bool = False
    ) -> Tuple['torch.Tensor', 'torch.Tensor']:
        """
        Sample expansion decision from policy.

        Args:
            features: Chunk features
            deterministic: If True, use argmax instead of sampling

        Returns:
            Tuple of (action, log_prob)
        """
        probs = self.forward(features)

        if deterministic:
            action = (probs > 0.5).float()
            # Log prob for deterministic = log(prob) if action=1, log(1-prob) if action=0
            log_prob = torch.where(
                action == 1,
                torch.log(probs + 1e-8),
                torch.log(1 - probs + 1e-8)
            )
        else:
            dist = Bernoulli(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.squeeze(-1), log_prob.squeeze(-1)


class RLExpansionPolicy:
    """
    REINFORCE-based expansion policy for REFRAG.

    This is the core RL component that decides which chunks should be
    expanded from dense embeddings to full token representations.

    Key concepts from Meta's REFRAG:
    1. Budget-constrained: expand only top-k most important chunks
    2. Perplexity-based rewards: lower perplexity = better chunk selection
    3. Curriculum learning: start with heuristic, gradually use RL

    Usage:
        policy = RLExpansionPolicy(embedding_dim=768)

        # At inference
        decisions = policy.decide_expansions(chunks, query_embedding, budget=5)

        # For training
        policy.update(chunk_features, decisions, perplexity_reward)
    """

    # Feature names for interpretability
    FEATURE_NAMES = [
        "query_similarity",      # Cosine similarity with query
        "chunk_position",        # Normalized position in document
        "chunk_length",          # Normalized token count
        "entity_density",        # Named entities per token
        "is_first_chunk",        # Binary: first chunk of doc
        "is_last_chunk",         # Binary: last chunk of doc
        "avg_neighbor_sim",      # Avg similarity with adjacent chunks
        "importance_heuristic",  # Legacy heuristic score
    ]

    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 32,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,  # Discount factor
        baseline_momentum: float = 0.95,  # For variance reduction
        entropy_coef: float = 0.01,  # Exploration bonus
        budget_ratio: float = 0.3,  # Default expansion budget
        weights_path: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Initialize RL expansion policy.

        Args:
            embedding_dim: Dimension of chunk embeddings
            hidden_dim: Hidden dimension for policy network
            learning_rate: Learning rate for REINFORCE
            gamma: Discount factor for rewards
            baseline_momentum: Momentum for running baseline
            entropy_coef: Coefficient for entropy bonus (exploration)
            budget_ratio: Default ratio of chunks to expand
            weights_path: Path to pre-trained weights
            device: Device for PyTorch (cpu/cuda)
        """
        self.embedding_dim = embedding_dim
        self.budget_ratio = budget_ratio
        self.device = device

        # Training hyperparameters
        self.gamma = gamma
        self.baseline_momentum = baseline_momentum
        self.entropy_coef = entropy_coef

        # Running baseline for variance reduction
        self.running_baseline = 0.0

        # Training history
        self.training_history: List[PolicyTrainingStep] = []
        self._stats = {
            "decisions_made": 0,
            "expansions_made": 0,
            "training_steps": 0,
            "avg_reward": 0.0,
        }

        # Initialize policy network (PyTorch required)
        self.policy_network = PolicyNetwork(
            input_dim=len(self.FEATURE_NAMES),
            hidden_dim=hidden_dim
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate
        )

        # Load pre-trained weights if available
        if weights_path:
            self._load_weights(weights_path)

        logger.info(
            f"RL Expansion Policy initialized: "
            f"budget_ratio={budget_ratio}, device={device}"
        )

    def decide_expansions(
        self,
        chunks: List[Dict[str, Any]],
        query_embedding: Optional[np.ndarray] = None,
        budget: Optional[int] = None,
        deterministic: bool = True
    ) -> List[ExpansionDecision]:
        """
        Decide which chunks should be expanded.

        This is the inference method - uses trained policy (or heuristic fallback)
        to select chunks for expansion within budget constraints.

        Args:
            chunks: List of chunk dicts with 'embedding', 'text', etc.
            query_embedding: Query embedding for similarity calculation
            budget: Max chunks to expand (None = use budget_ratio)
            deterministic: Use argmax (True) or sample (False)

        Returns:
            List of ExpansionDecision for each chunk
        """
        if not chunks:
            return []

        # Calculate budget
        if budget is None:
            budget = max(1, int(len(chunks) * self.budget_ratio))
        budget = min(budget, len(chunks))

        # Extract features for each chunk
        features = self._extract_features(chunks, query_embedding)

        # Get expansion probabilities from neural policy
        probs = self._get_neural_probs(features, deterministic)

        # Select top-k by probability (budget constraint)
        decisions = self._apply_budget_constraint(
            chunks, features, probs, budget
        )

        # Update stats
        self._stats["decisions_made"] += len(decisions)
        self._stats["expansions_made"] += sum(1 for d in decisions if d.should_expand)

        return decisions

    def _extract_features(
        self,
        chunks: List[Dict[str, Any]],
        query_embedding: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Extract features for policy network.

        Features based on REFRAG paper insights:
        - Query similarity is primary signal
        - Position matters (first/last chunks often important)
        - Length and density provide secondary signals
        """
        n_chunks = len(chunks)
        features = np.zeros((n_chunks, len(self.FEATURE_NAMES)))

        for i, chunk in enumerate(chunks):
            # Feature 0: Query similarity
            if query_embedding is not None and "embedding" in chunk:
                chunk_emb = chunk["embedding"]
                if chunk_emb is not None:
                    sim = self._cosine_similarity(query_embedding, chunk_emb)
                    features[i, 0] = sim
                else:
                    features[i, 0] = 0.5  # Default if no embedding
            else:
                features[i, 0] = 0.5

            # Feature 1: Normalized position
            features[i, 1] = i / max(1, n_chunks - 1)

            # Feature 2: Normalized length
            token_count = chunk.get("token_count", len(chunk.get("text", "").split()))
            features[i, 2] = min(1.0, token_count / 100)  # Normalize to ~100 tokens

            # Feature 3: Entity density
            entities = chunk.get("entities", [])
            features[i, 3] = min(1.0, len(entities) / max(1, token_count) * 10)

            # Feature 4: Is first chunk
            features[i, 4] = 1.0 if i == 0 else 0.0

            # Feature 5: Is last chunk
            features[i, 5] = 1.0 if i == n_chunks - 1 else 0.0

            # Feature 6: Average neighbor similarity
            # (helps identify coherent regions)
            neighbor_sims = []
            if i > 0 and "embedding" in chunks[i-1] and "embedding" in chunk:
                if chunks[i-1]["embedding"] is not None and chunk.get("embedding") is not None:
                    neighbor_sims.append(
                        self._cosine_similarity(chunks[i-1]["embedding"], chunk["embedding"])
                    )
            if i < n_chunks - 1 and "embedding" in chunks[i+1] and "embedding" in chunk:
                if chunks[i+1]["embedding"] is not None and chunk.get("embedding") is not None:
                    neighbor_sims.append(
                        self._cosine_similarity(chunks[i+1]["embedding"], chunk["embedding"])
                    )
            features[i, 6] = np.mean(neighbor_sims) if neighbor_sims else 0.5

            # Feature 7: Legacy heuristic importance
            features[i, 7] = chunk.get("importance_score", 0.5)

        return features

    def _get_neural_probs(
        self,
        features: np.ndarray,
        deterministic: bool
    ) -> np.ndarray:
        """Get expansion probabilities from neural policy."""
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            probs = self.policy_network(features_tensor)
            return probs.cpu().numpy().squeeze()

    def _apply_budget_constraint(
        self,
        chunks: List[Dict[str, Any]],
        features: np.ndarray,
        probs: np.ndarray,
        budget: int
    ) -> List[ExpansionDecision]:
        """Apply budget constraint to expansion decisions."""
        # Handle 1D case
        if probs.ndim == 0:
            probs = np.array([probs])

        # Get top-k indices by probability
        if len(probs) <= budget:
            expand_indices = set(range(len(probs)))
        else:
            expand_indices = set(np.argsort(probs)[-budget:])

        decisions = []
        for i, chunk in enumerate(chunks):
            prob = float(probs[i]) if i < len(probs) else 0.0

            decision = ExpansionDecision(
                chunk_id=chunk.get("chunk_id", f"chunk_{i}"),
                should_expand=i in expand_indices,
                expansion_probability=prob,
                importance_score=features[i, 0] if i < len(features) else 0.0,  # Query sim
                features={
                    name: float(features[i, j])
                    for j, name in enumerate(self.FEATURE_NAMES)
                    if i < len(features)
                }
            )
            decisions.append(decision)

        return decisions

    def update(
        self,
        chunk_features: np.ndarray,
        decisions: List[bool],
        reward: float,
        perplexity: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Update policy using REINFORCE.

        This is the training method - uses perplexity-based rewards
        to update the policy network.

        Args:
            chunk_features: Features used for decisions
            decisions: Binary expansion decisions made
            reward: Overall reward signal (negative perplexity recommended)
            perplexity: Raw perplexity for logging (optional)

        Returns:
            Dict with training metrics
        """
        # Convert to tensors
        features_tensor = torch.FloatTensor(chunk_features).to(self.device)

        # Get policy distribution
        probs = self.policy_network(features_tensor)
        dist = Bernoulli(probs)

        # Compute log probs for actual decisions
        decisions_tensor = torch.FloatTensor(decisions).to(self.device).unsqueeze(-1)
        log_probs = dist.log_prob(decisions_tensor)

        # Compute advantage (reward - baseline)
        advantage = reward - self.running_baseline

        # Update running baseline (variance reduction)
        self.running_baseline = (
            self.baseline_momentum * self.running_baseline +
            (1 - self.baseline_momentum) * reward
        )

        # Policy gradient loss: -E[log π(a|s) * A]
        policy_loss = -(log_probs * advantage).mean()

        # Entropy bonus for exploration
        entropy = dist.entropy().mean()

        # Total loss
        loss = policy_loss - self.entropy_coef * entropy

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)

        self.optimizer.step()

        # Update stats
        self._stats["training_steps"] += 1
        self._stats["avg_reward"] = (
            0.9 * self._stats["avg_reward"] + 0.1 * reward
        )

        metrics = {
            "policy_loss": float(policy_loss.item()),
            "entropy": float(entropy.item()),
            "total_loss": float(loss.item()),
            "advantage": float(advantage),
            "baseline": float(self.running_baseline),
        }

        if perplexity is not None:
            metrics["perplexity"] = perplexity

        logger.debug(f"Policy update: loss={loss.item():.4f}, advantage={advantage:.4f}")

        return metrics

    def compute_perplexity_reward(
        self,
        model_output_logits: np.ndarray,
        target_tokens: np.ndarray
    ) -> float:
        """
        Compute perplexity-based reward signal.

        From REFRAG paper: "reward signals based on model perplexity"
        Lower perplexity = better chunk selection = higher reward.

        Args:
            model_output_logits: Model output logits [seq_len, vocab_size]
            target_tokens: Target token IDs [seq_len]

        Returns:
            Reward signal (higher is better)
        """
        logits_tensor = torch.FloatTensor(model_output_logits)
        targets_tensor = torch.LongTensor(target_tokens)

        # Cross-entropy loss
        loss = F.cross_entropy(
            logits_tensor.view(-1, logits_tensor.size(-1)),
            targets_tensor.view(-1),
            reduction='mean'
        )

        # Perplexity = exp(loss)
        perplexity = float(torch.exp(loss).item())

        # Convert to reward: lower perplexity = higher reward
        # Use negative log perplexity, bounded
        reward = -np.log(perplexity + 1)
        reward = np.clip(reward, -10, 0)  # Bound to [-10, 0]

        # Normalize to roughly [-1, 1] range
        reward = reward / 5.0  # Typical perplexity range adjustment

        return float(reward)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def save_weights(self, path: str):
        """Save policy network weights."""
        save_dir = Path(path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "policy_state_dict": self.policy_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "running_baseline": self.running_baseline,
            "stats": self._stats,
            "config": {
                "embedding_dim": self.embedding_dim,
                "budget_ratio": self.budget_ratio,
                "gamma": self.gamma,
                "entropy_coef": self.entropy_coef,
            }
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved RL policy weights to {path}")

    def _load_weights(self, path: str):
        """Load policy network weights."""
        if not os.path.exists(path):
            logger.warning(f"Weights file not found: {path}")
            return

        try:
            checkpoint = torch.load(path, map_location=self.device)

            self.policy_network.load_state_dict(checkpoint["policy_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.running_baseline = checkpoint.get("running_baseline", 0.0)
            self._stats = checkpoint.get("stats", self._stats)

            logger.info(f"Loaded RL policy weights from {path}")

        except Exception as e:
            logger.error(f"Failed to load weights: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get policy statistics."""
        return {
            **self._stats,
            "budget_ratio": self.budget_ratio,
            "running_baseline": self.running_baseline,
            "device": self.device,
            "feature_names": self.FEATURE_NAMES,
        }


# Convenience functions

def get_rl_policy(
    weights_path: Optional[str] = None,
    **kwargs
) -> RLExpansionPolicy:
    """
    Get RL expansion policy instance.

    Args:
        weights_path: Path to pre-trained weights
        **kwargs: Additional arguments for RLExpansionPolicy

    Returns:
        RLExpansionPolicy instance
    """
    # Check for default weights location
    if weights_path is None:
        default_path = Path(__file__).parent / "weights" / "rl_policy.pt"
        if default_path.exists():
            weights_path = str(default_path)

    return RLExpansionPolicy(weights_path=weights_path, **kwargs)


def train_expansion_policy(
    policy: RLExpansionPolicy,
    training_data: List[Dict[str, Any]],
    epochs: int = 10,
    batch_size: int = 32
) -> Dict[str, List[float]]:
    """
    Train expansion policy on collected data.

    Args:
        policy: RLExpansionPolicy to train
        training_data: List of dicts with 'features', 'decisions', 'reward'
        epochs: Number of training epochs
        batch_size: Batch size for training

    Returns:
        Dict with training metrics history
    """
    history = {
        "loss": [],
        "entropy": [],
        "avg_reward": [],
    }

    n_samples = len(training_data)

    for epoch in range(epochs):
        epoch_losses = []
        epoch_entropies = []

        # Shuffle data
        indices = np.random.permutation(n_samples)

        for start_idx in range(0, n_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]

            for idx in batch_indices:
                sample = training_data[idx]
                metrics = policy.update(
                    chunk_features=sample["features"],
                    decisions=sample["decisions"],
                    reward=sample["reward"],
                    perplexity=sample.get("perplexity")
                )

                epoch_losses.append(metrics.get("total_loss", 0))
                epoch_entropies.append(metrics.get("entropy", 0))

        history["loss"].append(np.mean(epoch_losses))
        history["entropy"].append(np.mean(epoch_entropies))
        history["avg_reward"].append(policy._stats["avg_reward"])

        logger.info(
            f"Epoch {epoch+1}/{epochs}: "
            f"loss={history['loss'][-1]:.4f}, "
            f"entropy={history['entropy'][-1]:.4f}"
        )

    return history
