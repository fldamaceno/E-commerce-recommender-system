import numpy as np
import random
from typing import Dict, List

class EpsilonGreedy:
    def __init__(self, epsilon: float, n_features: int, learning_rate: float = 0.1):
        self.epsilon = epsilon
        self.lr = learning_rate
        self.n_features = n_features
        self.weights = {}  # item_id -> weight vector

    def _get_weights(self, item_id):
        """Return the weight vector for an item, initializing if needed."""
        if item_id not in self.weights:
            self.weights[item_id] = np.zeros(self.n_features)
        return self.weights[item_id]

    def _normalize(self, vector):
        """Normalize a vector safely (returns unchanged if norm is too small)."""
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 1e-8 else vector

    def predict(self, item_id, context_vector):
        """Return expected reward for item given its context."""
        weights = self._get_weights(item_id)
        context_vector = self._normalize(context_vector)
        return np.dot(weights, context_vector)

    def select_item(self, candidate_items_contexts: Dict[int, np.ndarray]) -> int:
        """
        Select an item using epsilon-greedy strategy.
        Vectorized for performance with large candidate sets.
        """
        # Exploration: pick random item
        if random.random() < self.epsilon:
            return random.choice(list(candidate_items_contexts.keys()))

        # Exploitation: compute scores in batch
        item_ids = np.array(list(candidate_items_contexts.keys()))
        contexts = np.stack([self._normalize(candidate_items_contexts[i]) for i in item_ids])

        # Ensure weights exist for all items
        for i in item_ids:
            self._get_weights(i)

        # Stack weights into matrix
        weights = np.stack([self.weights[i] for i in item_ids])

        # Compute dot products row-wise (fast)
        scores = np.einsum("ij,ij->i", weights, contexts)

        # Pick the item with highest score
        return item_ids[np.argmax(scores)]

    def select_top_k(self, candidate_items_contexts: Dict[int, np.ndarray], k: int) -> List[int]:
        """
        Select top k items efficiently using batch operations.
        Optional method for improved performance in recommender systems.
        """
        if len(candidate_items_contexts) <= k:
            return list(candidate_items_contexts.keys())
        
        # Exploration: pick random k items
        if random.random() < self.epsilon:
            all_items = list(candidate_items_contexts.keys())
            return random.sample(all_items, min(k, len(all_items)))
        
        # Exploitation: compute scores in batch and select top k
        item_ids = np.array(list(candidate_items_contexts.keys()))
        contexts = np.stack([self._normalize(candidate_items_contexts[i]) for i in item_ids])
        
        # Ensure weights exist for all items
        for i in item_ids:
            self._get_weights(i)
        
        # Stack weights into matrix
        weights = np.stack([self.weights[i] for i in item_ids])
        
        # Compute scores efficiently
        scores = np.einsum("ij,ij->i", weights, contexts)
        
        # Get top k indices (using argpartition for efficiency)
        top_k_indices = np.argpartition(scores, -k)[-k:]
        return item_ids[top_k_indices].tolist()

    def update(self, item_id: int, context_vector: np.ndarray, reward: float):
        """Update weights for an item after observing a reward."""
        context_vector = self._normalize(context_vector)
        prediction = self.predict(item_id, context_vector)
        error = reward - prediction

        self.weights[item_id] += self.lr * error * context_vector

        # Normalize weights lazily after update
        norm = np.linalg.norm(self.weights[item_id])
        if norm > 1e-8:
            self.weights[item_id] /= norm

