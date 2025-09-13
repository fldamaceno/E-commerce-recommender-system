import numpy as np
import random
from typing import Dict, List, Tuple
import numba

class EpsilonGreedy:
    def __init__(self, epsilon: float, n_features: int, learning_rate: float = 0.1):
        self.epsilon = epsilon
        self.lr = learning_rate
        self.n_features = n_features
        self.weights = {}  # item_id -> weight vector
        self._weight_cache = {}  # Cache for normalized weights
        self._cache_dirty = set()  # Track which weights need cache update

    def _get_weights(self, item_id):
        """Return the weight vector for an item, initializing if needed."""
        if item_id not in self.weights:
            self.weights[item_id] = np.zeros(self.n_features, dtype=np.float32)
            self._weight_cache[item_id] = np.zeros(self.n_features, dtype=np.float32)
        return self.weights[item_id]

    def _normalize_vector(self, vector):
        """Fast vector normalization with SIMD optimization."""
        norm = np.sqrt(np.dot(vector, vector))
        if norm > 1e-12:
            return vector * (1.0 / norm)
        return vector

    def _normalize_weights_batch(self, item_ids):
        """Normalize weights for multiple items efficiently."""
        for item_id in item_ids:
            if item_id in self._cache_dirty:
                weights = self.weights[item_id]
                norm = np.sqrt(np.dot(weights, weights))
                if norm > 1e-12:
                    self._weight_cache[item_id] = weights * (1.0 / norm)
                else:
                    self._weight_cache[item_id] = weights.copy()
                self._cache_dirty.remove(item_id)

    def predict_batch(self, item_ids: List[int], contexts: List[np.ndarray]) -> np.ndarray:
        """Ultra-fast batch prediction using precomputed normalized weights."""
        if not item_ids:
            return np.array([], dtype=np.float32)
        
        # Ensure weights are normalized
        self._normalize_weights_batch(item_ids)
        
        # Use list comprehension for faster context normalization
        contexts_norm = [self._normalize_vector(ctx) for ctx in contexts]
        
        # Compute scores using efficient dot products
        scores = np.empty(len(item_ids), dtype=np.float32)
        for i, item_id in enumerate(item_ids):
            scores[i] = np.dot(self._weight_cache[item_id], contexts_norm[i])
        
        return scores

    @staticmethod
    @numba.jit(nopython=True)
    def _compute_scores_numba(weights_matrix, contexts_norm):
        """Numba-accelerated score computation."""
        n = weights_matrix.shape[0]
        scores = np.empty(n, dtype=np.float32)
        for i in range(n):
            scores[i] = np.dot(weights_matrix[i], contexts_norm[i])
        return scores

    def select_item(self, candidate_items_contexts: Dict[int, np.ndarray]) -> int:
        """Optimized item selection for large candidate sets."""
        if not candidate_items_contexts:
            return None
            
        n_candidates = len(candidate_items_contexts)
        
        # Exploration with probability epsilon
        if random.random() < self.epsilon:
            return random.choice(list(candidate_items_contexts.keys()))

        # For very large candidate sets, use optimized batch processing
        item_ids = list(candidate_items_contexts.keys())
        contexts = list(candidate_items_contexts.values())
        
        scores = self.predict_batch(item_ids, contexts)
        return item_ids[np.argmax(scores)]

    def select_top_k(self, candidate_items_contexts: Dict[int, np.ndarray], k: int) -> List[int]:
        """Highly optimized top-k selection without candidate limitation."""
        if not candidate_items_contexts:
            return []
            
        n_candidates = len(candidate_items_contexts)
        
        if n_candidates <= k:
            return list(candidate_items_contexts.keys())
        
        # Exploration
        if random.random() < self.epsilon:
            all_items = list(candidate_items_contexts.keys())
            return random.sample(all_items, k)
        
        # Batch processing for exploitation
        item_ids = list(candidate_items_contexts.keys())
        contexts = list(candidate_items_contexts.values())
        
        scores = self.predict_batch(item_ids, contexts)
        
        # Use argpartition for O(n) complexity instead of O(n log n)
        if k < n_candidates // 2:
            # For small k, use argpartition on the larger end
            indices = np.argpartition(scores, -k)[-k:]
        else:
            # For large k, partition the smaller end
            indices = np.argpartition(scores, k)[:k]
        
        return [item_ids[i] for i in indices]

    def update_batch(self, updates: List[Tuple[int, np.ndarray, float]]):
        """Batch update for maximum efficiency."""
        for item_id, context_vector, reward in updates:
            self.update(item_id, context_vector, reward)

    def update(self, item_id: int, context_vector: np.ndarray, reward: float):
        """Single item update with optimized computation."""
        context_norm = self._normalize_vector(context_vector)
        
        # Get weights and compute prediction
        weights = self._get_weights(item_id)
        prediction = np.dot(weights, context_norm)
        error = reward - prediction

        # In-place update for maximum efficiency
        weights += self.lr * error * context_norm
        
        # Mark for cache update
        self._cache_dirty.add(item_id)