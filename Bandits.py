import numpy as np
import random
import torch


class EpsilonGreedy:
    def __init__(self, epsilon: float, n_features: int, learning_rate: float = 0.1, normalize_weights: bool = True):
        self.epsilon = epsilon
        self.lr = learning_rate
        self.n_features = n_features
        self.normalize_weights = normalize_weights
        self.weights = {}  # item_id -> vetor de pesos

    def _get_weights(self, item_id):
        if item_id not in self.weights:
            self.weights[item_id] = np.zeros(self.n_features)
        return self.weights[item_id]

    def _normalize(self, vector):
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def select_item(self, candidate_items_contexts):
        if random.random() < self.epsilon:
            return random.choice(list(candidate_items_contexts.keys()))

        scores = {}
        for item_id, context_vector in candidate_items_contexts.items():
            weights = self._get_weights(item_id)
            context_vector = self._normalize(context_vector)
            scores[item_id] = np.dot(weights, context_vector)

        return max(scores, key=scores.get)

    def update(self, item_id, context_vector, reward):
        weights = self._get_weights(item_id)
        context_vector = self._normalize(context_vector)
        prediction = np.dot(weights, context_vector)
        error = reward - prediction
        self.weights[item_id] += self.lr * error * context_vector

        # Normaliza os pesos apenas se a opção estiver ativada
        if self.normalize_weights:
            self.weights[item_id] = self._normalize(self.weights[item_id])


class CorrectedEpsilonGreedy:
    def __init__(self, epsilon: float, n_features: int, learning_rate: float = 0.1):
        """Mantém a lógica ORIGINAL do seu código que funcionava"""
        self.epsilon = epsilon
        self.lr = learning_rate
        self.n_features = n_features
        self.weights = {}  # item_id -> vetor de pesos (MESMO do original)
    
    def _get_weights(self, item_id):
        """MESMA lógica do original"""
        if item_id not in self.weights:
            self.weights[item_id] = np.zeros(self.n_features, dtype=np.float32)
        return self.weights[item_id]
    
    def _normalize(self, vector):
        """MESMA lógica do original"""
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector
    
    def select_item(self, candidate_items_contexts):
        """MESMA lógica do original - que estava funcionando"""
        import random
        
        if random.random() < self.epsilon:
            return random.choice(list(candidate_items_contexts.keys()))
        
        scores = {}
        for item_id, context_vector in candidate_items_contexts.items():
            weights = self._get_weights(item_id)
            context_normalized = self._normalize(context_vector)
            scores[item_id] = np.dot(weights, context_normalized)
        
        return max(scores, key=scores.get)
    
    def update(self, item_id, context_vector, reward):
        """MESMA lógica do original - que estava funcionando"""
        weights = self._get_weights(item_id)
        context_normalized = self._normalize(context_vector)
        prediction = np.dot(weights, context_normalized)
        error = reward - prediction
        self.weights[item_id] = weights + self.lr * error * context_normalized