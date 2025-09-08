import numpy as np
import random

class EpsilonGreedy:
    def __init__(self, epsilon: float, n_features: int, learning_rate: float = 0.1):
        self.epsilon = epsilon
        self.lr = learning_rate
        self.n_features = n_features
        self.weights = {}  # item_id -> vetor de pesos
        
    def _get_weights(self, item_id):
        # Verifica se o item_id já tem pesos, caso contrário inicializa com zeros
        if item_id not in self.weights:
            self.weights[item_id] = np.zeros(self.n_features)
        return self.weights[item_id]
    
    def predict(self, item_id, context_vector):
        weights = self._get_weights(item_id)
        # Normaliza o vetor de contexto para evitar problemas de escala
        context_vector = context_vector / np.linalg.norm(context_vector) if np.linalg.norm(context_vector) > 0 else context_vector
        # Calcula a previsão como o produto escalar entre os pesos e o vetor de contexto
        return np.dot(weights, context_vector)
    
    def select_item(self, candidate_items_contexts):
        if random.random() < self.epsilon:
            return random.choice(list(candidate_items_contexts.keys()))

        scores = {
            item_id: self.predict(item_id, context)
            for item_id, context in candidate_items_contexts.items()
        }
        return max(scores, key=scores.get)
        
    def update(self, item_id, context_vector, reward):
        weights = self._get_weights(item_id)
        prediction = self.predict(item_id, context_vector)
        error = reward - prediction
        self.weights[item_id] += self.lr * error * context_vector