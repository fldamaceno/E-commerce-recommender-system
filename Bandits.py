import numpy as np
import random

# ----------------------------
# Modelo EpsilonGreedy em PyTorch (vetorizado)
# ----------------------------
class EpsilonGreedyTorch:
    def __init__(
        self,
        epsilon: float,
        n_items: int,
        n_features: int,
        learning_rate: float = 0.1,
        normalize_weights: bool = True,
        device: str = None,
        seed: int = 42
    ):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.epsilon = float(epsilon)
        self.lr = float(learning_rate)
        self.n_items = int(n_items)
        self.n_features = int(n_features)
        self.normalize_weights = bool(normalize_weights)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Pesos por item: [n_items, n_features]
        self.weights = torch.zeros((self.n_items, self.n_features), dtype=torch.float32, device=self.device)
        # Opcional: inicializar pequeno ruído para quebrar simetrias
        self.weights += 1e-6 * torch.randn_like(self.weights)

    def _normalize_vecs(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Normaliza última dimensão dos tensores.
        Suporta shape [..., n_features]
        """
        norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        return x / (norm + eps)

    def select_top_k(self, candidate_item_indices: torch.LongTensor, candidate_contexts: torch.Tensor, top_k: int):
        """
        candidate_item_indices: tensor [m] dos índices dos itens candidatos (long tensor)
        candidate_contexts: tensor [m, n_features] (float)
        Retorna lista de top_k indices (em termos de índice global do item)
        """
        # Normaliza contextos (cada linha)
        contexts_norm = self._normalize_vecs(candidate_contexts)

        # Fetch weights para os candidatos: [m, n_features]
        candidate_weights = self.weights[candidate_item_indices]  # view

        # Scores: produto escalar por linha
        scores = torch.sum(candidate_weights * contexts_norm, dim=1)  # [m]

        # epsilon-greedy: com prob epsilon escolhe aleatório um conjunto?
        # Para top_k, implementamos exploração por item: com prob epsilon escolhemos aleatório entre candidatos restantes.
        # Uma maneira simples: com prob epsilon escolhemos top_k aleatórios, caso contrário top_k greedy.
        if random.random() < self.epsilon:
            # amostra sem reposição
            perm = torch.randperm(candidate_item_indices.size(0), device=self.device)[:top_k]
            chosen = candidate_item_indices[perm]
            return chosen.tolist()
        else:
            k = min(top_k, scores.size(0))
            topk = torch.topk(scores, k=k)
            chosen = candidate_item_indices[topk.indices]
            return chosen.tolist()

    def update_batch(self, item_indices: torch.LongTensor, contexts: torch.Tensor, rewards: torch.Tensor):
        """
        Atualiza múltiplos pares (item, context, reward) de forma vetorizada.
        - item_indices: [b] long tensor de índices globais dos itens
        - contexts: [b, n_features] tensor (float)
        - rewards: [b] float tensor com recompensa (0.0..1.0)
        Regra: w_i <- w_i + lr * (reward - dot(w_i, c_i)) * c_i
        Após atualização, opcionalmente normaliza pesos para cada item.
        """
        if item_indices.numel() == 0:
            return

        contexts = contexts.to(self.device).float()
        contexts_norm = self._normalize_vecs(contexts)  # manter consistência
        item_indices = item_indices.to(self.device)
        rewards = rewards.to(self.device).float()

        # Pesos atuais dos itens: [b, n_features]
        w = self.weights[item_indices]

        # Previsão
        preds = torch.sum(w * contexts_norm, dim=1)  # [b]
        errors = rewards - preds  # [b]

        # Atualização broadcasting: w + lr * errors[...,None] * context
        delta = (self.lr * errors).unsqueeze(1) * contexts_norm  # [b, n_features]
        # Emplace update
        self.weights[item_indices] = w + delta

        # Normaliza pesos por item se preciso
        if self.normalize_weights:
            self.weights[item_indices] = self._normalize_vecs(self.weights[item_indices])



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