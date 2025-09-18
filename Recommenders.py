import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Função de simulação vetorizada por usuário (torch)
# ----------------------------
def simular_recomendacao_top_k_epsilon_greedy_torch(
    context_events: pd.DataFrame,
    catalog_df: pd.DataFrame,
    model: EpsilonGreedyTorch,
    itemid_to_index: Dict[int,int],
    top_k: int = 5,
    warmup: int = 30,
    use_tqdm: bool = False
) -> pd.DataFrame:
    """
    context_events: DataFrame com colunas ['timestamp', 'itemid', 'context', 'event']
        - 'context' é array-like (lista/np.array) com n_features
    catalog_df: DataFrame com colunas ['itemid', 'timestamp', 'context'] (catalogo disponível ao longo do tempo)
    model: EpsilonGreedyTorch instanciado com n_items = len(itemid_to_index)
    itemid_to_index: dict map itemid->index global
    """
    history = []

    # Precompute: convert contexts in catalog to tensor matrix for fast slicing
    # Assumimos que catalog_df contém o catálogo completo usado (itemid pode repetir com timestamps)
    # Para cada evento filtramos catalog até timestamp do evento -> exemplos abaixo usam filtragem por timestamp
    # Para performance, transformaremos catalog_df em arrays numpy/tensor para iterações rápidas

    # Convert catalog to arrays
    catalog_df = catalog_df.copy()
    catalog_df['item_index'] = catalog_df['itemid'].map(itemid_to_index)
    catalog_df = catalog_df.dropna(subset=['item_index'])
    catalog_df['item_index'] = catalog_df['item_index'].astype(int)

    # Convert contexts in catalog to tensor once (we'll index rows)
    # We'll store contexts in a list aligned with catalog_df rows
    catalog_contexts = np.stack(catalog_df['context'].to_list()).astype(np.float32)  # [N_catalog_rows, n_features]
    catalog_timestamps = catalog_df['timestamp'].to_numpy()
    catalog_item_indices = catalog_df['item_index'].to_numpy(dtype=np.int64)

    # Preconvert context_events to list or arrays
    context_events = context_events.sort_values(by='timestamp').reset_index(drop=True)
    contexts_events_arr = np.stack(context_events['context'].to_list()).astype(np.float32)
    events_timestamps = context_events['timestamp'].to_numpy()
    events_itemids = context_events['itemid'].to_numpy()
    events_types = context_events['event'].to_numpy()

    rng = range(len(events_timestamps))
    if use_tqdm:
        rng = tqdm(rng, desc="Simulando eventos")

    for i in rng:
        ts = events_timestamps[i]
        true_item = int(events_itemids[i])
        contexto_event = contexts_events_arr[i]
        tipo_event = events_types[i]
        reward_val = {'view': 0.1, 'addtocart': 0.5, 'transaction': 1.0}.get(tipo_event, 0.0)

        # Warmup: só update com true_item quando disponível
        if i < warmup:
            if true_item in itemid_to_index:
                idx = itemid_to_index[true_item]
                model.update_batch(
                    item_indices=torch.tensor([idx], dtype=torch.long, device=model.device),
                    contexts=torch.tensor([contexto_event], dtype=torch.float32, device=model.device),
                    rewards=torch.tensor([reward_val], dtype=torch.float32, device=model.device)
                )
            continue

        # Filtra catálogo disponível até o timestamp do evento
        mask = catalog_timestamps <= ts
        if not mask.any():
            continue

        # Pegamos os candidatos disponíveis (por linha do catalog)
        cand_item_indices = catalog_item_indices[mask]         # shape [m]
        cand_contexts_rows = catalog_contexts[mask]           # shape [m, n_features]

        # Agrupar por item: pode haver múltiplas linhas com mesmo item_index, podemos deduplicar pegando a última ocorrência
        # Para eficiência: vamos dedup por item mantendo a última ocorrência (maior timestamp)
        # Construir dict item_index -> position (última)
        # Essa dedup típica em numpy:
        uniq_items, last_pos = np.unique(cand_item_indices[::-1], return_index=True)
        last_pos = (len(cand_item_indices)-1) - last_pos  # converte posição reversa para posição original
        sel_positions = last_pos[np.argsort(uniq_items)]  # ordena por item index para estabilidade
        candidate_indices_unique = cand_item_indices[sel_positions]  # item indices únicos
        candidate_contexts_unique = cand_contexts_rows[sel_positions]  # contexts correspondentes

        # Convert to torch tensors
        candidate_item_indices_tensor = torch.tensor(candidate_indices_unique, dtype=torch.long, device=model.device)
        candidate_contexts_tensor = torch.tensor(candidate_contexts_unique, dtype=torch.float32, device=model.device)

        # Seleciona top_k usando operação vetorizada do modelo
        topk_global_indices = model.select_top_k(candidate_item_indices_tensor, candidate_contexts_tensor, top_k=top_k)

        # Avalia recompensa: se true_item está entre topk
        recompensa = reward_val if true_item in topk_global_indices else 0.0

        # Atualiza modelo: para cada item do top_k, reward = reward_val se for o true_item else 0
        # Montamos batch de updates
        update_item_indices = []
        update_contexts = []
        update_rewards = []
        for itm in topk_global_indices:
            update_item_indices.append(itm)
            # acha posição do itm na candidate list para pegar contexto: candidate_item_indices_unique -> find
            # mais simples: use mapping de item->posição: (we have candidate_item_indices_unique array) -> find index
            pos = np.where(candidate_indices_unique == itm)[0][0]
            update_contexts.append(candidate_contexts_unique[pos])
            update_rewards.append(reward_val if itm == true_item else 0.0)

        # Extra: também atualiza o true_item com supervisão adicional se estiver no catálogo
        if true_item in itemid_to_index:
            true_idx = itemid_to_index[true_item]
            update_item_indices.append(true_idx)
            update_contexts.append(contexto_event)  # usar contexto do evento (ou do catálogo)
            update_rewards.append(reward_val)

        # Vetorizando update
        if len(update_item_indices) > 0:
            model.update_batch(
                item_indices=torch.tensor(update_item_indices, dtype=torch.long, device=model.device),
                contexts=torch.tensor(np.stack(update_contexts).astype(np.float32), dtype=torch.float32, device=model.device),
                rewards=torch.tensor(update_rewards, dtype=torch.float32, device=model.device)
            )

        history.append({
            'timestamp': ts,
            'true_item': true_item,
            'top_k': [int(x) for x in topk_global_indices],
            'reward': float(recompensa)
        })

    return pd.DataFrame(history)

def simular_recomendacao_top5_epsilon_greedy(df_merged, catalogo_df, model, top_k=5, warmup=30):
    historico = []

    # Converte contextos para np.array
    df_merged['context'] = df_merged['context'].apply(lambda x: np.array(x))
    catalogo_df['context'] = catalogo_df['context'].apply(lambda x: np.array(x))

    for i, row in enumerate(df_merged.itertuples(index=False)):
        evento_ts = row.timestamp
        true_item = row.itemid
        contexto_evento = row.context
        tipo_evento = row.event

        reward_val = {'view': 0.1, 'addtocart': 0.5, 'transaction': 1.0}.get(tipo_evento, 0.0)

        # Filtra catálogo disponível até o timestamp do evento
        disponiveis = catalogo_df[catalogo_df['timestamp'] <= evento_ts]
        if disponiveis.empty:
            continue

        # Mapeia itemid -> contexto
        candidatos_context = {
            r['itemid']: np.array(r['context']) for _, r in disponiveis.iterrows()
        }

        if i < warmup:
            # Fase de aprendizado supervisionado
            if true_item in candidatos_context:
                model.update(true_item, contexto_evento, reward_val)
            continue  # Não recomenda nessa fase

        # Copia dos candidatos para iteração segura
        candidatos_restantes = candidatos_context.copy()
        top_k_items = []

        # Seleção de itens com exploração e exploração
        for _ in range(min(top_k, len(candidatos_restantes))):
            # Usa a função select_item diretamente
            item_escolhido = model.select_item(candidatos_restantes)
            top_k_items.append(item_escolhido)
            candidatos_restantes.pop(item_escolhido)

        # Avalia recompensa total: se true_item está entre os top_k
        recompensa = reward_val if true_item in top_k_items else 0.0
        # Atualiza modelo com todos os top_k itens
        for item_id in top_k_items:
            r = reward_val if item_id == true_item else 0.0
            model.update(item_id, candidatos_context[item_id], r)
        # Aprendizado supervisionado adicional com true_item (se disponível)
        if true_item in candidatos_context:
            model.update(true_item, candidatos_context[true_item], reward_val)

        historico.append({
            'timestamp': evento_ts,
            'true_item': true_item,
            'top_k': top_k_items,
            'reward': recompensa
        })

    return pd.DataFrame(historico)


def simular_recomendacao_coseno(df_merged, catalogo_df):
    historico = []

    # Garante que todos os contextos são arrays numpy
    df_merged['context'] = df_merged['context'].apply(lambda x: np.array(x))
    catalogo_df['context'] = catalogo_df['context'].apply(lambda x: np.array(x))

    for _, row in df_merged.iterrows():
        evento_ts = row['timestamp']
        true_item = row['itemid']
        contexto_evento = row['context']
        tipo_evento = row['event']

        # Mapeia tipo de evento para recompensa
        reward_val = {'view': 0.01, 'addtocart': 0.1, 'transaction': 1.0}.get(tipo_evento, 0.0)

        # Filtra catálogo até o timestamp do evento
        disponiveis = catalogo_df[catalogo_df['timestamp'] <= evento_ts].copy()
        if disponiveis.empty:
            continue

        # Prepara matriz de contextos dos itens disponíveis
        X_catalogo = np.stack(disponiveis['context'].values)

        # Calcula similaridade coseno entre contexto do evento e todos os itens disponíveis
        sim_scores = cosine_similarity([contexto_evento], X_catalogo)[0]

        # Pega os top 5 índices mais similares
        top_indices = np.argsort(sim_scores)[-5:][::-1]
        top_5_items = disponiveis.iloc[top_indices]['itemid'].values

        # Recompensa se o true_item estiver entre os top 5
        recompensa = reward_val if true_item in top_5_items else 0.0

        historico.append({
            'timestamp': evento_ts,
            'true_item': true_item,
            'top_5': list(top_5_items),
            'reward': recompensa
        })

    return pd.DataFrame(historico)

def simular_recomendacao_top1_epsilongreedy(df_merged, catalogo_df, model):
    historico = []

    for _, row in df_merged.iterrows():
        true_item = row['itemid']
        context = row['context']
        event = row['event']
        reward = {'view': 0.01, 'addtocart': 0.1, 'transaction': 1}.get(event, 0.0)

        # Converte contexto se necessário
        if context is None:
            continue
        if not isinstance(context, np.ndarray):
            context = np.array(context)

        # Filtra catálogo disponível até o timestamp do evento
        disponiveis = catalogo_df[catalogo_df['timestamp'] <= row['timestamp']]
        if disponiveis.empty:
            continue

        # Mapeia item_id -> contexto (do próprio item)
        candidatos_context = {
            r['itemid']: np.array(r['context']) for _, r in disponiveis.iterrows()
        }

        item_escolhido = model.select_item(candidatos_context)
        recompensa_real = reward if item_escolhido == true_item else 0.0
        model.update(item_escolhido, candidatos_context[item_escolhido], recompensa_real)

        historico.append({
            'timestamp': row['timestamp'],
            'true_item': true_item,
            'escolhido': item_escolhido,
            'reward': recompensa_real
        })

    return pd.DataFrame(historico)