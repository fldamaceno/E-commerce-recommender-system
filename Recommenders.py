import numpy as np
import pandas as pd
from typing import Dict, List

def preprocess_catalog(catalogo_df: pd.DataFrame) -> Dict:
    """Preprocess catalog for efficient timestamp filtering."""
    catalogo_sorted = catalogo_df.sort_values('timestamp').reset_index(drop=True)
    return {
        'timestamps': catalogo_sorted['timestamp'].values,
        'contexts': catalogo_sorted['context'].values,
        'itemids': catalogo_sorted['itemid'].values,
        'df': catalogo_sorted
    }

def filter_catalog_by_timestamp(preprocessed_catalog: Dict, max_timestamp: float) -> Dict[int, np.ndarray]:
    """Efficiently filter catalog items by timestamp using binary search."""
    timestamps = preprocessed_catalog['timestamps']
    
    if len(timestamps) == 0:
        return {}
    
    # Binary search for the last valid index
    left, right = 0, len(timestamps) - 1
    while left <= right:
        mid = (left + right) // 2
        if timestamps[mid] <= max_timestamp:
            left = mid + 1
        else:
            right = mid - 1
    
    valid_count = right + 1
    if valid_count == 0:
        return {}
    
    # Build candidate dictionary
    candidatos_context = {}
    for j in range(valid_count):
        item_id = preprocessed_catalog['itemids'][j]
        context = preprocessed_catalog['contexts'][j]
        candidatos_context[item_id] = context
    
    return candidatos_context

def simular_recomendacao_top5_epsilon_greedy(df_merged: pd.DataFrame, catalogo_df: pd.DataFrame, 
                                            model, top_k: int = 5, warmup: int = 30) -> pd.DataFrame:
    """Simulate recommendations using epsilon-greedy strategy with performance optimizations."""
    historico = []
    reward_map = {'view': 0.1, 'addtocart': 0.5, 'transaction': 1.0}
    
    # Preprocess data
    if not hasattr(df_merged['context'].iloc[0], 'shape'):
        df_merged['context'] = df_merged['context'].apply(np.array)
    
    preprocessed_catalog = preprocess_catalog(catalogo_df)
    
    for i, row in enumerate(df_merged.itertuples(index=False)):
        evento_ts = row.timestamp
        true_item = row.itemid
        contexto_evento = row.context
        tipo_evento = row.event

        reward_val = reward_map.get(tipo_evento, 0.0)
        
        # Filter available catalog items
        candidatos_context = filter_catalog_by_timestamp(preprocessed_catalog, evento_ts)
        if not candidatos_context:
            continue

        if i < warmup:
            # Supervised learning phase
            if true_item in candidatos_context:
                model.update(true_item, contexto_evento, reward_val)
            continue

        # Select top k items using optimized method if available
        if hasattr(model, 'select_top_k') and callable(getattr(model, 'select_top_k')):
            top_k_items = model.select_top_k(candidatos_context, top_k)
        else:
            # Fallback to iterative selection
            temp_candidates = candidatos_context.copy()
            top_k_items = []
            for _ in range(min(top_k, len(temp_candidates))):
                chosen = model.select_item(temp_candidates)
                top_k_items.append(chosen)
                del temp_candidates[chosen]

        # Calculate reward and update model
        true_in_topk = true_item in top_k_items
        recompensa = reward_val if true_in_topk else 0.0
        
        # Update model for all recommended items
        for item_id in top_k_items:
            context = candidatos_context[item_id]
            r = reward_val if item_id == true_item else 0.0
            model.update(item_id, context, r)
        
        # Additional update for true item if not recommended
        if true_item in candidatos_context and not true_in_topk:
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