import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from bisect import bisect_right
import time
from collections import defaultdict

def preprocess_catalog_ultra_fast(catalogo_df: pd.DataFrame) -> Dict:
    """Extremely fast catalog preprocessing with memory mapping."""
    if catalogo_df.empty:
        return {}
    
    # Sort by timestamp for binary search
    catalogo_sorted = catalogo_df.sort_values('timestamp').reset_index(drop=True)
    
    # Precompute arrays for fastest possible access
    timestamps = catalogo_sorted['timestamp'].values
    itemids = catalogo_sorted['itemid'].values
    
    # Precompute contexts as numpy arrays
    if hasattr(catalogo_sorted['context'].iloc[0], 'shape'):
        contexts = catalogo_sorted['context'].values
    else:
        contexts = np.array([np.array(ctx, dtype=np.float32) for ctx in catalogo_sorted['context'].values])
    
    return {
        'timestamps': timestamps,
        'itemids': itemids,
        'contexts': contexts,
        'size': len(catalogo_sorted)
    }

def filter_catalog_ultra_fast(preprocessed_catalog: Dict, max_timestamp: float) -> Dict[int, np.ndarray]:
    """Ultra-fast catalog filtering with binary search and efficient dict creation."""
    timestamps = preprocessed_catalog['timestamps']
    
    if len(timestamps) == 0:
        return {}
    
    # Binary search for the cutoff index
    idx = bisect_right(timestamps, max_timestamp)
    if idx == 0:
        return {}
    
    # Create dictionary with direct array slicing (fastest method)
    itemids = preprocessed_catalog['itemids']
    contexts = preprocessed_catalog['contexts']
    
    # Use dictionary comprehension with pre-allocated arrays
    return {itemids[i]: contexts[i] for i in range(idx)}

def simular_recomendacao_top5_epsilon_greedy_unlimited(
    df_merged: pd.DataFrame, catalogo_df: pd.DataFrame, model, top_k: int = 5, warmup: int = 30
) -> pd.DataFrame:
    """Fully optimized recommender without event or candidate limitations."""
    historico = []
    reward_map = {'view': 0.1, 'addtocart': 0.5, 'transaction': 1.0}
    
    # Preprocess catalog once (major performance gain)
    preprocessed_catalog = preprocess_catalog_ultra_fast(catalogo_df)
    
    # Convert DataFrame to numpy arrays for ultra-fast access
    n_events = len(df_merged)
    timestamps = df_merged['timestamp'].values
    itemids = df_merged['itemid'].values
    events = df_merged['event'].values
    
    # Precompute contexts as numpy arrays
    if hasattr(df_merged['context'].iloc[0], 'shape'):
        contexts = df_merged['context'].values
    else:
        contexts = np.array([np.array(ctx, dtype=np.float32) for ctx in df_merged['context'].values])
    
    # Batch processing variables
    batch_updates = []
    BATCH_SIZE = 500  # Optimal batch size for performance
    
    # Precompute event rewards
    event_rewards = np.zeros(n_events, dtype=np.float32)
    for i in range(n_events):
        event_rewards[i] = reward_map.get(events[i], 0.0)
    
    print(f"Processing {n_events} events with unlimited candidates...")
    
    for i in range(n_events):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{n_events} events")
            
        evento_ts = timestamps[i]
        true_item = itemids[i]
        contexto_evento = contexts[i]
        reward_val = event_rewards[i]

        # Ultra-fast catalog filtering
        candidatos_context = filter_catalog_ultra_fast(preprocessed_catalog, evento_ts)
        if not candidatos_context:
            continue

        if i < warmup:
            # Supervised learning
            if true_item in candidatos_context:
                batch_updates.append((true_item, contexto_evento, reward_val))
            continue

        # Select top k items using optimized batch method
        if hasattr(model, 'select_top_k'):
            top_k_items = model.select_top_k(candidatos_context, top_k)
        else:
            # For extremely large candidate sets, use efficient iterative selection
            top_k_items = []
            candidate_keys = list(candidatos_context.keys())
            
            if len(candidate_keys) > 10000:
                # For massive candidate sets, use batch prediction to find top k
                scores = model.predict_batch(candidate_keys, 
                                           [candidatos_context[k] for k in candidate_keys])
                top_indices = np.argpartition(scores, -top_k)[-top_k:]
                top_k_items = [candidate_keys[idx] for idx in top_indices]
            else:
                # For reasonable candidate sets, use iterative selection
                temp_candidates = candidatos_context.copy()
                for _ in range(min(top_k, len(temp_candidates))):
                    chosen = model.select_item(temp_candidates)
                    top_k_items.append(chosen)
                    del temp_candidates[chosen]

        # Calculate reward
        true_in_topk = true_item in top_k_items
        recompensa = reward_val if true_in_topk else 0.0
        
        # Batch updates for recommended items
        for item_id in top_k_items:
            context = candidatos_context[item_id]
            r = reward_val if item_id == true_item else 0.0
            batch_updates.append((item_id, context, r))
        
        # Additional update for true item if not recommended
        if true_item in candidatos_context and not true_in_topk:
            batch_updates.append((true_item, candidatos_context[true_item], reward_val))

        historico.append({
            'timestamp': evento_ts,
            'true_item': true_item,
            'top_k': top_k_items,
            'reward': recompensa
        })
        
        # Process batch updates to prevent memory overflow
        if len(batch_updates) >= BATCH_SIZE:
            if hasattr(model, 'update_batch'):
                model.update_batch(batch_updates)
            else:
                for update in batch_updates:
                    model.update(*update)
            batch_updates = []
    
    # Process any remaining updates
    if batch_updates:
        if hasattr(model, 'update_batch'):
            model.update_batch(batch_updates)
        else:
            for update in batch_updates:
                model.update(*update)

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