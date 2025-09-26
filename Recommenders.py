import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
import Bandits
from typing import Dict, List, Tuple
from tqdm.auto import tqdm  # opcional para barra de progresso

def simular_recomendacao_corrigida(df_merged, catalogo_df, model, top_k=5, warmup=30):
    """
    Versão CORRIGIDA que mantém a LÓGICA ORIGINAL mas é otimizada
    """
    historico = []
    
    # Pré-processamento RÁPIDO mas mantendo a estrutura
    df_merged = df_merged.copy()
    catalogo_df = catalogo_df.copy()
    
    # Conversão eficiente de timestamps
    df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp'])
    catalogo_df['timestamp'] = pd.to_datetime(catalogo_df['timestamp'])
    
    # Garante que contextos são arrays numpy (otimização)
    if not isinstance(df_merged['context'].iloc[0], np.ndarray):
        df_merged['context'] = df_merged['context'].apply(
            lambda x: np.array(x, dtype=np.float32) if isinstance(x, (list, np.ndarray)) 
            else np.zeros(model.n_features, dtype=np.float32)
        )
    
    # Ordena catálogo para busca eficiente
    catalogo_df = catalogo_df.sort_values('timestamp')
    catalogo_timestamps = catalogo_df['timestamp'].values
    catalogo_items = catalogo_df['itemid'].values
    
    # Pre-converte contextos do catálogo
    catalogo_contexts = []
    for ctx in catalogo_df['context'].values:
        if isinstance(ctx, (list, np.ndarray)):
            catalogo_contexts.append(np.array(ctx, dtype=np.float32))
        else:
            catalogo_contexts.append(np.zeros(model.n_features, dtype=np.float32))
    
    # Mapa de recompensas (original)
    reward_map = {'view': 0.1, 'addtocart': 0.5, 'transaction': 1.0}
    
    n_interacoes = len(df_merged)
    
    with tqdm(total=n_interacoes, desc="Recomendações") as pbar:
        for i, row in enumerate(df_merged.itertuples(index=False)):
            evento_ts = row.timestamp
            true_item = row.itemid
            contexto_evento = row.context
            tipo_evento = row.event
            
            reward_val = reward_map.get(tipo_evento, 0.0)
            
            # FILTRAGEM OTIMIZADA (usando numpy)
            mask_disponiveis = catalogo_timestamps <= evento_ts
            indices_disponiveis = np.where(mask_disponiveis)[0]
            
            if len(indices_disponiveis) == 0:
                pbar.update(1)
                continue
            
            # Constrói dicionário de candidatos (MESMA ESTRUTURA ORIGINAL)
            candidatos_context = {}
            for idx in indices_disponiveis:
                item_id = catalogo_items[idx]
                context_vector = catalogo_contexts[idx]
                candidatos_context[item_id] = context_vector
            
            # FASE DE WARMUP - Lógica original
            if i < warmup:
                if true_item in candidatos_context:
                    model.update(true_item, contexto_evento, reward_val)
                pbar.update(1)
                continue
            
            # SELEÇÃO DE ITENS - LÓGICA ORIGINAL (que funcionava)
            candidatos_restantes = candidatos_context.copy()
            top_k_items = []
            
            for _ in range(min(top_k, len(candidatos_restantes))):
                if not candidatos_restantes:
                    break
                    
                # USA A MESMA LÓGICA DE SELEÇÃO ORIGINAL
                item_escolhido = model.select_item(candidatos_restantes)
                top_k_items.append(item_escolhido)
                candidatos_restantes.pop(item_escolhido)
            
            # AVALIA RECOMPENSA - Lógica original
            recompensa = reward_val if true_item in top_k_items else 0.0
            
            # ATUALIZAÇÃO - Lógica original (que aprendia corretamente)
            for item_id in top_k_items:
                r = reward_val if item_id == true_item else 0.0
                model.update(item_id, candidatos_context[item_id], r)
            
            # Aprendizado supervisionado adicional (original)
            if true_item not in top_k_items and true_item in candidatos_context:
                model.update(true_item, candidatos_context[true_item], reward_val)
            
            historico.append({
                'timestamp': evento_ts,
                'true_item': true_item,
                'top_k': top_k_items,
                'reward': recompensa,
                'event': tipo_evento
            })
            
            pbar.update(1)
    
    return pd.DataFrame(historico)

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

if __name__ == "__main__":
    print("Recommenders carregado corretamente")