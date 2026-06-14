import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

representativos_t6 = topic_model.get_representative_docs(6)

print("--- Documentos Representativos (Tópico 6) ---")
for i, doc in enumerate(representativos_t6):
    print(f"Representativo {i+1}: {doc[:150]}...")

docs_t6 = [doc for doc, topic in zip(docs, topics) if topic == 6]
total_docs_t6 = len(docs_t6)

docs_t6_sem_reps = [doc for doc in docs_t6 if doc not in representativos_t6]

tamanho_amostra = int(total_docs_t6 * 0.20)
np.random.seed(42) 
amostra_t6 = np.random.choice(docs_t6_sem_reps, size=tamanho_amostra, replace=False).tolist()

print(f"\nSorteada amostra de {tamanho_amostra} documentos (20% do total de {total_docs_t6}).")

print("Gerando embeddings para o cálculo de similaridade...")
emb_representativos = embedding_model.encode(representativos_t6, show_progress_bar=False)
emb_amostra = embedding_model.encode(amostra_t6, show_progress_bar=True)

print("Calculando notas de similaridade...")
matriz_similaridade = cosine_similarity(emb_amostra, emb_representativos)

scores_similaridade = np.mean(matriz_similaridade, axis=1)

df_amostra = pd.DataFrame({
    'Narrative': amostra_t6,
    'Score_Similaridade': scores_similaridade
})

print("\n=== Resultados Estatísticos da Amostra ===")
print(f"Média Geral da Amostra: {df_amostra['Score_Similaridade'].mean():.4f}")
print(f"Maior Similaridade Encontrada: {df_amostra['Score_Similaridade'].max():.4f}")
print(f"Menor Similaridade Encontrada: {df_amostra['Score_Similaridade'].min():.4f}")

print("\nTop 3 documentos da amostra mais próximos do núcleo:")
print(df_amostra.sort_values(by='Score_Similaridade', ascending=False).head(3).to_string())
