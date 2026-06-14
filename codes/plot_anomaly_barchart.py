import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

df_anomalias = pd.read_csv('ruido_anomalias_fumaca.csv')
documentos = df_anomalias['Narrative'].dropna().tolist()

vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documentos)

palavras = vectorizer.get_feature_names_out()

scores_maximos = tfidf_matrix.max(axis=0).toarray().ravel()

df_ranking = pd.DataFrame({
    'Termo': palavras,
    'Score_Max_TFIDF': scores_maximos
})

top_30_anomalias = df_ranking.sort_values(by='Score_Max_TFIDF', ascending=False).head(30)

plt.figure(figsize=(12, 10))

ax = sns.barplot(x='Score_Max_TFIDF', y='Termo', data=top_30_anomalias, palette='Reds_r')

plt.title('Top 30 Termos Críticos Identificados no Ruído (-1)\n(Pico de Relevância TF-IDF)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Score Máximo de TF-IDF (Intensidade no Relato)', fontsize=14, labelpad=10)
plt.ylabel('Termos / Palavras-Chave', fontsize=14, labelpad=10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

for index, value in enumerate(top_30_anomalias['Score_Max_TFIDF']):
    plt.text(value, index, f' {value:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.show()
