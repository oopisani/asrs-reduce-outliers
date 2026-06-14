df_anomalias = pd.read_csv('/content/ruido_anomalias_fumaca.csv')
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

print("=======================================================")
print("  TOP 30 PICOS DE ANOMALIA NO RUÍDO (-1)               ")
print("=======================================================")
print(top_30_anomalias.to_string(index=False))
