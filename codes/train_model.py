from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd

df = pd.read_csv(
    "path_of_your_dataset.csv",
    sep="\t"
)

docs = (
    df["Narrative"]
    .dropna()
    .astype(str)
    .tolist()
)

print(f"Total de relatos: {len(docs)}")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = embedding_model.encode(
    docs,
    show_progress_bar=True
)

umap_model = UMAP(
    n_neighbors=15,
    n_components=5, 
    min_dist=0.0,
    metric="cosine",
    random_state=42
)

hdbscan_model = HDBSCAN(
    min_cluster_size=30,
    metric="euclidean",
    prediction_data=True
)

termos_asrs = ['z', 'zz', 'zzz', 'zzzz', 'zzz1', 'zzz2', 'zzz3', 'x', 'xx', 'xxx', 'y', 'yy', 'yyy']

termos_aviacao_comuns = ['aircraft', 'flight', 'plane', 'told', 'said', 'just', 'like', 'got']

custom_stop_words = list(ENGLISH_STOP_WORDS) + termos_asrs + termos_aviacao_comuns

vectorizer_model = CountVectorizer(
    ngram_range=(1, 2),
    min_df=0.05,  
    max_df=0.85,  
    stop_words=custom_stop_words 
)

ctfidf_model = ClassTfidfTransformer(
    bm25_weighting=True
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    top_n_words=10,
    calculate_probabilities=True,
    verbose=True
)

print("Iniciando o treinamento do modelo base...")
topics, probs = topic_model.fit_transform(
    docs,
    embeddings
)

novos_topicos = topic_model.reduce_outliers(
    docs, 
    topics, 
    strategy="embeddings", 
    embeddings=embeddings
)

print("Atualizando representações dos tópicos...")
topic_model.update_topics(
    docs, 
    topics=novos_topicos,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model
)

topics = novos_topicos 

topic_info = topic_model.get_topic_info()

print(topic_info.head(20))

topic_info.to_csv("topic_info_reduzido.csv", index=False)

df_resultado = pd.DataFrame({
    "Narrative": docs,
    "Topic": topics
})

df_resultado.to_csv(
    "document_topics_reduzido.csv",
    index=False
)
  
print("\n--- Representação dos Tópicos ---")
for topic_id in topic_info["Topic"].head(20):
    if topic_id != -1:
        print(f"\nTópico {topic_id}")
        print(topic_model.get_topic(topic_id))
