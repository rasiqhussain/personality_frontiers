import os
import pandas as pd
import re
import contractions
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from umap import UMAP
from hdbscan import HDBSCAN
import nltk

nltk.download("punkt")

def run_topic_modeling(utterance_csv_path, output_prefix):
    df = pd.read_csv(utterance_csv_path)
    utterances = df["Utterance"].dropna().astype(str)

    punkt_params = PunktParameters()
    punkt = PunktSentenceTokenizer(punkt_params)

    def sent_tokenize_safe(text):
        return punkt.tokenize(text.strip())

    def preprocess_and_split(text):
        sents = sent_tokenize_safe(text)
        processed = []
        for s in sents:
            s = contractions.fix(s.lower())
            s = re.sub(r"[^a-z\s]", "", s)
            if len(s.split()) >= 3:
                processed.append(s.strip())
        return processed

    utterance_chunks = []
    for u in utterances:
        utterance_chunks.extend(preprocess_and_split(u))

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(utterance_chunks, show_progress_bar=True)

    custom_stop_words = list(set(ENGLISH_STOP_WORDS).union({
        "um", "uh", "yeah", "okay", "right", "hmm", "huh", "like", "just", "well",
        "dont", "didnt", "ive", "gonna", "kinda", "sorta", "hmmm"
    }))
    min_df_val = max(1, int(len(utterance_chunks) * 0.001))
    # or min_df_val = min(10, max(1, int(len(utterance_chunks) * 0.001))) 
    vectorizer_model = CountVectorizer(
        stop_words=custom_stop_words,
        min_df=min_df_val,
        max_df=0.99
    )
    #vectorizer_model = CountVectorizer(stop_words=custom_stop_words, min_df=1, max_df=1.0)
    umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=5, metric="euclidean", prediction_data=True)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        language="english",
        calculate_probabilities=True,
        nr_topics="auto"
    )

    topics, probs = topic_model.fit_transform(utterance_chunks, embeddings)

    output_dir = os.path.join(os.getcwd(), output_prefix)
    os.makedirs(output_dir, exist_ok=True)

    topic_keywords = []
    for topic_id in topic_model.get_topic_info()["Topic"].tolist():
        if topic_id == -1:
            continue
        keywords = topic_model.get_topic(topic_id)
        word_list = [w for w, _ in keywords[:10]]
        topic_keywords.append({
            "Topic": topic_id,
            "Keywords": word_list,
            "Keyword_String": ", ".join(word_list)
        })

    topic_df = pd.DataFrame(topic_keywords)
    topic_df.to_csv(os.path.join(output_dir, f"topic_keywords.csv"), index=False)

    umap_2d = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine", random_state=42)
    umap_embeddings_2d = umap_2d.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        umap_embeddings_2d[:, 0],
        umap_embeddings_2d[:, 1],
        c=topics,
        cmap="tab20",
        s=40,
        alpha=0.7
    )
    plt.colorbar(scatter, label="Topic")

    added = set()
    for idx, topic_id in enumerate(topics):
        if topic_id != -1 and topic_id not in added:
            x, y = umap_embeddings_2d[idx]
            keywords = ", ".join([w for w, _ in topic_model.get_topic(topic_id)[:3]])
            plt.text(x, y, f"T{topic_id}: {keywords}", fontsize=9, alpha=0.8)
            added.add(topic_id)

    plt.title("BERTopic: UMAP Topic Clusters with Labels")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"topic_clusters_umap.png"), dpi=300)
    plt.close()

    for topic_id in topic_model.get_topics().keys():
        if topic_id == -1:
            continue
        words = " ".join([w for w, _ in topic_model.get_topic(topic_id)])
        wc = WordCloud(width=400, height=300, background_color="white").generate(words)
        # the word cloud is without weights
        plt.figure()
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Topic {topic_id}")
        plt.savefig(os.path.join(output_dir, f"wordcloud_topic_{topic_id}.png"), dpi=300)
        plt.close()

    print(f"All outputs saved in：{output_dir}")
