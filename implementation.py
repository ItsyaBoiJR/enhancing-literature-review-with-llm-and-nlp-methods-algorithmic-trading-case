import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Dummy dataset of research paper abstracts
dummy_abstracts = [
    "This paper explores the use of reinforcement learning in algorithmic trading.",
    "A novel approach to portfolio optimization using deep learning.",
    "The impact of sentiment analysis on stock price prediction.",
    "Cryptocurrency trading strategies using machine learning models.",
    "A comparative study of traditional and machine learning methods in trading.",
    "The role of LSTMs in predicting financial time series.",
    "An analysis of high-frequency trading using neural networks.",
    "The evolution of algorithmic trading in the cryptocurrency market.",
    "A survey of machine learning applications in financial markets.",
    "The use of transformers for stock market prediction."
]

# Step 1: Preprocessing and TF-IDF Vectorization
def preprocess_and_vectorize(texts):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

# Step 2: Dimensionality Reduction using PCA
def reduce_dimensionality(tfidf_matrix, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_matrix = pca.fit_transform(tfidf_matrix.toarray())
    return reduced_matrix, pca

# Step 3: Clustering using KMeans
def cluster_documents(reduced_matrix, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(reduced_matrix)
    return labels, kmeans

# Step 4: Summarization using a Pre-trained LLM
def summarize_clusters(texts, labels, n_clusters):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summaries = []
    for cluster_id in range(n_clusters):
        cluster_texts = " ".join([texts[i] for i in range(len(texts)) if labels[i] == cluster_id])
        summary = summarizer(cluster_texts, max_length=50, min_length=25, do_sample=False)
        summaries.append((cluster_id, summary[0]['summary_text']))
    return summaries

# Step 5: Similarity Analysis
def compute_similarity(tfidf_matrix):
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

if __name__ == '__main__':
    # Step 1: Preprocess and vectorize abstracts
    tfidf_matrix, vectorizer = preprocess_and_vectorize(dummy_abstracts)
    
    # Step 2: Reduce dimensionality
    reduced_matrix, pca = reduce_dimensionality(tfidf_matrix)
    
    # Step 3: Cluster documents
    n_clusters = 3
    labels, kmeans = cluster_documents(reduced_matrix, n_clusters)
    
    # Step 4: Summarize clusters
    summaries = summarize_clusters(dummy_abstracts, labels, n_clusters)
    
    # Step 5: Compute similarity matrix
    similarity_matrix = compute_similarity(tfidf_matrix)
    
    # Output results
    print("Cluster Summaries:")
    for cluster_id, summary in summaries:
        print(f"Cluster {cluster_id}: {summary}")
    
    print("\nDocument Similarity Matrix:")
    print(similarity_matrix)