import joblib
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from .preprocessing import clean_text
from .embedings import embed


ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
kmeans = joblib.load(ARTIFACT_DIR / "kmeans_model.joblib")

with open(ARTIFACT_DIR / "cluster_labels.json", "r") as f:
    CLUSTER_LABELS = {int(k): v for k, v in json.load(f).items()}

def assign_cluster(text: str, top_n=3):
    """
    Assign text to the closest cluster(s).
    Args:
        text (str): Input article text.
        top_n (int): Number of closest clusters to return.
    Returns:
        assigned_cluster (int), top_clusters (list of (cluster_id, similarity)).
    """
    # 1. Preprocess
    cleaned = clean_text(text)

    # 2. Embed
    emb = embed(cleaned)

    # 3. Predict cluster assignment
    cluster_id = kmeans.predict(emb)[0]

    # 4. Compute similarities to all cluster centroids
    sims = cosine_similarity(emb, kmeans.cluster_centers_).flatten()
    top_idx = sims.argsort()[::-1][:top_n]

    return (
        CLUSTER_LABELS[cluster_id],
        [(CLUSTER_LABELS[i], float(sims[i])) for i in top_idx]
    )

