from fastapi import FastAPI
from pydantic import BaseModel
from .clustering import assign_cluster
from .llm_clustering import classify_with_llm
import logging

# Configure logging once
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Article Clustering API")

class ArticleInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(article: ArticleInput):
    logger.info("Received /predict request, text length=%d", len(article.text))
    cluster_id, top_clusters = assign_cluster(article.text)
    logger.info("Assigned to cluster=%s", cluster_id)
    return {
        "assigned_cluster": cluster_id,
        "top_clusters": [
            {"cluster_id": cid, "similarity": sim}
            for cid, sim in top_clusters
        ]
    }


@app.post("/llm-classify")
async def llm_classify(article: ArticleInput):
    logger.info("Received /llm-classify request, text length=%d", len(article.text))
    label = classify_with_llm(article.text)
    logger.info("LLM assigned cluster: %s", label)
    return {"assigned_cluster": label}

