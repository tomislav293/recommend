from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "all-MiniLM-L6-v2"
encoder = SentenceTransformer(model_name, device=device)

def embed(texts):
    """
    Compute embeddings for one or more texts.
    Args:
        texts (str or list of str): Input text(s).
    Returns:
        numpy.ndarray: Embedding(s).
    """
    if isinstance(texts, str):
        texts = [texts]
    return encoder.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=False
    )
