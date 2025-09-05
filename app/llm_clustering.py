import json
import os
from pathlib import Path
from openai import OpenAI



ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
with open(ARTIFACT_DIR / "llm_clusters.json", "r") as f:
    LLM_CLUSTERS = json.load(f)

def classify_with_llm(text: str):
    """
    Classify a new article into one of the LLM-discovered clusters.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = "We already clustered previous articles into these categories:\n\n"
    for cluster, examples in LLM_CLUSTERS.items():
        prompt += f"Category: {cluster}\n"
        for ex in examples[:3]:   # include a few representative examples
            prompt += f" - {ex}\n"
        prompt += "\n"

    prompt += f"\nClassify the following article into one of the above categories:\n{text}\n"
    prompt += "Return only the category name."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()
