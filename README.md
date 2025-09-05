## 📂 Project Structure
```
Recommend/
│
├── app/ # FastAPI app
│ ├── api.py # API endpoints
│ ├── clustering.py # KMeans clustering + assignment
│ ├── embeddings.py # Embedding generation
│ ├── llm_clustering.py # LLM-based classification
│ └── preprocessing.py # Text cleaning
│
├── tests
│ ├── test_api.py # API tests
│
│── Preprocessing_and_analysis.ipynb
│
├── artifacts/ # Saved models & embeddings (ignored in Git)
│
├── .env.example
├── articles_dataset.csv
├── requirements.txt
├── .gitignore
├── Documentation.md  # Project documentation
└── README.md

```

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/recommend.git
cd recommend
```
### Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate   # on Windows
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Environment variables
Create a .env file in the project root with your OpenAI API key:
```bash
OPENAI_API_KEY=your_openai_key_here
```

### Set up
Before running the API, you need to generate embeddings and train the clustering model.  
   Run the notebook in `notebooks/Preprocessing_and_analysis.ipynb` (or your preprocessing script).  
   This will create the following files inside `artifacts/`:
   - `kmeans_model.joblib`
   - `embeddings.joblib`
   - `cluster_labels.json`
   - (optional) `llm_clusters.json` if you run the LLM-based clustering

# Run the FastAPI app
From the project root:
```bash
uvicorn app.api:app --reload
```

FastAPI automatically provides Swagger UI:
```bash
http://127.0.0.1:8000/docs
```

1. Open the link in your browser.

2. You’ll see two endpoints:

- POST /predict → clustering with embeddings + KMeans

- POST /llm-classify → clustering with OpenAI LLM

3. Click “Try it out”, paste your input JSON, and hit Execute.

Example
```bash
{
  "text": "New fitness app improves marathon training efficiency"
}
```

