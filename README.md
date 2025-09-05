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
├── notebooks/
│ └── Preprocessing_and_analysis.ipynb # EDA + clustering experiments
│
├── artifacts/ # Saved models & embeddings (ignored in Git)
├── data/ # Raw dataset (ignored in Git)
│
├── requirements.txt
├── .gitignore
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

