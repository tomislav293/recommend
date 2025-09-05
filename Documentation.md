# 1. Data Preparation & Exploration

In the first step, the dataset was loaded into a Pandas DataFrame. The dataset contained 50 articles with fields such as id, date, title, summary, category, and url.

## Handling Data Types & Cleaning

- The id column was converted to string type
- The date column was converted into datetime format for easier time-series analysis
- Text based columns (title, summary, category, url) were explicitly converted to strings to avoid type mismatches
- Checked for missing values and duplicates

Since the dataset contains only categorical and text fields (no numerical measures beyond word counts), logarithmic transformations or normalization were not needed. Similarly, outlier detection was unnecessary, as there are no continuous numeric values that could distort the analysis.

## Exploratory Statistics

Exploratory statistics showed that the dataset contains 50 articles across three categories (Sports, Technology, Health). Category distribution was fairly balanced, with Sports having 20 articles, while Technology and Health each had 15. Temporal analysis showed that articles were published steadily across June–July 2024, with roughly one article per day per category.

Word count analysis revealed that titles are short and concise (average 3–4 words), while summaries are richer in content (average 5–6 words, with a maximum of 8). This indicates that summaries carry more descriptive information and will likely be more important for embedding and clustering.

# 2. Text preprocesing

## Cleaning and Normalization
To prepare the dataset for clustering, several text preprocessing steps were applied. First, a custom cleaning function was implemented. This function lowercased the text, removed URLs and punctuation, tokenized words using a regex pattern, removed English stopwords, and applied lemmatization. These steps ensure that texts are standardized and reduced to their semantic core, which improves the quality of embeddings by minimizing noise.

## Named Entity Recognition (NER)
Next, we extracted basic entities with spaCy. Using the en_core_web_sm model, we disabled unneeded components (tagger, parser, attribute_ruler, lemmatizer) for efficiency and focused only on entity recognition.
Entities were grouped into three categories:
- PERSON → names of people
- ORG → organizations and companies
- GPE → geopolitical entities (countries, cities, regions)

## Text embeddings
Finally, we converted the cleaned text into dense vector embeddings using SentenceTransformers. We chose the model all-MiniLM-L6-v2, a lightweight transformer producing 384-dimensional embeddings.
- Computation was performed on GPU (cuda) if available, otherwise CPU.
- A batch size of 64 ensured efficient processing.
- The resulting embeddings were stored as a NumPy array (X) with shape (50, 384).



# 3. Clustering & Ranking
To determine the number of clusters (k) for KMeans, we used two standard methods:
## Determine number of clusters 
### Elbow Method
The Elbow Method evaluates clustering inertia (within-cluster sum of squares) for different values of k. The inertia decreases as k increases, but the improvement diminishes after a certain point.

- In our plot, inertia drops steeply between k=2 and k=3, after which the curve becomes more gradual.
- This suggests that k=3 is a reasonable choice, as adding more clusters beyond this point yields diminishing returns.

### Silhouette Score
The Silhouette Score measures how well-separated the clusters are. It ranges from -1 (bad clustering) to 1 (perfect clustering).

- Our plot shows that the highest silhouette score is achieved at k=7, suggesting that clusters are more compact and better separated at this value.
- However, since we already know the dataset has 3 categories (Sports, Technology, Health), setting k=3 is more interpretable and consistent with the ground truth.
- Choosing k=7 would over-segment the dataset, creating artificial sub-clusters within the main categories.

### Final Choice of k
- k=3 was chosen for downstream clustering, balancing interpretability (matching known categories) with clustering quality.
- While k=7 had a higher silhouette score, it likely represents subtopics (e.g., “Running shoes” vs. “Tennis” inside Sports), which may be useful in larger datasets but are unnecessary here.

## Clustering
### KMeans Clustering
We first applied KMeans clustering with k=3, chosen based on Elbow and Silhouette analysis.

- The trained model was also saved with joblib for later reuse in the FastAPI app.
- Results: ARI = 0.255, NMI = 0.315 → alignment with true categories was weak.
- Interpretation: KMeans assumes spherical cluster shapes, which often fails for high-dimensional semantic embeddings.

### Agglomerative Clustering

Next, we applied hierarchical clustering (Ward linkage, Euclidean metric).

- This approach allows more flexible cluster boundaries.
- Results: ARI = 0.405, NMI = 0.512 → improved over KMeans.
- Interpretation: Articles were grouped more meaningfully, but still imperfect separation.

### NMF Topic Modeling

As a traditional topic modeling baseline, we ran Non-negative Matrix Factorization (NMF) on TF-IDF features.

- We extracted 6 latent topics, each represented by its top keywords.
- Some topics were interpretable (e.g., diet/health, tennis/sports), but others were noisy.
- Limitation: With only 50 articles, NMF struggled to form stable topics.

### LLM-based Clustering

Finally, we tested a Large Language Model (LLM) approach. Using GPT-4o-mini, we provided all article texts and asked it to group them into 3 categories.

- The model produced clusters with short labels (e.g., Technology, Health, Sports) and assigned document IDs.
- Any unassigned articles were marked as "Unassigned".
- Results: ARI = 0.964, NMI = 0.965 → nearly perfect alignment with ground-truth categories.
- Interpretation: The LLM leveraged background knowledge to correctly recognize categories, outperforming purely embedding-based clustering.


## Clustering Evaluation & Ranking

Within each cluster, we ranked articles by cosine similarity to the cluster centroid (or pseudo-centroid for Agglomerative and LLM).
- Top-ranked articles represent the most prototypical examples.
- For example, a sports cluster might have a Wimbledon article at the top, while Technology clusters highlight Android or AR-related articles.
- This ranking is useful for labeling clusters and retrieving representative content.

### Cluster Cohesion & Separation
We also computed internal cluster quality metrics:
- Cohesion: average similarity of articles within the same cluster.
- Separation: average similarity to articles in other clusters.

Findings:
- KMeans clusters had lower cohesion (0.13–0.23) and weaker separation.
- Agglomerative clustering improved cohesion (up to 0.33).
- LLM-based clusters were balanced, with cohesion around 0.14–0.17 and separation ~0.08 — consistent and interpretable.

### Mapping to True categories
We mapped clusters back to categories using majority voting:

- KMeans: {0: Technology, 1: Technology, 2: Sports} → no dedicated Health cluster.
- Agglomerative: {0: Technology, 1: Sports, 2: Sports} → again, Health scattered.

### Cluster alignment with true categories
To measure how well the clusters align with the ground truth categories (Sports, Technology, Health), we plotted confusion matrices for each method.

KMeans:
- Two clusters mapped to Technology, while Health was absorbed into Technology or misclassified.
- Sports formed a partial cluster, but overlap remained.
- Confirms the weakness observed in ARI/NMI scores.

Agglomerative:
- Performed better: Sports and Health were more distinct.
- Some overlap remained (e.g., Health vs. Technology).
- Balanced separation, consistent with improved ARI/NMI.

LLM-based:
- Achieved perfect separation: all 50 documents were assigned correctly to their categories.
- Confusion matrix was a clean diagonal.
- This validates that LLMs can exploit semantic understanding to cluster correctly, even with few examples.

## LLM Integration

### Cluster Summarization
We used GPT-4o-mini to generate human-readable summaries of clusters.

- First, we selected representative articles (either the first few summaries or the most central by cosine similarity).
- Then we prompted the LLM to write a short theme description for each cluster.


### LLM for Determining Cluster Count

We also tested using the LLM to directly decide the number of clusters.

The model was given all 50 article texts and asked:
- How many coherent categories exist?
- Give each cluster a label.
- Assign article indices to clusters.

The output was structured as JSON, making it easy to parse into df["cluster_llm"].

## API Endpoint
Finally, we implemented a FastAPI application that exposes clustering functionality as web endpoints.
This makes the system usable as a service for new, unseen articles.

The API is organized into modular components:

- preprocessing.py → cleaning and normalizing input text.
- embeddings.py → generating embeddings using SentenceTransformers.
- clustering.py → assigning text to the closest KMeans cluster.
- llm_clustering.py → classifying text into clusters using an LLM (OpenAI GPT-4o-mini).
- api.py → FastAPI app that ties everything together.

### Endpoints

/predict (ML-based clustering) - fast and efficient, suitable for large scale
- Preprocesses the input text.
- Generates embeddings with the MiniLM model.
- Assigns the article to the nearest KMeans cluster.
- Returns the assigned cluster and the top 3 closest clusters ranked by cosine similarity.

/llm-classify (LLM-based clustering) - more interpretable, slower, requires API access
- Uses a few-shot prompt with representative examples from prior LLM clustering.
- Assigns the input text to one of the discovered LLM clusters.
- Returns only the cluster label.


## Performance Considerations

- Embedding efficiency: Embeddings are computed in batches (32/64) to reduce latency. In production, async FastAPI endpoints or thread pools could be used to process multiple requests concurrently, with GPU acceleration if available.

- Memory usage:

- - Only cleaned text is stored (text_clean), avoiding unnecessary intermediate columns.

- - Models and embeddings are saved to disk (artifacts/) so they don’t need to be recomputed.

- - Cosine similarities are calculated on demand instead of storing full pairwise matrices.

- Scalability: For larger datasets, methods like MiniBatchKMeans or a vector database (FAISS, Pinecone) would make clustering and retrieval more memory-efficient.

## To Be Improved

Testing Coverage
- Add unit tests for preprocessing, embeddings, clustering, and ranking.
- Add integration tests for the LLM-based endpoints.


Logging & Monitoring
- Replace print with the logging module.
- Use structured logs (JSON logging) for better observability.
- Add monitoring hooks (e.g., request duration, clustering performance metrics).

Scalability
- Containerize with Docker for easy deployment
- Add Docker Compose for orchestrating app + database/artifacts
- Deploy to cloud (AWS/GCP/Azure) with autoscaling
