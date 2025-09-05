import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

URL_RE   = re.compile(r"http\S+|www\.\S+")
TOKEN_RE = re.compile(r"[a-z0-9]+")

def clean_text(s: str) -> str:
    s = s.lower()
    s = URL_RE.sub(" ", s)           # remove URLs
    # tokenization via regex (no punkt needed)
    tokens = TOKEN_RE.findall(s)
    # remove stopwords
    tokens = [t for t in tokens if t not in STOPWORDS]
    # lemmatize (noun default is fine for this task)
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)