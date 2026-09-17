"""
Content-based movie/show recommendation engine.

Built for a streaming-platform portfolio project. Demonstrates:
- TF-IDF vectorization of content metadata (genres + description)
- Cosine similarity for "more like this" recommendations
- A simple hybrid re-rank using a user's watch history (average profile vector)

Swap `CATALOG` for a real dataset (e.g. TMDB, MovieLens) to make this
production-realistic — the pipeline doesn't change.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------------------------------------------------------------------
# 1. Catalog (synthetic, stand-in for a real content table)
# ---------------------------------------------------------------------------
CATALOG = [
    {"id": 1, "title": "Signal Lost", "genres": "sci-fi thriller mystery",
     "description": "A comms officer on a deep-space station must find out why every ship that answers her signal disappears."},
    {"id": 2, "title": "Kitchen Confidential Nights", "genres": "drama comedy food",
     "description": "A chef rebuilding her career after a public scandal takes over a failing restaurant with a rebellious staff."},
    {"id": 3, "title": "The Long Take", "genres": "drama mystery crime",
     "description": "A film editor discovers evidence of a real murder hidden in unused footage from a decades-old movie."},
    {"id": 4, "title": "Paper Empires", "genres": "drama business thriller",
     "description": "Siblings fight for control of their late father's media conglomerate as old secrets resurface."},
    {"id": 5, "title": "Orbit Academy", "genres": "sci-fi comedy teen",
     "description": "Misfit cadets at a struggling space academy try to stop it from being shut down."},
    {"id": 6, "title": "Static and Bone", "genres": "horror mystery thriller",
     "description": "A radio host investigating a string of disappearances starts hearing the missing on her own broadcast."},
    {"id": 7, "title": "Second Helping", "genres": "comedy romance food",
     "description": "Two rival food-truck owners are forced into a business partnership after a citywide festival mishap."},
    {"id": 8, "title": "The Quiet Ledger", "genres": "drama crime business",
     "description": "A forensic accountant uncovers a fraud that reaches into her own family's company."},
    {"id": 9, "title": "Deep Field", "genres": "sci-fi drama mystery",
     "description": "An astronomer's data reveals a signal that shouldn't exist, and someone wants it buried."},
    {"id": 10, "title": "Off Menu", "genres": "comedy drama food",
     "description": "A washed-up celebrity chef is demoted to hosting a low-budget cooking show and slowly falls back in love with the craft."},
]

df = pd.DataFrame(CATALOG)
df["text"] = df["genres"] + " " + df["description"]

# ---------------------------------------------------------------------------
# 2. Vectorize content (TF-IDF over genres + synopsis)
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
content_matrix = vectorizer.fit_transform(df["text"])
similarity_matrix = cosine_similarity(content_matrix)


def recommend_similar(title: str, top_n: int = 3) -> pd.DataFrame:
    """'More like this' — pure content similarity to one title."""
    idx = df.index[df["title"] == title][0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:top_n]
    return df.iloc[[i for i, _ in scores]][["title", "genres"]].assign(
        score=[round(s, 3) for _, s in scores]
    )


def recommend_for_user(watch_history: list[str], top_n: int = 3) -> pd.DataFrame:
    """
    Build a user 'taste profile' by averaging the content vectors of
    everything they've watched, then rank the rest of the catalog
    against that profile. This is the same idea a real 'Because you
    watched...' row is built on, just without collaborative signals.
    """
    watched_idx = df.index[df["title"].isin(watch_history)]
    profile_vector = np.asarray(content_matrix[watched_idx].mean(axis=0))
    scores = cosine_similarity(profile_vector, content_matrix)[0]

    ranked = sorted(
        [(i, s) for i, s in enumerate(scores) if df.iloc[i]["title"] not in watch_history],
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    return df.iloc[[i for i, _ in ranked]][["title", "genres"]].assign(
        score=[round(s, 3) for _, s in ranked]
    )


if __name__ == "__main__":
    print("More like 'Signal Lost':")
    print(recommend_similar("Signal Lost"), "\n")

    print("Recommended for a user who watched Signal Lost + Deep Field:")
    print(recommend_for_user(["Signal Lost", "Deep Field"]))
