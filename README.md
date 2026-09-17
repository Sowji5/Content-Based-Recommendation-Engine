# Content-Based Recommendation Engine

A content-based recommendation system built to demonstrate how personalized
"Because You Watched…" style rows work on streaming platforms — plus a few
features real streaming apps typically *don't* expose to users.

**[Live interactive demo →](https://sowji5.github.io/Content-Based-Recommendation-Engine/ )**

## What it does

Given a small content catalog (title, genre, synopsis) and a user's watch
history, the engine builds a "taste profile" and recommends the closest
unwatched titles — the same core idea behind real recommendation shelves.

### Core technique

- **TF-IDF (Term Frequency – Inverse Document Frequency)** converts each
  show's genre + description into a numerical vector, weighting distinctive
  words (e.g. "sci-fi") higher than common ones.
- **Cosine similarity** measures how close two vectors are, used both to
  compare individual titles and to compare a user's averaged taste profile
  against the rest of the catalog.

### Features beyond a basic recommender

| Feature | What it does | Why it matters |
|---|---|---|
| **Explainability** | Surfaces the exact shared keywords driving each recommendation | Most real recommendation systems are black boxes — this one shows its work |
| **Diversity re-ranking (MMR)** | Uses Maximal Marginal Relevance to avoid recommending 4 near-identical titles in a row | Pure similarity ranking clusters; this balances relevance against variety |
| **Mood-based filtering** | Lets users browse by emotional register ("feel-good," "intense," "mind-bending") instead of genre | An axis most platforms don't offer at all |
| **Cold-start handling** | Falls back to catalog-wide popularity when a user has no watch history yet | Solves the "new user, no data" problem explicitly rather than returning nothing |

## Files

- `recommender.py` — Python implementation (pandas + scikit-learn). Includes
  `recommend_similar()`, `recommend_for_user()`, `explain()`, `diversify()`,
  `recommend_by_mood()`, and `recommend_cold_start()` as independently
  testable functions.
- `recommender-demo.html` — self-contained interactive browser demo (same
  logic reimplemented in vanilla JavaScript, no server or dependencies
  required).

## Running it locally

**Python engine:**
```bash
pip install pandas scikit-learn numpy
python recommender.py
```

**Interactive demo:**
Just open `recommender-demo.html` directly in any browser — no installation
needed.

## Why I built this

I work in marketing at a streaming startup, and wanted to understand — from
the inside out — how the recommendation systems that drive engagement and
retention on platforms like this actually work, by building a working (if
simplified) version myself.
