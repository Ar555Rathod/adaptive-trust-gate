"""Normalize raw datasets into standard interactions and items CSVs.

The output will always be:
- interactions.csv (userId, itemId, rating, timestamp)
- items.csv (itemId, metadata_text)
"""
import pandas as pd
from atg import config


def normalize_movielens():
    print(f"Normalizing MovieLens data from {config.RAW_DIR}...")
    
    # Load raw
    ratings = pd.read_csv(config.RATINGS_CSV)
    movies = pd.read_csv(config.MOVIES_CSV)
    
    # Optional tags
    tags = None
    if config.TAGS_CSV.exists():
        tags = pd.read_csv(config.TAGS_CSV)
        
    # Standardize interactions
    interactions = ratings.rename(columns={"movieId": "itemId"})
    interactions = interactions[["userId", "itemId", "rating", "timestamp"]]
    
    # Standardize items
    movies = movies.rename(columns={"movieId": "itemId"})
    
    # Aggregate tags by movie
    tags_per_movie = {}
    if tags is not None:
        grouped = tags.groupby("movieId")["tag"].apply(lambda s: " ".join(s.astype(str).str.lower()))
        tags_per_movie = grouped.to_dict()
        
    metadata = []
    for row in movies.itertuples():
        genre_tokens = str(row.genres).replace("|", " ").replace("(no genres listed)", "")
        tag_tokens = tags_per_movie.get(row.itemId, "")
        metadata_text = f"{genre_tokens} {tag_tokens}".strip()
        metadata.append({"itemId": row.itemId, "metadata_text": metadata_text})
        
    items = pd.DataFrame(metadata)
    return interactions, items


def _iter_gzip_json(path):
    import gzip
    import json

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_goodreads():
    print(f"Normalizing Goodreads data from {config.RAW_DIR}...")

    # UCSD "byGenre" dumps: gzipped JSON-lines, one book/interaction per line.
    books_path = config.RAW_DIR / "goodreads_books_young_adult.json.gz"
    interactions_path = config.RAW_DIR / "goodreads_interactions_young_adult.json.gz"

    if not interactions_path.exists() or not books_path.exists():
        raise FileNotFoundError(f"Goodreads dataset files missing in {config.RAW_DIR}")

    # Standardize interactions. The byGenre interactions dump has no numeric
    # rating for unrated "want to read" / "reading" entries -- drop those and
    # keep only actual ratings (1-5); "date_added" becomes the timestamp.
    interaction_rows = []
    for rec in _iter_gzip_json(interactions_path):
        rating = rec.get("rating")
        if not rating:
            continue
        interaction_rows.append(
            {
                "userId": rec["user_id"],
                "itemId": rec["book_id"],
                "rating": float(rating),
                "timestamp": pd.to_datetime(
                    rec.get("date_added"), errors="coerce"
                ),
            }
        )
    interactions = pd.DataFrame(interaction_rows)
    timestamps = interactions["timestamp"]
    interactions["timestamp"] = (
        timestamps.astype("int64") // 10**9
    ).where(timestamps.notna(), 0)
    interactions = interactions[["userId", "itemId", "rating", "timestamp"]]

    # Standardize items: the byGenre book dump only carries author IDs (no
    # names), so metadata text is built from title + description.
    metadata = []
    for rec in _iter_gzip_json(books_path):
        title = rec.get("title") or ""
        description = rec.get("description") or ""
        metadata_text = f"{title} {description}".strip()
        metadata.append({"itemId": rec["book_id"], "metadata_text": metadata_text})

    items = pd.DataFrame(metadata)
    return interactions, items


def main():
    if config.DATASET_NAME == "movielens":
        interactions, items = normalize_movielens()
    elif config.DATASET_NAME == "goodreads":
        interactions, items = normalize_goodreads()
    else:
        raise ValueError(f"Unknown dataset: {config.DATASET_NAME}")
        
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    interactions.to_csv(config.NORMALIZED_INTERACTIONS_CSV, index=False)
    items.to_csv(config.NORMALIZED_ITEMS_CSV, index=False)
    print(f"Saved normalized data to {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
