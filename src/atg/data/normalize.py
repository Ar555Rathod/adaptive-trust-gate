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
    print(f"Normalizing Goodreads '{config.GOODREADS_GENRE}' data from {config.RAW_DIR}...")

    # UCSD "byGenre" dumps: gzipped JSON-lines, one book/interaction per line.
    # Every genre slice shares this schema, so the genre only selects filenames.
    genre = config.GOODREADS_GENRE
    books_path = config.RAW_DIR / f"goodreads_books_{genre}.json.gz"
    interactions_path = config.RAW_DIR / f"goodreads_interactions_{genre}.json.gz"

    if not interactions_path.exists() or not books_path.exists():
        raise FileNotFoundError(
            f"Goodreads '{genre}' dataset files missing in {config.RAW_DIR}. "
            f"Expected {books_path.name} and {interactions_path.name}."
        )

    # rating for unrated "want to read" / "reading" entries -- drop those and
    # keep only actual ratings (1-5); "date_added" becomes the timestamp.
    import tempfile
    import os
    import csv

    fd, temp_csv = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["userId", "itemId", "rating", "date_added"])
        for rec in _iter_gzip_json(interactions_path):
            rating = rec.get("rating")
            if not rating:
                continue
            writer.writerow([
                rec["user_id"],
                rec["book_id"],
                float(rating),
                rec.get("date_added", "")
            ])

    print("Building interactions DataFrame...")
    interactions = pd.read_csv(temp_csv, dtype={"rating": "float32"})
    os.remove(temp_csv)

    print("Converting timestamps (this may take a minute)...")
    timestamps = pd.to_datetime(
        interactions["date_added"], 
        format="%a %b %d %H:%M:%S %z %Y", 
        errors="coerce",
        utc=True
    )
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


def normalize_movielens1m():
    """MovieLens-1M: '::'-separated, Latin-1, whole-star ratings, every user has
    at least 20 ratings (so, like ml-latest-small, the cold segment only exists
    through the simulated cold-start split in atg.data.splits).

    Item text is genres plus title. ML-1M ships no user tags, so this is the
    thinnest content signal of the datasets used -- deliberately so: it is the
    dense, weak-metadata condition.
    """
    print(f"Normalizing MovieLens-1M data from {config.RAW_DIR}...")
    interactions = pd.read_csv(
        config.RAW_DIR / "ratings.dat", sep="::", engine="python", encoding="latin-1",
        names=["userId", "itemId", "rating", "timestamp"])
    movies = pd.read_csv(
        config.RAW_DIR / "movies.dat", sep="::", engine="python", encoding="latin-1",
        names=["itemId", "title", "genres"])
    movies["metadata_text"] = (movies["genres"].str.replace("|", " ", regex=False)
                               + " " + movies["title"]).str.strip()
    return interactions[["userId", "itemId", "rating", "timestamp"]], movies[["itemId", "metadata_text"]]


def _amazon_artist(store):
    """'Shrimp City Slim  (Artist)    Format: Audio CD' -> 'Shrimp City Slim'."""
    if not store:
        return ""
    import re
    name = store.split("Format:")[0]
    return re.sub(r"\([^)]*\)", " ", name).strip()


def normalize_amazon():
    """Amazon Reviews 2023, one category.

    Ratings come from the 'rating_only' benchmark file (user, item, rating,
    timestamp in milliseconds) rather than the full review dump, which carries
    review text we do not use and is ~10x larger.

    Amazon allows a user to review the same item more than once. Left in, one
    (user, item) pair could land in both train and test, so only each pair's
    most recent rating is kept.

    Item text: title, artist (the 'store' field), record label, the genre path
    with each genre kept as a single token so 'Dance & Electronic' is one
    feature rather than three words, and the description. Only items that were
    actually rated are kept -- the content expert never queries the others.
    """
    cat = config.AMAZON_CATEGORY
    ratings_path = config.RAW_DIR / f"{cat}.csv.gz"
    meta_path = config.RAW_DIR / f"meta_{cat}.jsonl.gz"
    if not ratings_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Amazon '{cat}' files missing in {config.RAW_DIR}. Expected {ratings_path.name} "
            f"and {meta_path.name} -- run scripts/00_download_data.py.")

    print(f"Normalizing Amazon '{cat}' data from {config.RAW_DIR}...")
    r = pd.read_csv(ratings_path, dtype={"rating": "float32", "timestamp": "int64"})
    r = r.rename(columns={"user_id": "userId", "parent_asin": "itemId"})
    r["timestamp"] = r["timestamp"] // 1000          # milliseconds -> seconds
    n_raw = len(r)
    r = (r.sort_values(["userId", "itemId", "timestamp"], kind="mergesort")
          .drop_duplicates(["userId", "itemId"], keep="last"))
    print(f"  ratings: {n_raw:,} raw, {n_raw - len(r):,} repeat (user, item) pairs dropped, {len(r):,} kept")
    interactions = r[["userId", "itemId", "rating", "timestamp"]].reset_index(drop=True)

    rated = set(interactions["itemId"].unique())
    metadata = []
    for rec in _iter_gzip_json(meta_path):
        item = rec.get("parent_asin")
        if item not in rated:
            continue
        genres = [c.strip().replace(" ", "_") for c in (rec.get("categories") or [])]
        details = rec.get("details") or {}
        label = str(details.get("Label") or details.get("Manufacturer") or "").replace(" ", "_")
        parts = [rec.get("title") or "", _amazon_artist(rec.get("store")), label,
                 " ".join(genres), " ".join(rec.get("description") or [])]
        metadata.append({"itemId": item, "metadata_text": " ".join(p for p in parts if p).strip()})
    items = pd.DataFrame(metadata, columns=["itemId", "metadata_text"])
    print(f"  items: {len(rated):,} rated, {len(items):,} with metadata")
    return interactions, items


def main():
    if config.DATASET_NAME == "movielens":
        interactions, items = normalize_movielens()
    elif config.DATASET_NAME == "goodreads":
        interactions, items = normalize_goodreads()
    elif config.DATASET_NAME == "movielens1m":
        interactions, items = normalize_movielens1m()
    elif config.DATASET_NAME == "amazon":
        interactions, items = normalize_amazon()
    else:
        raise ValueError(f"Unknown dataset: {config.DATASET_NAME}")
        
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    interactions.to_csv(config.NORMALIZED_INTERACTIONS_CSV, index=False)
    items.to_csv(config.NORMALIZED_ITEMS_CSV, index=False)
    print(f"Saved normalized data to {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
