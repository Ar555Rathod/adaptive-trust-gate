"""Download the raw files for the dataset selected by ATG_DATASET.

    ATG_DATASET=goodreads   ATG_GOODREADS_GENRE=poetry        (default genre)
    ATG_DATASET=movielens                                     (ml-latest-small)
    ATG_DATASET=movielens1m
    ATG_DATASET=amazon      ATG_AMAZON_CATEGORY=CDs_and_Vinyl (default category)

Files already on disk are skipped, so this is safe to re-run. Everything comes
straight from the publishers' own hosts, so on Colab nothing needs to be
uploaded to Drive first.

For Amazon this fetches the 'rating_only' benchmark file (user, item, rating,
timestamp) rather than the full review dump: the pipeline never uses review
text, and the full file is ~10x larger (1.03 GB vs 94 MB for CDs_and_Vinyl).
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atg import config

UCSD_GOODREADS = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre"
UCSD_AMAZON = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"
GROUPLENS = "https://files.grouplens.org/datasets/movielens"


def fetch(url, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have  {dest.name}  ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  fetch {dest.name} ...", flush=True)
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    print(f"        {dest.stat().st_size / 1e6:.1f} MB")


def fetch_zip(url, zip_name: str, extracted_dir: Path):
    if extracted_dir.exists() and any(extracted_dir.iterdir()):
        print(f"  have  {extracted_dir.name}/")
        return
    zpath = extracted_dir.parent / zip_name
    fetch(url, zpath)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(extracted_dir.parent)
    print(f"  extracted -> {extracted_dir}")


def main():
    name, raw = config.DATASET_NAME, config.RAW_DIR
    print(f"dataset={name}  ->  {raw}")
    if name == "goodreads":
        g = config.GOODREADS_GENRE
        for kind in ("books", "interactions"):
            f = f"goodreads_{kind}_{g}.json.gz"
            fetch(f"{UCSD_GOODREADS}/{f}", raw / f)
    elif name == "amazon":
        c = config.AMAZON_CATEGORY
        fetch(f"{UCSD_AMAZON}/benchmark/0core/rating_only/{c}.csv.gz", raw / f"{c}.csv.gz")
        fetch(f"{UCSD_AMAZON}/raw/meta_categories/meta_{c}.jsonl.gz", raw / f"meta_{c}.jsonl.gz")
    elif name == "movielens1m":
        fetch_zip(f"{GROUPLENS}/ml-1m.zip", "ml-1m.zip", raw)
    elif name == "movielens":
        fetch_zip(f"{GROUPLENS}/ml-latest-small.zip", "ml-latest-small.zip", raw)
    else:
        raise SystemExit(f"no download recipe for ATG_DATASET={name}")


if __name__ == "__main__":
    main()
