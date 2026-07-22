"""
prepare_dataset.py
===================
Builds the MiniGPT training corpus from the Cornell Movie-Dialogs Corpus
(Danescu-Niculescu-Mizil & Lee, 2011), a public research dataset released
by Cornell University for academic NLP research.

Instead of scraping full copyrighted screenplays, we use this pre-existing,
widely-used research corpus and filter it down to a curated set of
Star Wars and similar sci-fi/space-adventure films. This keeps the dataset
legitimate, reproducible, and appropriately sized for a laptop-trainable
character-level GPT (roughly 600-800KB of text).

What this script does:
    1. Downloads the raw corpus files (movie_lines.txt, movie_conversations.txt,
       movie_titles_metadata.txt) from a public GitHub mirror of the corpus.
    2. Filters to a curated list of movie IDs (Star Wars trilogy + similar
       sci-fi/adventure films already present in the corpus).
    3. Reconstructs conversations in their original turn-by-turn order using
       movie_conversations.txt (not just raw unordered lines).
    4. Formats each turn as "CHARACTER: line of dialogue" (screenplay-style),
       separating conversations with a blank line.
    5. Writes the final corpus to data/input.txt.

Run:
    python data/prepare_dataset.py
"""

import os
import urllib.request
from typing import Dict, List, Tuple

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

# Public GitHub mirror of the Cornell Movie-Dialogs Corpus raw files.
BASE_URL = (
    "https://raw.githubusercontent.com/SudharshanShanmugasundaram/Chatbot/"
    "master/data/cornell%20movie-dialogs%20corpus"
)

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "input.txt")

FILES = ["movie_lines.txt", "movie_conversations.txt", "movie_titles_metadata.txt"]

# Curated movie IDs: the Star Wars original trilogy plus a set of similar
# sci-fi / space-adventure films already present in the corpus. This keeps
# the dataset thematically coherent (spaceships, heroes, villains, banter)
# while giving the model enough data to actually learn dialogue patterns.
SELECTED_MOVIES: Dict[str, str] = {
    "m529": "Star Wars: A New Hope",
    "m337": "Star Wars: The Empire Strikes Back",
    "m489": "Star Wars: Return of the Jedi",
    "m583": "Star Trek V: The Final Frontier",
    "m68": "Galaxy Quest",
    "m584": "Tron",
    "m544": "Superman",
    "m549": "The Terminator",
    "m547": "Terminator 2: Judgment Day",
    "m614": "X-Men",
    "m34": "Blade Runner",
    "m253": "Back to the Future",
    "m433": "The Matrix",
    "m411": "Jurassic Park",
    "m15": "Aliens",
    "m236": "Alien",
    "m5": "The Fifth Element",
    "m97": "Independence Day",
    "m125": "Men in Black",
    "m126": "Minority Report",
    "m196": "Star Trek: First Contact",
    "m221": "Total Recall",
    "m237": "Alien vs. Predator",
    "m304": "Contact",
    "m365": "Gattaca",
    "m409": "Jurassic Park III",
    "m410": "The Lost World: Jurassic Park",
    "m473": "Planet of the Apes",
    "m478": "Predator",
    "m530": "Starship Troopers",
}

LINE_SEP = " +++$+++ "


def download_corpus() -> None:
    """Download raw corpus files if not already present locally."""
    os.makedirs(RAW_DIR, exist_ok=True)
    for fname in FILES:
        dest = os.path.join(RAW_DIR, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"[skip] {fname} already downloaded")
            continue
        url = f"{BASE_URL}/{fname}"
        print(f"[download] {fname} <- {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"[done] {fname} ({os.path.getsize(dest):,} bytes)")


def load_lines(path: str) -> Dict[str, Tuple[str, str, str]]:
    """
    Parse movie_lines.txt into a dict mapping lineID -> (movieID, character, text).

    Each row in the raw file has the format:
        lineID +++$+++ characterID +++$+++ movieID +++$+++ characterName +++$+++ text
    """
    lines: Dict[str, Tuple[str, str, str]] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for row in f:
            parts = row.rstrip("\n").split(LINE_SEP)
            if len(parts) < 5:
                continue
            line_id, _char_id, movie_id, char_name, text = parts[:5]
            if movie_id in SELECTED_MOVIES:
                lines[line_id] = (movie_id, char_name.strip(), text.strip())
    return lines


def load_conversations(path: str) -> List[List[str]]:
    """
    Parse movie_conversations.txt into a list of conversations, where each
    conversation is a list of lineIDs in chronological (as-spoken) order.

    Each row has the format:
        characterID1 +++$+++ characterID2 +++$+++ movieID +++$+++ ['L1','L2',...]
    """
    conversations: List[List[str]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for row in f:
            parts = row.rstrip("\n").split(LINE_SEP)
            if len(parts) < 4:
                continue
            movie_id = parts[2]
            if movie_id not in SELECTED_MOVIES:
                continue
            # The 4th field looks like: "['L194', 'L195', 'L196', 'L197']"
            raw_ids = parts[3].strip("[]").replace("'", "").replace('"', "")
            line_ids = [x.strip() for x in raw_ids.split(",") if x.strip()]
            if line_ids:
                conversations.append(line_ids)
    return conversations


def build_corpus(lines: Dict[str, Tuple[str, str, str]],
                  conversations: List[List[str]]) -> str:
    """Reconstruct conversations as 'CHARACTER: text' turns, in order."""
    blocks: List[str] = []
    for convo in conversations:
        turns = []
        for line_id in convo:
            if line_id in lines:
                _movie_id, char_name, text = lines[line_id]
                if text:
                    turns.append(f"{char_name}: {text}")
        if len(turns) >= 2:  # keep only genuine back-and-forth exchanges
            blocks.append("\n".join(turns))
    return "\n\n".join(blocks)


def main() -> None:
    download_corpus()

    lines_path = os.path.join(RAW_DIR, "movie_lines.txt")
    convos_path = os.path.join(RAW_DIR, "movie_conversations.txt")

    print("[parse] loading lines...")
    lines = load_lines(lines_path)
    print(f"[parse] {len(lines):,} dialogue lines from {len(SELECTED_MOVIES)} movies")

    print("[parse] loading conversations...")
    conversations = load_conversations(convos_path)
    print(f"[parse] {len(conversations):,} conversations")

    print("[build] reconstructing corpus...")
    corpus = build_corpus(lines, conversations)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(corpus)

    n_chars = len(corpus)
    n_vocab = len(set(corpus))
    print(f"[done] wrote {OUTPUT_PATH}")
    print(f"       {n_chars:,} characters, {n_vocab} unique characters")
    print("\nMovies included:")
    for mid, name in SELECTED_MOVIES.items():
        print(f"  - {name}")


if __name__ == "__main__":
    main()
