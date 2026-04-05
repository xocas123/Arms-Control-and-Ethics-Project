"""Segment arms-control relevant passages from general debate speeches."""
import re
import pandas as pd
import numpy as np

ARMS_CONTROL_KEYWORDS = [
    "disarmament", "arms control", "nuclear weapon", "nuclear disarmament",
    "non-proliferation", "nonproliferation", "chemical weapon", "biological weapon",
    "landmine", "cluster munition", "arms trade", "arms transfer", "conventional weapon",
    "TPNW", "NPT", "ATT", "Ottawa", "CCW", "BWC", "CWC", "CTBT",
    "humanitarian consequence", "humanitarian disarmament",
    "deterrence", "deterrent", "strategic stability", "nuclear umbrella",
    "verification", "safeguards", "inspection regime",
    "First Committee", "Conference on Disarmament",
    "IAEA", "OPCW", "Biological Weapons Convention",
    "nuclear-free", "nuclear-weapon-free zone",
    "arms race", "strategic forces", "nuclear doctrine",
    "drone", "unmanned aerial", "unmanned combat", "UAV", "UCAV",
    "remotely piloted", "autonomous weapon", "lethal autonomous",
    "LAWS", "killer robot", "meaningful human control",
]

def split_into_paragraphs(text: str) -> list:
    """Split text into paragraphs."""
    # Split on double newlines first, then single newlines if needed
    paragraphs = re.split(r'\n\s*\n', text)
    if len(paragraphs) < 3:
        # Try single newlines
        paragraphs = text.split('\n')
    return [p.strip() for p in paragraphs if len(p.strip()) > 50]


def find_keyword_matches(text: str, keywords: list = ARMS_CONTROL_KEYWORDS) -> list:
    """Find which keywords appear in text (case-insensitive)."""
    text_lower = text.lower()
    return [kw for kw in keywords if kw.lower() in text_lower]


def segment_arms_control(df: pd.DataFrame,
                          method: str = "keyword",
                          min_segment_length: int = 100,
                          context_sentences: int = 2,
                          text_col: str = "text") -> pd.DataFrame:
    """
    Extract arms-control relevant passages from full speech text.

    method='keyword': extract paragraphs with keyword matches + context paragraphs

    Returns DataFrame with same schema + columns:
    - segment_text: str (extracted passage, or full text if no segments found)
    - keyword_matches: list[str]
    - n_matches: int
    - is_full_speech: bool (True if no segments found, using full speech)
    """
    results = []

    for _, row in df.iterrows():
        text = row.get(text_col, "")
        if not text or len(text) < min_segment_length:
            continue

        paragraphs = split_into_paragraphs(text)

        # Find paragraphs with keyword matches
        matched_indices = []
        all_matches = []

        for i, para in enumerate(paragraphs):
            matches = find_keyword_matches(para)
            if matches:
                matched_indices.append(i)
                all_matches.extend(matches)

        if not matched_indices:
            # No keywords found — use full speech with flag
            new_row = dict(row)
            new_row["segment_text"] = text
            new_row["keyword_matches"] = []
            new_row["n_matches"] = 0
            new_row["is_full_speech"] = True
            results.append(new_row)
        else:
            # Extract matched paragraphs + context
            selected_indices = set()
            for idx in matched_indices:
                for offset in range(-context_sentences, context_sentences + 1):
                    if 0 <= idx + offset < len(paragraphs):
                        selected_indices.add(idx + offset)

            selected_paras = [paragraphs[i] for i in sorted(selected_indices)]
            segment_text = "\n\n".join(selected_paras)

            new_row = dict(row)
            new_row["segment_text"] = segment_text
            new_row["keyword_matches"] = list(set(all_matches))
            new_row["n_matches"] = len(set(all_matches))
            new_row["is_full_speech"] = False
            results.append(new_row)

    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        print(f"[Segment] {len(result_df)} documents, avg {result_df['n_matches'].mean():.1f} keyword matches per doc")
    return result_df
