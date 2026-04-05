"""
Configuration loader for the arms control NLP pipeline.
Loads config.yaml and resolves paths relative to the project root.
"""

import os
import yaml
from pathlib import Path


class Config:
    """Simple configuration container loaded from config.yaml."""

    def __init__(self, config_dict: dict, project_root: Path):
        self._data = config_dict
        self.project_root = project_root

        # Scalar fields
        self.year_start: int = config_dict.get("year_start", 1970)
        self.year_end: int = config_dict.get("year_end", 2023)
        self.focus_start: int = config_dict.get("focus_start", 2000)
        self.focus_end: int = config_dict.get("focus_end", 2023)
        self.embedding_model: str = config_dict.get("embedding_model", "all-MiniLM-L6-v2")
        self.bertopic_min_topic_size: int = config_dict.get("bertopic_min_topic_size", 15)
        self.lda_k_range: list = config_dict.get("lda_k_range", [5, 10, 15, 20, 25, 30])
        self.dtm_n_topics: int = config_dict.get("dtm_n_topics", 15)
        self.umap_n_neighbors: int = config_dict.get("umap_n_neighbors", 15)
        self.umap_n_components: int = config_dict.get("umap_n_components", 5)
        self.hdbscan_min_cluster_size: int = config_dict.get("hdbscan_min_cluster_size", 15)
        self.keyword_window_size: int = config_dict.get("keyword_window_size", 3)
        self.embedding_similarity_threshold: float = config_dict.get(
            "embedding_similarity_threshold", 0.35
        )
        self.random_seed: int = config_dict.get("random_seed", 42)

        # Weights sub-dict
        self.weights: dict = config_dict.get(
            "weights",
            {
                "treaty_anchor_similarity": 0.30,
                "voting_score": 0.25,
                "humanitarian_topic_pct": 0.20,
                "commitment_strength": 0.15,
                "care_harm_loading": 0.10,
            },
        )

        # Resolved paths
        self.data_dir: Path = project_root / config_dict.get("data_dir", "data/")
        self.output_dir: Path = project_root / config_dict.get("output_dir", "output/")
        self.ungdc_path: Path = project_root / config_dict.get(
            "ungdc_path", "data/raw/ungdc/"
        )
        self.treaties_path: Path = project_root / config_dict.get(
            "treaties_path", "data/raw/treaties/"
        )
        self.voting_path: Path = project_root / config_dict.get(
            "voting_path", "data/raw/unvotes/"
        )
        self.processed_path: Path = self.data_dir / "processed"
        self.country_groups_path: Path = self.data_dir / "raw" / "country_groups.json"

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __repr__(self):
        return f"Config(years={self.year_start}-{self.year_end})"


def load_config(config_path: str = None) -> Config:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to config.yaml. Defaults to config.yaml in the project root
        (two levels up from this file: src/config.py → project root).

    Returns
    -------
    Config
    """
    if config_path is None:
        # Resolve relative to this file's location
        this_dir = Path(__file__).parent
        project_root = this_dir.parent
        config_path = project_root / "config.yaml"
    else:
        config_path = Path(config_path)
        project_root = config_path.parent

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return Config(data, project_root)


# Module-level singleton (lazy-loaded)
_config: Config = None


def get_config() -> Config:
    """Return the module-level Config singleton, loading it if necessary."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
