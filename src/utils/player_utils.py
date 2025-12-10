"""
Utilities for mapping player IDs to readable names.
"""

import json
import os
from typing import Dict, List

import pandas as pd


class PlayerNameMapper:
    """
    Manage persistent mapping from raw player IDs to readable names.
    """

    def __init__(
        self,
        map_path: str = "src/config/player_map.json",
        candidate_names: List[str] | None = None,
    ):
        """
        Args:
            map_path (str): Path to JSON file storing mappings.
            candidate_names (list): Pool of names to use for new players.
        """
        self.map_path = map_path
        self.name_map: Dict[str, str] = {}
        self.candidate_names = candidate_names or [
            "Alice",
            "Beth",
            "Chloe",
            "Diana",
            "Ella",
            "Fiona",
            "Grace",
            "Hannah",
            "Isla",
            "Jasmine",
            "Katie",
            "Laura",
            "Megan",
            "Nina",
            "Olivia",
            "Paige",
            "Rachel",
            "Sophie",
        ]
        self._load_map()

    def _load_map(self) -> None:
        """Load existing mapping from JSON file if it exists."""
        if os.path.exists(self.map_path):
            with open(self.map_path, "r", encoding="utf-8") as f:
                self.name_map = json.load(f)
        else:
            self.name_map = {}

    def _save_map(self) -> None:
        """Save current mapping to JSON file."""
        os.makedirs(os.path.dirname(self.map_path), exist_ok=True)
        with open(self.map_path, "w", encoding="utf-8") as f:
            json.dump(self.name_map, f, indent=4)

    def update_mapping(self, df: pd.DataFrame, col: str = "player_name") -> None:
        """
        Update the mapping with any new IDs found in the DataFrame.

        Args:
            df (pd.DataFrame): Input data
            col (str): Column containing player IDs
        """
        unique_ids = df[col].unique().tolist()
        used_names = set(self.name_map.values())

        for pid in unique_ids:
            if pid not in self.name_map:
                # pick the next unused candidate
                next_name = next(
                    (n for n in self.candidate_names if n not in used_names),
                    f"Player_{len(self.name_map) + 1}",
                )
                self.name_map[pid] = next_name
                used_names.add(next_name)

        self._save_map()

    def apply_mapping(self, df: pd.DataFrame, col: str = "player_name") -> pd.DataFrame:
        """
        Apply mapping to DataFrame.

        Args:
            df (pd.DataFrame): Input data
            col (str): Column to remap

        Returns:
            pd.DataFrame: DataFrame with readable names
        """
        self.update_mapping(df, col)
        df = df.copy()
        df[col] = df[col].map(self.name_map)
        return df
