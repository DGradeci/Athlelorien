"""
Utilities for mapping player IDs to readable names.
"""

import json
import os
import re
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
            "Abigail",
            "Ada",
            "Amber",
            "Amelia",
            "Annie",
            "April",
            "Ariana",
            "Astrid",
            "Audrey",
            "Autumn",
            "Beatrice",
            "Bella",
            "Bianca",
            "Bridget",
            "Brooke",
            "Caitlin",
            "Camille",
            "Cara",
            "Carly",
            "Catherine",
            "Celeste",
            "Charlotte",
            "Clara",
            "Daisy",
            "Danielle",
            "Darcey",
            "Edith",
            "Elena",
            "Eliza",
            "Elsie",
            "Emma",
            "Erin",
            "Esme",
            "Eva",
            "Felicity",
            "Francesca",
            "Freya",
            "Gemma",
            "Georgia",
            "Harriet",
            "Heidi",
            "Imogen",
            "Iris",
            "Ivy",
            "Joanna",
            "Juliet",
            "Keira",
            "Leona",
            "Lila",
            "Lucy"
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
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

        unique_ids = [pid for pid in df[col].dropna().unique().tolist() if str(pid).strip() != ""]
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
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

        self.update_mapping(df, col)
        df = df.copy()
        raw = df[col]
        mapped = raw.astype(str).map(self.name_map)
        df[col] = mapped.where(mapped.notna(), raw)
        return df

    def upgrade_placeholders(self, pattern: str = r"Player_\d+") -> int:
        """
        Replace existing placeholder names like 'Player_21' with unused candidate names.
        Returns how many entries were updated.
        """
        placeholder_ids = [pid for pid, nm in self.name_map.items()
                           if re.fullmatch(pattern, str(nm))]
        if not placeholder_ids:
            return 0
    
        used_real = {nm for nm in self.name_map.values()
                     if not re.fullmatch(pattern, str(nm))}
    
        available = [n for n in self.candidate_names if n not in used_real]
        k = min(len(placeholder_ids), len(available))
    
        for pid, new_name in zip(sorted(placeholder_ids), available[:k]):
            self.name_map[pid] = new_name
    
        self._save_map()
        return k