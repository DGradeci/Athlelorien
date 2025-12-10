"""
Utility functions for fetching football pitch polygons from Overpass (OSM)
and saving them as JSON.

This provides a simple callable function:
    fetch_all_pitches(outfile)

which you can import into notebooks or other modules.

Example
-------
from src.config.arenas import ARENAS
from src.utils.pitch_fetcher import fetch_all_pitches

fetch_all_pitches("toppserien_pitches.json", arenas=ARENAS)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

from config.arenas import ARENAS
from utils.osm_client import OverpassClient
from utils.pitch_finder import PitchFinder


def fetch_all_pitches(
    outfile: str | Path = "toppserien_pitches.json",
    arenas: Dict[str, tuple[float, float]] = ARENAS,
    radius: int = 300,
    sleep_secs: float = 2.0,
) -> Dict[str, Any]:
    """
    Fetch nearest pitch polygon for each arena.

    Parameters
    ----------
    outfile : str or Path
        Where to save the resulting JSON file.
    arenas : dict
        Mapping of stadium name → (lat, lon)
    radius : int
        Search radius in metres.
    sleep_secs : float
        Pause between requests to avoid rate limiting.

    Returns
    -------
    dict
        The full structure written to JSON file.
    """

    client = OverpassClient()
    finder = PitchFinder(client)

    results: Dict[str, Dict[str, Any]] = {}

    for name, (lat, lon) in arenas.items():
        print(f"\nFetching pitch for {name} ({lat}, {lon})...")

        try:
            poly = finder.get_nearest_pitch_polygon(lat, lon, radius=radius)
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            continue

        if poly is None:
            print("  No pitch found nearby.")
            continue

        coords = poly["coords"]
        print(
            f"  Found way {poly['id']} with {len(coords)} unique vertices. "
            f"Tags: {poly.get('tags', {})}"
        )

        results[name] = {
            "way_id": poly["id"],
            "tags": poly.get("tags", {}),
            "coords": coords,
        }

        time.sleep(sleep_secs)

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    with outfile.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} pitch polygons to {outfile}")
    return results
