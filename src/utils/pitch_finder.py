"""
High-level interface for fetching football pitch polygons from OSM via Overpass.
"""

from typing import Any, Dict, List, Optional, Tuple

from .geo import clean_polygon_coords, haversine
from .osm_client import OverpassClient


class PitchFinder:
    """
    Helper to query the nearest football / soccer pitch polygon to a given point.
    """

    def __init__(self, client: OverpassClient) -> None:
        self.client = client

    @staticmethod
    def _build_query(lat: float, lon: float, radius: int) -> str:
        """
        Build an Overpass QL query for pitches within `radius` metres of (lat, lon).
        """
        return f"""
        [out:json][timeout:25];
        (
          way["leisure"="pitch"]["sport"~"football|soccer"](around:{radius},{lat},{lon});
          way["leisure"="pitch"](around:{radius},{lat},{lon});
        );
        out body;
        >;
        out skel qt;
        """

    def get_nearest_pitch_polygon(
        self, lat: float, lon: float, radius: int = 250
    ) -> Optional[Dict[str, Any]]:
        """
        Query OSM via Overpass and return the nearest pitch polygon to (lat, lon).

        Returns
        -------
        dict or None
            Example:

            {
              "id": way_id,
              "tags": {...},
              "coords": [(lat1, lon1), (lat2, lon2), ...]  # cleaned unique corners
            }
        """
        query = self._build_query(lat, lon, radius)
        data = self.client.run(query)

        node_lookup: Dict[int, Tuple[float, float]] = {}
        ways: List[Dict[str, Any]] = []

        for el in data.get("elements", []):
            if el["type"] == "node":
                node_lookup[el["id"]] = (el["lat"], el["lon"])
            elif el["type"] == "way":
                ways.append(el)

        polygons: List[Dict[str, Any]] = []

        for way in ways:
            raw_coords = [
                node_lookup[nid] for nid in way.get("nodes", []) if nid in node_lookup
            ]

            if len(raw_coords) >= 3:
                coords_clean = clean_polygon_coords(raw_coords)
                if len(coords_clean) >= 3:
                    polygons.append(
                        {
                            "id": way["id"],
                            "tags": way.get("tags", {}),
                            "coords": coords_clean,
                        }
                    )

        if not polygons:
            return None

        def centroid(poly_coords: List[Tuple[float, float]]) -> Tuple[float, float]:
            lats = [c[0] for c in poly_coords]
            lons = [c[1] for c in poly_coords]
            return (sum(lats) / len(lats), sum(lons) / len(lons))

        best: Optional[Dict[str, Any]] = None
        best_dist = float("inf")

        for poly in polygons:
            clat, clon = centroid(poly["coords"])
            d = haversine(clat, clon, lat, lon)
            if d < best_dist:
                best_dist = d
                best = poly

        return best
