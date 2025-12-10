"""
Geographic helper functions: haversine distance and polygon cleaning.
"""

from math import atan2, cos, radians, sin
from typing import List, Tuple


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two points on Earth in metres.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of first point in decimal degrees.
    lat2, lon2 : float
        Latitude and longitude of second point in decimal degrees.
    """
    R = 6371000.0  # mean Earth radius in metres

    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    a = sin(dphi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2.0) ** 2
    c = 2.0 * atan2(a**0.5, (1.0 - a) ** 0.5)

    return R * c


def clean_polygon_coords(
    coords: List[Tuple[float, float]], tol: float = 1e-9
) -> List[Tuple[float, float]]:
    """
    Clean a polygon coordinate list:

    - Remove consecutive duplicate coordinates
    - Drop the closing node if it is (almost) identical to the first

    Converts [A, B, C, D, A] into [A, B, C, D].

    Parameters
    ----------
    coords : list of (lat, lon)
    tol : float
        Tolerance for treating two coordinates as identical.
    """
    if not coords:
        return coords

    cleaned = [coords[0]]
    for lat, lon in coords[1:]:
        lat0, lon0 = cleaned[-1]
        if abs(lat - lat0) > tol or abs(lon - lon0) > tol:
            cleaned.append((lat, lon))

    # Drop last if same as first
    lat_first, lon_first = cleaned[0]
    lat_last, lon_last = cleaned[-1]
    if abs(lat_first - lat_last) <= tol and abs(lon_first - lon_last) <= tol:
        cleaned = cleaned[:-1]

    return cleaned
