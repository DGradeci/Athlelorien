"""
Thin client for the Overpass (OpenStreetMap) API with retry logic.
"""

import time
from typing import Any, Dict, List, Optional

import requests


# Overpass API endpoints – you can add/remove as needed
OVERPASS_URLS: List[str] = [
    "https://overpass-api.de/api/interpreter",
    # "https://overpass.kumi.systems/api/interpreter",
    # "https://overpass.openstreetmap.fr/api/interpreter",
]

# Be nice and identify your app + contact
HEADERS: Dict[str, str] = {
    "User-Agent": "toppserien-gps-analysis/0.1 (contact: your-email@example.com)"
}


class OverpassClient:
    """
    Simple Overpass API client with retries and exponential backoff.
    """

    def __init__(self, max_retries: int = 5, base_sleep: float = 2.0) -> None:
        self.max_retries = max_retries
        self.base_sleep = base_sleep

    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute an Overpass QL query and return decoded JSON.

        Retries on 429 and 5xx responses, and on network errors, across
        all configured OVERPASS_URLS.
        """
        last_error: Optional[Exception] = None

        for url in OVERPASS_URLS:
            sleep = self.base_sleep

            for attempt in range(self.max_retries):
                try:
                    resp = requests.post(
                        url,
                        data={"data": query},
                        headers=HEADERS,
                        timeout=60,
                    )

                    if resp.status_code == 429 or 500 <= resp.status_code < 600:
                        # Rate limit or transient server error -> backoff & retry
                        print(
                            f"Overpass {url} returned {resp.status_code}. "
                            f"Retrying in {sleep:.1f}s "
                            f"(attempt {attempt + 1}/{self.max_retries})..."
                        )
                        time.sleep(sleep)
                        sleep *= 2
                        continue

                    resp.raise_for_status()
                    return resp.json()

                except requests.exceptions.RequestException as e:
                    last_error = e
                    print(
                        f"Error calling {url}: {e}. "
                        f"Retrying in {sleep:.1f}s "
                        f"(attempt {attempt + 1}/{self.max_retries})..."
                    )
                    time.sleep(sleep)
                    sleep *= 2

            print(f"Giving up on {url} after {self.max_retries} attempts.")

        raise RuntimeError(f"All Overpass endpoints failed. Last error: {last_error}")
