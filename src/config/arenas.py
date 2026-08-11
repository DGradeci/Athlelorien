"""
Hard-coded coordinates for Toppserien arenas.

These are approximate centres of the main match pitch / stadium.
"""

from typing import Dict, Tuple

ARENAS: Dict[str, Tuple[float, float]] = {
    "Arna Idrettspark": (60.426389, 5.471667),
    "Avaldsnes Idrettssenter": (59.356667, 5.264167),
    "Brann Stadion": (60.366800, 5.358400),
    "KFUM-Arena": (59.889749, 10.783302),
    "Klepp Stadion": (58.775556, 5.633333),
    "Lerkendal stadion": (63.412281, 10.404834),
    "Sofiemyr stadion": (59.800556, 10.818611),
    "LSK-hallen": (59.962222, 11.068889),
    "Kringsjå kunstgress": (59.965556, 10.726944),
    "Nadderud stadion": (59.921366, 10.581050),
    "Røa kunstgress": (59.941809, 10.634959),
    "Koteng Arena": (63.445278, 10.451667),
    "Stemmemyren": (60.425833, 5.305556),
    "Intility Arena": (59.917778, 10.806667),
}


ARENA_ALIASES: Dict[str, str] = {
    "Avaldsnes 1": "Avaldsnes Idrettssenter",
    "Brann stadion": "Brann Stadion",
    "Klepp stadion": "Klepp Stadion",
    "Kringsjå kunstgress": "Kringsjå kunstgress",
    "Røa kunstgress": "Røa kunstgress",
    "Stemmemyren kunstgress": "Stemmemyren",
}
