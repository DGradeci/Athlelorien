import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import fsspec
from pathlib import Path

from utils.day_loader import DayDataLoader
from utils.player_utils import PlayerNameMapper


def test_day_loader_load_day_1hz_mean(tmp_path: Path):
    day_dir = tmp_path / "2024-01-01"
    day_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "time": ["00:00:00", "00:00:00", "00:00:01", "00:00:01"],
            "player_name": ["id1", "id2", "id1", "id2"],
            "lat": [0.0, 0.0, 0.0, 0.0],
            "lon": [0.0, 0.0, 0.0, 0.0],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )

    for player, group in df.groupby("player_name"):
        path = day_dir / f"{player}.parquet"
        table = pa.Table.from_pandas(group)
        pq.write_table(table, path)

    fs = fsspec.filesystem("file")
    mapper = PlayerNameMapper(str(tmp_path / "player_map.json"))
    loader = DayDataLoader(fs, mapper)

    result = loader.load_day_1hz(str(day_dir), method="mean")

    assert set(result["player_name"]) == {"Abigail", "Ada"}
    assert result["value"].dtype == float
    assert result.shape[0] == 4
    assert result["timestamp"].notna().all()
