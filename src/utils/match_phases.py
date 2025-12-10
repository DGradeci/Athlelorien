import pandas as pd


def label_match_phases(df, kickoff_time, halftime_break_min=15):
    df = df.copy()

    kickoff = kickoff_time
    end_1H = kickoff + pd.Timedelta(minutes=45)
    start_2H = end_1H + pd.Timedelta(minutes=halftime_break_min)
    end_2H = start_2H + pd.Timedelta(minutes=45)

    t = df["timestamp"]

    df["match_phase"] = "post"
    df.loc[t < kickoff, "match_phase"] = "pre"
    df.loc[(t >= kickoff) & (t < end_1H), "match_phase"] = "1H"
    df.loc[(t >= end_1H) & (t < start_2H), "match_phase"] = "HT"
    df.loc[(t >= start_2H) & (t < end_2H), "match_phase"] = "2H"

    return df
