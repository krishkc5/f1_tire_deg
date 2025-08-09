from pathlib import Path
import sys

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def read_laps() -> pd.DataFrame:
    p = RAW / "laps_hun_2022.csv"
    if not p.exists():
        raise FileNotFoundError("raw laps not found; run scripts/01_fetch_data.py first")
    df = pd.read_csv(p, low_memory=False)
    # ensure timedeltas are parsed if present
    for col in ("LapTime", "Time", "PitInTime", "PitOutTime"):
        if col in df.columns:
            df[col] = pd.to_timedelta(df[col], errors="coerce")
    return df


def is_green(track_status: pd.Series) -> pd.Series:
    # Keep laps with TrackStatus == "1" (green). If missing, assume green.
    s = track_status.astype(str).str.strip()
    return (track_status.isna()) | (s == "1")


def drop_in_out(df: pd.DataFrame) -> pd.DataFrame:
    # Out-lap: has PitOutTime on this lap; In-lap: has PitInTime on this lap.
    return df.loc[df["PitOutTime"].isna() & df["PitInTime"].isna()].copy()


def per_driver_pace_filter(df: pd.DataFrame) -> pd.DataFrame:
    # Remove obvious slow laps per driver using a robust upper fence (median + 1.5*IQR).
    x = df.copy()
    x["LapTime_s"] = x["LapTime"].dt.total_seconds()
    keep_idx = []
    for drv, g in x.groupby("Driver", dropna=False):
        gt = g.dropna(subset=["LapTime_s"])
        if gt.empty:
            continue
        q1 = np.percentile(gt["LapTime_s"], 25)
        q3 = np.percentile(gt["LapTime_s"], 75)
        fence = q3 + 1.5 * (q3 - q1)
        keep_idx.append(gt.loc[gt["LapTime_s"] <= fence].index)
    keep_idx = x.index.intersection(pd.Index(np.concatenate(keep_idx))) if keep_idx else x.index
    return x.loc[keep_idx].copy()


def clean_laps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # basic columns we rely on
    need = ["Driver", "Team", "LapNumber", "LapTime", "Compound", "Stint", "TyreLife", "PitInTime", "PitOutTime", "TrackStatus"]
    for c in need:
        if c not in df.columns:
            df[c] = pd.NA

    # filter
    df = df.loc[df["LapTime"].notna()]                 # valid laptime
    df = df.loc[is_green(df["TrackStatus"])]           # green only
    df = drop_in_out(df)                                # not in/out laps
    df = per_driver_pace_filter(df)                     # remove big slow outliers
    # tidy types
    if df["TyreLife"].dtype.kind not in "fi":
        df["TyreLife"] = pd.to_numeric(df["TyreLife"], errors="coerce")
    df["LapTime_s"] = df["LapTime"].dt.total_seconds()
    return df


def stint_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["Driver", "Stint", "Compound"], dropna=False)
    out = g.agg(
        laps=("LapNumber", "count"),
        start_lap=("LapNumber", "min"),
        end_lap=("LapNumber", "max"),
        mean_pace_s=("LapTime_s", "mean"),
        median_pace_s=("LapTime_s", "median"),
    ).reset_index()
    return out.sort_values(["Driver", "Stint"])


def simple_deg_fit(df: pd.DataFrame) -> pd.DataFrame:
    # Linear model per compound: LapTime_s ~ a + b * TyreLife
    # (pooled over drivers; we’ll refine later if needed)
    z = df.dropna(subset=["LapTime_s", "TyreLife", "Compound"]).copy()
    results = []
    for comp, g in z.groupby("Compound"):
        if len(g) < 20 or g["TyreLife"].nunique() < 3:
            continue
        x = g["TyreLife"].to_numpy()
        y = g["LapTime_s"].to_numpy()
        b, a = np.polyfit(x, y, 1)  # slope b (sec/lap-age), intercept a
        # store a few diagnostics
        yhat = a + b * x
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        results.append({"Compound": comp, "intercept_s": float(a), "slope_s_per_lap": float(b), "n": int(len(g)), "rmse_s": rmse})
    return pd.DataFrame(results).sort_values("Compound")


def main():
    try:
        OUT.mkdir(parents=True, exist_ok=True)

        raw = read_laps()
        cl = clean_laps(raw)
        st = stint_summary(cl)
        deg = simple_deg_fit(cl)

        cl.to_csv(OUT / "clean_laps_hun_2022.csv", index=False)
        st.to_csv(OUT / "stint_summary_hun_2022.csv", index=False)
        deg.to_csv(OUT / "deg_compound_fit_hun_2022.csv", index=False)

        print(f"ok: clean laps -> {OUT.relative_to(ROOT)}/clean_laps_hun_2022.csv [{len(cl)} rows]")
        print(f"ok: stint summary -> {OUT.relative_to(ROOT)}/stint_summary_hun_2022.csv [{len(st)} rows]")
        print(f"ok: degradation fit -> {OUT.relative_to(ROOT)}/deg_compound_fit_hun_2022.csv")
        if not deg.empty:
            for _, r in deg.iterrows():
                print(f"  {r['Compound']}: slope {r['slope_s_per_lap']:.4f} s/lap-age, rmse {r['rmse_s']:.3f}, n={int(r['n'])}")

    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
