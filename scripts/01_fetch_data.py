from pathlib import Path
import sys

import pandas as pd
import fastf1


def root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_paths(r: Path):
    cache = r / "fastf1_cache"
    out = r / "data" / "raw"
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    return cache, out


def load_race(year: int, grand_prix: str, session_code: str, cache: Path):
    fastf1.Cache.enable_cache(str(cache))
    s = fastf1.get_session(year, grand_prix, session_code)  # "R" for race
    s.load()  # timing, laps, weather
    return s


def pit_table_from_laps(laps: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in [
        "Driver", "Team", "LapNumber", "PitInTime", "PitOutTime",
        "Stint", "Compound", "TyreLife", "FreshTyre", "Time", "LapTime"
    ] if c in laps.columns]

    pitlaps = laps.loc[laps.get("PitInTime").notna(), cols].copy()
    pitlaps.rename(columns={"LapNumber": "InLap",
                            "PitInTime": "InTime",
                            "PitOutTime": "OutTime"}, inplace=True)
    pitlaps.sort_values(["Driver", "InLap"], inplace=True)
    pitlaps["Stop"] = pitlaps.groupby("Driver").cumcount() + 1

    order = ["Driver", "Team", "Stop", "InLap", "InTime", "OutTime"]
    order += [c for c in pitlaps.columns if c not in order]
    return pitlaps[order]


def main():
    try:
        r = root_dir()
        cache, out = ensure_paths(r)

        session = load_race(2022, "Hungary", "R", cache)
        laps = session.laps

        laps_path = out / "laps_hun_2022.csv"
        pits_path = out / "pitstops_hun_2022.csv"

        laps.to_csv(laps_path, index=False)
        pit_table_from_laps(laps).to_csv(pits_path, index=False)

        drivers = ", ".join(sorted(laps["Driver"].unique()))
        print("ok: 2022 Hungarian GP (Race) loaded")
        print(f"ok: laps -> {laps_path.relative_to(r)} [{len(laps)} rows]")
        print(f"ok: pits -> {pits_path.relative_to(r)}")
        print(f"ok: drivers -> {drivers}")
        print(f"cache: {cache.relative_to(r)}")

    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
