"""Stage 1 data audit: what is in the feed, and what will bite us later.

Run:  python -m src.audit --feed data/raw/delhi_gtfs_trimmed --label "Delhi OTD (trimmed)"

The audit answers five questions, in this order, because each one decides
something in a later stage:

1. Shape and dtypes per table      -> how much data, and are IDs text or numbers?
2. Missing values per column       -> which features are even computable?
3. Duplicates (rows and keys)      -> Stage 2 must drop these or clusters double-count.
4. Coordinate ranges + envelope    -> Stage 2's bounding box and Stage 3's H3 grid.
5. Referential integrity + times   -> can we build delays, dwell times, headways at all?

Nothing is fixed here. The audit only *reports*, so Stage 2 can justify each
cleaning threshold against a measured number rather than a guess.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.io import OUTPUTS_DIR, load_gtfs, resolve_feed_dir

# ---------------------------------------------------------------------------
# Rough envelope for Delhi NCT, used ONLY to count obviously-outside points.
# It is deliberately generous: Delhi's urban area spills past the NCT boundary.
# Stage 2 replaces this with a bbox justified by the data itself.
# Flagged as approximate in the output so nobody mistakes it for a legal boundary.
# ---------------------------------------------------------------------------
CITY_ENVELOPE = {
    "lat_min": 28.30,
    "lat_max": 28.95,
    "lon_min": 76.75,
    "lon_max": 77.50,
    "is_approximate": True,
    "note": "Generous envelope around Delhi NCT; not an administrative boundary.",
}


def audit_table(df: pd.DataFrame, name: str) -> dict:
    """Shape, duplicates and per-column missingness for one table."""
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "n_columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_per_column": {c: int(df[c].isna().sum()) for c in df.columns},
        "nunique_per_column": {c: int(df[c].nunique(dropna=True)) for c in df.columns},
    }


def audit_stops(stops: pd.DataFrame) -> dict:
    """Coordinate sanity for the stop table (the spine of every spatial feature)."""
    lat = pd.to_numeric(stops["stop_lat"], errors="coerce")
    lon = pd.to_numeric(stops["stop_lon"], errors="coerce")

    in_envelope = (
        lat.between(CITY_ENVELOPE["lat_min"], CITY_ENVELOPE["lat_max"])
        & lon.between(CITY_ENVELOPE["lon_min"], CITY_ENVELOPE["lon_max"])
        & lat.notna()
        & lon.notna()
    )

    out = {
        "n_stops": int(len(stops)),
        "n_duplicate_stop_ids": int(stops["stop_id"].duplicated().sum()),
        "n_missing_coords": int((lat.isna() | lon.isna()).sum()),
        "lat_min": float(lat.min()) if lat.notna().any() else None,
        "lat_max": float(lat.max()) if lat.notna().any() else None,
        "lon_min": float(lon.min()) if lon.notna().any() else None,
        "lon_max": float(lon.max()) if lon.notna().any() else None,
        # Exact duplicate coordinates often mean two platforms of one stop.
        "n_duplicate_coordinate_pairs": int(
            pd.DataFrame({"lat": lat, "lon": lon}).duplicated().sum()
        ),
        "n_outside_envelope": int((~in_envelope).sum()),
        "pct_outside_envelope": round(100 * (~in_envelope).mean(), 3),
        "envelope": CITY_ENVELOPE,
    }
    return out


def audit_integrity(feed: dict[str, pd.DataFrame]) -> dict:
    """Referential integrity, key duplicates and whether times exist at all."""
    stops, routes, trips, st = feed["stops"], feed["routes"], feed["trips"], feed["stop_times"]

    st_stop_ids = set(st["stop_id"].dropna().astype(str))
    st_trip_ids = set(st["trip_id"].dropna().astype(str))
    known_stops = set(stops["stop_id"].dropna().astype(str))
    known_trips = set(trips["trip_id"].dropna().astype(str))
    known_routes = set(routes["route_id"].dropna().astype(str))

    # stop_sequence must increase along a trip; a violation means the stop order
    # cannot be trusted, which breaks every segment/speed feature in Stage 3.
    seq = pd.to_numeric(st["stop_sequence"], errors="coerce")
    st_seq = pd.DataFrame({"trip_id": st["trip_id"], "seq": seq}).dropna()
    monotonic_by_trip = st_seq.groupby("trip_id")["seq"].apply(
        lambda s: bool(s.is_monotonic_increasing)
    )

    times_present = {"arrival_time", "departure_time"}.issubset(st.columns)

    trips_per_route = trips.groupby("route_id").size()
    stops_per_trip = st.groupby("trip_id").size()

    return {
        # Stops referenced by stop_times that are not in stops.txt (and vice versa)
        "stop_ids_in_stop_times_missing_from_stops": int(len(st_stop_ids - known_stops)),
        "stop_ids_in_stops_never_used": int(len(known_stops - st_stop_ids)),
        "trip_ids_in_stop_times_missing_from_trips": int(len(st_trip_ids - known_trips)),
        "route_ids_in_trips_missing_from_routes": int(
            len(set(trips["route_id"].dropna().astype(str)) - known_routes)
        ),
        "duplicate_trip_stop_sequence_keys": int(
            st.duplicated(subset=["trip_id", "stop_sequence"]).sum()
        ),
        "trips_with_nonmonotonic_stop_sequence": int((~monotonic_by_trip).sum()),
        "n_stop_sequence_missing": int(seq.isna().sum()),
        "has_arrival_departure_times": bool(times_present),
        "trips_per_route": {
            "min": int(trips_per_route.min()),
            "median": float(trips_per_route.median()),
            "max": int(trips_per_route.max()),
            "routes_with_one_trip": int((trips_per_route == 1).sum()),
            "n_routes": int(len(trips_per_route)),
        },
        "stops_per_trip": {
            "min": int(stops_per_trip.min()),
            "median": float(stops_per_trip.median()),
            "max": int(stops_per_trip.max()),
        },
    }


def audit_calendar_and_agency(feed: dict[str, pd.DataFrame]) -> dict:
    """Service calendar span and the timezone we must convert GPS times into."""
    out: dict = {}
    if "agency" in feed:
        agency = feed["agency"]
        out["agencies"] = int(len(agency))
        out["agency_timezones"] = sorted(agency["agency_timezone"].dropna().unique().tolist())
        out["agency_names"] = sorted(agency["agency_name"].dropna().unique().tolist())
    if "calendar" in feed:
        cal = feed["calendar"]
        days = [c for c in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday") if c in cal]
        out["n_services"] = int(cal["service_id"].nunique())
        out["start_date_min"] = str(cal["start_date"].min()) if "start_date" in cal else None
        out["end_date_max"] = str(cal["end_date"].max()) if "end_date" in cal else None
        # pandas 3 no longer accepts a whole DataFrame in to_numeric, so convert per column.
        day_flags = cal[days].apply(pd.to_numeric, errors="coerce").fillna(0)
        out["service_days_any"] = {d: int(day_flags[d].sum()) for d in days}
        out["services_running_seven_days"] = (
            int(day_flags.eq(1).all(axis=1).sum()) if days else None
        )
    return out


def build_findings(report: dict) -> list[str]:
    """Turn measurements into plain-language warnings a student can act on."""
    findings: list[str] = []
    stops = report["stops"]
    integ = report["integrity"]

    if not integ["has_arrival_departure_times"]:
        findings.append(
            "stop_times has NO arrival_time/departure_time columns: this feed cannot "
            "give schedule delay, dwell time or headway. Get the full official zip."
        )
    if integ["trips_per_route"]["routes_with_one_trip"] > 0:
        findings.append(
            f"{integ['trips_per_route']['routes_with_one_trip']} of "
            f"{integ['trips_per_route']['n_routes']} routes have exactly one trip, so "
            "this is a trimmed sample, not a working timetable. Recurring temporal "
            "patterns cannot be learned from it."
        )
    if stops["n_duplicate_coordinate_pairs"] > 0:
        findings.append(
            f"{stops['n_duplicate_coordinate_pairs']} stops share an identical "
            "coordinate pair - likely platforms of one physical stop. Decide in "
            "Stage 3 whether to merge them into one node."
        )
    if stops["n_outside_envelope"] > 0:
        findings.append(
            f"{stops['n_outside_envelope']} stops ({stops['pct_outside_envelope']}%) "
            "fall outside the generous Delhi envelope - inspect them before using "
            "the envelope as a Stage 2 filter."
        )
    if integ["stop_ids_in_stop_times_missing_from_stops"] > 0:
        findings.append(
            f"{integ['stop_ids_in_stop_times_missing_from_stops']} stop_ids used in "
            "stop_times are absent from stops.txt - those rows must be dropped in Stage 2."
        )
    if integ["duplicate_trip_stop_sequence_keys"] > 0:
        findings.append(
            f"{integ['duplicate_trip_stop_sequence_keys']} duplicate (trip_id, "
            "stop_sequence) keys - de-duplicate before ordering stops."
        )
    if report.get("calendar", {}).get("agents_note"):
        findings.append(report["calendar"]["agents_note"])
    if not findings:
        findings.append("No blocking issues found by the automated checks.")
    return findings


def run_audit(feed_path: str | Path, label: str) -> dict:
    """Run every check and return one nested dict (also the JSON we save)."""
    feed_dir = resolve_feed_dir(feed_path)
    feed = load_gtfs(feed_dir)

    report = {
        "stage": 1,
        "label": label,
        "feed_path": str(feed_dir),
        "audited_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "synthetic": False,  # this is a real published feed; synthetic GPS is added in Stage 2/3
        "tables": {name: audit_table(df, name) for name, df in feed.items()},
        "stops": audit_stops(feed["stops"]),
        "integrity": audit_integrity(feed),
        "calendar": audit_calendar_and_agency(feed),
    }
    report["findings"] = build_findings(report)
    return report


def print_report(report: dict) -> None:
    """Compact console summary."""
    print(f"\nFleetSense Stage 1 audit - {report['label']}")
    print(f"path: {report['feed_path']}")
    print(f"run : {report['audited_at_utc']}   synthetic: {report['synthetic']}")

    rows = [
        {"table": name, "rows": t["rows"], "cols": t["n_columns"], "dup_rows": t["duplicate_rows"]}
        for name, t in report["tables"].items()
    ]
    print("\n-- tables --")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n-- coordinates --")
    s = report["stops"]
    print(
        f"stops={s['n_stops']}  lat=[{s['lat_min']:.5f}, {s['lat_max']:.5f}]  "
        f"lon=[{s['lon_min']:.5f}, {s['lon_max']:.5f}]  "
        f"outside_envelope={s['n_outside_envelope']} ({s['pct_outside_envelope']}%)"
    )

    print("\n-- integrity --")
    i = report["integrity"]
    print(f"arrival/departure times present : {i['has_arrival_departure_times']}")
    print(f"routes={i['trips_per_route']['n_routes']}  "
          f"trips/route median={i['trips_per_route']['median']:.1f}  "
          f"routes_with_one_trip={i['trips_per_route']['routes_with_one_trip']}")
    print(f"stops/trip median={i['stops_per_trip']['median']:.1f}  "
          f"duplicate (trip,seq)={i['duplicate_trip_stop_sequence_keys']}  "
          f"nonmonotonic trips={i['trips_with_nonmonotonic_stop_sequence']}")

    print("\n-- findings --")
    for n, f in enumerate(report["findings"], 1):
        print(f"{n}. {f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FleetSense Stage 1 GTFS data audit")
    parser.add_argument("--feed", default="data/raw/delhi_gtfs_trimmed",
                        help="Directory or .zip containing the GTFS files")
    parser.add_argument("--label", default="Delhi OTD (trimmed public copy)")
    parser.add_argument("--out", default=str(OUTPUTS_DIR / "data_audit.json"))
    args = parser.parse_args()

    report = run_audit(args.feed, args.label)
    print_report(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
