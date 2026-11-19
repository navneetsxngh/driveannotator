# Label-engine tests for 01_build_dataset.ipynb (moved out of the notebook).
# Executed by dev/test_notebook.py inside the notebook's namespace.
# ==== Cell 10b: unit tests (synthetic scenes, no network) ====
# [TESTS]
def _scene(rows, t_us=300_000, cameras=CAMERA_NAMES):
    """observe()-like frame; also usable as previous_positions() output (has t_us)."""
    df = pd.DataFrame(rows, columns=["track", "cls", "x", "y", "vis_cam", "h"])
    df["z"] = 0.0
    df["range"] = np.hypot(df["x"].astype(float), df["y"].astype(float))
    df["t_label_us"] = np.full(len(df), t_us, dtype=np.int64)
    df["t_us"] = np.full(len(df), t_us, dtype=np.int64)
    for cam in cameras:
        df[f"vis_{cam}"] = (df["vis_cam"] == cam).astype(bool)
        df[f"h_{cam}"] = np.where(df["vis_cam"] == cam, df["h"], 0.0)
    df["visible"] = df["vis_cam"].notna().astype(bool)
    return df.drop(columns=["vis_cam", "h"])

def run_label_tests():
    c = CFG
    assert [bucket(n, c.COUNT_BUCKETS) for n in (1, 2, 3, 4, 7, 8, 50)] == ["1", "2-3", "2-3", "4-7", "4-7", "8+", "8+"]
    assert [bucket(d, c.DIST_BUCKETS) for d in (0.5, 5, 7, 15, 39, 41)] == ["<5m", "<5m", "5-10m", "10-20m", "20-40m", "40m+"]
    # sun elevation: Berlin-ish noon in June is high, midnight in December is far below horizon
    assert sun_elevation_deg(51.2, 10.4, 15, 6, 13) > 55
    assert sun_elevation_deg(51.2, 10.4, 15, 12, 0) < -50
    assert scene_lighting(sun_elevation_deg(51.2, 10.4, 15, 12, 17), 0.3, c) == "night"   # Dec 17:00 Germany, bright image
    assert scene_lighting(sun_elevation_deg(51.2, 10.4, 15, 12, 16), 0.3, c) == "twilight" # sunset ~16:15
    assert scene_lighting(sun_elevation_deg(51.2, 10.4, 15, 11, 10), 0.2, c) == "daylight" # overcast morning
    assert scene_lighting(7.9, 0.09, c) == "night", "low sun + pitch-dark frame -> night"
    assert scene_lighting(40.0, 0.09, c) == "daylight", "high sun: dark frame (tunnel/underpass) stays daylight"
    assert scene_lighting(None, 0.5, c) == "daylight" and scene_lighting(None, 0.05, c) == "night"
    assert scene_lighting(None, 0.2, c) == "twilight"
    assert clip_sun_elevation("Atlantis", 6, 12, c) is None and clip_sun_elevation("Germany", None, 12, c) is None

    # objects_at: tracks are asynchronous -> one row per track, nearest within tolerance
    obs = pd.DataFrame({"t_us": [0, 3_000, 100_000, 103_000, 200_000, 900_000],
                        "track": ["a", "b", "a", "b", "a", "a"], "x": [1, 2, 3, 4, 5, 6]})
    got = objects_at(obs, 101_000, 60_000)
    assert got["track"].tolist() == ["a", "b"] and got["t_us"].tolist() == [100_000, 103_000], got
    assert objects_at(obs, 2_000_000, 60_000) is None, "outside coverage -> None"
    empty = objects_at(obs, 550_000, 60_000)
    assert empty is not None and empty.empty, "covered but nothing near -> empty frame (empty scene)"

    # empty scene
    t = build_target(_scene([]), None, 0.0, "daylight", c, CAMERA_NAMES)
    assert t == {"lighting": "daylight", "road_users": {}, "cameras": {}, "events": []}, t
    assert validate_target(t) == []

    # pedestrian ahead in path at 12 m, car in path at 8 m, invisible truck
    now = _scene([("p1", "pedestrian", 12.0, 0.5, "front_wide", 80),
                  ("c1", "car", 8.0, -0.3, "front_wide", 200),
                  ("t1", "truck", 20.0, 0.0, None, 0)])
    t = build_target(now, None, 0.0, "daylight", c, CAMERA_NAMES)
    types = [e["type"] for e in t["events"]]
    assert types == ["pedestrian_in_path", "lead_vehicle_close"], types
    assert t["road_users"] == {"car": "1", "pedestrian": "1"}, "invisible truck must not be counted"
    assert t["cameras"] == {"front_wide": ["car", "pedestrian"]}
    assert t["events"][0] == {"type": "pedestrian_in_path", "camera": "front_wide", "distance": "10-20m"}
    assert validate_target(t) == []

    # curved road: ped at x=20,y=4 is IN path for left curvature k=0.02 (centre y=4), OUT for straight
    now = _scene([("p1", "pedestrian", 20.0, 4.0, "front_wide", 50)])
    assert [e["type"] for e in compute_events(now, None, 0.02, c, CAMERA_NAMES)] == ["pedestrian_in_path"]
    assert compute_events(now, None, 0.0, c, CAMERA_NAMES) == []

    # cut-in: car moves from y=2.9 (t=0) to y=2.6 (t=0.3 s) -> 1.0 m/s toward corridor
    now = _scene([("c1", "car", 15.0, 2.6, "front_wide", 90)], t_us=300_000)
    prev = _scene([("c1", "car", 15.0, 2.9, "front_wide", 90)], t_us=0)
    assert [e["type"] for e in compute_events(now, prev, 0.0, c, CAMERA_NAMES)] == ["vehicle_cut_in"]
    # same car moving AWAY (2.6 -> 2.9) -> no cut-in
    away_now = _scene([("c1", "car", 15.0, 2.9, "front_wide", 90)], t_us=300_000)
    away_prev = _scene([("c1", "car", 15.0, 2.6, "front_wide", 90)], t_us=0)
    assert compute_events(away_now, away_prev, 0.0, c, CAMERA_NAMES) == []
    # slow drift (0.1 m/s) -> no cut-in
    prev_slow = _scene([("c1", "car", 15.0, 2.63, "front_wide", 90)], t_us=0)
    assert compute_events(now, prev_slow, 0.0, c, CAMERA_NAMES) == []
    # per-track time gap too small (50 ms) -> speed not trusted -> no cut-in
    prev_close = _scene([("c1", "car", 15.0, 2.9, "front_wide", 90)], t_us=250_000)
    assert compute_events(now, prev_close, 0.0, c, CAMERA_NAMES) == []

    # VRU beside ego, seen in cross_right -> vru_near_ego, best camera = largest box
    now = _scene([("p1", "pedestrian", 1.0, -3.0, "cross_right", 300)])
    ev = compute_events(now, None, 0.0, c, CAMERA_NAMES)
    assert ev == [{"type": "vru_near_ego", "camera": "cross_right", "distance": "<5m"}], ev

    # counts bucket by unique tracks
    now = _scene([(f"c{i}", "car", 30.0 + i, 5.0, "front_wide", 40) for i in range(5)])
    assert build_target(now, None, 0.0, "daylight", c, CAMERA_NAMES)["road_users"] == {"car": "4-7"}

    # validator catches errors
    bad = {"lighting": "sunny", "road_users": {"car": "13"}, "cameras": {"front_wide": ["bus"]},
           "events": [{"type": "vru_near_ego", "camera": "roof", "distance": "1m"}]}
    errs = validate_target(bad)
    assert len(errs) >= 5, errs
    assert parse_and_validate("{not json")[1][0].startswith("invalid JSON")
    assert validate_target({"road_users": {}, "lighting": "night", "cameras": {}, "events": []}), "key order enforced"
    return "all label-engine tests passed"

print(run_label_tests())
print("\n✅ Cell 10 OK")
