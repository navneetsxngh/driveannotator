# Runs the generated notebook end-to-end against a fake PhysicalAI-AV interface.
import json, os, sys, shutil, tempfile, re, time
from pathlib import Path
from types import SimpleNamespace

os.environ["MPLBACKEND"] = "Agg"
NB = Path(sys.argv[1])
cells = [c for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
src = {}
for c in cells:
    s = "".join(c["source"])
    m = re.match(r"# ==== Cell (\w+):", s)
    src[m.group(1)] = s
print("code cells:", list(src))

work = Path(tempfile.mkdtemp(prefix="nbtest_"))
os.chdir(work)
G = {"__name__": "__main__"}


def run(cell_id, patch=None):
    s = src[cell_id]
    if patch:
        for a, b in patch:
            assert a in s, f"patch target missing in cell {cell_id}: {a!r}"
            s = s.replace(a, b)
    print(f"\n######## running cell {cell_id} ########")
    exec(compile(s, f"<cell {cell_id}>", "exec"), G)


run("1")
run("2")
run("6"); run("8"); run("9"); run("10")  # pure defs
exec(compile((Path(__file__).parent / "dataset_label_tests.py").read_text(encoding="utf-8"),
             "dataset_label_tests.py", "exec"), G)  # label-engine tests (kept outside the notebook)

np, pd = G["np"], G["pd"]
from scipy.spatial.transform import Rotation, RigidTransform
from physical_ai_av.egomotion import EgomotionState

# ---------------------------------------------------------------- unit tests: schema helpers
raw = pd.DataFrame({
    "timestamp": [0.0, 0.1], "obj_id": ["a", "b"], "label_class": ["Automobile", "Person"],
    "center": [[1, 2, 3], [4, 5, 6]], "size": [[4, 2, 1.5], [0.5, 0.5, 1.8]],
    "orientation": [[0, 0, 0, 1], [0, 0, 0, 1]],
})
ex = G["expand_vector_columns"](raw)
cm = G["resolve_obstacle_columns"](ex)
assert cm["x"] == "center_x" and cm["length"] == "size_x" and cm["track_id"] == "obj_id" and cm["qw"] == "orientation_w", cm
try:
    G["resolve_obstacle_columns"](raw.drop(columns=["obj_id"]).pipe(G["expand_vector_columns"]))
    raise SystemExit("expected KeyError")
except KeyError as e:
    assert "track_id" in str(e)
assert G["detect_time_scale"](np.array([0.5, 10.0, 19.0]), (0, 20_000_000)) == 1e6
assert G["proportional_quota"]({"US": 90, "DE": 9, "JP": 1}, 10) == {"US": 8, "DE": 1, "JP": 1}
print("schema helper tests OK")

# ---------------------------------------------------------------- synthetic world
YAW, SPEED = 0.3, 8.0
def ego_interp():
    t = np.arange(0, 20_000_001, 10_000, dtype=np.int64)
    s = t / 1e6 * SPEED
    n = len(t)
    q = Rotation.from_euler("z", np.full((n, 1), YAW)).as_quat()
    df = pd.DataFrame({"timestamp": t, "x": s * np.cos(YAW), "y": s * np.sin(YAW), "z": 0.0,
                       "qx": q[:, 0], "qy": q[:, 1], "qz": q[:, 2], "qw": q[:, 3],
                       "vx": SPEED * np.cos(YAW), "vy": SPEED * np.sin(YAW), "vz": 0.0,
                       "ax": 0.0, "ay": 0.0, "az": 0.0, "curvature": 0.0})
    return EgomotionState.from_egomotion_df(df).create_interpolator(t)

EGO = ego_interp()

class Pinhole:
    def __init__(self, w=1920, h=1080, f=900.0):
        self.width, self.height, self.f = w, h, f
    def ray2pixel(self, ray):
        ray = np.asarray(ray, float)
        return np.stack([self.f * ray[..., 0] / ray[..., 2] + self.width / 2,
                         self.f * ray[..., 1] / ray[..., 2] + self.height / 2], axis=-1)
    def is_out_of_bounds(self, px):
        return (px[..., 0] < 0) | (px[..., 0] >= self.width) | (px[..., 1] < 0) | (px[..., 1] >= self.height)

CAM_YAW = {"camera_cross_left_120fov": 60, "camera_front_wide_120fov": 0, "camera_cross_right_120fov": -60,
           "camera_rear_left_70fov": 150, "camera_rear_right_70fov": -150, "camera_front_tele_30fov": 0}
R_BASE = Rotation.from_matrix([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])  # cam z->rig x, cam x->rig -y, cam y->rig -z
EXTR = SimpleNamespace(sensor_poses={k: RigidTransform.from_components(
    rotation=Rotation.from_euler("z", v, degrees=True) * R_BASE, translation=[1.5, 0, 1.5])
    for k, v in CAM_YAW.items()})
INTR = SimpleNamespace(camera_models={k: Pinhole() for k in CAM_YAW})

def world(t_s, x_rig, y_rig):
    """anchor coords of a point given in rig@t_s."""
    p = EGO(np.array([int(t_s * 1e6)])).pose[0]
    return p.apply([x_rig, y_rig, 0.8])

def obstacles_raw():
    """Mirrors the real PAI obstacle.offline schema: rig frame, per-track async microsecond timestamps."""
    rows = []
    ident = [0.0, 0.0, 0.0, 1.0]       # all boxes aligned with ego heading
    def add(t_us, src, tid, cls, rig_xyz, size):
        rows.append((np.uint64(t_us), src, tid, *rig_xyz, *size, *ident, cls, "rig", np.uint64(t_us)))
    SRC, SRC2 = "scene:obstacles:autolabels:v2", "scene:obstacles:other:v1"
    for k in range(0, 200):
        for j, tid in enumerate(["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "lead", "ped", "cutin", "pole"]):
            t_us = k * 100_000 + j * 7_000 + 49_000          # each track has its own clock offset
            if t_us > 19_990_000:
                continue
            t_s = t_us / 1e6
            pose = EGO(np.array([t_us])).pose[0]
            if tid.startswith("p") and tid not in ("ped", "pole"):
                jj = int(tid[1:]); xa = 30.0 * jj
                anchor = [xa * np.cos(YAW) - 4.0 * np.sin(YAW), xa * np.sin(YAW) + 4.0 * np.cos(YAW), 0.8]
                add(t_us, SRC, tid, "automobile", pose.inv().apply(anchor).tolist(), (4.5, 1.9, 1.5))
            elif tid == "lead":
                add(t_us, SRC, tid, "automobile", (8.0, 0.0, 0.8), (4.5, 1.9, 1.5))
                if k % 2 == 0:   # same object from a second, sparser source -> must not be double counted
                    add(t_us + 1_000, SRC2, "0", "automobile", (8.0, 0.0, 0.8), (4.5, 1.9, 1.5))
            elif tid == "ped":
                add(t_us, SRC, tid, "person", (1.0, -4.0, 0.8), (0.6, 0.6, 1.7))
            elif tid == "cutin":
                lat = 3.4 - max(0.0, min(t_s - 5.4, 1.0)) * 1.2
                add(t_us, SRC, tid, "automobile", (15.0, lat, 0.8), (4.5, 1.9, 1.5))
            else:
                add(t_us, SRC, tid, "protruding_object", (30.0, 6.0, 0.8), (0.3, 0.3, 3.0))
    cols = ["timestamp_us", "source", "track_id", "center_x", "center_y", "center_z", "size_x", "size_y", "size_z",
            "orientation_x", "orientation_y", "orientation_z", "orientation_w", "label_class",
            "reference_frame", "reference_frame_timestamp_us"]
    return pd.DataFrame(rows, columns=cols)

OBS_RAW = obstacles_raw()

class FakeReader:
    def __init__(self):
        self.timestamps = np.arange(0, 20_000_000, 33_333, dtype=np.int64)
    def decode_images_from_timestamps(self, req):
        idx = np.searchsorted(self.timestamps, req, side="right") - 1
        frames = np.zeros((len(req), 1080, 1920, 3), dtype=np.uint8)
        frames[...] = (np.arange(len(req)) * 7 % 200 + 40).astype(np.uint8)[:, None, None, None]
        return frames, self.timestamps[idx]
    def close(self):
        pass

class NS(SimpleNamespace):
    pass

class FakeAVDI:
    def __init__(self):
        cams = {n.upper(): n for n in CAM_YAW}
        self.features = SimpleNamespace(
            CAMERA=NS(**cams, ALL=set(CAM_YAW)),
            LABELS=NS(EGOMOTION="egomotion", OBSTACLE_OFFLINE="obstacle.offline", ALL={"egomotion", "obstacle.offline"}),
            CALIBRATION=NS(SENSOR_EXTRINSICS="sensor_extrinsics", CAMERA_INTRINSICS="camera_intrinsics",
                           ALL={"sensor_extrinsics", "camera_intrinsics"}),
        )
        self.features.ALL = set(CAM_YAW) | {"egomotion", "obstacle.offline", "sensor_extrinsics", "camera_intrinsics"}
        self.calls = []
    def get_clip_feature(self, clip_id, feature, maybe_stream=False):
        self.calls.append((clip_id, feature))
        if feature == "egomotion": return EGO
        if feature == "sensor_extrinsics": return EXTR
        if feature == "camera_intrinsics": return INTR
        if feature == "obstacle.offline":
            if clip_id == "clip-nolabels": raise KeyError("no obstacle")
            return {"obstacle": OBS_RAW.copy()}
        if feature in CAM_YAW: return FakeReader()
        raise KeyError(feature)

FAKE = FakeAVDI()
clips = ["clip-train-1", "clip-val-1", "clip-test-1", "clip-nolabels"]
G.update(
    avdi=FAKE, make_avdi=lambda: FAKE, HF_TOKEN="x", DATASET_REVISION="deadbeef",
    CLIP_INDEX=pd.DataFrame({"chunk": [1, 2, 3, 1], "split": ["train", "val", "test", "train"]}, index=clips),
    FEATURE_PRESENCE=pd.DataFrame({f: True for f in FAKE.features.ALL}, index=clips),
    DATA_COLLECTION=pd.DataFrame({"country": ["US", "DE", "US", "US"]}, index=clips),
    CAM_FEATURE={n: getattr(FAKE.features.CAMERA, a) for _, n, a in G["CFG"].CAMERAS},
    EGO_FEATURE="egomotion", EXTR_FEATURE="sensor_extrinsics", INTR_FEATURE="camera_intrinsics",
    OBS_FEATURE="obstacle.offline",
)
G["CFG"].N_CLIPS = {"train": 2, "val": 1, "test": 1}
G["CFG"].NUM_WORKERS = 2

run("3")
G["RUN_META"]["dataset_revision"] = "deadbeef"
run("6")
run("6b")
run("8b")
assert G["OBSTACLE_FRAME"] == "rig", G["OBSTACLE_FRAME"]
assert G["OBS_TIME_SCALE"] == 1.0, G["OBS_TIME_SCALE"]
assert G["OBS_SOURCE"] == "scene:obstacles:autolabels:v2", G["OBS_SOURCE"]

# fallback heuristic (files without reference_frame): anchor vs rig must be told apart
obs_rig, _ = G["normalize_obstacles"](OBS_RAW, G["OBS_COLMAP"], 1.0, G["CFG"].CLASS_MAP, source=G["OBS_SOURCE"])
anchor_pts = np.stack([EGO(np.array([t])).pose[0].apply(p) for t, p in
                       zip(obs_rig["frame_t_us"], obs_rig[["x", "y", "z"]].to_numpy())])
obs_anchor = obs_rig.copy(); obs_anchor[["x", "y", "z"]] = anchor_pts
assert G["detect_obstacle_frame"](obs_rig, EGO)[0] == "rig"
assert G["detect_obstacle_frame"](obs_anchor, EGO)[0] == "anchor"
print("frame detection anchor/rig OK")


run("9b", patch=[("VISUAL_CHECK_PASSED = False", "VISUAL_CHECK_PASSED = True")])
seen = G["seen"]
by = seen.assign(tid=seen["track"].str.split("|").str[-1]).set_index("tid")
assert by.index.is_unique, "duplicate tracks -> source filter failed"
assert by.at["lead", "vis_front_wide"] and not by.at["lead", "vis_rear_left"], "lead car must be front-only"
assert by.at["ped", "vis_cross_right"] and not by.at["ped", "vis_cross_left"], "right pedestrian must be in cross_right"
assert by.at["lead", "x"] > 7.5 and abs(by.at["lead", "y"]) < 0.2, by.loc["lead", ["x", "y"]]
print("geometry/visibility OK")

run("11")
run("11b")
run("12")
run("13")
recs = G["records"]
assert {k: len(v) for k, v in recs.items()} == {"train": 5, "val": 5, "test": 5}, {k: len(v) for k, v in recs.items()}
by_t = {r["t0_us"]: r["target"] for r in recs["train"]}
print(json.dumps(by_t[6_000_000]))
types6 = [e["type"] for e in by_t[6_000_000]["events"]]
assert "vehicle_cut_in" in types6, types6
assert "lead_vehicle_close" in types6 and "vru_near_ego" in types6, types6
assert "vehicle_cut_in" not in [e["type"] for e in by_t[14_000_000]["events"]]
assert by_t[6_000_000]["cameras"]["cross_right"] == ["pedestrian"]
assert "other_vehicle" not in by_t[6_000_000]["road_users"]  # pole ignored
meta_skip = json.loads((G["DIRS"]["clips"] / "clip-nolabels.json").read_text())
assert meta_skip["status"] == "skipped", meta_skip
r0 = recs["train"][0]
assert [f["camera_id"] for f in r0["frames"]] == sorted(f["camera_id"] for f in r0["frames"])
img = G["read_frame"](r0, r0["frames"][0])
assert img.size[0] * img.size[1] <= G["CFG"].STORE_MAX_PIXELS and img.size[0] * img.size[1] >= G["CFG"].MODEL_MAX_PIXELS, img.size
print("stored image size", img.size)

run("14")

# resume: running build again must not reprocess
n_calls = len(FAKE.calls)
run("12")
assert len(FAKE.calls) == n_calls, "resume reprocessed clips"
print("resume OK")

# cross-process lock: fresh lock -> busy; stale lock -> taken over and built
row = {"clip_id": "clip-locktest", "split": "train", "chunk": 9, "country": "US"}
lock = G["DIRS"]["clips"] / "clip-locktest.lock"
lock.write_text("pid=1")
assert G["process_clip"](row)["status"] == "busy"
old = time.time() - 3600
os.utime(lock, (old, old))
assert G["process_clip"](row)["status"] == "ok" and not lock.exists()
assert not list(G["DIRS"]["frames"].glob("*.tmp")) and not list(Path(G["CFG"].LOCAL_WORK).glob("*.zip"))
(G["DIRS"]["clips"] / "clip-locktest.json").unlink()
(G["DIRS"]["frames"] / "clip-locktest.zip").unlink()
print("lock + temp cleanup OK")

# fingerprint lock
G["CFG"].PATH_HALF_WIDTH_M = 2.5
try:
    run("3"); raise SystemExit("fingerprint lock failed")
except RuntimeError as e:
    assert "fingerprint" in str(e)
G["CFG"].PATH_HALF_WIDTH_M = 1.8
print("fingerprint lock OK")

# review + gold
rv = G["DIRS"]["review"] / "test_reviews.jsonl"
t0, t1 = recs["test"][0], recs["test"][1]
edited = dict(t1["target"]); edited["lighting"] = "night"
rv.write_text("\n".join(json.dumps(x) for x in [
    {"sample_id": t0["sample_id"], "decision": "approved", "label": t0["target"], "auto_label": t0["target"], "reviewer_id": "r", "utc": "u"},
    {"sample_id": t1["sample_id"], "decision": "edited", "label": edited, "auto_label": t1["target"], "reviewer_id": "r", "utc": "u"},
    {"sample_id": recs["test"][2]["sample_id"], "decision": "rejected", "label": None, "auto_label": None, "reviewer_id": "r", "utc": "u"},
]) + "\n")
G["load_reviews"] = lambda: {json.loads(l)["sample_id"]: json.loads(l) for l in rv.read_text().splitlines()}
run("16")
gold = [json.loads(l) for l in (G["DIRS"]["manifests"] / "test_gold.jsonl").read_text().splitlines()]
assert len(gold) == 2 and gold[1]["target"]["lighting"] == "night"
run("17")
print("\n\nALL NOTEBOOK TESTS PASSED in", work)
