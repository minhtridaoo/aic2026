"""keyframe thứ n -> frame_idx (giá trị cột 2 của CSV nộp bài).

    python -X utf8 _kf_sang_frame.py L26_V192 100 125
"""
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as C

VIDEO = sys.argv[1].upper()
DAU = int(sys.argv[2])
CUOI = int(sys.argv[3])

cfg = C.load_config("config_xem.yaml")
events, _, _, rep = C.nap_du_lieu(cfg, video_ids=[VIDEO])
theo_n = {}
for e in events:
    n = int(str(e.event_id).split("#")[1])
    theo_n[n] = e
print(f"{VIDEO}: {len(events)} keyframe")
for n in range(DAU, CUOI + 1):
    e = theo_n.get(n)
    if e:
        print(f"  kf {n:4d}  frame_idx {e.keyframe_idx:7d}  t={e.t_start:7.2f}s")
