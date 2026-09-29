#!/usr/bin/env python
"""Kiểm `event_id -> ảnh` trên TÊN FILE THẬT của corpus (việc 2.3).

    python -X utf8 kiem_resolver_anh.py                    # dùng qa.image_roots trong config
    python -X utf8 kiem_resolver_anh.py --goc "D:\\...\\keyframes" --so 20

VÌ SAO CẦN
----------
Tầng QA VLM chỉ chạy được nếu `EventImageResolver` dịch được `L22_V006#00209`
thành đúng file ảnh. Ảnh dev set do ta tự đặt tên (`001_L22_V006_n209.jpg`), còn
corpus của BTC đặt tên theo quy ước khác. Chưa ai đối chiếu hai quy ước đó, nên
ngày thi thiếu ảnh sẽ kéo QA từ 0.1800 về 0.1067 (sau bản sửa 1.1 thì không sập
nữa, nhưng vẫn mất điểm).

BA PHÉP KIỂM
------------
  1. resolve N event_id ngẫu nhiên từ map-keyframes THẬT -> phải ra file có thật
  2. đếm: số ảnh trong thư mục video == số keyframe trong map-keyframes
  3. **so byte** ảnh dev set với ảnh corpus của cùng `(video, n)` — đây là phép
     kiểm mạnh nhất: giống byte nghĩa là hai quy ước đánh số TRÙNG NHAU, chứ
     không chỉ "có file ở đúng đường dẫn".

Phép 3 chỉ chạy được với các video có mặt ở cả hai bên; corpus tải về một phần
thì nó tự bỏ qua phần thiếu và nói rõ đã kiểm được bao nhiêu.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as cau_hinh
from aic2026.data_plane.image_resolver import (EventImageResolver,
                                               ImageResolutionError,
                                               parse_event_id)

TEN_ANH_DEV = re.compile(
    r"^\d+_(?P<video>[A-Za-z0-9_-]+)_n(?P<n>\d+)\.(?:jpe?g|png|webp)$", re.I)


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--goc", action="append", default=[],
                   help="thư mục keyframe corpus; lặp được. Không có thì lấy "
                        "`qa.image_roots` trong config")
    p.add_argument("--so", type=int, default=20, help="số event_id lấy mẫu")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def _bam(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    args = _args()
    config = cau_hinh.load_config(args.config)
    roots = [Path(x) for x in (args.goc or (config.get("qa") or {}).get(
        "image_roots") or [])]
    roots = [r for r in roots if r.is_dir()]
    if not roots:
        print("❌ Chưa có thư mục ảnh nào để kiểm.")
        print("   Đặt `qa.image_roots` trong config.yaml, hoặc truyền --goc.")
        print("   Ngày thi trên Kaggle: trỏ vào thư mục mount keyframe (việc 2.1).")
        sys.exit(1)
    print("Thư mục ảnh:")
    for r in roots:
        con = sorted(x.name for x in r.iterdir() if x.is_dir())
        print(f"  {r}  ({len(con)} thư mục video, vd {con[:3]})")

    # Video nào thật sự có ảnh ở đây -> chỉ nạp map-keyframes của chúng.
    co_anh = {}
    for r in roots:
        for d in r.iterdir():
            if d.is_dir():
                co_anh.setdefault(d.name.upper(), d)
    if not co_anh:
        print("❌ Không có thư mục con nào theo tên video (mong dạng L21_V001/).")
        sys.exit(1)

    events, _, _, rep = cau_hinh.nap_du_lieu(config,
                                             video_ids=sorted(co_anh))
    print(f"\nmap-keyframes: {rep.events:,d} event / {rep.videos_da_nap} video "
          f"(giao với thư mục ảnh)")
    if not events:
        print("❌ Không nạp được event nào cho các video có ảnh.")
        sys.exit(1)

    resolver = EventImageResolver(roots=roots)  # KHÔNG dùng dev_image_dir: đang
    # kiểm đúng quy ước của corpus, không phải quy ước ảnh dev.

    # --- phép 1: resolve mẫu ngẫu nhiên ------------------------------------- #
    random.seed(args.seed)
    mau = random.sample(events, min(args.so, len(events)))
    ok, loi = 0, []
    for e in mau:
        try:
            path = resolver.resolve(e.event_id)
            ok += 1 if path.is_file() else 0
        except (ImageResolutionError, ValueError) as exc:
            loi.append((e.event_id, str(exc).splitlines()[0]))
    print(f"\n1) resolve {len(mau)} event_id ngẫu nhiên: {ok}/{len(mau)} ra file thật")
    for eid, msg in loi[:5]:
        print(f"     ❌ {eid}: {msg}")

    # --- phép 2: đếm ảnh vs đếm keyframe ------------------------------------ #
    theo_video = {}
    for e in events:
        theo_video.setdefault(e.video_id.upper(), []).append(e)
    lech = []
    for vid, ds in sorted(theo_video.items()):
        thu_muc = co_anh.get(vid)
        n_anh = sum(1 for x in thu_muc.iterdir()
                    if x.suffix.lower().lstrip(".") in EventImageResolver.EXTENSIONS)
        if n_anh != len(ds):
            lech.append((vid, len(ds), n_anh))
    print(f"2) đếm ảnh == đếm keyframe: {len(theo_video) - len(lech)}/"
          f"{len(theo_video)} video khớp")
    for vid, n_kf, n_anh in lech[:5]:
        print(f"     ❌ {vid}: {n_kf} keyframe nhưng {n_anh} ảnh")

    # --- phép 3: so byte với ảnh dev set ------------------------------------ #
    dev_dir = (config.get("qa") or {}).get("dev_image_dir")
    giong, khac, bo_qua = 0, [], 0
    if dev_dir and Path(dev_dir).is_dir():
        for anh in sorted(Path(dev_dir).iterdir()):
            m = TEN_ANH_DEV.fullmatch(anh.name)
            if not m:
                continue
            video, n = m.group("video").upper(), int(m.group("n"))
            if video not in co_anh:
                bo_qua += 1
                continue
            try:
                cua_corpus = resolver.resolve(f"{video}#{n:05d}")
            except ImageResolutionError:
                khac.append((anh.name, "corpus không có file tương ứng"))
                continue
            if _bam(anh) == _bam(cua_corpus):
                giong += 1
            else:
                khac.append((anh.name, f"khác byte với {cua_corpus.name}"))
        print(f"3) so byte ảnh dev vs corpus: {giong} giống · {len(khac)} khác "
              f"· {bo_qua} bỏ qua (video chưa tải)")
        for ten, vi_sao in khac[:5]:
            print(f"     ❌ {ten}: {vi_sao}")
        if giong and not khac:
            print("     -> hai quy ước đánh số TRÙNG NHAU: `n` trong event_id ="
                  " số thứ tự file ảnh của corpus")
    else:
        print("3) so byte ảnh dev vs corpus: bỏ qua (không có `qa.dev_image_dir`)")

    hong = bool(loi) or bool(lech) or bool(khac)
    print("\n  NGHIỆM THU 2.3: " + ("❌ CÓ SAI LỆCH — xem trên"
                                    if hong else "✅ resolver đọc đúng tên file thật"))
    if not hong and giong == 0:
        print("  ⚠ Phép 3 chưa kiểm được câu nào (chưa tải video của dev set),")
        print("    nên chỉ chứng minh 'có file ở đúng đường dẫn', chưa chứng minh")
        print("    hai quy ước đánh số trùng nhau.")
    sys.exit(1 if hong else 0)


if __name__ == "__main__":
    main()
