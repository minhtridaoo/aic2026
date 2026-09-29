"""Đọc bộ dữ liệu thật của BTC vào các kiểu dữ liệu của `aic2026`.

Đây là cây cầu duy nhất giữa dữ liệu trên đĩa và phần còn lại của hệ thống.
Mọi module khác chỉ thấy `Event` / `Candidate`, không biết gì về đường dẫn.

BA THÀNH PHẦN ĐỌC ĐƯỢC
----------------------
    map-keyframes/L21_V001.csv    n, pts_time, fps, frame_idx
    clip-features-32/L21_V001.npy (N, 512) float16, ĐÃ chuẩn hoá L2
    media-info/L21_V001.json      title, description, keywords, author, ...

Số dòng của CSV và số hàng của .npy khớp nhau theo từng video (audit v4.7.4:
`All row counts match: 873 / 873`), nên hàng thứ i của .npy là keyframe thứ i
của CSV. Loader vẫn kiểm lại từng video thay vì tin, vì lệch một hàng là lệch
toàn bộ kết quả mà không có triệu chứng nào nhìn thấy được.

MỘT KEYFRAME = MỘT EVENT
------------------------
Ta KHÔNG gộp keyframe thành cửa sổ 10 giây ở bước này. Lý do: BTC cấp feature
theo từng keyframe, gộp lại là tự tay làm mờ thông tin trước khi đo được gì.
Khoảng cách keyframe (trung vị 2.16s) đã đóng vai trò "cửa sổ" sẵn rồi.
`t_end` của một event lấy bằng `pts_time` của keyframe kế tiếp, để các event
lợp kín trục thời gian thay vì là các điểm rời rạc — DP căn chuỗi cần khoảng
cách thời gian có nghĩa.

VỀ `frame_idx` — vì sao nó là ID nộp bài, và phần nào vẫn chưa chắc
------------------------------------------------------------------
`keyframe_idx` của Event lấy thẳng từ cột `frame_idx` trong CSV.

Bằng chứng ỦNG HỘ (thể lệ vòng sơ tuyển, đọc 2026-08-17): BTC chấm KIS bằng
`I(v = GT_v ∧ id ∈ [s, e])` với ví dụ "L01_V001, khung hình từ 500 đến 510" —
tức `frame_id` là **số khung trong video gốc**, không phải số thứ tự file ảnh.
Mục 3 của thể lệ nói rõ tên file keyframe đặt theo thứ tự tăng dần còn "vị trí
(frame index) tương ứng của mỗi keyframe được ghi trong file metadata". Bảng
`map-keyframes` chính là ánh xạ đó, nên cột `frame_idx` là đại lượng cần nộp.

Phần CHƯA chắc, giữ nguyên cảnh báo cũ: thể lệ không viết thành một câu đồng
nhất "cột `frame_idx` = `frame_id` nộp bài", nên đây là suy luận mạnh chứ không
phải trích dẫn. Và audit v4.7.4 ghi 614 bước trùng trên 192 video, với bằng
chứng within-video đã BÁC BỎ một global offset đồng nhất — tức dữ liệu có chỗ
lấn cấn riêng của nó. Đừng áp một phép hiệu chỉnh ngầm nào lên cột này; xem
`README.md` mục canonical manifest.

PHỤ THUỘC
---------
Module này (và chỉ module này, cùng `stores.NumpyVectorStore`) cần `numpy`.
Phần còn lại của gói vẫn chạy trên stdlib thuần.
"""
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .models import Event

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - phụ thuộc cứng của riêng file này
    raise ImportError(
        "aic2026.core.dataset cần numpy. Cài bằng: pip install numpy"
    ) from exc


VIDEO_ID_PATTERN = re.compile(r"(?i)(L\d{2,3}_V\d{3,5})")

# CHỈ dùng cho ma trận rỗng (không nạp được video nào) — lúc đó số chiều không
# suy ra được từ đâu cả. Đường chạy thật luôn lấy số chiều từ chính file .npy.
DIM_MAC_DINH = 512

# Cùng bộ luật với `classify_component` trong audit v4.7.4 của nhóm trưởng, để
# hai bên không hiểu khác nhau về "thế nào là file clip feature".
_TABLE_EXTS = {".csv", ".tsv", ".txt", ".parquet", ".json", ".jsonl"}


def phan_loai(relative_path: str, ext: str) -> str:
    low = relative_path.lower()
    if ext == ".npy":
        return "clip_features" if any(
            t in low for t in ("clip", "feature", "embedding")) else "npy_unknown"
    if ext in _TABLE_EXTS:
        if "object" in low:
            return "objects"
        if "map" in low and "keyframe" in low:
            return "keyframe_mapping"
        if any(t in low for t in ("media-info", "media_info", "metadata")):
            return "metadata"
    return "other"


# --------------------------------------------------------------------------- #
# Dò đường dẫn
# --------------------------------------------------------------------------- #
@dataclass
class VideoPaths:
    """Ba file của một video. Bất kỳ cái nào cũng có thể thiếu."""
    video_id: str
    clip: Optional[Path] = None
    mapping: Optional[Path] = None
    meta: Optional[Path] = None

    @property
    def du_toi_thieu(self) -> bool:
        """Đủ để dựng Event: cần mapping (khung/thời gian) và clip (vector)."""
        return self.clip is not None and self.mapping is not None


def scan(root: str | Path,
         goc_feature: Optional[str | Path] = None) -> Dict[str, VideoPaths]:
    """Duyệt `root`, gom file theo video_id. Không quan tâm thư mục lồng sâu bao nhiêu.

    Nhờ vậy trỏ vào bản tải nhẹ hay trỏ thẳng vào dataset gốc trên Kaggle đều chạy.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Không thấy thư mục dữ liệu: {root}")

    ket_qua: Dict[str, VideoPaths] = {}
    for thu_muc, _, ten_files in os.walk(root):
        for ten in ten_files:
            p = Path(thu_muc) / ten
            m = VIDEO_ID_PATTERN.search(ten)
            if not m:
                continue
            vid = m.group(1).upper()
            nhom = phan_loai(str(p.relative_to(root)), p.suffix.lower())
            vp = ket_qua.setdefault(vid, VideoPaths(video_id=vid))
            if nhom == "clip_features":
                vp.clip = p
            elif nhom == "keyframe_mapping":
                vp.mapping = p
            elif nhom == "metadata":
                vp.meta = p

    if goc_feature is not None:
        _thay_feature(ket_qua, goc_feature)
    return ket_qua


def _thay_feature(ket_qua: Dict[str, VideoPaths],
                  goc_feature: str | Path) -> None:
    """Thay vector ảnh bằng bộ feature ở thư mục khác (vd SigLIP2 tự encode).

    VÌ SAO PHẢI TÁCH THƯ MỤC
    ------------------------
    Không thể giải nén bộ feature mới vào chung `root`: `phan_loai` coi MỌI file
    `.npy` có chữ `clip`/`feature`/`embedding` trong đường dẫn là clip_features,
    còn vòng lặp trên gán `vp.clip = p` — file quét sau ghi đè file quét trước,
    không một cảnh báo nào. Kết quả sẽ là mỗi video dùng một không gian vector
    khác nhau tuỳ thứ tự `os.walk`.

    Video KHÔNG có feature thay thế bị đặt `clip = None`, tức báo THIẾU FILE chứ
    KHÔNG lặng lẽ tụt về vector CLIP cũ. Trộn 512 chiều với 768 chiều trong cùng
    một ma trận là chuyện phải nổ ngay, không phải chuyện để đoán sau.
    """
    goc = Path(goc_feature)
    if not goc.exists():
        raise FileNotFoundError(f"Không thấy thư mục feature: {goc}")

    thay: Dict[str, Path] = {}
    for thu_muc, _, ten_files in os.walk(goc):
        for ten in ten_files:
            p = Path(thu_muc) / ten
            if p.suffix.lower() != ".npy":
                continue
            m = VIDEO_ID_PATTERN.search(ten)
            if not m:
                continue
            vid = m.group(1).upper()
            cu = thay.get(vid)
            if cu is not None and cu != p:
                raise ValueError(
                    f"{goc}: có HAI file .npy cho {vid} — {cu.name} và {p.name}.\n"
                    "  Không đoán file nào đúng. Xoá bản dư rồi chạy lại.")
            thay[vid] = p

    if not thay:
        raise ValueError(
            f"{goc}: không thấy file .npy nào có video_id trong tên.\n"
            "  Bộ feature phải đặt tên theo video, vd L21_V001.npy.")

    for vid, vp in ket_qua.items():
        vp.clip = thay.get(vid)


# --------------------------------------------------------------------------- #
# Đọc từng loại file
# --------------------------------------------------------------------------- #
@dataclass
class KeyframeRow:
    n: int              # thứ tự keyframe trong video, đếm từ 1
    pts_time: float     # giây
    fps: float
    frame_idx: int      # số khung trong video gốc — ứng viên cho ID nộp bài


def load_map_keyframes(path: str | Path) -> List[KeyframeRow]:
    """Đọc map-keyframes CSV. Cột: n, pts_time, fps, frame_idx."""
    rows: List[KeyframeRow] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for i, r in enumerate(csv.DictReader(fh), start=2):  # dòng 1 là header
            try:
                rows.append(KeyframeRow(
                    n=int(r["n"]),
                    pts_time=float(r["pts_time"]),
                    fps=float(r["fps"]),
                    frame_idx=int(r["frame_idx"]),
                ))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path} dòng {i}: đọc lỗi ({exc})") from exc
    rows.sort(key=lambda k: k.n)
    return rows


@dataclass
class VideoMeta:
    """media-info JSON. Nguồn text tiếng Việt DUY NHẤT trong bộ dữ liệu.

    Bộ data BTC không có OCR và không có ASR, nên đây là thứ duy nhất cứu được
    hai kênh chữ. Nhược điểm: nó ở mức VIDEO, không định vị được khung hình.
    Dùng để thu hẹp "video nào", không dùng để chọn "khung nào".
    """
    video_id: str
    title: str = ""
    description: str = ""
    author: str = ""
    keywords: List[str] = field(default_factory=list)
    length: Optional[int] = None
    publish_date: str = ""
    watch_url: str = ""

    def text(self, kem_description: bool = False) -> str:
        """Gộp thành một chuỗi để đưa vào kho chữ.

        Mặc định BỎ `description` vì nó đầy link, hashtag và lời quảng cáo kênh
        lặp lại y hệt nhau ở hàng trăm video — thứ đó làm loãng chỉ mục chứ
        không giúp phân biệt. Bật lên khi nào đo được là nó có ích.
        """
        phan = [self.title, " ".join(self.keywords), self.author]
        if kem_description:
            phan.append(self.description)
        return " ".join(p for p in phan if p).strip()


def load_media_info(path: str | Path, video_id: str) -> VideoMeta:
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    kw = d.get("keywords") or []
    if isinstance(kw, str):
        kw = [kw]
    return VideoMeta(
        video_id=video_id,
        title=str(d.get("title", "") or ""),
        description=str(d.get("description", "") or ""),
        author=str(d.get("author", "") or ""),
        keywords=[str(k) for k in kw],
        length=d.get("length") if isinstance(d.get("length"), int) else None,
        publish_date=str(d.get("publish_date", "") or ""),
        watch_url=str(d.get("watch_url", "") or ""),
    )


def load_clip_features(path: str | Path, dtype=np.float32,
                       chuan_hoa_lai: bool = True) -> "np.ndarray":
    """Đọc .npy (N, D) float16 -> ma trận dtype yêu cầu.

    D lấy từ chính file, KHÔNG cố định 512: CLIP của BTC là 512, SigLIP2 base là
    768, so400m là 1152.

    Ép sang float32 vì matmul float16 trên CPU chậm hơn nhiều; 177k x 512
    float32 = 346 MiB, vẫn thoải mái (768 chiều = 520 MiB).

    `chuan_hoa_lai` (mặc định BẬT) — vì sao cần dù nguồn đã chuẩn hoá L2:

    Audit ghi `all_possible_normalized: True`, tức BTC có ý định để chuẩn bằng 1.
    Nhưng lưu ở float16 làm chuẩn thực tế dao động 0.99953 – 1.00044. Hệ quả đo
    được: cosine của một vector với CHÍNH NÓ bằng |v|² = 0.99906, trong khi một
    keyframe gần-giống có |v| = 1.00031 lại đạt 0.99941 — tức là khung hình khác
    vượt lên trên chính nó chỉ vì sai số lưu trữ. Đã bắt được 3/200 trường hợp
    như vậy trong `kiem_chung_thuc_te.py`.

    Chuẩn hoá lại khiến tích vô hướng đúng bằng cosine, và điểm số giữa các
    video so sánh được với nhau — điều mà tầng `optimize` dựa vào để xếp slot.
    """
    a = np.load(str(path))
    if a.ndim != 2:
        raise ValueError(f"{path}: mong (N, D), nhận {a.shape}")
    m = np.ascontiguousarray(a, dtype=dtype)
    if chuan_hoa_lai and m.size:
        n = np.linalg.norm(m, axis=1, keepdims=True)
        m = m / np.where(n == 0, 1.0, n)
    return m


# --------------------------------------------------------------------------- #
# Dựng Event
# --------------------------------------------------------------------------- #
@dataclass
class LoadReport:
    """Đã nạp được gì, bỏ qua cái gì và vì sao. In ra để khỏi phải đoán."""
    videos_tim_thay: int = 0
    videos_da_nap: int = 0
    events: int = 0
    thieu_file: List[str] = field(default_factory=list)
    lech_so_hang: List[Tuple[str, int, int]] = field(default_factory=list)
    loi_khac: List[Tuple[str, str]] = field(default_factory=list)
    fps_phan_bo: Dict[float, int] = field(default_factory=dict)

    def tom_tat(self) -> str:
        d = [f"video tìm thấy : {self.videos_tim_thay}",
             f"video đã nạp   : {self.videos_da_nap}",
             f"event dựng được: {self.events:,d}"]
        if self.fps_phan_bo:
            top = sorted(self.fps_phan_bo.items(), key=lambda kv: -kv[1])[:4]
            d.append("fps            : " +
                     ", ".join(f"{f:g} ({n} video)" for f, n in top))
        if self.thieu_file:
            d.append(f"⚠ thiếu file   : {len(self.thieu_file)} video "
                     f"(vd: {', '.join(self.thieu_file[:3])})")
        if self.lech_so_hang:
            d.append(f"⚠ LỆCH SỐ HÀNG: {len(self.lech_so_hang)} video — "
                     "csv và npy không khớp, ĐÃ BỎ QUA")
            for vid, n_csv, n_npy in self.lech_so_hang[:3]:
                d.append(f"    {vid}: csv={n_csv} npy={n_npy}")
        if self.loi_khac:
            d.append(f"⚠ lỗi khác     : {len(self.loi_khac)}")
            for vid, msg in self.loi_khac[:3]:
                d.append(f"    {vid}: {msg}")
        return "\n".join("  " + x for x in d)


def events_cua_video(video_id: str, rows: Sequence[KeyframeRow],
                     meta: Optional[VideoMeta] = None,
                     meta_vao_kenh_chu: bool = False) -> List[Event]:
    """Mỗi keyframe -> một Event. `t_end` nối tới keyframe kế tiếp."""
    events: List[Event] = []
    chu = meta.text() if (meta and meta_vao_kenh_chu) else ""
    for i, r in enumerate(rows):
        ke_tiep = rows[i + 1] if i + 1 < len(rows) else None
        if ke_tiep is not None:
            t_end = ke_tiep.pts_time
            end_frame = max(r.frame_idx, ke_tiep.frame_idx - 1)
        else:
            # Keyframe cuối: không biết video còn dài bao nhiêu, lấy khoảng cách
            # của bước trước làm ước lượng thay vì để độ dài bằng 0.
            khoang = (r.pts_time - rows[i - 1].pts_time) if i > 0 else 1.0
            t_end = r.pts_time + max(khoang, 0.04)
            end_frame = r.frame_idx + max(int(round(khoang * r.fps)), 1)
        events.append(Event(
            video_id=video_id,
            event_id=f"{video_id}#{r.n:05d}",
            t_start=r.pts_time,
            t_end=t_end,
            start_frame=r.frame_idx,
            end_frame=end_frame,
            keyframe_idx=r.frame_idx,   # ⚠ chưa chắc là ID nộp bài, xem docstring
            visual_text="",             # có embedding thật rồi, không cần text thay thế
            ocr_text="",                # bộ data KHÔNG có OCR
            asr_text=chu,               # tạm mượn kênh này cho text metadata nếu bật
        ))
    return events


def load_dataset(root: str | Path,
                 video_ids: Optional[Iterable[str]] = None,
                 gioi_han_video: Optional[int] = None,
                 meta_vao_kenh_chu: bool = False,
                 dtype=np.float32,
                 goc_feature: Optional[str | Path] = None,
                 ) -> Tuple[List[Event], "np.ndarray", Dict[str, VideoMeta], LoadReport]:
    """Nạp dữ liệu thật.

    Trả về `(events, embeddings, metas, report)` trong đó `embeddings[i]` là
    vector của `events[i]` — cùng thứ tự, đó là hợp đồng giữa hàm này và
    `MultiChannelRetriever.index`.

    Embedding KHÔNG gắn vào từng Event: 177.321 list Python 512 phần tử sẽ ngốn
    vài GB RAM. Giữ thành một ma trận numpy duy nhất là 363 MB.

    Args:
        root: thư mục chứa dữ liệu (dò tự động, lồng sâu bao nhiêu cũng được)
        goc_feature: thư mục RIÊNG chứa vector ảnh thay thế (vd SigLIP2 tự
            encode), đặt tên theo video `L21_V001.npy`. `root` vẫn cấp
            map-keyframes và media-info. Video nào thiếu file ở đây sẽ bị báo
            THIẾU, không tụt về vector CLIP cũ — xem `_thay_feature`.
        video_ids: chỉ nạp các video này; None = tất cả
        gioi_han_video: chỉ nạp N video đầu (theo thứ tự tên) — để thử nhanh
        meta_vao_kenh_chu: đưa title+keywords vào kênh chữ (mặc định TẮT — và
            đây là quyết định ĐO ĐƯỢC, không phải mặc định bỏ ngỏ)

            Đo 2026-08-08 (`thu_nghiem_tang_tim.py`, dev set 60 câu KIS, ghép
            cặp):

                tắt (nền)  điểm 0.1233   R@100 28.3%   video-R@100 68.3%
                BẬT        điểm 0.0933   R@100 20.0%   video-R@100 51.7%
                           Δ = -0.0300 ± 0.0104  -> hại thật, vượt 2·SE

            Vì sao hại: truy vấn dev set là TIẾNG ANH, còn title/keywords của
            BTC là TIẾNG VIỆT. `InMemoryTextStore` khớp TF-IDF theo token thô nên
            gần như không có token chung, nhưng RRF vẫn cấp cho kênh chữ đúng
            trọng số 1/(60+rank) như kênh visual. Kết quả: mỗi câu bị chèn thêm
            ~50 event gần như ngẫu nhiên, đẩy ứng viên visual tốt ra khỏi top.
            Đo được: 3.000 ứng viên có điểm chữ > 0, tức kênh chạy hết công suất
            — hại chứ không phải không hoạt động.

            Còn chậm 40×: 283s so với 7s cho 60 câu (TF-IDF trên 177.321 văn bản).

            Khi nào đáng bật lại: nếu truy vấn giữ nguyên TIẾNG VIỆT (khớp ngôn
            ngữ với metadata), hoặc thay `InMemoryTextStore` bằng Elasticsearch
            có analyzer tiếng Việt, và hạ trọng số kênh chữ trong fusion. Cả ba
            đều là thay đổi thật, phải đo lại chứ không suy ra được từ đây.
    """
    paths = scan(root, goc_feature=goc_feature)
    rep = LoadReport(videos_tim_thay=len(paths))

    chon = sorted(paths)
    if video_ids is not None:
        muon = {v.upper() for v in video_ids}
        chon = [v for v in chon if v in muon]
    if gioi_han_video is not None:
        chon = chon[:gioi_han_video]

    tat_ca_events: List[Event] = []
    khoi: List["np.ndarray"] = []

    metas: Dict[str, VideoMeta] = {}
    for vid in chon:
        vp = paths[vid]
        if not vp.du_toi_thieu:
            rep.thieu_file.append(vid)
            continue
        try:
            rows = load_map_keyframes(vp.mapping)
            mat = load_clip_features(vp.clip, dtype=dtype)
        except Exception as exc:                      # noqa: BLE001 - báo cáo, không sập
            rep.loi_khac.append((vid, str(exc)))
            continue

        if len(rows) != mat.shape[0]:
            # Bỏ qua hẳn. Cắt bớt cho khớp là cách chắc chắn nhất để gán nhầm
            # vector cho keyframe mà không ai phát hiện ra.
            rep.lech_so_hang.append((vid, len(rows), int(mat.shape[0])))
            continue

        meta = None
        if vp.meta is not None:
            try:
                meta = load_media_info(vp.meta, vid)
                metas[vid] = meta
            except Exception as exc:                  # noqa: BLE001
                rep.loi_khac.append((vid, f"media-info: {exc}"))

        evs = events_cua_video(vid, rows, meta, meta_vao_kenh_chu)
        tat_ca_events.extend(evs)
        khoi.append(mat)
        rep.videos_da_nap += 1
        if rows:
            f = round(rows[0].fps, 3)
            rep.fps_phan_bo[f] = rep.fps_phan_bo.get(f, 0) + 1

    if khoi:
        # Số chiều LẤY TỪ DỮ LIỆU, không giả định 512. CLIP của BTC là 512 nhưng
        # SigLIP2 base là 768 và so400m là 1152. `vstack` tự nổ nếu hai video có
        # số chiều khác nhau — đúng chỗ ta muốn nó nổ.
        embeddings = np.vstack(khoi)
    else:
        embeddings = np.zeros((0, DIM_MAC_DINH), dtype=dtype)
    rep.events = len(tat_ca_events)
    if len(tat_ca_events) != embeddings.shape[0]:      # pragma: no cover - bất biến
        raise AssertionError(
            f"event {len(tat_ca_events)} != embedding {embeddings.shape[0]}")
    return tat_ca_events, embeddings, metas, rep
