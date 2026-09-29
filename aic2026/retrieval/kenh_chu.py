"""KÊNH CHỮ OCR/ASR theo KHOẢNG SHOT — lấp kênh mà hệ thống dựng sẵn nhưng rỗng.

VÌ SAO ĐÁNG LÀM, VÀ VÌ SAO ĐÚNG LÚC NÀY
---------------------------------------
`MultiChannelRetriever` có sẵn ba kênh visual/ocr/asr và RRF đã nối từ đầu.
Nhưng bộ dữ liệu BTC **không cấp OCR lẫn ASR**, nên hai kênh đó luôn rỗng và
mọi thứ đổ hết lên vai encoder ảnh.

Soi bài nộp vòng thử nghiệm (20/08, 5,6/12) thấy rất rõ chỗ đau: 6 câu sai hẳn
video — **1, 2, 8, 14, 21, 23** — đều là MẨU TIN NGẮN trong bản tin *60 Giây*.
Encoder ảnh không thể phân biệt "đàn hổ con quý hiếm" với "phóng tàu vũ trụ"
khi cả hai đều là cảnh studio + phóng sự ngắn. Nhưng phát thanh viên **nói ra
miệng** đúng những chữ đó. Kênh ASR bắt được đúng nhóm câu này. Ba câu QA của
đề cũng đều là câu ĐỌC CHỮ trong khung hình -> kênh OCR.

INDEX Ở MỨC KHOẢNG, KHÔNG PHẢI MỨC KEYFRAME — CÓ LÝ DO ĐO ĐƯỢC
--------------------------------------------------------------
Cám dỗ đầu tiên là đổ text vào `Event.ocr_text` cho cả 177.321 keyframe. Đừng.
Đã đo ngày 08/08 (`thu_nghiem_tang_tim.py`, dev 60 câu, ghép cặp):

    kênh chữ TẮT   điểm 0.1233   R@100 28.3%
    kênh chữ BẬT   điểm 0.0933   R@100 20.0%    Δ = -0.0300 ± 0.0104

Hại thật, vượt 2·SE. Hai nguyên nhân, và cả hai lặp lại y nguyên nếu làm ẩu ở
đây: (1) một khoảng shot trải ~9,5 keyframe nên cùng một đoạn text bị nhân bản
9,5 lần, làm IDF méo hoàn toàn; (2) `InMemoryTextStore` chấm TF-IDF trên
177.321 tài liệu chậm gấp **40 lần** (283s so với 7s cho 60 câu).

Nên ở đây: chấm trên **18.660 khoảng**, rồi mới nở ra event. Nhanh hơn ~10 lần,
IDF đúng, và trọng số trong RRF là một con số nhìn thấy được chứ không phải
"kênh nào cũng 1/(60+rank)" như lần trước.

CHẤM BẰNG IDF, KHÔNG PHẢI ĐẾM TỪ TRÙNG
--------------------------------------
Hệ của bạn ấy dùng `|q ∩ doc| / |q|`. Với tiếng Việt thì hỏng: "một", "người",
"trong", "của" có mặt ở gần như mọi khoảng, nên câu nào cũng "trùng" 40% và thứ
hạng gần như ngẫu nhiên. Ở đây dùng coverage CÓ TRỌNG SỐ IDF, tức từ hiếm
("Lausanne", "cực quang", "panna cotta") quyết định, từ chức năng gần như không
đóng góp. Không cần bảng stopword tay — IDF tự lo, và tự lo đúng theo kho.
"""
from __future__ import annotations

import bisect
import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

COT_BAT_BUOC = ("video_id", "start_pts_time_technical",
                "end_pts_time_technical", "ocr_text_compact",
                "asr_text_compact")


@dataclass
class CauHinhKenhChu:
    """Mọi núm đều TẮT theo mặc định. Bật là một quyết định phải đo."""

    bat: bool = False
    duong_dan: Optional[str] = None
    # Trọng số trong RRF hợp nhất. Đặt < 1 vì kênh chữ là bằng chứng BỔ SUNG:
    # nó rất mạnh khi trúng từ hiếm và hoàn toàn câm khi truy vấn thuần thị
    # giác. Cho nó nói to bằng encoder ảnh là tái lập đúng lỗi 08/08.
    trong_so: float = 0.6
    pool_khoang: int = 150          # lấy bao nhiêu khoảng trước khi nở ra event
    toi_da_event_moi_khoang: int = 30
    # Bỏ qua khoảng có điểm dưới ngưỡng: khớp một từ chức năng không phải bằng
    # chứng, mà là nhiễu được cấp một suất trong RRF.
    nguong_diem: float = 0.15
    # 🔴 CHUẨN HOÁ ĐIỂM THEO CÁI GÌ — thêm 2026-08-21 sau khi sơ tuyển đợt 1
    # được 3,8/12 (vòng thử nghiệm 8,8/12 trên CÙNG kho 873 video).
    #
    #   "truy_van"  điểm = IDF khớp / IDF cả truy vấn  → "tài liệu này phủ được
    #               bao nhiêu phần câu hỏi". Đây là hành vi cũ, và nó ĐÚNG khi
    #               truy vấn ngắn còn tài liệu dài (câu bản tin vs ASR 759 ký tự
    #               mỗi khoảng): mọi từ của câu hỏi đều phải xuất hiện.
    #
    #   "chong_lan" điểm = IDF khớp / min(IDF truy vấn, IDF tài liệu) — hệ số
    #               chồng lấn. Chuẩn hoá theo BÊN NGẮN HƠN, nên đối xứng.
    #
    # Vì sao phải có: câu đề đợt 1 tả cảnh dài 40–90 âm tiết, còn tiêu đề video
    # chỉ 6–12 âm tiết. `query-p1-5-kis` khớp đúng 5 âm tiết hiếm (mực/xào/đậu/
    # hà/lan) với tiêu đề "MỰC XÀO ĐẬU HÀ LAN" — tức là phủ gần TRỌN tiêu đề,
    # bằng chứng gần như chắc chắn — nhưng chia cho IDF của cả 60 âm tiết còn
    # lại thì ra ~0,08, dưới `nguong_diem` 0,15, nên bị vứt. Kết quả: kênh chữ
    # CÂM HOÀN TOÀN ở mọi câu tả cảnh dài, đúng 17/25 câu của đợt 1.
    #
    # Mặc định giữ "truy_van" để không lặng lẽ đổi số đã đo của vòng thử nghiệm.
    chuan_hoa_diem: str = "truy_van"
    # Sàn KHỐI LƯỢNG IDF tuyệt đối của phần khớp (không phải tỉ lệ). Cần vì
    # "chong_lan" chuẩn hoá theo bên ngắn hơn, nên một tiêu đề 4 âm tiết phổ
    # thông ("CANH CÁ HỒNG") bị phủ trọn bởi vài âm tiết vu vơ trong câu đề dài
    # cũng đạt tỉ lệ 0,3–0,4 — cao ngang một cú trúng thật. Tỉ lệ không phân
    # biệt được "phủ trọn 4 âm tiết rẻ tiền" với "phủ trọn 5 âm tiết hiếm";
    # khối lượng IDF thì có. Đo trên đợt 1: cú trúng thật `L26_V035` đạt 22,6
    # còn rác cao nhất chỉ 9,0.
    #
    # 0.0 = tắt, giữ nguyên hành vi cũ.
    toi_thieu_idf: float = 0.0
    dung_ocr: bool = True
    dung_asr: bool = True
    # Khoảng của Stage20 là các đoạn "peak", KHÔNG phủ kín trục thời gian: đo
    # thật trên corpus, chỉ 9.978/177.321 event (5,6%) rơi đúng vào trong một
    # khoảng. Nới dung sai cho phép gán event nằm NGOÀI khoảng nhưng cách nó
    # dưới `dung_sai_giay`.
    #
    # 0 = chặt tuyệt đối. Nới ra là đánh đổi có thật: lời thoại tràn qua ranh
    # giới shot (người nói không dừng lại khi máy quay cắt cảnh) nên vài giây
    # là hợp lý, nhưng nới quá thì gán lời của cảnh khác. Phải ĐO, đừng đoán.
    dung_sai_giay: float = 0.0


@dataclass
class BaoCaoNap:
    so_khoang: int = 0
    so_video: int = 0
    co_ocr: int = 0
    co_asr: int = 0
    so_token: int = 0
    bo_qua: int = 0
    canh_bao: List[str] = field(default_factory=list)

    def mo_ta(self) -> str:
        d = [f"{self.so_khoang:,d} khoảng / {self.so_video} video",
             f"có OCR {self.co_ocr:,d} ({self.co_ocr / max(1, self.so_khoang):.0%})",
             f"có ASR {self.co_asr:,d} ({self.co_asr / max(1, self.so_khoang):.0%})",
             f"{self.so_token:,d} token khác nhau"]
        if self.bo_qua:
            d.append(f"⚠ bỏ {self.bo_qua} dòng thiếu trường")
        return "  ·  ".join(d)


# --------------------------------------------------------------------------- #
# Tách từ
# --------------------------------------------------------------------------- #
_KHONG_CHU = re.compile(r"[^\w]+", re.UNICODE)


def chuan_hoa(s) -> str:
    return " ".join(_KHONG_CHU.sub(" ", unicodedata.normalize(
        "NFKC", str(s or "")).casefold()).split())


def _chuoi(x) -> str:
    """Ô rỗng -> "". Bẫy thật: `pandas.read_csv` trả `NaN` cho ô trống, và
    `str(NaN)` là `"nan"` — một chuỗi KHÔNG rỗng. Không chặn thì "nan" thành
    một token có mặt ở 13% số khoảng, độ phủ báo 100% thay vì 78%, và mọi truy
    vấn chứa chữ "nan" (khá nhiều trong tiếng Việt) khớp bừa."""
    if x is None:
        return ""
    if isinstance(x, float) and x != x:            # NaN
        return ""
    t = str(x).strip()
    return "" if t.lower() in ("nan", "none", "<na>") else t


def tach_tu(s) -> List[str]:
    """Token = âm tiết. Tiếng Việt ghép nhiều âm tiết thành từ, nhưng tách theo
    âm tiết vẫn khớp tốt vì IDF sẽ đề cao đúng những âm tiết hiếm
    ("lausanne", "quang", "cotta") và dìm những âm tiết phổ biến."""
    return [t for t in chuan_hoa(s).split() if len(t) >= 2 or t.isdigit()]


# --------------------------------------------------------------------------- #
# Kho
# --------------------------------------------------------------------------- #
class KhoChuKhoang:
    """OCR/ASR theo khoảng shot + chỉ mục ngược + chấm coverage có IDF."""

    def __init__(self, khoang: Sequence[Dict], cfg: Optional[CauHinhKenhChu] = None):
        self.cfg = cfg or CauHinhKenhChu()
        self.khoang: List[Dict] = list(khoang)
        self.bao_cao = BaoCaoNap(so_khoang=len(self.khoang))

        self._nguoc: Dict[str, List[int]] = defaultdict(list)
        self._idf: Dict[str, float] = {}
        self._theo_video: Dict[str, List[Tuple[float, float, int]]] = defaultdict(list)
        self._event_theo_khoang: Dict[int, List[str]] = {}
        self._xay()

    # ---- dựng chỉ mục ----------------------------------------------------- #
    def _xay(self) -> None:
        df: Dict[str, int] = defaultdict(int)
        for i, k in enumerate(self.khoang):
            phan = []
            if self.cfg.dung_ocr:
                phan.append(k.get("ocr_text_compact", ""))
            if self.cfg.dung_asr:
                phan.append(k.get("asr_text_compact", ""))
            toks = set(tach_tu(" ".join(str(x or "") for x in phan)))
            k["_tokens"] = toks
            for t in toks:
                self._nguoc[t].append(i)
                df[t] += 1
            vid = str(k.get("video_id", "")).upper()
            self._theo_video[vid].append(
                (float(k.get("start_pts_time_technical") or 0.0),
                 float(k.get("end_pts_time_technical") or 0.0), i))
            if str(k.get("ocr_text_compact") or "").strip():
                self.bao_cao.co_ocr += 1
            if str(k.get("asr_text_compact") or "").strip():
                self.bao_cao.co_asr += 1

        n = max(1, len(self.khoang))
        # IDF làm mượt kiểu BM25: từ có mặt ở mọi khoảng -> ~0, từ hiếm -> lớn.
        self._idf = {t: math.log(1.0 + (n - c + 0.5) / (c + 0.5))
                     for t, c in df.items()}
        # Tổng khối IDF của TỪNG tài liệu — mẫu số của "chong_lan". Tính một lần
        # ở đây thay vì trong `cham_khoang`: nó không phụ thuộc truy vấn.
        self._idf_tai_lieu = [
            sum(self._idf.get(t, 0.0) for t in k.get("_tokens") or ())
            for k in self.khoang]
        for vid in self._theo_video:
            self._theo_video[vid].sort()
        self.bao_cao.so_video = len(self._theo_video)
        self.bao_cao.so_token = len(self._idf)

    # ---- nạp -------------------------------------------------------------- #
    @classmethod
    def tu_bang(cls, rows: Iterable[Dict],
                cfg: Optional[CauHinhKenhChu] = None) -> "KhoChuKhoang":
        sach, bo = [], 0
        for r in rows:
            if not all(c in r for c in COT_BAT_BUOC):
                bo += 1
                continue
            sach.append({c: _chuoi(r[c]) if c.endswith("_compact") else r[c]
                         for c in COT_BAT_BUOC})
        kho = cls(sach, cfg)
        kho.bao_cao.bo_qua = bo
        if bo:
            kho.bao_cao.canh_bao.append(
                f"{bo} dòng thiếu trường bắt buộc, đã bỏ")
        return kho

    @classmethod
    def tu_file(cls, duong_dan: str | Path,
                cfg: Optional[CauHinhKenhChu] = None) -> "KhoChuKhoang":
        """`.parquet` (cần pyarrow) · `.csv` · `.json` / `.jsonl`.

        Nhận nhiều định dạng CÓ CHỦ Ý: parquet chỉ đọc được khi có pyarrow, mà
        pyarrow không phải lúc nào cũng có ở máy dev — không nên vì thế mà
        không test được lớp này.
        """
        p = Path(duong_dan)
        if not p.exists():
            raise FileNotFoundError(
                f"Không thấy dữ liệu kênh chữ: {p}\n"
                f"  Trên Kaggle: Add Input dataset `nguyenminhtric/20-note` rồi "
                f"trỏ `retrieval.kenh_chu.duong_dan` vào "
                f"`shadow_evidence_pack_registry.parquet`.")
        hau = p.suffix.lower()
        if hau in (".json", ".jsonl", ".ndjson"):
            txt = p.read_text(encoding="utf-8")
            rows = ([json.loads(x) for x in txt.splitlines() if x.strip()]
                    if hau != ".json" else json.loads(txt))
            if isinstance(rows, dict):
                rows = rows.get("rows") or rows.get("data") or []
            return cls.tu_bang(rows, cfg)

        import pandas as pd
        if hau == ".csv":
            df = pd.read_csv(p)
        elif hau in (".parquet", ".pq"):
            try:
                df = pd.read_parquet(p)
            except ImportError as exc:
                raise ImportError(
                    "Đọc .parquet cần `pyarrow`. Trên Kaggle có sẵn; ở máy dev "
                    "thì `pip install pyarrow`, hoặc xuất sang .csv/.jsonl."
                ) from exc
        else:
            raise ValueError(f"Định dạng chưa hỗ trợ: {hau}")

        thieu = [c for c in COT_BAT_BUOC if c not in df.columns]
        if thieu:
            raise RuntimeError(
                f"{p.name} thiếu cột {thieu}.\n"
                f"  Cột đang có: {sorted(df.columns)[:20]}\n"
                f"  Đây phải là `shadow_evidence_pack_registry.parquet` của "
                f"Stage20, không phải frame/asr/temporal payload.")
        return cls.tu_bang(df[list(COT_BAT_BUOC)].to_dict("records"), cfg)

    # ---- gắn event -------------------------------------------------------- #
    def gan_event(self, events: Sequence) -> int:
        """Gắn mỗi event vào khoảng CHỨA `t_start` của nó. Trả số event gắn được.

        Tìm nhị phân trên danh sách khoảng đã sắp của từng video. Khoảng của
        Stage20 dựng bằng AutoShot nên không phủ kín 100% trục thời gian; event
        rơi vào kẽ hở thì KHÔNG gắn — thà bỏ còn hơn gán cho khoảng lân cận rồi
        gán nhầm lời thoại của cảnh khác.
        """
        self._event_theo_khoang = defaultdict(list)
        gan = 0
        for e in events:
            vid = str(getattr(e, "video_id", "")).upper()
            ds = self._theo_video.get(vid)
            if not ds:
                continue
            t = float(getattr(e, "t_start", 0.0))
            j = bisect.bisect_right(ds, (t, float("inf"), 1 << 62)) - 1
            dsai = float(self.cfg.dung_sai_giay)
            # Xét cả khoảng đứng trước lẫn khoảng đứng sau `t`: với dung sai > 0
            # thì khoảng gần nhất có thể nằm ở phía sau.
            tot, cach_tot = None, None
            for k in (j, j + 1):
                if not 0 <= k < len(ds):
                    continue
                s, en, idx = ds[k]
                cach = 0.0 if s <= t <= en else (s - t if t < s else t - en)
                if cach <= dsai and (cach_tot is None or cach < cach_tot):
                    tot, cach_tot = idx, cach
            if tot is not None:
                self._event_theo_khoang[tot].append(e.event_id)
                gan += 1
        return gan

    # ---- truy vấn --------------------------------------------------------- #
    def cham_khoang(self, truy_van: str) -> List[Tuple[int, float]]:
        """[(chỉ số khoảng, điểm 0..1)] đã sắp giảm dần. Chỉ duyệt khoảng có từ chung."""
        q = [t for t in set(tach_tu(truy_van)) if t in self._idf]
        if not q:
            return []
        tong = sum(self._idf[t] for t in q)
        if tong <= 0:
            return []
        diem: Dict[int, float] = defaultdict(float)
        for t in q:
            w = self._idf[t]
            for i in self._nguoc.get(t, ()):
                diem[i] += w

        if self.cfg.chuan_hoa_diem == "chong_lan":
            def _mau(i: int) -> float:
                # min(bên truy vấn, bên tài liệu): chuẩn hoá theo bên NGẮN hơn.
                # `or tong` để tài liệu rỗng (không token nào) không chia cho 0.
                return min(tong, self._idf_tai_lieu[i] or tong)
        elif self.cfg.chuan_hoa_diem == "truy_van":
            def _mau(i: int) -> float:
                return tong
        else:
            raise ValueError(
                f"chuan_hoa_diem={self.cfg.chuan_hoa_diem!r} không hợp lệ — "
                f"chỉ nhận 'truy_van' hoặc 'chong_lan'.")

        ra = []
        san = float(self.cfg.toi_thieu_idf)
        for i, s in diem.items():
            if s < san:
                continue
            d = s / _mau(i)
            if d >= self.cfg.nguong_diem:
                # Cắt về 1.0: với "chong_lan", tài liệu ngắn hơn truy vấn và
                # được phủ trọn thì tỉ số bằng đúng 1, nhưng sai số dấu phẩy
                # động có thể nhích lên 1.0000000000000002 — điểm > 1 sẽ phá
                # mọi giả định "0..1" ở tầng hợp nhất.
                ra.append((i, min(1.0, d)))
        ra.sort(key=lambda kv: (-kv[1], kv[0]))
        return ra[:self.cfg.pool_khoang]

    def khoi_idf_khop(self, truy_van: str) -> List[Tuple[int, float, float]]:
        """[(khoảng, khối IDF khớp, tỉ lệ chồng lấn)] — để CHỌN NGƯỠNG bằng số.

        Không dùng trong đường chạy chính. Tồn tại vì `toi_thieu_idf` là một
        con số phải rút ra từ dữ liệu: xem cú trúng thật cách rác bao xa.
        """
        q = [t for t in set(tach_tu(truy_van)) if t in self._idf]
        tong = sum(self._idf[t] for t in q)
        if not q or tong <= 0:
            return []
        diem: Dict[int, float] = defaultdict(float)
        for t in q:
            for i in self._nguoc.get(t, ()):
                diem[i] += self._idf[t]
        ra = [(i, s, s / min(tong, self._idf_tai_lieu[i] or tong))
              for i, s in diem.items()]
        ra.sort(key=lambda kv: -kv[1])
        return ra

    def xep_hang_event(self, truy_van: str,
                       top_k: int = 300) -> List[Tuple[str, float]]:
        """[(event_id, điểm)] — nở khoảng ra event, giữ thứ tự khoảng."""
        if not self._event_theo_khoang:
            return []
        ra: List[Tuple[str, float]] = []
        for i, s in self.cham_khoang(truy_van):
            for eid in self._event_theo_khoang.get(
                    i, ())[:self.cfg.toi_da_event_moi_khoang]:
                ra.append((eid, s))
                if len(ra) >= top_k:
                    return ra
        return ra

    def van_ban_cua_event(self, event_id: str) -> Dict[str, str]:
        """OCR/ASR của khoảng chứa event — để IN RA khi soi kết quả."""
        for i, ds in self._event_theo_khoang.items():
            if event_id in ds:
                k = self.khoang[i]
                return {"ocr": str(k.get("ocr_text_compact") or ""),
                        "asr": str(k.get("asr_text_compact") or "")}
        return {"ocr": "", "asr": ""}
