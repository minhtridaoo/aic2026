"""Đọc config YAML rồi dựng ĐÚNG hệ thống mà config mô tả.

VÌ SAO CÓ FILE NÀY (việc 1.5, 2026-08-16)
-----------------------------------------
Trước file này, `config.example.yaml` KHÔNG được dòng code nào đọc: mỗi script tự
dựng `Pipeline` theo cách riêng. Hậu quả đo được ngày 15/08 khi diễn tập nộp bài:
`nop_bai_thu.py` xuất ra zip có đáp án QA **tiếng Anh** (`Blue`/`Red`/`White`),
chạy **base CLIP** thay vì SigLIP2 so400m, và bật **portfolio** — cả ba đều trái
cấu hình đã chốt. Sai lệch đó không có triệu chứng nào: zip vẫn hợp lệ, validator
vẫn `ok=True`.

Nên chỗ nối "config -> hệ thống" phải là MỘT hàm, và mọi đường nộp bài phải đi
qua nó. `dung_pipeline` bên dưới là hàm đó.

Nguyên tắc giữ nguyên như phần còn lại của repo: thiếu thứ gì thì NÉM LỖI kèm
cách sửa, không âm thầm tụt về mặc định — vì một mặc định âm thầm đúng là kiểu
lỗi vừa mất trọn điểm QA ở trên.

    from aic2026 import config as cfg

    c = cfg.load_config()                       # config.yaml, hoặc .example
    events, emb, metas, rep = cfg.nap_du_lieu(c)
    p = cfg.dung_pipeline(c, events, emb)

Đường dẫn trong YAML được phép ghi tương đối; `load_config` quy về tuyệt đối theo
gốc repo, nên chạy script từ thư mục nào cũng cho cùng một hệ thống.
"""
from __future__ import annotations

import copy
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .core.dataset import load_dataset
from .core.de_bai import NGON_NGU_HOP_LE
from .core.dich import OpusMTEnViTranslator
from .core.pipeline import Pipeline, PortfolioConfig
from .core.providers import (CLIPTextEncoder, MCLIPTextEncoder,
                             RuleBasedProvider, SigLIP2TextEncoder,
                             build_encoder, build_llm)
from .core.stores import NumpyVectorStore
from .core.temporal_dp import DPParams
from .data_plane.image_resolver import EventImageResolver
from .optimize.calibration import RankCalibrator
from .retrieval.service import RetrievalPolicy

# `config.yaml` = lớp phủ của MÁY NÀY (đường dẫn dữ liệu), thường chỉ có
# `extends: config.example.yaml` + vài khoá. `config.example.yaml` là bản chuẩn
# đã chốt. Tách hai lớp để đường dẫn máy không phải sao chép cả file cấu hình —
# sao chép là cách sai lệch quay lại.
TEN_FILE_MAC_DINH = ("config.yaml", "config.example.yaml")

# Provider VLM nào cần biến môi trường nào. Khoá KHÔNG bao giờ đọc từ file.
#
# Mỗi provider là một tuple các NHÓM; trong một nhóm chỉ cần MỘT biến (hoặc), còn
# các nhóm thì phải đủ CẢ (và). Cloudflare cần đúng hai thứ khác nhau — token và
# account id — nên gộp chúng thành "hoặc" sẽ báo đủ khoá khi mới có một nửa.
KHOA_MOI_TRUONG: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "gemini_vlm": (("GEMINI_API_KEY", "GOOGLE_API_KEY"),),
    "groq_vlm": (("GROQ_API_KEY",),),
    "cloudflare_vlm": (("CLOUDFLARE_API_TOKEN",), ("CLOUDFLARE_ACCOUNT_ID",)),
    "anthropic_vlm": (("ANTHROPIC_API_KEY",),),
}


def bien_moi_truong(provider: str) -> List[str]:
    """Danh sách phẳng các biến môi trường liên quan tới một provider (để in ra)."""
    return [x for nhom in KHOA_MOI_TRUONG.get(provider, ()) for x in nhom]

# Khoá nào trong YAML là đường dẫn -> quy về tuyệt đối theo gốc repo.
_KHOA_DUONG_DAN = (
    ("offline", "raw_video_root"),
    ("offline", "btc_data_root"),
    ("offline", "image_feature_root"),
    ("offline", "manifest_dir"),
    ("offline", "vector_bundle_dir"),
    ("encoder", "cache_path"),
    ("encoder", "vietnamese_cache_path"),
    ("qa", "dev_image_dir"),
)


class CauHinhSai(ValueError):
    """Config đọc được nhưng nội dung không dùng được (thiếu khoá, sai giá trị)."""


# --------------------------------------------------------------------------- #
# Đọc file
# --------------------------------------------------------------------------- #
def goc_repo() -> Path:
    """Thư mục chứa `config.example.yaml` (một cấp trên package)."""
    return Path(__file__).resolve().parent.parent


def tim_file_config(path: Optional[str | os.PathLike] = None) -> Path:
    """Đường dẫn file config sẽ dùng. `config.yaml` thắng `config.example.yaml`."""
    if path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (goc_repo() / p) if not p.exists() else p.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Không thấy file config: {p}")
        return p.resolve()
    for ten in TEN_FILE_MAC_DINH:
        p = goc_repo() / ten
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(
        "Không thấy config nào trong " + str(goc_repo()) + ": thiếu cả "
        + " và ".join(TEN_FILE_MAC_DINH))


def _doc_yaml(path: Path) -> Dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
        raise RuntimeError(
            "Thiếu gói `pyyaml` để đọc config. `pip install pyyaml`.\n"
            "  (Chỉ tầng đọc config cần gói này; core vẫn chạy stdlib thuần.)"
        ) from exc
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise CauHinhSai(f"{path}: gốc YAML phải là mapping, đang là "
                         f"{type(data).__name__}")
    return data


def _la_list_mapping(x) -> bool:
    return isinstance(x, list) and bool(x) and all(isinstance(i, dict) for i in x)


def _gop_sau(goc: Dict, phu: Dict) -> Dict:
    """Trộn sâu: `phu` ghi đè `goc`, mapping thì gộp tiếp, còn lại thay hẳn.

    LIST CÁC MAPPING được gộp THEO CHỈ SỐ, không thay hẳn. Cần vì lớp phủ của máy
    chỉ muốn đặt một khoá trong một mục của danh sách (ví dụ `image_feature_root`
    của encoder phụ) — thay hẳn thì mọi khoá còn lại (provider, model_id, cache)
    biến mất và encoder phụ tụt về mặc định sai mà không có triệu chứng.
    Danh sách thường (số, chuỗi) vẫn thay hẳn, ví dụ `qa.image_roots`.
    """
    ra = copy.deepcopy(goc)
    for khoa, gia_tri in (phu or {}).items():
        cu = ra.get(khoa)
        if isinstance(cu, dict) and isinstance(gia_tri, dict):
            ra[khoa] = _gop_sau(cu, gia_tri)
        elif _la_list_mapping(cu) and _la_list_mapping(gia_tri):
            gop = [copy.deepcopy(x) for x in cu]
            for i, muc in enumerate(gia_tri):
                if i < len(gop):
                    gop[i] = _gop_sau(gop[i], muc)
                else:
                    gop.append(copy.deepcopy(muc))
            ra[khoa] = gop
        else:
            ra[khoa] = copy.deepcopy(gia_tri)
    return ra


def _tuyet_doi(gia_tri, goc: Path):
    if gia_tri in (None, ""):
        return gia_tri
    p = Path(str(gia_tri)).expanduser()
    return str(p if p.is_absolute() else (goc / p))


def _giai_duong_dan(config: Dict) -> Dict:
    """Quy mọi khoá đường dẫn về tuyệt đối theo gốc repo (chạy từ đâu cũng như nhau)."""
    goc = goc_repo()
    ra = copy.deepcopy(config)
    for muc, khoa in _KHOA_DUONG_DAN:
        phan = ra.get(muc)
        if isinstance(phan, dict) and phan.get(khoa) not in (None, ""):
            phan[khoa] = _tuyet_doi(phan[khoa], goc)
    qa = ra.get("qa")
    if isinstance(qa, dict):
        roots = qa.get("image_roots") or []
        if isinstance(roots, (str, os.PathLike)):
            roots = [roots]
        qa["image_roots"] = [_tuyet_doi(r, goc) for r in roots if str(r)]
    llm = ra.get("llm")
    if isinstance(llm, dict):
        for phan in [llm, llm.get("primary"), *(llm.get("fallbacks") or [])]:
            if isinstance(phan, dict) and phan.get("cache_path"):
                phan["cache_path"] = _tuyet_doi(phan["cache_path"], goc)
    hn = (ra.get("retrieval") or {}).get("hop_nhat")
    if isinstance(hn, dict):
        for muc in hn.get("phu") or []:
            if not isinstance(muc, dict):
                continue
            for khoa in ("cache_path", "image_feature_root"):
                if muc.get(khoa):
                    muc[khoa] = _tuyet_doi(muc[khoa], goc)
    return ra


def load_config(path: Optional[str | os.PathLike] = None,
                _da_qua: Sequence[Path] = ()) -> Dict:
    """Đọc config (theo `extends` nếu có) rồi quy đường dẫn về tuyệt đối."""
    file_config = tim_file_config(path)
    if file_config in tuple(_da_qua):
        raise CauHinhSai(
            "`extends` vòng tròn: " + " -> ".join(str(x) for x in
                                                  (*_da_qua, file_config)))
    data = _doc_yaml(file_config)
    cha = data.pop("extends", None)
    if cha:
        cha_path = Path(str(cha)).expanduser()
        if not cha_path.is_absolute():
            cha_path = file_config.parent / cha_path
        goc = load_config(cha_path, _da_qua=(*_da_qua, file_config))
        # `_giai_duong_dan` của cha đã chạy; gộp xong chạy lại cũng không đổi gì
        # vì đường tuyệt đối là điểm bất động của phép quy đổi.
        ke_thua = [*(goc.pop("_nguon_ke_thua", None) or []),
                   goc.pop("_nguon_config", str(cha_path))]
        data = _gop_sau(goc, data)
        data["_nguon_ke_thua"] = ke_thua
    # Ghi ĐÈ, không setdefault: file ngoài cùng mới là file người ta gọi, còn tên
    # các file cha nằm ở `_nguon_ke_thua` để in ra khi cần lần vết.
    data["_nguon_config"] = str(file_config)
    return _giai_duong_dan(data)


# --------------------------------------------------------------------------- #
# Dữ liệu
# --------------------------------------------------------------------------- #
def duong_du_lieu(config: Dict) -> Tuple[Path, Optional[Path]]:
    """(thư mục bundle nhẹ của BTC, thư mục vector ảnh tự encode hoặc None)."""
    offline = config.get("offline") or {}
    root = offline.get("btc_data_root")
    if not root:
        raise CauHinhSai(
            "Thiếu `offline.btc_data_root` — không biết nạp 873 video từ đâu.\n"
            "  Đặt nó trong config.yaml (lớp phủ của máy này), ví dụ:\n"
            "      offline:\n"
            "        btc_data_root: 'D:\\COMP\\AIC2026\\data cuộc thi\\aic2026_data_nhe'")
    root = Path(root)
    if not root.is_dir():
        raise CauHinhSai(f"`offline.btc_data_root` không phải thư mục: {root}\n"
                         "  -> chạy kaggle_lay_data_nhe.py rồi giải nén về đây.")

    kind = (config.get("encoder") or {}).get("provider", "hash")
    feat = offline.get("image_feature_root")
    if kind == "siglip2" and not feat:
        raise CauHinhSai(
            "`encoder.provider: siglip2` nhưng thiếu `offline.image_feature_root`.\n"
            "  Vector ảnh SigLIP2 phải nằm ở thư mục RIÊNG, không giải nén lẫn vào\n"
            "  bundle nhẹ (xem dataset._thay_feature: trộn hai bộ = mỗi video một\n"
            "  không gian vector).\n"
            "  -> chạy kaggle_encode_siglip2.py, tải zip về, trỏ khoá này vào đó.")
    if not feat:
        return root, None
    feat = Path(feat)
    if not feat.is_dir():
        raise CauHinhSai(
            f"`offline.image_feature_root` không phải thư mục: {feat}\n"
            "  -> chạy kaggle_encode_siglip2.py trên Kaggle GPU rồi giải nén vào đây.")
    return root, feat


def nap_du_lieu(config: Dict, **kwargs):
    """`load_dataset` với đúng hai thư mục mà config chỉ định."""
    root, feat = duong_du_lieu(config)
    return load_dataset(root, goc_feature=feat, **kwargs)


# --------------------------------------------------------------------------- #
# Thành phần
# --------------------------------------------------------------------------- #
def dung_encoder(config: Dict):
    """Encoder chữ theo `encoder:` — `build_encoder` đã kiểm chéo cache vs MODEL_ID."""
    return build_encoder(config)


def _encoder_tieng_viet(config: Dict, enc):
    """Text tower cho câu TIẾNG VIỆT, cùng không gian với vector ảnh đang dùng.

    SigLIP2 vốn đa ngữ nên chính nó là đường đúng. Với CLIP ViT-B/32 thì phải
    dùng mCLIP (đã căn vào cùng không gian ảnh 512 chiều) — gửi tiếng Việt thẳng
    vào CLIP gốc đã đo là xếp ngẫu nhiên (R@100 = 0.0%).
    """
    if isinstance(enc, SigLIP2TextEncoder):
        return enc
    if isinstance(enc, CLIPTextEncoder) and not isinstance(enc, MCLIPTextEncoder):
        cache = (config.get("encoder") or {}).get("vietnamese_cache_path")
        cache = cache if cache and Path(cache).is_file() else None
        return MCLIPTextEncoder(cache_path=cache, cho_phep_live=True)
    return None


def provider_vlm_dang_bat(config: Dict) -> List[Dict]:
    """Danh sách provider VLM sẽ thật sự được gọi, theo đúng thứ tự failover."""
    llm = config.get("llm") or {}
    kind = llm.get("provider", "rule_based")
    if kind == "failover_vlm":
        primary = llm.get("primary") or {}
        muc = [primary, *(llm.get("fallbacks") or [])]
        return [m for m in muc
                if m.get("enabled", True) and m.get("provider") in KHOA_MOI_TRUONG]
    if kind in KHOA_MOI_TRUONG and llm.get("enabled", True):
        return [llm]
    return []


def thieu_khoa_api(config: Dict) -> List[str]:
    """Provider nào đang bật mà thiếu biến môi trường khoá.

    Dùng để chặn TRƯỚC khi nạp 177.321 vector: thiếu khoá là lỗi cấu hình, và
    `Pipeline` cố tình ném thay vì âm thầm chạy QA rẻ suốt cả buổi thi.
    """
    ra = []
    for muc in provider_vlm_dang_bat(config):
        ten = muc.get("provider", "")
        thieu = [nhom for nhom in KHOA_MOI_TRUONG.get(ten, ())
                 if not any(os.environ.get(x) for x in nhom)]
        if thieu:
            ra.append(f"{ten}: cần "
                      + " và ".join(" hoặc ".join(nhom) for nhom in thieu))
    return ra


def canh_bao_cau_hinh(config: Dict) -> List[str]:
    """Các kết hợp cấu hình HỢP LỆ về cú pháp nhưng đã đo là sai trong thực tế.

    Không ném lỗi: đây là cảnh báo cho người vận hành, vì mỗi mục dưới đây có thể
    là lựa chọn có chủ ý. Nhưng phải IN RA, vì cả hai lỗi bên dưới đều thuộc loại
    "chạy trơn, điểm thấp, không có triệu chứng".
    """
    ra: List[str] = []
    qa = config.get("qa") or {}
    for muc in provider_vlm_dang_bat(config):
        if muc.get("provider") != "gemini_vlm":
            continue
        batch = int(muc.get("batch_max_images", 1) or 1)
        timeout = float(muc.get("timeout_seconds", 60) or 60)
        # Đo thật 16/08: batch 10 ảnh = 22,8s cho 1 request (~2,3s/ảnh). Lấy
        # 3s/ảnh + 10s dựng kết nối làm sàn an toàn.
        can = 3.0 * batch + 10.0
        if batch >= 2 and timeout < can:
            ra.append(
                f"gemini_vlm: batch_max_images={batch} nhưng timeout_seconds="
                f"{timeout:g} — đo thật batch 10 ảnh mất 22,8s, nên cần ≥ "
                f"{can:g}s. Quá hạn thì QA rơi xuống provider dự phòng YẾU HƠN "
                f"mà đáp án không có dấu hiệu gì.")
    if provider_vlm_dang_bat(config) and int(qa.get("vlm_top_n", 10) or 0):
        co_anh = bool(qa.get("image_roots")) or bool(
            qa.get("dev_image_dir") and Path(str(qa["dev_image_dir"])).is_dir())
        if not co_anh:
            ra.append(
                "qa: đã bật VLM nhưng `image_roots` rỗng và không có "
                "`dev_image_dir` — không có ảnh nào để trả lời, QA sẽ hạ xuống "
                "zero-shot (hoặc ném lỗi nếu tắt fallback).")
    port = config.get("portfolio") or {}
    if port.get("enabled") and not port.get("calibrator"):
        ra.append("portfolio: bật mà `calibrator: none` — optimizer không có xác "
                  "suất hiệu chỉnh thì phân bổ slot gần như vô nghĩa.")
    return ra


def dung_llm(config: Dict, qa_cheap: bool = False):
    """Provider trả lời QA. `qa_cheap=True` là ghi đè CÓ Ý THỨC về QA zero-shot."""
    if qa_cheap:
        return RuleBasedProvider()
    llm = config.get("llm") or {}
    if not llm or llm.get("provider", "rule_based") == "rule_based":
        return RuleBasedProvider()
    return build_llm(config)


def dung_image_resolver(config: Dict) -> EventImageResolver:
    """Resolver ảnh theo `qa.image_roots` + `qa.dev_image_dir`.

    Thư mục ảnh dev chỉ có trên máy làm nhãn; trên Kaggle nó không tồn tại nên
    thiếu nó là cảnh báo, không phải lỗi. Thiếu CẢ HAI thì `resolve()` sẽ ném
    `ImageResolutionError` ở đúng chỗ có ngữ cảnh, nên không chặn ở đây.
    """
    qa = config.get("qa") or {}
    roots = qa.get("image_roots") or []
    dev = qa.get("dev_image_dir")
    if dev and not Path(dev).is_dir():
        warnings.warn(
            f"`qa.dev_image_dir` không tồn tại, bỏ qua: {dev}", RuntimeWarning,
            stacklevel=2)
        dev = None
    return EventImageResolver(roots=roots, dev_image_dir=dev)


def dung_portfolio(config: Dict) -> PortfolioConfig:
    """`portfolio:` -> `PortfolioConfig`. Mặc định TẮT (chưa vượt 2·SE)."""
    p = config.get("portfolio") or {}
    bat = bool(p.get("enabled", False))
    kwargs: Dict = {}
    for khoa in ("kis_tol_frames", "trake_tol_frames", "trake_paths_per_video",
                 "n_answers"):
        if p.get(khoa) is not None:
            kwargs[khoa] = int(p[khoa])
    if p.get("mass") is not None:
        kwargs["mass"] = float(p["mass"])
    if p.get("trake_jitters"):
        kwargs["trake_jitters"] = tuple(int(x) for x in p["trake_jitters"])
    if p.get("frame_offsets"):
        kwargs["frame_offsets"] = tuple(int(x) for x in p["frame_offsets"])
    ten_calib = str(p.get("calibrator", "rank") or "").lower()
    if bat and ten_calib in ("rank", "rankcalibrator"):
        kwargs["calibrator"] = RankCalibrator()
    elif bat and ten_calib not in ("", "none", "khong"):
        raise CauHinhSai(f"`portfolio.calibrator` không biết: {ten_calib!r} "
                         "(có: rank | none)")
    return PortfolioConfig(enabled=bat, **kwargs)


def chinh_sach_truy_hoi(config: Dict) -> RetrievalPolicy:
    r = config.get("retrieval") or {}
    hn = r.get("hop_nhat") or {}
    return RetrievalPolicy(
        pool_per_query=int(r.get("pool_per_query", 300)),
        candidate_videos=int(r.get("candidate_videos", 20)),
        output_top_k=int(r.get("top_k", 100)),
        fuse_pool=int(hn.get("pool", 500)),
        fuse_rrf_k=float(hn.get("rrf_k", 60.0)),
        fuse_trong_so_chinh=float(hn.get("trong_so_chinh", 2.0)))


def nap_hop_nhat(config: Dict, events) -> List[Tuple[object, float]]:
    """Dựng các retriever PHỤ cho `retrieval.hop_nhat` (việc 4.4a).

    Mỗi mục là một checkpoint KHÁC, với thư mục vector ảnh và cache truy vấn
    riêng. Tốn thêm RAM đúng bằng ma trận của nó (base-224: ~545 MB cho 177.321
    khung) và ~5 giây nạp; đổi lại KIS 0.3400 -> 0.3750 (Δ +0.0350 ± 0.0127).

    Retriever phụ được index bằng CHÍNH events của nó, không phải `events` của
    đường chính — nếu hai bên lệch thứ tự thì vector sẽ gắn sai khung mà không có
    triệu chứng. Hợp nhất về sau chỉ nối theo `event_id`, nên lệch thứ tự không
    thành vấn đề. `events` truyền vào chỉ để kiểm số lượng.
    """
    r = config.get("retrieval") or {}
    hn = r.get("hop_nhat") or {}
    if not hn.get("enabled"):
        return []
    from .core.retrieval import MultiChannelRetriever

    ra: List[Tuple[object, float]] = []
    for muc in hn.get("phu") or []:
        if muc.get("enabled", True) is False:
            continue
        goc_feature = muc.get("image_feature_root")
        if not goc_feature or not Path(str(goc_feature)).is_dir():
            # Thiếu bộ vector ảnh của checkpoint phụ thì BỎ NÓ, không im lặng
            # dùng vector của checkpoint khác — trộn hai không gian là sai mà
            # không có triệu chứng.
            warnings.warn(
                f"hop_nhat: bỏ {muc.get('model_id')} — chưa có thư mục vector "
                f"ảnh ({goc_feature or 'chưa đặt image_feature_root'}). "
                f"Đặt nó trong config.yaml.", RuntimeWarning, stacklevel=2)
            continue
        con = {"encoder": {k: muc[k] for k in
                           ("provider", "model_id", "cache_path", "live", "dim")
                           if k in muc}}
        enc = build_encoder(con)
        root, _ = duong_du_lieu(config)
        e2, emb2, _, rep2 = load_dataset(root, goc_feature=goc_feature)
        if emb2.size and enc.dim is not None and emb2.shape[1] != enc.dim:
            raise CauHinhSai(
                f"hop_nhat {muc.get('model_id')}: vector ảnh {emb2.shape[1]} "
                f"chiều nhưng vector chữ {enc.dim} chiều — hai bên phải cùng một "
                f"checkpoint.")
        if events is not None and rep2.events != len(list(events)):
            warnings.warn(
                f"hop_nhat: encoder phụ có {rep2.events} event, đường chính có "
                f"{len(list(events))} — hợp nhất theo event_id nên vẫn chạy, "
                f"nhưng nên kiểm lại hai bộ vector.", RuntimeWarning,
                stacklevel=2)
        retriever = MultiChannelRetriever(enc)
        retriever.visual = NumpyVectorStore(
            khu_modality_gap=bool(muc.get(
                "khu_modality_gap",
                (config.get("retrieval") or {}).get("khu_modality_gap", True))))
        retriever.index(e2, embeddings=emb2)
        ra.append((retriever, float(muc.get("trong_so", 1.0))))
    return ra


def dp_params(config: Dict) -> DPParams:
    t = config.get("temporal_dp") or {}
    kwargs = {khoa: float(t[khoa]) for khoa in
              ("lam", "gap_max", "penalty_miss",
               "modality_confidence_threshold", "miss_factor")
              if t.get(khoa) is not None}
    return DPParams(**kwargs)


def so_dong_nop(config: Dict) -> int:
    return int((config.get("online") or {}).get("max_submission_rows", 100))


def ngon_ngu_truy_van(config: Dict) -> str:
    """Bản ngôn ngữ nào của đề được dùng để TRUY HỒI (`truy_van.ngon_ngu`).

    Không phải khoá trang trí: đo trên dev set 120 câu KIS, SigLIP2 base-224,
    bản tiếng Việt ra 0.1750 còn bản tiếng Anh ra 0.2950 (R@1 2,4% so với 8,8%).
    Hệ thống KHÔNG tự phát hiện chuyện đó lúc chạy — câu tiếng Việt được
    `ChonEncoderTheoNgonNgu` định tuyến về chính encoder đang dùng và chạy trơn
    tru. Nên lựa chọn phải nằm ở config, và phải được in ra trước khi nộp.
    """
    ng = str((config.get("truy_van") or {}).get("ngon_ngu", "en")).lower()
    if ng not in NGON_NGU_HOP_LE:
        raise CauHinhSai(
            f"`truy_van.ngon_ngu` không biết: {ng!r} "
            f"(có: {' | '.join(NGON_NGU_HOP_LE)})")
    return ng


def _sinh_ung_vien(config: Dict):
    """Bộ sinh đáp án ứng viên cho QA zero-shot, theo `qa.answer_candidates`.

    `auto` (mặc định) = giữ nguyên hành vi cũ: `core.qa.dung_sinh_ung_vien()`
    chọn theo BIẾN MÔI TRƯỜNG (có `ANTHROPIC_API_KEY` thì gọi Claude sinh ứng
    viên). Ghi ra đây để lựa chọn đó thấy được trong config chứ không phải một
    tác dụng phụ của môi trường — nhưng KHÔNG đổi mặc định, vì mọi con số QA cheap
    đã đo đều đo với `auto`.

    `luat` = chỉ bảng luật, offline, không tốn API. Đáng chọn khi cần tái lập
    hoặc khi `answer_translation: bang_tay` (ứng viên do LLM sinh ra thường nằm
    ngoài bảng tay nên sẽ giữ nguyên tiếng Anh).
    """
    che_do = str((config.get("qa") or {}).get("answer_candidates") or "auto").lower()
    if che_do == "auto":
        return None
    from .core.qa import SinhBangLLM, SinhTheoLuat
    if che_do in ("luat", "rule", "rules"):
        return SinhTheoLuat()
    if che_do in ("llm", "claude"):
        return SinhBangLLM()
    raise CauHinhSai(f"`qa.answer_candidates` không biết: {che_do!r} "
                     "(có: auto | luat | llm)")


def _visual_qa(config: Dict, enc):
    """QA zero-shot dựng TAY khi config đòi khác mặc định của `Pipeline`.

    Trả `None` = để `Pipeline` tự dựng đúng như hôm nay (đường đã đo). Chỉ dựng
    tay khi config nói rõ một trong hai điều:

      * `answer_translation: bang_tay` — chỉ bảng tay màu/số, không nạp model
        dịch. Rẻ, chạy offline, nhưng câu MỞ giữ nguyên tiếng Anh.
      * `answer_candidates` khác `auto` — chốt bộ sinh ứng viên thay vì để biến
        môi trường quyết.
    """
    qa = config.get("qa") or {}
    che_do = str(qa.get("answer_translation") or "opus_mt").lower()
    if che_do in ("opus_mt", "opusmt", "opus-mt"):
        bang_tay = False
    elif che_do in ("bang_tay", "bangtay", "hand_table"):
        bang_tay = True
    else:
        raise CauHinhSai(f"`qa.answer_translation` không biết: {che_do!r} "
                         "(có: opus_mt | bang_tay)")
    sinh = _sinh_ung_vien(config)
    if not bang_tay and sinh is None:
        return None
    if not isinstance(enc, CLIPTextEncoder):
        return None
    from .core.qa import ClipZeroShotQA, DichDapAnVi, dung_sinh_ung_vien
    vi = str(qa.get("output_language", "vi")).lower() == "vi"
    dich = None
    if vi:
        dich = DichDapAnVi(None if bang_tay else OpusMTEnViTranslator())
    return ClipZeroShotQA(enc, sinh or dung_sinh_ung_vien(), dich_dap_an=dich)


# --------------------------------------------------------------------------- #
# Hệ thống
# --------------------------------------------------------------------------- #
def _mot_kenh_chu(kc: Dict):
    """Một mục cấu hình -> `(kho, trọng số)` hoặc `(None, 0.0)`.

    KHÔNG BAO GIỜ ném ra ngoài. Kênh chữ là bằng chứng BỔ SUNG; biến nó thành
    điều kiện để hệ thống khởi động được là tự tạo một điểm chết mới giữa buổi
    thi. Thiếu file thì cảnh báo TO rồi chạy tiếp bằng encoder ảnh.
    """
    if not kc.get("bat"):
        return None, 0.0
    duong_dan = kc.get("duong_dan")
    ten = str(kc.get("ten") or kc.get("kieu") or "kênh chữ")
    if not duong_dan:
        warnings.warn(f"{ten}: bat=true nhưng thiếu `duong_dan` — BỎ QUA.",
                      RuntimeWarning, stacklevel=3)
        return None, 0.0
    trong_so = float(kc.get("trong_so", 2.0))

    # `kieu: fts` — OCR theo TỪNG KEYFRAME từ Stage11 (`ocr_fts.sqlite`), khớp
    # 1:1 với 177.321 event. Khác `kieu: khoang` (Stage20) vốn theo khoảng shot.
    if str(kc.get("kieu", "khoang")).lower() == "fts":
        from .retrieval.kho_ocr_fts import CauHinhOcrFts, KhoOcrFts
        cfg = CauHinhOcrFts(
            bat=True, duong_dan=str(duong_dan),
            bang=str(kc.get("bang", "OCR_ALL")), trong_so=trong_so,
            pool=int(kc.get("pool", 300)),
            bo_so_ngan=bool(kc.get("bo_so_ngan", True)))
        try:
            kho = KhoOcrFts(cfg.duong_dan, cfg)
        except Exception as exc:                              # noqa: BLE001
            warnings.warn(f"{ten} KHÔNG nạp được ({exc}) — chạy tiếp không có "
                          f"nó.", RuntimeWarning, stacklevel=3)
            return None, 0.0
        print(f"  {ten} (w={trong_so}): {kho.bao_cao.mo_ta()}")
        return kho, trong_so

    from .retrieval.kenh_chu import CauHinhKenhChu, KhoChuKhoang
    cfg = CauHinhKenhChu(
        bat=True, duong_dan=str(duong_dan), trong_so=trong_so,
        pool_khoang=int(kc.get("pool_khoang", 150)),
        toi_da_event_moi_khoang=int(kc.get("toi_da_event_moi_khoang", 30)),
        nguong_diem=float(kc.get("nguong_diem", 0.15)),
        chuan_hoa_diem=str(kc.get("chuan_hoa_diem", "truy_van")),
        dung_ocr=bool(kc.get("dung_ocr", True)),
        dung_asr=bool(kc.get("dung_asr", True)),
        dung_sai_giay=float(kc.get("dung_sai_giay", 0.0)))
    try:
        kho = KhoChuKhoang.tu_file(duong_dan, cfg)
    except Exception as exc:                                  # noqa: BLE001
        warnings.warn(f"{ten} KHÔNG nạp được ({exc}) — chạy tiếp không có nó.",
                      RuntimeWarning, stacklevel=3)
        return None, 0.0
    print(f"  {ten} (w={trong_so}): {kho.bao_cao.mo_ta()}")
    for c in kho.bao_cao.canh_bao:
        print(f"    ⚠ {c}")
    return kho, trong_so


def nap_kenh_chu(config: Dict):
    """`retrieval.kenh_chu` -> DANH SÁCH `[(kho, trọng số)]`.

    Nhận cả hai dạng: MỘT mapping (một kênh) hoặc DANH SÁCH mapping (nhiều
    kênh). Nhiều kênh vì OCR và ASR là hai loại bằng chứng khác nhau — chữ in
    trên màn hình so với lời nói — và xứng đáng hai trọng số riêng.
    """
    kc = (config.get("retrieval") or {}).get("kenh_chu")
    if not kc:
        return []
    muc = kc if isinstance(kc, list) else [kc]
    ra = []
    for m in muc:
        if not isinstance(m, dict):
            continue
        kho, w = _mot_kenh_chu(m)
        if kho is not None and w:
            ra.append((kho, w))
    return ra


def dung_don_slot(config: Dict):
    """`retrieval.don_slot` -> `CauHinhDonSlot`. Mặc định TẮT."""
    from .optimize.don_slot import CauHinhDonSlot
    d = (config.get("retrieval") or {}).get("don_slot") or {}
    hn = d.get("han_ngach")
    return CauHinhDonSlot(
        bat=bool(d.get("bat", False)),
        han_ngach=tuple(int(x) for x in hn) if hn else CauHinhDonSlot.han_ngach,
        cach_chon=str(d.get("cach_chon", "tron")).lower(),
        toi_thieu_ung_vien=int(d.get("toi_thieu_ung_vien", 20)))


def dung_pipeline(config: Dict, events, embeddings=None, *, encoder=None,
                  llm=None, portfolio: Optional[PortfolioConfig] = None,
                  qa_cheap: bool = False,
                  retrieval_phu: Optional[Sequence] = None) -> Pipeline:
    """Dựng `Pipeline` ĐÚNG như config mô tả, rồi index luôn.

    Đây là điểm nối duy nhất giữa file YAML và hệ thống. Mọi đường nộp bài phải
    đi qua đây; script nào tự gọi `Pipeline(...)` là script đó có thể trôi khỏi
    cấu hình thi mà không ai thấy (đúng lỗi 1.5 ngày 15/08).

    `qa_cheap` và các tham số ghi đè khác chỉ để phục vụ thử nghiệm/không có khoá
    API — chúng được in ra trong `mo_ta_cau_hinh` để không lẫn với cấu hình thật.
    """
    enc = encoder if encoder is not None else dung_encoder(config)
    if embeddings is not None and getattr(embeddings, "size", 0):
        dim_anh = int(embeddings.shape[1])
        if enc.dim is not None and dim_anh != enc.dim:
            raise CauHinhSai(
                f"Vector ảnh {dim_anh} chiều, vector chữ {enc.dim} chiều.\n"
                "  Hai bên phải cùng MỘT checkpoint. Kiểm `encoder.model_id` và\n"
                "  `offline.image_feature_root` trong config, và MODEL_ID lúc\n"
                "  chạy kaggle_encode_siglip2.py.")

    qa = config.get("qa") or {}
    provider = llm if llm is not None else dung_llm(config, qa_cheap=qa_cheap)
    _kc = nap_kenh_chu(config)
    can_anh = bool(getattr(provider, "supports_images", False))
    p = Pipeline(
        enc, provider,
        dp_params=dp_params(config),
        fusion=str((config.get("retrieval") or {}).get("fusion", "rrf")),
        portfolio=portfolio if portfolio is not None else dung_portfolio(config),
        visual_qa=_visual_qa(config, enc),
        vietnamese_encoder=_encoder_tieng_viet(config, enc),
        retrieval_policy=chinh_sach_truy_hoi(config),
        # Truyền `retrieval_phu` để A/B hai cấu hình dùng CHUNG một bộ vector phụ:
        # gọi `nap_hop_nhat` hai lần là nạp lại 545 MB lần nữa cho không.
        retrieval_phu=(nap_hop_nhat(config, events) if retrieval_phu is None
                       else retrieval_phu),
        kenh_chu=_kc,
        don_slot_cfg=dung_don_slot(config),
        qa_output_language=str(qa.get("output_language", "vi")).lower(),
        qa_answer_translator=(OpusMTEnViTranslator()
                              if str(qa.get("output_language", "vi")).lower() == "vi"
                              else None),
        image_resolver=dung_image_resolver(config) if can_anh else None,
        vlm_fallback_to_zero_shot=bool(qa.get("vlm_fallback_to_zero_shot", True)),
        qa_vlm_top_n=qa.get("vlm_top_n", 10),
        qa_vlm_question_types=qa.get("vlm_question_types", ("color", "open")),
        qa_tail_answer=str(qa.get("tail_answer", "vlm")).lower())
    p.retriever.visual = NumpyVectorStore(
        khu_modality_gap=bool(
            (config.get("retrieval") or {}).get("khu_modality_gap", True)))
    p.index(events, embeddings=embeddings)
    return p


def mo_ta_cau_hinh(config: Dict, *, qa_cheap: bool = False) -> List[str]:
    """Vài dòng để script in ra TRƯỚC khi chạy — nhìn là biết đang nộp bằng gì."""
    enc = config.get("encoder") or {}
    qa = config.get("qa") or {}
    port = config.get("portfolio") or {}
    vlm = [m.get("provider", "?") + "/" + str(m.get("model", "?"))
           for m in provider_vlm_dang_bat(config)]
    _, feat = (None, (config.get("offline") or {}).get("image_feature_root"))
    dong = [
        f"config          : {config.get('_nguon_config', '(dict trực tiếp)')}",
        f"encoder         : {enc.get('provider', 'hash')} "
        f"{enc.get('model_id', '')}".rstrip(),
        f"vector ảnh      : {feat or (config.get('offline') or {}).get('btc_data_root')}",
        f"đáp án QA       : {str(qa.get('output_language', 'vi')).lower()} "
        f"(dịch: {qa.get('answer_translation') or 'opus_mt'})",
        f"QA VLM          : {' -> '.join(vlm) if vlm else 'không (QA zero-shot)'}"
        + ("  [BỊ GHI ĐÈ --qa-cheap]" if qa_cheap and vlm else ""),
        f"vlm_top_n       : {qa.get('vlm_top_n', 10)} · loại câu: "
        f"{list(qa.get('vlm_question_types', ('color', 'open')))}",
        f"portfolio       : {'BẬT' if port.get('enabled') else 'tắt'}",
        f"ngôn ngữ đề     : {ngon_ngu_truy_van(config)} (bản dùng để truy hồi)",
        f"số dòng/truy vấn: {so_dong_nop(config)}",
    ]
    if config.get("_nguon_ke_thua"):
        dong.insert(1, "kế thừa từ     : "
                    + " <- ".join(str(x) for x in config["_nguon_ke_thua"]))
    return dong
