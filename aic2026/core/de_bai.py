"""Đọc ĐỀ THI: chọn ngôn ngữ truy vấn, và tách chuỗi sự kiện TRAKE.

VÌ SAO MODULE NÀY TỒN TẠI
-------------------------
Ngày thi, BTC cấp đề dưới dạng một file. Hai chi tiết của file đó có thể xoá
điểm mà không để lại một triệu chứng nào nhìn thấy được:

1. NGÔN NGỮ TRUY VẤN
   Đề có cả bản tiếng Việt lẫn tiếng Anh. Mọi con số đã đo của hệ thống (KIS
   0.3750) đo trên bản TIẾNG ANH. Đo lại đúng dev set đó bằng bản tiếng Việt,
   SigLIP2 base-224: **0.1750 so với 0.2950 — mất 41%** (R@1 8,8% -> 2,4%).

   Nhưng `Pipeline` KHÔNG báo lỗi khi nhận tiếng Việt: `ChonEncoderTheoNgonNgu`
   định tuyến câu tiếng Việt về chính encoder đang dùng (xem
   `config._encoder_tieng_viet`), nên hệ thống chạy trơn tru và chỉ có điểm là
   thấp. Bộ đọc cũ lấy văn bản bằng `q.get("text") or q.get("_mo_ta_vi")` —
   nếu file đề của BTC đặt bản tiếng Việt dưới đúng khoá tên `text` thì nó lấy
   bản tiếng Việt, không nói gì, và mất 41% điểm KIS.

   => Ở đây chọn CÓ Ý THỨC theo chính sách (`truy_van.ngon_ngu`), và trả về bản
   ghi để nơi gọi IN RA đã dùng bản nào cho từng câu.

2. CHUỖI SỰ KIỆN TRAKE
   Bộ đọc cũ làm `for x in q["sub_events"]`. Nếu giá trị là một CHUỖI như
   ``"a → b"`` thì Python lặp theo KÝ TỰ: ra 5 sub-event ``a``, ``" "``, ``→``,
   ``" "``, ``b``. Không ném lỗi. `n_events` sai kéo theo mọi dòng TRAKE sai số
   cột, và theo luật BTC thì sai số khoảnh khắc là R-Score = 0 cho cả truy vấn.

   => `tach_sub_events` nhận cả hai dạng, và ném lỗi khi không tách được thay vì
   trả về rác.

NHẬN DẠNG NGÔN NGỮ: TÊN KHOÁ THẮNG NỘI DUNG
-------------------------------------------
Đoán theo DẤU tiếng Việt (`co_dau_tieng_viet`) sai được ở CẢ HAI chiều:

  * tiếng Việt viết không dấu bị xếp thành 'en'
    ``{"text": "Mot nguoi nong dan...", "text_en": "A farmer..."}``
  * tiếng Anh có danh từ riêng Việt bị xếp thành 'vi' — và đây không phải
    trường hợp hiếm với bộ dữ liệu này: "a metro train at Nhổn station" có
    dấu ổ, "Pavel Bittner" thì không, nhưng "Phú Quốc"/"Đà Nẵng" thì có.

Nên khi tên khoá đã nói rõ ngôn ngữ (`text_en`, `_mo_ta_vi`, `query_vi`) thì
`the_ngon_ngu` thắng, và giữa các ứng viên cùng ngôn ngữ thì khoá gắn nhãn được
ưu tiên hơn khoá trung tính (`text`). Chỉ khi khoá trung tính thì mới đoán theo
dấu.

Đó vẫn là phỏng đoán. Vì vậy mọi hàm ở đây trả về `BanGhiNgonNgu` để nơi gọi
IN RA đã dùng bản nào cho từng câu — đừng bỏ phần in ra đó, nó là thứ duy nhất
biến một lỗi câm thành một lỗi nhìn thấy được.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Sequence, Tuple

from .dich import co_dau_tieng_viet

NGON_NGU_HOP_LE = ("en", "vi", "tu_dong")

# Thứ tự = độ ưu tiên khi nhiều khoá cùng có mặt. Đặt các tên trung tính
# (`text`, `query`) TRƯỚC vì đó là tên hay gặp nhất; ngôn ngữ không suy ra từ
# tên khoá mà suy ra từ nội dung, nên thứ tự này chỉ quyết định khi hoà.
KHOA_VAN_BAN: Tuple[str, ...] = (
    "text", "text_en", "text_vi",
    "query", "query_en", "query_vi",
    "description", "description_en", "description_vi",
    "mo_ta", "mo_ta_en", "mo_ta_vi", "_mo_ta_vi", "_mo_ta_en",
    "cau_truy_van", "noi_dung",
)
KHOA_CAU_HOI: Tuple[str, ...] = (
    "question_vi", "cau_hoi_vi", "_cau_hoi_vi",
    "question", "question_en", "cau_hoi", "_cau_hoi",
)
KHOA_SU_KIEN: Tuple[str, ...] = (
    "sub_events", "subevents", "events", "su_kien", "cac_su_kien",
    "khoanh_khac", "moments",
)

# Khoá KHÔNG bao giờ là văn bản truy vấn, dùng cho phép quét dự phòng bên dưới.
#
# 🔴 `ten_file` thêm 2026-08-20 sau khi nó XOÁ SẠCH một lần chạy đề thật. Bộ đọc
# `doc_de_txt` ghi `"ten_file": "query-p1-1-kis"` vào mỗi câu. Chuỗi đó dài 14 ký
# tự (ngưỡng dự phòng là 12) và không có dấu tiếng Việt, nên `ngon_ngu_cua` xếp
# nó là 'en'; chính sách cũng là 'en'; nó THẮNG bản mô tả tiếng Việt thật. Cả 21
# câu KIS/QA đi tìm trên 177.321 khung bằng đúng chuỗi "query-p1-1-kis".
# Không có gì nổ, zip vẫn hợp lệ, chỉ có điểm là rác.
# Vì sao chưa bao giờ lộ: tên năm ngoái `query-1-kis` dài 11 ký tự — DƯỚI ngưỡng
# 12. Mã gói `p1` của năm nay đẩy nó qua ngưỡng. Đó là may, không phải thiết kế.
KHOA_BO_QUA = frozenset({
    "qid", "id", "task", "type", "loai", "answer", "dap_an", "gt",
    "ground_truth", "video_id", "frame_id",
    "ten_file", "ten_tep", "file", "filename", "ten", "name", "n_events",
})

# Dấu tách chuỗi sự kiện TRAKE. KHÔNG dùng dấu phẩy: một sự kiện đơn lẻ hoàn
# toàn có thể chứa dấu phẩy ("người đàn ông, mặc áo đỏ, bước vào"), tách theo
# phẩy sẽ băm nhỏ đúng thứ ta cần giữ nguyên.
DAU_TACH = re.compile(r"\s*(?:→|->|=>|;|\||\r?\n)\s*")

# Tên file BTC dùng cho từng truy vấn: `query-<N>-<task>`. Nguồn: hướng dẫn nộp
# bài AIC2025 (`Tài liệu/info cuộc thi/info-aic25.md`) — BTC phát đề thành các
# file `query-1-kis.txt`, `query-3-qa.txt`... và bài nộp phải là file CSV TRÙNG
# TÊN đặt trong thư mục `submission/`.
#
# 🔴 SỬA 2026-08-20, có bằng chứng từ bộ đề THẬT (vòng thử nghiệm AIC2026,
# `data cuộc thi/THUNGHIEM-bo-de-thi/`): BTC chèn thêm MÃ GÓI vào giữa —
# `query-p1-1-kis.txt`, `query-p1-16-trake.txt`. Mẫu cũ không khớp nên cả 24/24
# câu bị báo "LỆCH". Đó là cảnh báo SAI, và cảnh báo sai còn nguy hơn không có
# cảnh báo: nó dạy người đọc bỏ qua đúng dòng lẽ ra phải chặn họ lại. Nhóm gói
# để tuỳ chọn nên cả hai dạng (`query-1-kis`, `query-p1-1-kis`) đều khớp.
MAU_TEN_BTC = re.compile(
    r"(?i)^query[-_](?:(?P<goi>[a-z][a-z0-9]*)[-_])?"
    r"(?P<so>\d+)[-_](?P<loai>kis|qa|trake)$")


@dataclass
class BanGhiNgonNgu:
    """Đã dùng bản nào cho một truy vấn — để in ra, không phải để quyết định."""
    qid: str
    khoa: str
    ngon_ngu: str            # 'en' | 'vi'
    theo_chinh_sach: bool
    canh_bao: str = ""
    # Chuỗi THẬT SỰ đem đi truy hồi, cắt ngắn. Thêm 2026-08-20: hôm đó hệ thống
    # đi tìm bằng `"query-p1-1-kis"` cho cả 21 câu và bản in vẫn xanh, vì nó chỉ
    # ghi TÊN KHOÁ chứ không ghi NỘI DUNG. Tên khoá `'ten_file'` không nói lên
    # điều gì với mắt người đang đọc lướt 24 dòng; nội dung thì nói ngay.
    xem_truoc: str = ""

    def mo_ta(self) -> str:
        dau = "✓" if self.theo_chinh_sach else "⚠"
        dong = f"{dau} {self.qid:18s} {self.ngon_ngu}  (khoá {self.khoa!r})"
        if self.xem_truoc:
            dong += f"  « {self.xem_truoc} »"
        return dong


def ngon_ngu_cua(text: str) -> str:
    """'vi' nếu có dấu tiếng Việt, ngược lại 'en'. Xem giới hạn ở docstring."""
    return "vi" if co_dau_tieng_viet(str(text)) else "en"


def the_ngon_ngu(khoa: str) -> str:
    """Ngôn ngữ suy từ TÊN KHOÁ (`text_en`, `_mo_ta_vi`), hoặc '' nếu trung tính.

    Tên khoá đáng tin hơn phép đoán theo dấu, vì phép đoán mù với tiếng Việt
    viết không dấu: `{"text": "Mot nguoi nong dan...", "text_en": "A farmer..."}`
    bị xếp cả hai thành 'en', và nếu chỉ xét nội dung thì `text` (đứng trước)
    thắng — đúng bản KHÔNG nên dùng.
    """
    k = str(khoa).lower()
    for ng in ("en", "vi"):
        if k.endswith("_" + ng):
            return ng
    return ""


def ngon_ngu_hieu_luc(khoa: str, text: str) -> str:
    """Tên khoá thắng nội dung; không có tên gắn nhãn thì mới đoán theo dấu."""
    return the_ngon_ngu(khoa) or ngon_ngu_cua(text)


def _sach(gia_tri: Any) -> str:
    return str(gia_tri).strip() if isinstance(gia_tri, str) else ""


def _xem_truoc(text: str, n: int = 56) -> str:
    """Cắt ngắn để in kèm bản ghi — đủ để mắt người nhận ra nhầm trường."""
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[:n - 1] + "…"


def _ung_vien_theo_khoa(q: Dict, khoa: Sequence[str]
                        ) -> Iterator[Tuple[str, str, str]]:
    for k in khoa:
        t = _sach(q.get(k))
        if t:
            yield (k, t, ngon_ngu_hieu_luc(k, t))


def _ung_vien_du_phong(q: Dict, da_xet: Sequence[str]
                       ) -> Iterator[Tuple[str, str, str]]:
    """Quét các khoá LẠ — phòng khi BTC đặt tên trường khác mọi dự đoán.

    Kết quả luôn xếp SAU các khoá đã biết, và luôn được báo ra ngoài qua
    `BanGhiNgonNgu` để người vận hành thấy hệ thống đã bám vào trường nào.

    HAI ĐIỀU KIỆN, không phải một
    -----------------------------
    Ngưỡng 12 ký tự MỘT MÌNH là không đủ, và đã hỏng thật ngày 20/08: khoá
    `ten_file` = ``"query-p1-1-kis"`` dài 14 ký tự nên lọt, rồi thắng cả bản mô
    tả thật (xem `KHOA_BO_QUA`). Chặn theo danh sách tên khoá là bịt một lỗ; lỗ
    tiếp theo là khoá lạ nào đó chưa ai nghĩ ra.

    Nên thêm điều kiện theo HÌNH DẠNG giá trị: một mô tả cảnh luôn là một CÂU —
    ít nhất 3 từ. Mã, đường dẫn, nhãn, tên file thì không. Điều kiện này không
    phụ thuộc vào việc đoán trước tên khoá của BTC.
    """
    bo = set(da_xet) | KHOA_BO_QUA | set(KHOA_SU_KIEN)
    for k, v in q.items():
        if k in bo or k.startswith("_ghi_chu"):
            continue
        t = _sach(v)
        if len(t) >= 12 and len(t.split()) >= 3:
            yield (k, t, ngon_ngu_hieu_luc(k, t))


def ung_vien_van_ban(q: Dict) -> List[Tuple[str, str, str]]:
    """[(khoá, văn bản, ngôn ngữ)] theo thứ tự ưu tiên, đã bỏ khoá rỗng."""
    ra = list(_ung_vien_theo_khoa(q, KHOA_VAN_BAN))
    ra += list(_ung_vien_du_phong(q, [k for k, _, _ in ra] + list(KHOA_VAN_BAN)
                                  + list(KHOA_CAU_HOI)))
    return ra


def chon_van_ban(q: Dict, chinh_sach: str = "en", qid: str = ""
                 ) -> Tuple[str, BanGhiNgonNgu]:
    """Chọn văn bản truy hồi theo chính sách ngôn ngữ.

    Không bao giờ im lặng: khi phải đi chệch chính sách (đề chỉ có một bản),
    `BanGhiNgonNgu.canh_bao` được điền và `theo_chinh_sach=False`.
    """
    if chinh_sach not in NGON_NGU_HOP_LE:
        raise ValueError(
            f"chính sách ngôn ngữ không biết: {chinh_sach!r} "
            f"(có: {' | '.join(NGON_NGU_HOP_LE)})")
    ung_vien = ung_vien_van_ban(q)
    if not ung_vien:
        raise ValueError(
            f"{qid or q.get('qid', '?')}: không tìm thấy trường văn bản nào.\n"
            f"  Khoá đã thử: {', '.join(KHOA_VAN_BAN)}\n"
            f"  Khoá có trong câu này: {', '.join(sorted(map(str, q)))}")

    if chinh_sach == "tu_dong":
        k, t, ng = ung_vien[0]
        return t, BanGhiNgonNgu(qid, k, ng, True, xem_truoc=_xem_truoc(t))

    hop = [x for x in ung_vien if x[2] == chinh_sach]
    if hop:
        # Khoá GẮN NHÃN đứng trước khoá trung tính (sort ổn định giữ thứ tự cũ
        # trong từng nhóm): `text_en` đáng tin hơn `text` kể cả khi phép đoán
        # theo dấu đã xếp cả hai vào cùng một ngôn ngữ.
        hop.sort(key=lambda x: 0 if the_ngon_ngu(x[0]) == chinh_sach else 1)
        k, t, ng = hop[0]
        return t, BanGhiNgonNgu(qid, k, ng, True, xem_truoc=_xem_truoc(t))

    k, t, ng = ung_vien[0]
    them = ("  Đo 120 câu KIS, so400m + hợp nhất: bản tiếng Việt 0.2883 so với "
            "0.3750 của bản người gõ tay. `truy_van.viet_lai` bù lại phần lớn "
            "(0.3450) — kiểm xem nó có bật không." if chinh_sach == "en"
            else "")
    return t, BanGhiNgonNgu(
        qid, k, ng, False, xem_truoc=_xem_truoc(t),
        canh_bao=(f"{qid}: chính sách `{chinh_sach}` nhưng đề chỉ có bản "
                  f"`{ng}` (khoá {k!r}) — đang dùng bản đó.{them}"))


def chon_cau_hoi(q: Dict) -> str:
    """Câu hỏi QA gửi cho VLM. Ưu tiên bản TIẾNG VIỆT, không theo `chinh_sach`.

    Lý do tách khỏi `chon_van_ban`: câu hỏi không tham gia chấm cosine (xem
    `Query.question_vi`), nên chọn bản tiếng Anh cho nó không đem lại gì, trong
    khi đáp án tiếng Việt đúng ngôn ngữ đầu ra thì đỡ một vòng dịch.
    """
    for _, t, ng in _ung_vien_theo_khoa(q, KHOA_CAU_HOI):
        if ng == "vi":
            return t
    for _, t, _ in _ung_vien_theo_khoa(q, KHOA_CAU_HOI):
        return t
    return ""


def _mot_su_kien(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        text = _sach(x.get("text") or x.get("mo_ta") or x.get("noi_dung"))
        d: Dict[str, Any] = {"text": text}
        if x.get("modality_hint"):
            d["modality_hint"] = x["modality_hint"]
        return d
    return {"text": _sach(x) or str(x).strip()}


def tach_sub_events(gia_tri: Any, qid: str = "") -> List[Dict[str, Any]]:
    """Chuẩn hoá `sub_events` về [{'text': ...}, ...].

    Nhận CHUỖI (tách theo ``→ -> => ; |`` hoặc xuống dòng) lẫn danh sách. Ném
    lỗi thay vì trả rác — xem mục 2 ở docstring module: để lọt một chuỗi vào
    vòng lặp cũ là ra một sub-event cho MỖI KÝ TỰ, im lặng.
    """
    ten = qid or "?"
    if gia_tri is None or (isinstance(gia_tri, str) and not gia_tri.strip()):
        raise ValueError(f"{ten}: TRAKE không có `sub_events`")

    if isinstance(gia_tri, str):
        phan = [p for p in (x.strip() for x in DAU_TACH.split(gia_tri)) if p]
        if len(phan) < 2:
            raise ValueError(
                f"{ten}: `sub_events` là một chuỗi không tách được thành nhiều "
                f"sự kiện: {gia_tri[:80]!r}\n"
                f"  Dấu tách nhận: → -> => ; | và xuống dòng.\n"
                f"  Sửa file đề thành danh sách, hoặc chèn dấu tách. ĐỪNG để "
                f"chuỗi đi tiếp: lặp một chuỗi trong Python là lặp theo KÝ TỰ.")
        return [{"text": p} for p in phan]

    if isinstance(gia_tri, (list, tuple)):
        tho = list(gia_tri)
        # Chữ ký của lỗi lặp-theo-ký-tự, phòng khi rác đến từ nơi khác. Kiểm
        # TRƯỚC khi lọc: `list("a → b")` có 5 phần tử nhưng hai phần tử khoảng
        # trắng bị lọc mất, còn 3 — lọc xong mới kiểm là để lọt.
        if len(tho) >= 4 and all(isinstance(x, str) and len(x) <= 1
                                 for x in tho):
            raise ValueError(
                f"{ten}: `sub_events` gồm {len(tho)} phần tử đều dài <= 1 ký tự "
                f"— gần như chắc chắn là một chuỗi đã bị lặp theo ký tự.")
        ra = [d for d in (_mot_su_kien(x) for x in tho) if d.get("text")]
        if not ra:
            raise ValueError(f"{ten}: `sub_events` rỗng sau khi chuẩn hoá")
        return ra

    raise ValueError(
        f"{ten}: `sub_events` phải là chuỗi hoặc danh sách, "
        f"nhận {type(gia_tri).__name__}")


def ten_file_nop(qid: str, ten_file: str = "") -> str:
    """Tên file CSV cho một truy vấn. PHẢI trùng tên file đề của BTC.

    Codabench ghép file nộp với truy vấn THEO TÊN. Sai tên là truy vấn đó 0 điểm
    — và vẫn trừ một trong ba lần nộp. Không có cách nào phát hiện lúc chạy: zip
    vẫn hợp lệ, mọi phép kiểm định dạng vẫn xanh.

    Vì vậy quy tắc ở đây là **không bao giờ tự sáng tác tên**:

      * `ten_file` do người dùng ghi trong file đề  -> dùng nguyên văn
      * `qid` đã đúng dạng `query-3-qa`             -> `query-3-qa.csv`
      * còn lại                                     -> `query-<qid>.csv`

    Trường hợp thứ ba là trường hợp NGUY HIỂM (ví dụ qid `qa-1` của dev set cho
    ra `query-qa-1.csv` trong khi BTC đợi `query-1-qa.csv`), nên nó luôn kèm
    cảnh báo từ `canh_bao_ten_file`.
    """
    if ten_file:
        ten = str(ten_file).strip()
        return ten if ten.lower().endswith(".csv") else f"{ten}.csv"
    ten = str(qid).strip()
    if MAU_TEN_BTC.match(ten):
        return f"{ten}.csv"
    return f"query-{ten}.csv"


def canh_bao_ten_file(ten: str) -> str:
    """'' nếu tên khớp quy ước BTC, ngược lại là câu cảnh báo để IN RA."""
    goc = str(ten)
    goc = goc[:-4] if goc.lower().endswith(".csv") else goc
    if MAU_TEN_BTC.match(goc):
        return ""
    return (f"{ten}: KHÔNG khớp quy ước `query-[<gói>-]<số>-<kis|qa|trake>.csv` của BTC "
            f"(nguồn: hướng dẫn nộp bài AIC2025). Codabench ghép file với truy "
            f"vấn THEO TÊN — sai tên là truy vấn đó 0 điểm và vẫn trừ 1 lần nộp. "
            f"Đặt `qid` đúng bằng tên file đề, hoặc thêm khoá `ten_file`.")


def chon_sub_events(q: Dict, chinh_sach: str = "en", qid: str = ""
                    ) -> Tuple[List[Dict[str, Any]], BanGhiNgonNgu]:
    """Chuỗi sự kiện TRAKE, chọn theo cùng chính sách ngôn ngữ như `chon_van_ban`.

    Một câu TRAKE thường có bản tiếng Anh dạng danh sách (`sub_events`) và bản
    tiếng Việt dạng chuỗi ngăn bằng "→" (`_mo_ta_vi`). Không chọn có ý thức thì
    ngôn ngữ của TRAKE trôi khác ngôn ngữ của KIS/QA trong cùng một lần nộp.
    """
    ten = qid or str(q.get("qid", "?"))
    thu = (*KHOA_SU_KIEN, "_mo_ta_vi", "mo_ta_vi", "su_kien_vi",
           "sub_events_vi", "text_vi", "query_vi")
    ung_vien: List[Tuple[str, List[Dict[str, Any]], str]] = []
    da_xet = set()
    for k in thu:
        if k in da_xet or k not in q or q[k] in (None, "", [], ()):
            continue
        da_xet.add(k)
        try:
            se = tach_sub_events(q[k], ten)
        except ValueError:
            continue
        ung_vien.append(
            (k, se, ngon_ngu_hieu_luc(k, " ".join(d["text"] for d in se))))
    if not ung_vien:
        raise ValueError(
            f"{ten}: TRAKE không có chuỗi sự kiện đọc được "
            f"(đã thử: {', '.join(thu)}). Nếu đề để nó dạng chuỗi, dấu tách "
            f"nhận được là: → -> => ; | và xuống dòng.")

    if chinh_sach != "tu_dong":
        hop = [x for x in ung_vien if x[2] == chinh_sach]
        if hop:
            hop.sort(key=lambda x: 0 if the_ngon_ngu(x[0]) == chinh_sach else 1)
            k, se, ng = hop[0]
            return se, BanGhiNgonNgu(ten, k, ng, True)
        k, se, ng = ung_vien[0]
        return se, BanGhiNgonNgu(
            ten, k, ng, False,
            canh_bao=(f"{ten}: chính sách `{chinh_sach}` nhưng chuỗi sự kiện "
                      f"chỉ có bản `{ng}` (khoá {k!r}) — đang dùng bản đó."))
    k, se, ng = ung_vien[0]
    return se, BanGhiNgonNgu(ten, k, ng, True)
