"""Đọc thư mục file đề `.txt` của BTC -> JSON cho `nop_bai.py` (việc ngày thi).

VÌ SAO CẦN
----------
Đến 19/08 vẫn chưa có công cụ nào làm việc này: ngày thi phải tự gõ JSON cho
~170 câu dưới áp lực đồng hồ. Đó đúng là chỗ dễ sai `qid` nhất, mà sai `qid` là
tên file CSV sai, là **câu đó 0 điểm và vẫn trừ một lần nộp** — không phép kiểm
định dạng nào bắt được vì zip vẫn hợp lệ (xem `de_bai.ten_file_nop`).

Ở đây `qid` lấy NGUYÊN VĂN tên file BTC (bỏ đuôi `.txt`), nên tên file nộp tự
khớp. Đó là toàn bộ lý do tồn tại của module này.

NGUYÊN TẮC: ĐOÁN THÌ PHẢI NÓI
-----------------------------
Chưa ai thấy file đề thật, nên mọi thứ ngoài `qid` đều là suy đoán:

  * loại câu suy từ tên file (`...-kis`, `...-qa`, `...-trake`)
  * câu QA: dòng CUỐI kết thúc bằng `?` là câu hỏi, phần còn lại là mô tả cảnh
  * TRAKE: mỗi dòng (hoặc mục đánh số / ngăn bằng `→`) là một sự kiện

Mỗi lần phải đoán, `BanGhiDoc.canh_bao` được điền và `doc_thu_muc` trả về để
bên gọi IN RA. Người phải nhìn được nó đã hiểu đề thế nào TRƯỚC khi chạy 20
phút truy hồi — chứ không phải phát hiện sau khi đã nộp.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# `query-3-qa.txt` -> ("3", "qa"); `query-p1-16-trake.txt` -> ("16", "trake").
# Nhóm `goi` là MÃ GÓI, tuỳ chọn: bộ đề thật của vòng thử nghiệm AIC2026 đặt tên
# `query-p1-<N>-<loại>.txt`. Xem `de_bai.MAU_TEN_BTC` — hai mẫu phải đi cùng
# nhau, lệch nhau là một bên báo hợp lệ còn bên kia báo lệch.
MAU_TEN = re.compile(
    r"^query[-_](?:(?P<goi>[a-z][a-z0-9]*)[-_])?"
    r"(?P<so>\d+)[-_](?P<loai>kis|qa|trake)$", re.IGNORECASE)
# "1. ..." / "1) ..." / "- ..." / "* ..." ở đầu dòng: đánh số của người ra đề.
MAU_DANH_SO = re.compile(r"^\s*(?:\d+\s*[.)]|[-*•])\s*")
NGAN_SU_KIEN = re.compile(r"\s*(?:->|→|;)\s*")
# Nhãn sự kiện TRAKE của BTC: "E1:", "E 2 .", "e3)". CHỈ dùng để NHẬN BIẾT dòng
# nào là sự kiện — KHÔNG dùng con số trong nhãn làm thứ tự hay làm số sự kiện.
# Bộ đề thật `query-p1-18-trake.txt` đánh nhầm thành E1/E2/E2/E4: nếu tin vào
# nhãn thì hoặc mất một sự kiện (gộp hai "E2") hoặc thành 4 sự kiện lệch chỗ.
# Thứ tự XUẤT HIỆN mới là thứ tự thời gian, và đó là thứ BTC chấm.
#
# 🔴 DẤU PHÂN CÁCH LÀ TUỲ CHỌN — sửa 2026-08-21 khi đề SƠ TUYỂN ĐỢT 1 về tay.
# Vòng thử nghiệm BTC ghi `E1: Khoảnh khắc...`; đợt 1 ghi `E1 Khoảnh khắc...`
# KHÔNG có dấu hai chấm. Mẫu cũ đòi `[:.)-]` nên không khớp dòng nào, rơi về
# nhánh "mỗi dòng là một sự kiện" và đếm CẢ CÂU DẪN -> 4 sự kiện cho một câu
# BTC hỏi 3. Luật ghi rõ số Frame ID phải khớp chính xác số events, nên đó là
# R-Score 0 cho cả truy vấn, mà zip vẫn hợp lệ và mọi phép kiểm vẫn xanh.
#
# Không có `[:.)-]` thì BẮT BUỘC phải có khoảng trắng: `E1Khoảnh` không phải
# nhãn, và điều kiện "ít nhất HAI dòng mang nhãn" ở `_su_kien` vẫn chặn việc
# một câu KIS lỡ chứa "E1 " bị hiểu thành TRAKE.
MAU_NHAN_SU_KIEN = re.compile(r"^\s*E\s*\d+\s*(?:[:.)\-]\s*|\s+)",
                              re.IGNORECASE)

# 🔴 DÒNG TRƯỚC `E1` LÀ CÂU DẪN HAY LÀ SỰ KIỆN? — sửa 2026-08-21 sau khi BTC
# TỪ CHỐI CHẤM `query-p1-16-trake` của sơ tuyển đợt 1:
#
#     "Row 1: Expected 4 frame IDs, got 3" × 100 dòng
#
# File đề đợt 1 có 4 dòng: một dòng không nhãn rồi E1/E2/E3. Bản trước coi MỌI
# dòng trước nhãn đầu tiên là câu dẫn -> nộp 3 khung, BTC đòi 4. So ba file TRAKE
# vòng thử nghiệm với file đợt 1 thì ranh giới hiện ra rất rõ:
#
#   thử-16  "Đoạn video múa lân ..., tìm các sự kiện sau:"        -> câu dẫn
#   thử-18  "Trong đoạn video nấu ăn ..., gồm các khoảnh khắc:"   -> câu dẫn
#   thử-4   (không có dòng nào trước E1)                          -> —
#   đợt1-16 "Đoạn video bắt đầu bằng ảnh cận đầu một con lân       -> SỰ KIỆN
#            trắng, mũi đỏ, bên cạnh lá cờ trắng viền đỏ."
#
# Câu dẫn của BTC KẾT BẰNG DẤU HAI CHẤM và tuyên bố sắp liệt kê ("tìm các sự
# kiện sau", "gồm các khoảnh khắc"). Dòng đợt 1 kết bằng dấu chấm và tả một
# khoảnh khắc hình ảnh cụ thể — nên nó được đánh số là sự kiện thứ nhất, còn
# E1/E2/E3 là sự kiện 2/3/4.
#
# Nhận nhầm theo chiều nào cũng là R-Score 0 cho cả câu, nên mặc định phải là
# chiều ÍT MẤT MÁT HƠN: dòng không thoả hai dấu hiệu trên thì TÍNH LÀ SỰ KIỆN,
# và `BanGhiDoc.canh_bao` hét lên để người còn đếm lại bằng mắt.
MAU_CAU_DAN = re.compile(r"[:：]\s*$")
MAU_TU_CAU_DAN = re.compile(
    r"(?:tìm|gồm|bao gồm|liệt kê|xác định|đánh dấu)\s+(?:các|những)"
    r"|(?:sự kiện|khoảnh khắc|mốc thời gian)\s+sau",
    re.IGNORECASE)


def _la_cau_dan(dong: str) -> bool:
    """Dòng này là CÂU DẪN (khung cảnh chung) chứ không phải một sự kiện?"""
    return bool(MAU_CAU_DAN.search(dong) or MAU_TU_CAU_DAN.search(dong))


@dataclass
class BanGhiDoc:
    """Đã đọc được gì từ MỘT file đề, và chỗ nào là suy đoán."""

    qid: str
    task: str
    nguon: str
    so_ky_tu: int = 0
    so_su_kien: int = 0
    co_cau_hoi: bool = False
    canh_bao: List[str] = field(default_factory=list)

    def mo_ta(self) -> str:
        dau = "⚠" if self.canh_bao else "✓"
        chi_tiet = f"{self.so_ky_tu} ký tự"
        if self.task == "trake":
            chi_tiet += f" · {self.so_su_kien} sự kiện"
        elif self.task == "qa":
            chi_tiet += " · có câu hỏi" if self.co_cau_hoi else " · KHÔNG thấy câu hỏi"
        return f"{dau} {self.qid:22s} {self.task:6s} {chi_tiet}"


def _loai_tu_ten(ten: str) -> Tuple[str, Optional[str]]:
    """(task, cảnh báo). Tên không đúng mẫu BTC thì đoán theo từ khoá."""
    khop = MAU_TEN.match(ten)
    if khop:
        return khop.group("loai").lower(), None
    thap = ten.lower()
    for t in ("trake", "qa", "kis"):
        if t in thap:
            return t, (f"tên file {ten!r} không đúng mẫu "
                       f"`query-[<gói>-]<số>-<loại>` của BTC; đoán loại `{t}` "
                       f"theo từ khoá. Kiểm lại tên file nộp.")
    return "kis", (f"tên file {ten!r} không cho biết loại câu; TẠM coi là `kis`. "
                   f"Sai loại là sai cả cách chấm — kiểm lại trước khi nộp.")


def _tach_cau_hoi(dong: List[str]) -> Tuple[str, str]:
    """(mô tả cảnh, câu hỏi). Câu hỏi là dòng CUỐI kết thúc bằng `?`.

    Nhánh dự phòng (thêm 2026-08-20): dòng nào CHỨA `?` mà không kết thúc bằng
    `?` vẫn tính là câu hỏi, và cả dòng được giữ nguyên làm câu hỏi. Bộ đề thật
    có `query-p1-15-qa.txt`:

        ... Hỏi xã này có tên là gì? (tại thời điểm đó)

    Dấu `?` nằm giữa dòng vì BTC chú thích thêm trong ngoặc. Bản cũ không thấy
    câu hỏi nào -> `_cau_hoi_vi` rỗng -> `pipeline` lấy `query.text`, tức nguyên
    cả đoạn tả cảnh, làm câu hỏi cho VLM. Giữ NGUYÊN dòng thay vì cắt ở dấu `?`
    là cố ý: phần trong ngoặc ("tại thời điểm đó") thu hẹp đáp án, bỏ đi là hỏng
    câu hỏi.
    """
    for i in range(len(dong) - 1, -1, -1):
        if dong[i].rstrip().endswith("?"):
            return " ".join(dong[:i] + dong[i + 1:]).strip(), dong[i].strip()
    for i in range(len(dong) - 1, -1, -1):
        if "?" in dong[i]:
            return " ".join(dong).strip(), dong[i].strip()
    return " ".join(dong).strip(), ""


def _su_kien(dong: List[str]) -> Tuple[List[str], str]:
    """Sự kiện TRAKE -> `(danh sách sự kiện, câu mở đầu)`.

    🔴 VÌ SAO PHẢI CÓ NHÁNH NHÃN `E<n>:` (sửa 2026-08-20)
    ------------------------------------------------------
    Bản cũ coi MỌI dòng không rỗng là một sự kiện. Trên bộ đề THẬT của vòng thử
    nghiệm AIC2026 điều đó sai ở 2/3 câu TRAKE, và sai theo kiểu tệ nhất:

        query-p1-16-trake.txt
            Đoạn video múa lân một con lân màu vàng đen trắng, tìm các sự kiện sau:
            E1: ...  E2: ...  E3: ...  E4: ...

    Dòng đầu là CÂU DẪN, không phải sự kiện. Bản cũ đọc ra **5 sự kiện** cho một
    câu BTC hỏi **4**. Hậu quả không phải "kém một chút": luật ghi rõ số Frame ID
    phải khớp chính xác số events, nên file đó **R-Score 0 cho cả truy vấn**, mà
    zip vẫn hợp lệ và mọi phép kiểm định dạng vẫn xanh. Đúng chủng loại lỗi câm
    mà module này sinh ra để chặn.

    Quy tắc: hễ có ÍT NHẤT HAI dòng mang nhãn `E<n>` thì CHỈ những dòng đó là sự
    kiện; phần trước nhãn đầu tiên là câu mở đầu; dòng không có nhãn nằm SAU một
    nhãn được nối vào sự kiện đang mở (đề xuống dòng giữa chừng). Đòi hai nhãn
    để một câu KIS lỡ có chữ "E1" ở đâu đó không bị hiểu thành TRAKE một sự kiện.

    Nhãn CHỈ để nhận biết, không để đánh số: `query-p1-18-trake.txt` đánh nhầm
    E1/E2/E2/E4. Thứ tự xuất hiện mới là thứ tự thời gian.

    Câu mở đầu được trả RIÊNG chứ không vứt đi: nó là manh mối cấp VIDEO
    ("múa lân", "món ăn về nấm") và TRAKE chấm 0 nếu sai video — xem
    `doc_mot_file` để biết nó được dùng thế nào.
    """
    sach = [MAU_DANH_SO.sub("", d).strip() for d in dong]
    sach = [d for d in sach if d]

    if sum(1 for d in sach if MAU_NHAN_SU_KIEN.match(d)) >= 2:
        mo_dau: List[str] = []
        truoc: List[str] = []   # dòng trước `E1` mà KHÔNG phải câu dẫn
        su_kien: List[str] = []
        for d in sach:
            if MAU_NHAN_SU_KIEN.match(d):
                su_kien.append(MAU_NHAN_SU_KIEN.sub("", d).strip())
            elif su_kien:
                su_kien[-1] = f"{su_kien[-1]} {d}".strip()
            elif _la_cau_dan(d):
                mo_dau.append(d)
            else:
                truoc.append(d)
        # `truoc` đứng TRƯỚC vì BTC đánh số theo thứ tự thời gian, và dòng không
        # nhãn nằm trên E1 trong file đề là khoảnh khắc mở màn.
        return truoc + su_kien, " ".join(mo_dau).strip()

    if len(sach) > 1:
        return sach, ""
    if sach:
        return [x.strip() for x in NGAN_SU_KIEN.split(sach[0]) if x.strip()], ""
    return [], ""


def doc_mot_file(path: Path) -> Tuple[Dict, BanGhiDoc]:
    """Một file `.txt` -> (mục truy vấn, bản ghi để in ra)."""
    qid = path.stem.strip()
    task, canh_bao_ten = _loai_tu_ten(qid)
    noi_dung = path.read_text(encoding="utf-8-sig", errors="replace")
    dong = [d.strip() for d in noi_dung.splitlines() if d.strip()]

    bg = BanGhiDoc(qid=qid, task=task, nguon=str(path),
                   so_ky_tu=len(noi_dung.strip()))
    if canh_bao_ten:
        bg.canh_bao.append(canh_bao_ten)
    if not dong:
        bg.canh_bao.append(f"{qid}: FILE RỖNG — truy vấn này sẽ không có nội dung")

    # `ten_file` ghi thẳng bằng tên BTC: `ten_file_nop` dùng nguyên văn, khỏi
    # phụ thuộc vào việc `qid` có khớp mẫu hay không.
    muc: Dict = {"qid": qid, "task": task, "ten_file": qid}

    if task == "trake":
        se, mo_dau = _su_kien(dong)
        # Câu mở đầu được GHÉP vào đầu từng sự kiện, không phải chỉ ghi lại.
        # Lý do: `reasoning/trake.py` encode RIÊNG text của từng sự kiện rồi lấy
        # hợp các pool đó ra danh sách video ứng viên — `query.text` không được
        # dùng đến chỗ nào trong đường TRAKE (kiểm bằng cách đọc `solve()`).
        # Nên nếu không ghép, "Khoảnh khắc 4 chân hoàn toàn chạm đất đầu tiên"
        # đi tìm một mình trên 177.321 khung, không còn chữ nào nói đây là múa
        # lân. Mà TRAKE sai video là 0 điểm cả câu, nên độ chính xác cấp video
        # đáng giá hơn độ sắc nét giữa các sự kiện trong cùng video.
        # ⚠ CHƯA ĐO — dev set TRAKE tự làm không có câu nào kiểu câu-dẫn + E1..En.
        # Muốn tắt thì xoá tiền tố trong file JSON sinh ra, `_mo_dau` giữ nguyên
        # văn để làm việc đó.
        if mo_dau:
            se = [f"{mo_dau} {x}".strip() for x in se]
            muc["_mo_dau"] = mo_dau
        muc["sub_events"] = se
        muc["text"] = " ".join(dong)
        bg.so_su_kien = len(se)
        if mo_dau:
            bg.canh_bao.append(
                f"{qid}: dòng đầu kết bằng ':' nên là CÂU DẪN, không tính là "
                f"sự kiện -> {len(se)} sự kiện. Đã ghép câu dẫn vào đầu mỗi sự "
                f"kiện làm ngữ cảnh cấp video. ĐẾM LẠI BẰNG MẮT trong đề: sai "
                f"số sự kiện là R-Score 0 cho cả câu.")
        n_nhan = sum(1 for d in dong
                     if MAU_NHAN_SU_KIEN.match(MAU_DANH_SO.sub("", d).strip()))
        if n_nhan >= 2 and len(se) > n_nhan:
            bg.canh_bao.append(
                f"{qid}: có {len(se) - n_nhan} dòng KHÔNG mang nhãn E<n> đứng "
                f"trước `E1` và KHÔNG kết bằng ':' -> ĐÃ TÍNH LÀ SỰ KIỆN "
                f"(=> {len(se)} sự kiện, nộp {len(se)} khung/dòng). Đây đúng là "
                f"chỗ BTC bác `query-p1-16-trake` đợt 1 ('Expected 4 frame IDs, "
                f"got 3'). ĐẾM LẠI BẰNG MẮT.")
        if len(se) < 2:
            bg.canh_bao.append(
                f"{qid}: chỉ tách được {len(se)} sự kiện. TRAKE cần đúng số "
                f"khoảnh khắc BTC hỏi — sai `n_events` là R-Score 0 cho cả câu.")
    elif task == "qa":
        mo_ta, cau_hoi = _tach_cau_hoi(dong)
        muc["text"] = mo_ta or " ".join(dong)
        bg.co_cau_hoi = bool(cau_hoi)
        if cau_hoi:
            muc["_cau_hoi_vi"] = cau_hoi
        else:
            bg.canh_bao.append(
                f"{qid}: không thấy dòng nào kết thúc bằng '?'. Không có câu hỏi "
                f"thì tầng đáp án đoán mò — mở file kiểm và thêm `_cau_hoi_vi`.")
    else:
        muc["text"] = " ".join(dong)

    return muc, bg


def doc_thu_muc(thu_muc: str | Path, mau: str = "*.txt"
                ) -> Tuple[List[Dict], List[BanGhiDoc]]:
    """Cả thư mục đề -> (danh sách mục, bản ghi). Sắp theo SỐ trong tên file."""
    goc = Path(thu_muc)
    if not goc.is_dir():
        raise NotADirectoryError(f"Không phải thư mục: {goc}")
    files = sorted(goc.glob(mau))
    if not files:
        raise FileNotFoundError(f"{goc}: không có file nào khớp {mau!r}")

    def khoa(p: Path):
        k = MAU_TEN.match(p.stem)
        return ((0, k.group("goi") or "", int(k.group("so")), k.group("loai"))
                if k else (1, "", 0, p.stem.lower()))

    muc, ban_ghi = [], []
    for p in sorted(files, key=khoa):
        m, bg = doc_mot_file(p)
        muc.append(m)
        ban_ghi.append(bg)

    trung = {q for q in (m["qid"] for m in muc)
             if sum(1 for x in muc if x["qid"] == q) > 1}
    if trung:
        raise ValueError(
            f"qid trùng nhau: {sorted(trung)}. Hai câu ghi đè lên cùng một file "
            f"CSV nộp — phải sửa tên file đề trước khi chạy.")
    return muc, ban_ghi


def ghi_json(muc: List[Dict], ra: str | Path) -> Path:
    p = Path(ra)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"queries": muc}, ensure_ascii=False, indent=1),
                 encoding="utf-8")
    return p
