"""DỒN SLOT — phân bổ 100 dòng nộp theo VIDEO thay vì rải đều.

VẤN ĐỀ, ĐO ĐƯỢC TRÊN CHÍNH BÀI NỘP 7,6 ĐIỂM
-------------------------------------------
KIS chỉ ăn điểm khi `frame_idx` nằm TRONG cửa sổ `[s,e]` của BTC. Cửa sổ đo
được trên dev set là ~107 khung ≈ 4,3 giây, tức chỉ khoảng **2 keyframe** lọt
vào. Đúng video là điều kiện CẦN, không phải đủ.

Soi bài nộp đạt 7,6/12 (21/08), cột "số khung của video hạng 1 có mặt trong
100 dòng":

    q6  cuốn chay   L26_V056     1 / 146 keyframe   (1%)
    q21 Lausanne    L21_V025     3 / 286            (1%)
    q12 bánh rán    L26_V279     5 / 156            (3%)
    q17 trao quà    L30_V091     7 / 134            (5%)
    q7  chim rừng   L29_V023    11 / 441            (2%)

Với 1 khung trong một video 146 keyframe thì xác suất trúng cửa sổ gần bằng 0 —
**dù đã tìm đúng video và xếp nó hạng 1**. Đây là mất điểm ở tầng PHÂN BỔ, không
phải ở tầng tìm.

VÌ SAO LƯỚI AN TOÀN CŨ GẦN NHƯ VÔ GIÁ TRỊ
-----------------------------------------
`submission.diversify_tail` rải phần đuôi vòng tròn qua các video khác nhau, để
hedge R@20/50/100. Hợp lý khi độ chính xác cấp video còn thấp. Nhưng đo thật:
95 slot đuôi đang chia cho ~30 video, mỗi video ~3 khung. Ba khung trong một
video 300 keyframe cũng chỉ ~1% cơ hội. Nghĩa là ta đang trả 95 slot để mua 30
tấm vé gần như chắc chắn trượt, thay vì mua 2-3 tấm vé thật.

CÁCH LÀM
--------
1. Xếp hạng VIDEO theo hạng tốt nhất của khung thuộc nó trong danh sách ứng viên.
2. Cấp hạn ngạch giảm dần cho vài video đầu (mặc định 55/20/12/8/5).
3. NỞ mỗi video ra TOÀN BỘ timeline của nó rồi chấm lại bằng đúng phép biến đổi
   của `search` (`score_many`), lấy đủ hạn ngạch.
4. Xen kẽ theo hạn ngạch, nhưng KHUNG HẠNG 1 GỐC LUÔN ĐỨNG ĐẦU.

Điểm 4 không phải chi tiết: `Final = (1/5)·Σ_{k∈{1,5,20,50,100}} R@k` nên slot 1
được tính trong cả 5 số hạng. Đổi khung hạng 1 là đánh cược vào chính chỗ đắt
nhất, nên lớp này KHÔNG đụng vào nó.

⚠ MỘT KẾT QUẢ PHỦ ĐỊNH CŨ, VÀ VÌ SAO Ở ĐÂY KHÁC
-----------------------------------------------
Việc 4.3 (08/08) đã thử "RRF xếp video + nở timeline + round-robin" và đo ra
THUA: KIS 67 -> 61 câu chạm khung với V=20. Hai khác biệt then chốt:

  * V=20 là rải, không phải dồn. Ở đây V mặc định 5 và hạn ngạch dốc.
  * Lúc đó chưa có kênh OCR/ASR nên độ chính xác cấp video thấp; dồn vào một
    video sai là mất trắng. Sau 21/08 thì video hạng 1 đúng ở phần lớn câu —
    tiền đề đã đổi.

🔴 ĐÃ ĐO 21/08 — KẾT QUẢ PHỦ ĐỊNH. GIỮ `bat=False`.
--------------------------------------------------
`do_don_slot.py`, 120 câu KIS dev có ground truth CẤP KHUNG, ghép cặp:

    tắt (nền)                 Final 0.2633   R@100 45.0%
    30/12/8  (50 slot)              0.2650         42.5%   +0.0017  ≈ nhiễu
    20/10/6  (36 slot)              0.2617         43.3%   -0.0016  ≈
    12/6     (18 slot)              0.2600         43.3%   -0.0033  ≈
    20       (chỉ video 1)          0.2500         43.3%   -0.0133
    55/20/12/8/5 · trộn             0.2200         29.2%   -0.0433  ❌
    55/20/12/8/5 · đều              0.1267         20.8%   -0.1367  ❌
    80/15/5  · trộn                 0.2067         27.5%   -0.0567  ❌

Chẩn đoán ban đầu SAI ở đúng một chỗ: tôi cho rằng "3 khung trong một video 300
keyframe là tấm vé chắc trượt, nên 95 slot đuôi gần như vô giá trị". Số liệu
bác bỏ điều đó — R@100 sập từ 45% xuống 29% khi dồn hết 100 slot, tức **16%
số câu có khung đúng nằm ở video xếp NGOÀI top-5**. Cái đuôi đang mua thứ thật.

Vì sao dồn không bù lại được: trên dev set, video hạng 1 chỉ đúng khoảng 45%
số câu. Cấp 30 slot cho một video sai là đốt 30 slot. Phần thắng ở 45% kia
không đủ trả cho phần thua ở 55%.

KHI NÀO ĐÁNG THỬ LẠI
--------------------
Đúng một điều kiện: khi **độ chính xác cấp video cao hơn hẳn**. Điều kiện đó có
thật trên đề BTC — kênh OCR/ASR đưa video đúng lên hạng 1 ở phần lớn câu, và
lớp `ghim` (`so_dong`) chính là dồn slot CÓ ĐIỀU KIỆN, đã ăn +1,4 điểm thật
trên leaderboard. Nói cách khác: dồn slot đúng, nhưng chỉ khi biết chắc video.

Hướng còn mở: dồn CÓ ĐIỀU KIỆN theo tín hiệu kênh chữ (chỉ dồn khi OCR/ASR
cũng chỉ vào video hạng 1). Chưa cài, chưa đo.

⚠ Và một cảnh báo về chính phép đo trên: dev set KHÔNG phải thước tin được cho
mọi thứ. Cùng ngày, chính nó nói kênh OCR/ASR làm HẠI (−0.0250 ± 0.0084) trong
khi leaderboard thật nói kênh đó ăn **+1,6/12**. Truy vấn dev là mô tả cảnh
người gõ; đề BTC là bản tin đầy danh từ riêng. Hai phân bố khác nhau.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class CauHinhDonSlot:
    bat: bool = False
    # Hạn ngạch cho video hạng 1, 2, 3... Tổng nên <= 100; phần thiếu để dành
    # cho danh sách ứng viên gốc, tức lưới an toàn vẫn còn.
    han_ngach: Sequence[int] = (55, 20, 12, 8, 5)
    # Trong mỗi video, lấy khung theo ĐIỂM hay rải ĐỀU theo thời gian?
    #   "diem"  — top theo điểm. Rủi ro: khung gần nhau điểm gần nhau nên dễ
    #             chụm vào một cảnh, bỏ trống phần còn lại của video.
    #   "deu"   — rải đều trên trục thời gian. Phủ tốt nhưng bỏ qua tín hiệu.
    #   "tron"  — nửa đầu hạn ngạch theo điểm, nửa sau rải đều. MẶC ĐỊNH.
    cach_chon: str = "tron"
    # Không dồn khi danh sách ứng viên quá ngắn (truy vấn hỏng, corpus nhỏ).
    toi_thieu_ung_vien: int = 20


@dataclass
class BaoCaoDonSlot:
    video: List[Tuple[str, int]] = field(default_factory=list)
    them: int = 0
    giu_hang_1: bool = True

    def mo_ta(self) -> str:
        v = " · ".join(f"{v}:{n}" for v, n in self.video[:5])
        return f"dồn slot [{v}] · thêm {self.them} khung mới"


def _thu_tu_video(candidates: Sequence) -> List[str]:
    """Video theo HẠNG TỐT NHẤT của khung thuộc nó (không theo số lượng khung).

    Theo số lượng là sai: một video dài tự nhiên có nhiều khung lọt top mà
    không nhất thiết khớp hơn.
    """
    thay: Dict[str, int] = {}
    for i, c in enumerate(candidates):
        v = c.event.video_id
        if v not in thay:
            thay[v] = i
    return sorted(thay, key=lambda v: thay[v])


def _chon_trong_video(xep: List[Tuple[object, float]], n: int,
                      cach: str) -> List[object]:
    """`xep` = [(event, điểm)] đã sắp giảm dần theo điểm. Trả `n` event."""
    if n <= 0 or not xep:
        return []
    if len(xep) <= n:
        return [e for e, _ in xep]
    if cach == "diem":
        return [e for e, _ in xep[:n]]

    theo_tg = sorted((e for e, _ in xep), key=lambda e: e.t_start)
    if cach == "deu":
        buoc = len(theo_tg) / float(n)
        return [theo_tg[min(len(theo_tg) - 1, int(i * buoc))] for i in range(n)]

    # "tron": nửa theo điểm (bám tín hiệu), nửa rải đều (bảo hiểm phủ).
    n1 = n // 2
    lay = [e for e, _ in xep[:n1]]
    da = {id(e) for e in lay}
    con = [e for e in theo_tg if id(e) not in da]
    n2 = n - len(lay)
    if con and n2 > 0:
        buoc = len(con) / float(n2)
        lay += [con[min(len(con) - 1, int(i * buoc))] for i in range(n2)]
    # Khử trùng giữ thứ tự.
    ra, thay = [], set()
    for e in lay:
        if e.event_id not in thay:
            thay.add(e.event_id)
            ra.append(e)
    return ra[:n]


def don_slot(candidates: Sequence, service, encoder_vector,
             cfg: Optional[CauHinhDonSlot] = None,
             ) -> Tuple[List, BaoCaoDonSlot]:
    """Trả `(danh sách Candidate mới, báo cáo)`. Không đủ điều kiện thì trả nguyên.

    `service`  : `RetrievalService` — cần `full_timeline` và kho vector.
    `encoder_vector`: vector truy vấn đã encode, để `score_many` chấm lại
        timeline bằng ĐÚNG phép biến đổi của `search` (có khử modality gap).
        `None` thì bỏ qua bước nở — chấm sai hình học còn tệ hơn không nở.
    """
    from aic2026.core.models import Candidate

    cfg = cfg or CauHinhDonSlot()
    bc = BaoCaoDonSlot()
    if (not cfg.bat or encoder_vector is None
            or len(candidates) < cfg.toi_thieu_ung_vien):
        return list(candidates), bc

    store = service.retriever.visual
    thu_tu = _thu_tu_video(candidates)
    lay_theo_video: List[List] = []

    for vid, quota in zip(thu_tu, cfg.han_ngach):
        timeline = service.full_timeline(vid)
        if not timeline:
            continue
        diem = store.score_many([e.event_id for e in timeline], encoder_vector)
        xep = sorted(((e, diem[e.event_id]) for e in timeline
                      if e.event_id in diem),
                     key=lambda kv: (-kv[1], kv[0].t_start))
        chon = _chon_trong_video(xep, quota, cfg.cach_chon)
        if chon:
            lay_theo_video.append(chon)
            bc.video.append((vid, len(chon)))

    if not lay_theo_video:
        return list(candidates), bc

    # Khung hạng 1 GỐC luôn đứng đầu: slot 1 được tính trong cả 5 số hạng của
    # `Final`, nên đó là chỗ đắt nhất và lớp này không đánh cược ở đó.
    ra: List = [candidates[0]]
    da = {candidates[0].event.event_id}

    # Xen kẽ theo hạn ngạch để video hạng 1 chiếm phần đầu nhưng video hạng 2-3
    # vẫn có mặt sớm — R@5/R@20 vẫn còn cửa nếu hạng 1 sai.
    i = 0
    while len(ra) < 100 and any(i < len(x) for x in lay_theo_video):
        for nhom in lay_theo_video:
            if i < len(nhom) and len(ra) < 100:
                e = nhom[i]
                if e.event_id not in da:
                    da.add(e.event_id)
                    ra.append(Candidate(event=e, score=0.0,
                                        modality_scores={"don_slot": 1.0}))
        i += 1

    # Còn chỗ thì nối phần ứng viên gốc chưa dùng — lưới an toàn.
    for c in candidates:
        if len(ra) >= 100:
            break
        if c.event.event_id not in da:
            da.add(c.event.event_id)
            ra.append(c)

    bc.them = sum(1 for c in ra
                  if c.event.event_id not in
                  {x.event.event_id for x in candidates})
    return ra, bc
