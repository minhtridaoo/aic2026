"""GHIM VIDEO — đưa tri thức người vận hành đã XÁC MINH vào bài nộp.

VÌ SAO CÓ LỚP NÀY
-----------------
Vòng sơ tuyển AIC2026 mở 19:30–22:30 (3 giờ, có người ngồi máy), tối đa 3 lần
nộp mỗi gói. Nghĩa là bài nộp không phải là đầu ra của một hàm chạy một phát —
nó là kết quả của một VÒNG LẶP người–máy. Nhưng cho tới 20/08 hệ thống không có
đường nào để nhét lại một điều người đã biết chắc vào kết quả.

Ví dụ thật, đề vòng thử nghiệm 20/08. Câu `query-p1-19-qa` hỏi hai câu thơ trong
**đình thần Nguyễn Trung Trực tại Kiên Giang**. Tra `media-info` của BTC ra ĐÚNG
MỘT video trong 873 video có cụm đó trong tiêu đề (`L24_V011`). Đây không phải
phỏng đoán — nó là một phép tra vét cạn trên trường dữ liệu BTC cấp. Trong khi
đó tầng truy hồi ảnh xếp `L28_V007` lên hạng 1 và không có `L24_V011` ở đâu
trong 100 dòng. Không ghim được thì câu đó chắc chắn 0 điểm, dù đáp án đã nằm
trong tay.

CÁI NÀY KHÔNG PHẢI GIAN LẬN, VÀ CŨNG KHÔNG PHẢI PHÉP MÀU
--------------------------------------------------------
Không gian lận: `media-info` nằm trong bộ dữ liệu BTC phát cho mọi đội.
Không phép màu: ghim chỉ nói ĐÚNG VIDEO. Chọn khung nào trong video đó vẫn là
việc của encoder — hệ thống chấm lại toàn bộ timeline của video được ghim bằng
đúng truy vấn đó rồi xếp hạng. Với KIS, đúng video mà sai khung vẫn là 0 điểm.

THIẾT KẾ: GHÉP VÀO ĐẦU, KHÔNG THAY THẾ
--------------------------------------
`gop()` đặt các dòng của video được ghim lên TRƯỚC, rồi nối phần kết quả gốc
phía sau (đã khử trùng). Lý do là hình dạng của thang điểm:
`Final = (1/5)·Σ_{k∈{1,5,20,50,100}} R@k` với `R@k = max` trên 100 dòng đầu.
Slot 1 được tính trong cả 5 số hạng, slot 51 chỉ trong 1 — nên niềm tin cao nhất
phải đứng đầu. Nhưng vì mọi dòng tới 100 đều còn giá trị, VỨT kết quả gốc đi là
tự cắt lưới an toàn: người ghim sai thì vẫn còn nguyên bài của máy phía sau.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class GhimTruyVan:
    """Điều người vận hành khẳng định về MỘT truy vấn."""

    qid: str
    videos: List[str] = field(default_factory=list)
    # NHÓM CHƯƠNG TRÌNH được phép, dạng tiền tố `L24`. Thêm 2026-08-21 sau khi
    # sơ tuyển đợt 1 được 3,8/12.
    #
    # Kho 873 video của BTC không phải một đống phim rời — nó là ĐÚNG 10 chương
    # trình, và cỡ nhóm lệch nhau hàng chục lần:
    #
    #   L21/L22  60 Giây (bản tin)          29 + 31
    #   L23      Cúp Truyền hình (đua xe)        25
    #   L24      Lân Sư Rồng                     43
    #   L25      Bí quyết ôn thi THPT            88
    #   L26      Món ngon mỗi ngày              498   ← 57% cả kho
    #   L27      Việt Nam đi là ghiền            16
    #   L28/L29  Mê Kông (phóng sự)         24 + 23
    #   L30      Lan tỏa năng lượng tích cực     96
    #
    # Một câu tả cảnh múa lân trên cột thì KHÔNG THỂ nằm ngoài L24. Nói điều đó
    # ra là cắt không gian tìm từ 873 xuống 43 video — mạnh gần bằng ghim thẳng
    # một video, nhưng rủi ro thấp hơn hẳn: người vận hành nhận ra "đây là múa
    # lân" chắc chắn hơn nhiều so với đoán đúng tập nào trong 43 tập.
    #
    # Đây là chỗ giữa: `videos` là cược tất tay, `nhom` là thu hẹp có kiểm soát.
    nhom: List[str] = field(default_factory=list)
    # KHUNG do người CHỌN BẰNG MẮT trên `xem_anh.py`, mỗi mục là
    # `[video, frame]` (KIS/QA) hoặc `[video, f1, f2, ..., fN]` (TRAKE).
    # Chúng được đặt lên TRƯỚC tất cả, đúng thứ tự người bấm.
    #
    # Đây là nấc mạnh nhất của lớp ghim, và lý do nằm ở thang điểm chứ không ở
    # thuật toán: `Final = (1/5)·Σ_{k∈{1,5,20,50,100}} R@k`. Cùng một khung,
    # đứng hạng 100 được 0,2 điểm, đứng hạng 1 được 1,0. Máy đã tìm ra khung
    # đúng trong 57,5% số câu (R@100) nhưng chỉ ăn 46,7% điểm (đo thật trên
    # leaderboard 20/08) — toàn bộ phần chênh đó là việc XẾP LẠI THỨ TỰ, và
    # người nhìn ảnh làm được còn máy thì không.
    khung: List[List] = field(default_factory=list)
    # Đáp án QA áp cho MỌI dòng của câu đó. Dùng khi đáp án suy ra được từ tri
    # thức ngoài (câu thơ, tên riêng) chứ không cần VLM đọc ảnh.
    answer: str = ""
    # NHIỀU đáp án ứng viên, rải luân phiên xuống các dòng. Thêm 2026-08-21.
    #
    # Câu đếm số ("không tính bảng chú giải, có bao nhiêu vị trí động đất cấp độ
    # 4?") có không gian đáp án nhỏ và rời rạc, mà `answer` một chuỗi thì đặt
    # cược đúng một lần. Đợt 1 nộp "trắng"/"vàng" cho câu đếm số — sai chắc
    # chắn, tức 0 điểm dù khung có đúng.
    #
    # Vì sao rải được mà không phạm luật khử trùng: dòng trùng nhau tính theo
    # (video, khung), nên các keyframe LIỀN NHAU trong cùng một shot là các dòng
    # HỢP LỆ KHÁC NHAU. Cửa sổ chấm của BTC phủ cả shot, nên nếu shot đúng nằm
    # trong top thì mỗi keyframe của nó là một lần đoán riêng.
    #
    # Đây là cược có kỳ vọng dương chứ không phải phép màu: R@1 vẫn phụ thuộc
    # dòng đầu đoán trúng, chỉ R@5..R@100 mới được hưởng.
    answers: List[str] = field(default_factory=list)
    ghi_chu: str = ""
    # BAO NHIÊU trong 100 slot dành cho video được ghim. Đây là núm ĐẶT CƯỢC,
    # nên đặt mặc định theo hình dạng thang điểm chứ không theo cảm tính:
    #
    #   Final = (1/5)·Σ_{k∈{1,5,20,50,100}} R@k
    #
    # Cho ghim 50 slot đầu = nếu ghim ĐÚNG thì ăn trọn 4/5 số hạng (k=1,5,20,50)
    # ngay cả khi bài của máy vô dụng; nếu ghim SAI thì vẫn còn 50 slot của máy
    # để cứu số hạng k=100, tức giữ được 1/5 điểm thay vì mất trắng.
    # Đặt 100 chỉ khi bài của máy cho câu đó rõ ràng là rác (đã nhìn tận mắt).
    so_dong: int = 50

    def mo_ta(self) -> str:
        phan = [f"{self.qid}:"]
        if self.khung:
            phan.append(f"{len(self.khung)} KHUNG người chọn")
        if self.videos:
            phan.append(f"video {', '.join(self.videos)} ({self.so_dong}/100 slot)")
        elif self.nhom:
            phan.append(f"nhóm {'+'.join(self.nhom)} ({self.so_dong}/100 slot)")
        if not self.khung and not self.videos and not self.nhom:
            phan.append("chỉ đè đáp án")
        if self.answer:
            phan.append(f'đáp án "{self.answer[:60]}"')
        elif self.answers:
            phan.append(f"{len(self.answers)} đáp án rải luân phiên "
                        f"{self.answers[:6]}")
        if self.ghi_chu:
            phan.append(f"vì {self.ghi_chu}")
        return "  ·  ".join(phan)


def doc_ghim(path: str | Path) -> Dict[str, GhimTruyVan]:
    """Đọc file JSON ghim.

        {"query-p1-19-qa": {"videos": ["L24_V011"],
                            "answer": "...",
                            "ghi_chu": "video duy nhất có 'Nguyễn Trung Trực'"}}

    Chấp `"video"` (số ít, một chuỗi) lẫn `"videos"` (danh sách) — gõ vội lúc
    thi thì hay ra dạng số ít.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "ghim" in data:
        data = data["ghim"]
    ra: Dict[str, GhimTruyVan] = {}
    for qid, muc in data.items():
        # Khoá bắt đầu bằng `_` là chú thích của người viết file, không phải
        # truy vấn. File ghim là thứ đọc lại vào 21 giờ tối lúc còn 1 lần nộp,
        # nên nó PHẢI chứa được lý do — chú thích quan trọng ngang dữ liệu.
        if str(qid).startswith("_"):
            continue
        if isinstance(muc, str):
            muc = {"videos": [muc]}
        vids = muc.get("videos") or muc.get("video") or []
        if isinstance(vids, str):
            vids = [vids]
        vids = [str(v).strip().upper() for v in vids if str(v).strip()]
        nhom = muc.get("nhom") or muc.get("nhoms") or []
        if isinstance(nhom, str):
            nhom = [nhom]
        nhom = [str(v).strip().upper() for v in nhom if str(v).strip()]
        for n in nhom:
            # `L24_V011` lọt vào `nhom` là lỗi gõ vội hay gặp, và nó KHÔNG báo
            # lỗi ở đâu khác: tiền tố đó vẫn khớp đúng một video, nên "thu hẹp
            # có kiểm soát" lặng lẽ biến thành cược tất tay.
            if "_" in n:
                raise ValueError(
                    f"{qid}: `nhom` nhận TIỀN TỐ chương trình (vd 'L24'), "
                    f"không phải tên video ({n!r}). Muốn ghim đúng video này "
                    f"thì đưa nó vào `videos`.")
        if vids and nhom:
            raise ValueError(
                f"{qid}: đặt cả `videos` lẫn `nhom` thì `nhom` không có tác "
                f"dụng gì — bỏ bớt một cái để file ghim nói đúng điều đang làm.")
        # `answers` là ALIAS của `answer` (28/08): trường trong dataclass tên
        # `answers` nên gõ số nhiều vào FILE là lỗi tự nhiên nhất — và bản cũ
        # BỎ QUA IM LẶNG, tức file ghim ghi 2 ứng viên đáp án mà bài nộp chạy
        # như không có gì. Đã mắc thật ở `query-p2-23-qa` đợt 2.
        answer_tho = muc.get("answer")
        if answer_tho in (None, "", []) and muc.get("answers") is not None:
            answer_tho = muc.get("answers")
        # Khoá lạ giờ HÉT lên thay vì trôi qua. Khoá bắt đầu bằng `_` là chỗ
        # người vận hành ghi chú tự do, luôn được phép.
        biet = {"videos", "video", "nhom", "nhoms", "answer", "answers",
                "khung", "frames", "so_dong", "ghi_chu"}
        la = sorted(k for k in muc if not str(k).startswith("_")
                    and k not in biet)
        if la:
            raise ValueError(
                f"{qid}: khoá không hiểu {la} trong file ghim. Khoá hợp lệ: "
                f"{sorted(biet)}. (Muốn ghi chú tự do thì đặt tên bắt đầu bằng "
                f"'_'.) Bản cũ bỏ qua im lặng nên một khoá gõ sai làm ghim mất "
                f"tác dụng mà không báo gì.")
        answers: List[str] = []
        if isinstance(answer_tho, (list, tuple)):
            answers = [str(x).strip() for x in answer_tho if str(x).strip()]
            answer = ""
        else:
            answer = str(answer_tho or "").strip()
        for a in ([answer] if answer else []) + answers:
            if len(a) > 100:
                raise ValueError(
                    f"{qid}: đáp án ghim dài {len(a)} ký tự, luật BTC cho tối "
                    f"đa 100. Rút gọn trước khi nộp.")
        khung = []
        for x in (muc.get("khung") or muc.get("frames") or []):
            if isinstance(x, str):                # "L26_V004|4803" từ trang xem
                x = x.split("|")
            if not isinstance(x, (list, tuple)) or len(x) < 2:
                raise ValueError(
                    f"{qid}: mỗi mục `khung` phải là [video, frame, ...], "
                    f"nhận {x!r}")
            khung.append([str(x[0]).strip().upper()]
                         + [int(v) for v in x[1:]])
        so_dong = int(muc.get("so_dong", 50))
        if not 1 <= so_dong <= 100:
            raise ValueError(f"{qid}: `so_dong` phải trong 1..100, nhận {so_dong}")
        ra[str(qid)] = GhimTruyVan(qid=str(qid), videos=vids, nhom=nhom,
                                   answer=answer, answers=answers,
                                   ghi_chu=str(muc.get("ghi_chu") or ""),
                                   so_dong=so_dong, khung=khung)
    return ra


def loc_video(events: Sequence[Any], embeddings,
              videos: Sequence[str]) -> Tuple[List[Any], Any]:
    """Cắt (events, embeddings) xuống chỉ các video được ghim, GIỮ ĐÚNG THỨ TỰ.

    Hợp đồng giữa `events[i]` và `embeddings[i]` là thứ cả `retrieval.index`
    dựa vào; cắt lệch một hàng là gán nhầm vector cho keyframe, sai âm thầm.
    Vì vậy ở đây chỉ số được dựng một lần rồi dùng cho cả hai.
    """
    muon = {str(v).upper() for v in videos}
    idx = [i for i, e in enumerate(events) if e.video_id.upper() in muon]
    if not idx:
        raise LookupError(
            f"không có event nào thuộc {sorted(muon)}. Kiểm lại tên video — "
            f"phải đúng dạng `L24_V011`, không có đuôi `.mp4`.")
    ev = [events[i] for i in idx]
    if embeddings is None:
        return ev, None
    return ev, embeddings[idx]


def _row_tu_khung(task: str, k: Sequence, answer: str = "") -> Dict:
    """`[video, frame...]` -> một dòng nộp bài đúng dạng của `task`."""
    video, khung = str(k[0]), [int(x) for x in k[1:]]
    if task == "trake":
        return {"video_id": video, "frames": khung}
    if len(khung) != 1:
        raise ValueError(
            f"{task.upper()} cần đúng MỘT frame mỗi dòng, nhận {khung}. "
            f"(Nhiều frame chỉ đúng với TRAKE — kiểm lại loại câu.)")
    r = {"video_id": video, "frame_idx": khung[0]}
    if task == "qa":
        r["answer"] = answer
    return r


def _khoa(task: str, row: Dict) -> Tuple:
    """Khoá nhận dạng MỘT giả thuyết, để khử trùng khi ghép hai danh sách."""
    if task == "trake":
        return (row["video_id"], tuple(int(f) for f in row["frames"]))
    return (row["video_id"], int(row["frame_idx"]))


def gop(rows_ghim: Sequence[Dict], rows_goc: Sequence[Dict], task: str,
        max_rows: int = 100, answer: str = "",
        answers: Optional[Sequence[str]] = None) -> List[Dict]:
    """Dòng ghim lên đầu, dòng gốc nối sau, khử trùng, cắt còn `max_rows`.

    `answer` khác rỗng thì ĐÈ lên đáp án của mọi dòng — kể cả dòng gốc. Với QA,
    R-Score chỉ cộng khi khung nằm trong cửa sổ đúng VÀ đáp án đúng, nên nếu đã
    biết chắc đáp án thì dùng nó cho tất cả các dòng là tối ưu: mỗi dòng còn lại
    chỉ mất đi cơ hội đúng nhờ ăn may của một đáp án khác.

    `answers` (nhiều ứng viên) thì rải LUÂN PHIÊN theo thứ tự dòng: dòng 1 lấy
    ứng viên 1, dòng 2 lấy ứng viên 2, ... Cố ý không gom theo cụm: các dòng đầu
    là các keyframe điểm cao nhất, nên rải luân phiên đảm bảo mỗi ứng viên đều
    được thử ở một trong những khung tốt nhất, thay vì ứng viên cuối chỉ còn
    những khung mà chính encoder cũng không tin.
    """
    ra: List[Dict] = []
    da_co = set()
    for r in list(rows_ghim) + list(rows_goc):
        k = _khoa(task, r)
        if k in da_co:
            continue
        da_co.add(k)
        r = dict(r)
        if task == "qa":
            if answers:
                r["answer"] = answers[len(ra) % len(answers)]
            elif answer:
                r["answer"] = answer
        ra.append(r)
        if len(ra) >= max_rows:
            break
    return ra


def ap_dung(pipeline, query, ghim: GhimTruyVan, rows_goc: Sequence[Dict],
            events: Sequence[Any], embeddings, top_k: int = 100,
            ) -> Tuple[List[Dict], str]:
    """Chạy lại `query` CHỈ trên các video được ghim rồi ghép lên đầu `rows_goc`.

    Trả `(rows, dòng-báo-cáo)`. Pipeline bị RE-INDEX trong hàm này; bên gọi phải
    tự nạp lại chỉ mục đầy đủ sau khi dùng xong — xem `nop_bai.py`.
    """
    # KHUNG người chọn đứng TRƯỚC MỌI THỨ — kể cả trước video được ghim. Người
    # đã nhìn thấy ảnh thì đó là bằng chứng mạnh nhất đang có; máy chỉ xếp phần
    # còn lại phía sau làm lưới an toàn.
    rows_nguoi = [_row_tu_khung(query.task, k, ghim.answer) for k in ghim.khung]

    # `nhom` nở ra danh sách video ở ĐÂY chứ không ở `doc_ghim`: chỉ tới lúc này
    # mới biết kho thật có những video nào. Nở sớm thì file ghim phải biết trước
    # nội dung kho, mà kho thì đổi theo từng đợt data BTC phát.
    videos = list(ghim.videos)
    mo_ta_dich = "+".join(videos)
    if not videos and ghim.nhom:
        tien_to = tuple(f"{n}_" for n in ghim.nhom)
        videos = sorted({str(e.video_id).upper() for e in events
                         if str(e.video_id).upper().startswith(tien_to)})
        if not videos:
            raise LookupError(
                f"{ghim.qid}: nhóm {ghim.nhom} không khớp video nào trong kho. "
                f"Tiền tố phải đúng dạng `L24` (kho đang có "
                f"{sorted({str(e.video_id).split('_')[0] for e in events})}).")
        mo_ta_dich = f"nhóm {'+'.join(ghim.nhom)} ({len(videos)} video)"

    if not videos:
        _va_dap_an(query.task, rows_nguoi, ghim, (), rows_goc)
        rows = gop(rows_nguoi, rows_goc, query.task, top_k, ghim.answer,
                   ghim.answers)
        bao = (f"{ghim.qid}: {len(rows_nguoi)} khung người chọn lên đầu"
               if rows_nguoi else f"{ghim.qid}: chỉ đè đáp án")
        return rows, bao

    ev, em = loc_video(events, embeddings, videos)
    # `reset()` TRƯỚC `index()` là bắt buộc: `index()` cộng dồn, nên thiếu dòng
    # này thì 177.321 khung cũ vẫn còn nguyên và "ghim" chẳng ghim gì cả.
    pipeline.retriever.reset()
    pipeline.index(ev, em)

    # 🔴 `pipeline.index()` KHÔNG cắt được encoder phụ của `retrieval.hop_nhat`.
    # Các retriever phụ được dựng một lần trong `config.nap_hop_nhat` với chỉ
    # mục ĐẦY ĐỦ 177.321 khung và `pipeline.index` chỉ dựng lại đường chính; tệ
    # hơn, `RetrievalService._hop_nhat` có nhánh dự phòng
    # `self._su_kien.get(eid) or cand.event` nên event lạ đi thẳng ra kết quả.
    # Đã quan sát thật 20/08: ghim L24_V011 mà CSV vẫn mở đầu bằng L28_V007.
    #
    # Bỏ encoder phụ trong lượt ghim. Cái giá là mất phần hợp nhất cho việc xếp
    # hạng KHUNG BÊN TRONG video (đo được: 0.3750 -> 0.3400 trên toàn cục), đổi
    # lấy việc ghim thật sự có hiệu lực. Đánh đổi này đúng theo định nghĩa: đã
    # ghim tức là câu hỏi "video nào" coi như đã trả lời, chỉ còn "khung nào".
    phu_cu = None
    svc = getattr(pipeline, "retrieval_service", None)
    if svc is not None and getattr(svc, "phu", None):
        phu_cu, svc.phu = svc.phu, []
    # TRAKE: `TrakeConfig.paths_per_video` mặc định 8, tức một video sinh tối đa
    # 8 alignment. Hợp lý khi phải chia 100 slot cho ~12 video ứng viên; VÔ LÝ
    # khi đã ghim đúng một video — 42/50 slot sẽ bỏ trống. Ghim rồi thì toàn bộ
    # ngân sách nên đổ vào các cách căn thời gian KHÁC NHAU trong video đó, vì
    # đó chính là chỗ bất định còn lại (keyframe BTC cách nhau trung vị 2,16s
    # còn cửa sổ chấm TRAKE dưới 10 khung ≈ 0,4s).
    #
    # CHỈ khi ghim ĐÚNG MỘT video. Với `nhom` thì còn hàng chục video ứng viên,
    # nên chỗ bất định vẫn là "video nào" — bơm `paths_per_video` lúc đó sẽ để
    # một video nuốt sạch 100 slot, đúng cái mà `nhom` sinh ra để tránh.
    if (query.task == "trake" and len(videos) == 1
            and getattr(pipeline, "trake_solver", None)):
        import dataclasses
        s = pipeline.trake_solver
        s.config = dataclasses.replace(
            s.config, paths_per_video=max(s.config.paths_per_video,
                                          ghim.so_dong))
    try:
        rows_ghim = pipeline.answer(query, top_k=min(ghim.so_dong, top_k))
    finally:
        if phu_cu is not None:
            svc.phu = phu_cu

    # Lưới cuối: dù đường truy hồi có đổi thế nào, dòng "ghim" phải thuộc video
    # được ghim. Nếu không thì đây không còn là ghim, chỉ là chạy lại tốn giờ.
    muon = {v.upper() for v in videos}
    la = [r for r in rows_ghim if str(r["video_id"]).upper() not in muon]
    rows_ghim = [r for r in rows_ghim if str(r["video_id"]).upper() in muon]
    if la:
        raise RuntimeError(
            f"{ghim.qid}: {len(la)} dòng ghim rơi ra ngoài video được ghim "
            f"(vd {la[0]['video_id']}). Chỉ mục chưa được cắt sạch — xem "
            f"phần bỏ encoder phụ trong `ap_dung`.")

    _va_dap_an(query.task, rows_nguoi, ghim, rows_ghim, rows_goc)
    rows = gop(rows_nguoi + rows_ghim[:ghim.so_dong], rows_goc, query.task,
               top_k, ghim.answer, ghim.answers)

    hang_cu = _hang_dau_tien(rows_goc, query.task, videos)
    # Nói rõ CẢ HAI phần. Bản trước chỉ báo phần video, nên khi ghim thêm
    # `khung` thì bản in trông y hệt lúc không có khung — không nhìn ra được
    # phần đắt nhất (khung đúng lên hạng 1) có chạy hay không.
    phan_khung = (f"{len(rows_nguoi)} KHUNG người chọn + " if rows_nguoi else "")
    bao = (f"{ghim.qid}: {phan_khung}{len(rows_ghim)} dòng của "
           f"{mo_ta_dich} lên đầu ({len(ev)} keyframe được chấm "
           f"lại); trước khi ghim dòng đầu tiên trúng đích đứng hạng "
           + (f"{hang_cu}" if hang_cu else "KHÔNG có trong 100 dòng"))
    return rows, bao


def _va_dap_an(task: str, rows_nguoi: List[Dict], ghim: "Ghim",
               rows_ghim: Sequence[Dict], rows_goc: Sequence[Dict]) -> None:
    """Vá đáp án RỖNG cho các dòng `khung` của câu QA. Sửa tại chỗ.

    🔴 LỖI ĐO ĐƯỢC 28/08, sơ tuyển đợt 2, `query-p2-23-qa`. `khung` được thêm
    ngày 21/08 cho KIS — nơi một dòng chỉ cần `(video, frame)` — nên
    `_row_tu_khung` lấy đáp án từ `ghim.answer`, và `ghim.answer` rỗng khi người
    vận hành CHỦ Ý không đè đáp án (vì đáp án phải nhìn ảnh mới ra). Kết quả:

        đáp án sau ghim: ['', '15', '25']      <- 4 dòng ĐẦU rỗng
        CSV: `L25_V012,15268,`                 <- 0 điểm chắc chắn

    Tệ hơn cả mất điểm: `validate_csv` KHÔNG coi trường rỗng là lỗi định dạng,
    nên zip vẫn báo "hợp lệ" và bài nộp trông không có gì bất thường. Đúng
    chủng loại lỗi câm mà repo này chống suốt tháng 8. Ở ca đo được, máy vốn đã
    để đúng video ở HẠNG 1 với đáp án `15`, nên ghim khung đổi 1,0 điểm thành 0.

    Vá bằng đáp án của dòng ghim hạng cao nhất (đó là khung mà VLM tin nhất
    trong đúng video được ghim), không có thì lấy của bài gốc. KHÔNG bịa: nếu cả
    hai nguồn đều rỗng thì để nguyên rỗng để bên gọi còn thấy mà sửa.
    """
    if task != "qa" or not rows_nguoi or ghim.answer:
        return

    def _dau(nguon):
        return next((str(r.get("answer") or "").strip() for r in nguon
                     if str(r.get("answer") or "").strip()), "")

    dap = _dau(rows_ghim) or _dau(rows_goc)
    if not dap:
        return
    for r in rows_nguoi:
        if not str(r.get("answer") or "").strip():
            r["answer"] = dap


def _hang_dau_tien(rows: Sequence[Dict], task: str,
                   videos: Sequence[str]) -> Optional[int]:
    muon = {str(v).upper() for v in videos}
    for i, r in enumerate(rows, start=1):
        if str(r["video_id"]).upper() in muon:
            return i
    return None
