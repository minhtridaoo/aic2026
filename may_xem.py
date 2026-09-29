#!/usr/bin/env python
"""MÁY XEM — đưa MẮT NGƯỜI vào vòng lặp, tại chỗ, nhiều người cùng lúc.

    python -X utf8 may_xem.py --config config_ocr_asr.yaml \
        --de data/de_dot2.json --ghim data/ghim_dot2.json --sau 300

Rồi mở `http://127.0.0.1:8000`. Muốn đồng đội cùng bấm thì `--host 0.0.0.0` và
họ mở `http://<IP máy này>:8000` trong cùng mạng LAN.

VÌ SAO ĐÂY LÀ VIỆC ĐÁNG GIÁ NHẤT CÒN LẠI
----------------------------------------
Số đo của chính hệ thống này (120 câu KIS, đúng cấu hình thi) và của bảng xếp
hạng thật:

    R@100  = 57,5%     khung ĐÚNG đã nằm trong 100 dòng máy trả về
    Final  = 0,3450    nhưng chỉ ăn chừng này, vì nó không đứng hạng 1
    (đề thật 20/08: nộp 5,6/12 = 0,467 điểm/câu trên cùng dạng chênh lệch)

`Final = (1/5)·Σ_{k∈{1,5,20,50,100}} R@k` với `R@k` lấy max trên tiền tố. Cùng
một khung: đứng hạng 100 được 0,2 điểm, đứng hạng 1 được 1,0. Kéo nó lên hạng 1
KHÔNG cần mô hình tốt hơn — chỉ cần một người nhìn thấy nó.

Trần của việc nhìn, đo bằng `chan_doan_qa_truot.py` theo độ sâu pool D:

    D          100    200    500   1000   2000
    KIS /120    67     73     90     97    106      (56% -> 75% -> 88%)
    QA  /30     12     16     19     20     24

Nên `--sau` mặc định 300 chứ không phải 100: xem sâu hơn là cách RẺ NHẤT để
nâng trần, và nó không tốn một request API nào.

VÌ SAO LÀ SERVER CHỨ KHÔNG PHẢI HTML TĨNH (`xem_anh.py`)
--------------------------------------------------------
`xem_anh.py` nhúng ảnh base64 vào HTML để chạy được trên Kaggle rồi tải về. Nó
đúng cho hoàn cảnh đó, nhưng đêm thi thì có ba chỗ chết người:

  1. Vòng lặp Kaggle (sinh HTML -> tải zip -> mở) mất hàng chục phút. Có 3 giờ.
  2. Người bấm xong phải TỰ TAY dán JSON vào file ghim. Đó là thao tác gõ tay
     lúc 22 giờ, đúng loại việc sinh ra lỗi câm mà repo này chống suốt tháng 8.
  3. 300 ảnh base64 cho 24 câu là hàng trăm MB HTML.

Ở đây ảnh nằm dưới đĩa (28,66 GiB keyframe của BTC), server đọc thẳng, thu nhỏ
trong bộ nhớ, và mỗi cú bấm được GHI NGAY vào file ghim. Không có bước gõ tay
nào giữa mắt người và bài nộp.

HAI CHẾ ĐỘ XEM, ỨNG VỚI HAI KIỂU TRƯỢT ĐÃ CHẨN ĐOÁN (`chan_doan_qa_truot.py`)
-----------------------------------------------------------------------------
  * `/q/<qid>`  — lưới ứng viên của máy. Nhắm nhóm A (máy đã chạm khung đúng,
    chỉ xếp sai hạng): 56% câu KIS.
  * `/v/<vid>`  — TOÀN BỘ keyframe của một video theo thời gian. Nhắm nhóm B
    (đúng video, sai khung): 22% câu KIS, hạng trung vị 358 — tức máy không
    bao giờ tự đưa lên top, nhưng người đã biết video thì lướt 200 ảnh là thấy.
    Đây cũng là chế độ dùng sau khi OCR/ASR/tiêu đề đã ghim được video.

FILE GHIM LÀ NGUỒN SỰ THẬT DUY NHẤT
-----------------------------------
Mỗi cú bấm ghi lại `khung` của câu đó trong file `--ghim`, giữ nguyên mọi khoá
khác (`videos`, `nhom`, `answer`, `so_dong`, `ghi_chu`) do người viết tay. Sau
đó nộp bằng đúng đường cũ, không có đường tắt nào:

    python -X utf8 nop_bai.py --config config_ocr_asr.yaml \
        --truy-van data/de_dot2.json --ghim data/ghim_dot2.json \
        --ra out/NOP/dot2_B.zip

`aic2026/core/ghim.py::ap_dung` đặt KHUNG người chọn lên trước tất cả, rồi mới
tới phần của máy làm lưới an toàn.

⚠ MỘT CÂU = MỘT NGƯỜI. Lưu là GHI ĐÈ `khung` của câu đó, nên hai người cùng bấm
một câu sẽ giẫm chân nhau. Trang chủ có ô nhận việc để thấy ai đang làm câu nào.
"""
import argparse
import html
import io
import json
import os
import socket
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as cau_hinh
from aic2026.core import dataset
from aic2026.core import viet_lai_truy_van
from aic2026.data_plane.image_resolver import (EventImageResolver,
                                               ImageResolutionError)
from nop_bai import doc_truy_van

GOC = os.path.dirname(os.path.abspath(__file__))


def ghi_ghim(path, ghim_tho, qid, khung, ai=""):
    """Cập nhật `khung` của MỘT câu trong file ghim, tại chỗ, nguyên tử.

    Ba điều bắt buộc, cả ba đều rút ra từ lỗi đã gặp:

    1. **Giữ nguyên các khoá khác.** `videos` / `nhom` / `answer` / `so_dong` /
       `ghi_chu` là thứ người vận hành viết tay sau khi tra OCR/ASR/tiêu đề
       (xem `data/ghim_dot1.json`). Ghi đè cả mục là xoá công của họ.
    2. **Ghi qua file tạm rồi `os.replace`.** Người bấm ảnh liên tục, mỗi cú
       bấm là một lượt ghi; nếu tiến trình bị Ctrl+C đúng lúc ghi thì file ghim
       thành JSON dở dang và `nop_bai.py` không đọc được — mất TẤT CẢ lựa chọn
       của cả buổi. `os.replace` là nguyên tử trên cùng ổ đĩa.
    3. **Bỏ khoá `khung` khi danh sách rỗng** thay vì để `[]`, để file ghim đọc
       lên vẫn nói đúng điều đang có.
    """
    muc = dict(ghim_tho.get(qid) or {})
    if khung:
        muc["khung"] = [[str(k[0]).upper()] + [int(x) for x in k[1:]]
                        for k in khung]
    else:
        muc.pop("khung", None)
    if ai:
        muc["_nguoi"] = ai
    muc["_luc"] = time.strftime("%d/%m %H:%M")
    if muc:
        ghim_tho[qid] = muc
    else:
        ghim_tho.pop(qid, None)
    tam = str(path) + ".tam"
    with open(tam, "w", encoding="utf-8") as fh:
        json.dump(ghim_tho, fh, ensure_ascii=False, indent=2)
    os.replace(tam, path)
    return ghim_tho


def _args():
    p = argparse.ArgumentParser(description="Server xem ảnh cho vòng lặp người–máy")
    p.add_argument("--config", default=None)
    p.add_argument("--de", required=True, help="file đề JSON (doc_de_thi.py)")
    p.add_argument("--ghim", default=os.path.join(GOC, "data", "ghim_xem.json"),
                   help="file ghim để GHI vào; đọc lại nếu đã có")
    p.add_argument("--sau", type=int, default=300,
                   help="độ sâu pool đem ra cho người xem (mặc định 300)")
    p.add_argument("--cong", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1",
                   help="0.0.0.0 để đồng đội trong LAN mở được")
    p.add_argument("--rong", type=int, default=210, help="bề ngang ảnh thu nhỏ")
    p.add_argument("--chi", default="", help="chỉ nạp các qid này, ngăn bằng phẩy")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# KHO — nạp một lần, dùng suốt buổi
# --------------------------------------------------------------------------- #
class Kho:
    """Dữ liệu + pipeline + trạng thái người bấm. Một thể hiện cho cả server."""

    def __init__(self, a):
        self.a = a
        self.config = cau_hinh.load_config(a.config)
        self.goc_data = (self.config.get("offline") or {}).get("btc_data_root")
        self.resolver = EventImageResolver.from_config(self.config)
        if not self.resolver.roots:
            raise SystemExit(
                "❌ `qa.image_roots` rỗng — không có ảnh thì trang này vô nghĩa.\n"
                "   Thêm vào config:\n"
                "     qa:\n"
                "       image_roots: ['D:\\COMP\\AIC2026\\data cuộc thi\\keyframes']\n"
                "   rồi chạy `python kiem_resolver_anh.py` để kiểm trước.")

        chinh_sach = cau_hinh.ngon_ngu_truy_van(self.config)
        self.truy_van, ban_ghi, _ = doc_truy_van(a.de, chinh_sach)
        chi = {x.strip() for x in a.chi.split(",") if x.strip()}
        if chi:
            giu = [i for i, q in enumerate(self.truy_van) if q.qid in chi]
            self.truy_van = [self.truy_van[i] for i in giu]
            ban_ghi = [ban_ghi[i] for i in giu]

        # Truy vấn phải đi qua ĐÚNG bước viết lại như `nop_bai.py`, nếu không thì
        # ứng viên người xem sẽ khác ứng viên được nộp — người bấm đúng trên một
        # danh sách không tồn tại trong bài nộp. Cache đĩa nên gần như 0 request.
        lech = [bg for bg in ban_ghi if not bg.theo_chinh_sach]
        if lech:
            bo, vi_sao = viet_lai_truy_van.tu_config(self.config)
            if bo is None:
                print(f"  ⚠ KHÔNG viết lại truy vấn ({vi_sao}) — pool người xem "
                      f"sẽ KHÁC pool của nop_bai.py nếu ở đó viết lại được.")
            else:
                print(f"  Viết lại truy vấn bằng {vi_sao} ...", flush=True)
                for dong in viet_lai_truy_van.ap_dung(bo, self.truy_van, ban_ghi):
                    print(f"     {dong}")

        t = time.time()
        print("Đang nạp dữ liệu ...", flush=True)
        events, emb, _metas, rep = cau_hinh.nap_du_lieu(self.config)
        print(f"  {rep.events:,d} event / {rep.videos_da_nap} video "
              f"({time.time() - t:.1f}s)")
        # qa_cheap: trang này chỉ cần ỨNG VIÊN KHUNG. Gọi VLM ở đây là đốt quota
        # cho những đáp án mà `nop_bai.py` lát nữa sẽ tính lại.
        self.p = cau_hinh.dung_pipeline(self.config, events, emb, qa_cheap=True)
        self.events = events
        print(f"  encoder: {type(self.p.base_encoder).__name__} "
              f"({self.p.base_encoder.dim} chiều)")

        self._rows = {}          # qid -> [dòng]
        self._kf = {}            # video -> [KeyframeRow]
        self._f2n = {}           # video -> {frame_idx: n}
        self._paths = None
        self._thumb = {}         # (path, w) -> bytes JPEG
        self._khoa = threading.Lock()
        self._khoa_anh = threading.Lock()
        # `Pipeline` giữ trạng thái (chỉ mục, last_trace) và KHÔNG an toàn khi
        # nhiều luồng gọi cùng lúc. Server này là ThreadingHTTPServer và hai
        # người mở hai câu cùng lúc là chuyện bình thường, nên mọi lượt chấm
        # phải xếp hàng. Chỉ chậm ở lần đầu mỗi câu — sau đó `_rows` đã có.
        self._khoa_chay = threading.Lock()
        self.nhan = {}           # qid -> tên người nhận việc
        self.chon = {}           # qid -> [[video, frame...], ...]
        self._doc_ghim_cu()

    # -- đề ----------------------------------------------------------------- #
    def q(self, qid):
        for q in self.truy_van:
            if q.qid == qid:
                return q
        return None

    def de_text(self, q):
        if q.task == "trake":
            return " → ".join(se.text for se in q.sub_events)
        return q.text

    def n_events(self, q):
        return len(q.sub_events) if q.task == "trake" else 1

    # -- ứng viên ----------------------------------------------------------- #
    def rows(self, qid):
        """Top-`--sau` dòng của máy cho câu này. Tính một lần rồi giữ."""
        with self._khoa:
            if qid in self._rows:
                return self._rows[qid]
        q = self.q(qid)
        with self._khoa_chay:
            with self._khoa:                      # người khác vừa chấm xong
                if qid in self._rows:
                    return self._rows[qid]
            t = time.time()
            rows = self.p.answer(q, top_k=self.a.sau)
            # `flush` không phải chi tiết vặt: stdout ra pipe/Tee-Object là đệm
            # theo KHỐI, nên không có nó thì người vận hành nhìn màn hình trống
            # suốt lúc nung và tưởng máy treo, giữa 3 giờ làm bài.
            print(f"  {qid}: {len(rows)} ứng viên ({time.time() - t:.1f}s)",
                  flush=True)
            with self._khoa:
                self._rows[qid] = rows
        return rows

    def da_co(self, qid):
        with self._khoa:
            return qid in self._rows

    def nong(self):
        """Tính trước pool của MỌI câu, ngay sau khi server lên.

        Đo thật 25/08 trên máy này: câu ĐẦU TIÊN mất 37s (gần hết là một lần
        `dataset.scan` 873 video + nạp map-keyframes của các video trong kết
        quả), các câu sau chỉ 2,1s, và trang đã tính rồi thì 3ms.
        24 câu × 2,1s ≈ 50s — nếu để người bấm mới tính thì đó là 50 giây
        ngồi chờ RẢI RÁC giữa buổi làm bài 3 giờ, đúng lúc đang cần nghĩ.
        Nung sẵn ở đây thì người không bao giờ chờ.
        """
        t = time.time()
        for q in self.truy_van:
            try:
                self.rows(q.qid)
            except Exception as exc:                          # noqa: BLE001
                print(f"  ⚠ {q.qid}: không tính được pool ({exc})")
        print(f"\n  ✅ đã nung sẵn {len(self._rows)}/{len(self.truy_van)} câu "
              f"trong {time.time() - t:.0f}s — mở trang là hiện ngay.\n",
              flush=True)

    # -- ảnh ---------------------------------------------------------------- #
    def keyframes(self, vid):
        """[KeyframeRow] của một video, theo thứ tự thời gian."""
        vid = vid.upper()
        with self._khoa:
            if vid in self._kf:
                return self._kf[vid]
            if self._paths is None:
                self._paths = dataset.scan(self.goc_data)
        vp = self._paths.get(vid)
        rows = (dataset.load_map_keyframes(vp.mapping)
                if vp is not None and vp.mapping else [])
        with self._khoa:
            self._kf[vid] = rows
        return rows

    def n_cua(self, vid, frame_idx):
        """frame_idx (số khung video gốc) -> n (số thứ tự keyframe = tên file).

        Hai thứ này KHÁC nhau: `L26_V004` có khung 4803 nhưng ảnh là `061.jpg`.
        Nhầm chỗ này thì trang hiện toàn ảnh sai mà không báo lỗi gì.
        """
        vid = vid.upper()
        with self._khoa:
            bang = self._f2n.get(vid)
        if bang is None:
            bang = {int(r.frame_idx): int(r.n) for r in self.keyframes(vid)}
            with self._khoa:
                self._f2n[vid] = bang
        return bang.get(int(frame_idx))

    def duong_anh(self, vid, n):
        try:
            return self.resolver.resolve(f"{vid.upper()}#{int(n):05d}")
        except (ImageResolutionError, ValueError):
            return None

    def thumb(self, vid, n, rong):
        """JPEG đã thu nhỏ. `rong <= 0` = trả nguyên ảnh gốc."""
        path = self.duong_anh(vid, n)
        if path is None:
            return None
        khoa = (str(path), int(rong))
        with self._khoa_anh:
            if khoa in self._thumb:
                return self._thumb[khoa]
        if rong <= 0:
            with open(path, "rb") as fh:
                data = fh.read()
        else:
            try:
                from PIL import Image
                im = Image.open(path).convert("RGB")
                w, h = im.size
                if w > rong:
                    im = im.resize((rong, max(1, round(h * rong / w))),
                                   Image.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=74)
                data = buf.getvalue()
            except Exception:                                    # noqa: BLE001
                return None
        with self._khoa_anh:
            # Trần thô để một buổi thi 3 giờ không nuốt hết RAM. 6000 ảnh
            # ~12 KB = 72 MB, thừa cho 24 câu × 300 ảnh đã xem qua.
            if len(self._thumb) > 6000:
                self._thumb.clear()
            self._thumb[khoa] = data
        return data

    def nong_anh(self, cap):
        """Dựng sẵn ảnh thu nhỏ cho một danh sách `(video, n)`, chạy nền.

        Trình duyệt đã `loading="lazy"` nên chỉ xin ảnh đang nhìn thấy; nhưng
        người xem CUỘN rất nhanh (300 ảnh/câu), và mỗi ảnh là một lần giải mã
        JPEG 1280×720 (~20ms). Nung trước bằng vài luồng thì lúc cuộn tới ảnh
        đã nằm trong bộ nhớ. Bỏ qua mọi lỗi: đây là tối ưu, không phải chức năng.
        """
        def chay():
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as bom:
                for vid, n in cap:
                    if n is not None:
                        bom.submit(self.thumb, vid, n, self.a.rong)
        threading.Thread(target=chay, daemon=True).start()

    # -- ghim --------------------------------------------------------------- #
    def _doc_ghim_cu(self):
        self.ghim_tho = {}
        if os.path.exists(self.a.ghim):
            with open(self.a.ghim, "r", encoding="utf-8") as fh:
                self.ghim_tho = json.load(fh)
            for qid, muc in self.ghim_tho.items():
                if str(qid).startswith("_") or not isinstance(muc, dict):
                    continue
                khung = []
                for x in (muc.get("khung") or []):
                    if isinstance(x, str):
                        x = x.split("|")
                    khung.append([str(x[0]).upper()] + [int(v) for v in x[1:]])
                if khung:
                    self.chon[qid] = khung
            n = sum(len(v) for v in self.chon.values())
            print(f"  đọc lại {self.a.ghim}: {len(self.chon)} câu đã có "
                  f"{n} khung người chọn")

    def luu(self, qid, khung, ai=""):
        """Ghi `khung` của MỘT câu vào file ghim, giữ nguyên các khoá khác."""
        with self._khoa:
            self.chon[qid] = khung
            ghi_ghim(self.a.ghim, self.ghim_tho, qid, khung, ai)
        return len(khung)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
CSS = """
:root{color-scheme:dark}
*{box-sizing:border-box}
body{background:#14161a;color:#e6e8eb;font:14px/1.55 system-ui,sans-serif;
     margin:0;padding:16px 20px 120px}
a{color:#7fb2ff;text-decoration:none}
a:hover{text-decoration:underline}
h1{font-size:18px;margin:0 0 4px}
.de{background:#1d2026;border-left:3px solid #7fb2ff;padding:10px 14px;
    margin:10px 0 14px;border-radius:0 6px 6px 0;max-width:1150px}
.thanh{position:sticky;top:0;background:#14161ae8;backdrop-filter:blur(6px);
       padding:8px 0 10px;margin-bottom:6px;z-index:9;border-bottom:1px solid #2b3038}
.luoi{display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--w),1fr));gap:9px}
.o{background:#1d2026;border:2px solid #2b3038;border-radius:7px;padding:4px;
   cursor:pointer;position:relative}
.o:hover{border-color:#4a5568}
.o.chon{border-color:#4ade80;background:#16281d}
.o img{width:100%;display:block;border-radius:4px;min-height:40px;background:#22262e}
.o .nhan{font:11px ui-monospace,monospace;color:#98a2b3;padding:3px 2px 0;
         display:flex;justify-content:space-between;gap:6px}
.o.chon .nhan{color:#4ade80}
.thutu{position:absolute;top:6px;left:6px;background:#4ade80;color:#0b2016;
       font-weight:700;font-size:12px;border-radius:4px;padding:1px 6px}
.mat{background:#2a1d1d;color:#f0a0a0;font:11px monospace;padding:20px 4px;
     text-align:center;border-radius:4px}
table{border-collapse:collapse;width:100%;max-width:1150px}
td,th{border-bottom:1px solid #2b3038;padding:7px 10px;text-align:left;font-size:13px}
th{color:#98a2b3;font-weight:600;font-size:12px}
tr:hover td{background:#1a1d23}
button{background:#2b3038;color:#e6e8eb;border:0;border-radius:5px;padding:6px 13px;
       cursor:pointer;font-size:13px;margin-right:7px}
button:hover{background:#3a4150}
button.chinh{background:#2f6b45}
button.chinh:hover{background:#3a8355}
input{background:#1d2026;color:#e6e8eb;border:1px solid #2b3038;border-radius:5px;
      padding:6px 9px;font-size:13px}
.mo{color:#98a2b3;font-size:12px}
.xanh{color:#4ade80}
.vang{color:#fbbf24}
"""

JS_CHUNG = """
function ten(){
  let t = localStorage.getItem('ai');
  if(!t){ t = prompt('Tên bạn (để biết ai đang làm câu nào):') || 'ẩn danh';
          localStorage.setItem('ai', t); }
  return t;
}
async function gui(url, du_lieu){
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'},
                             body: JSON.stringify(du_lieu)});
  return r.json();
}
"""


def _trang(tieu_de, than, css_bien=""):
    return f"""<!doctype html><html lang="vi"><meta charset="utf-8">
<title>{html.escape(tieu_de)}</title><style>{CSS}</style>
<body style="{css_bien}">{than}</body></html>"""


def trang_chu(kho):
    hang = []
    for q in kho.truy_van:
        n = len(kho.chon.get(q.qid, []))
        ai = kho.nhan.get(q.qid, "")
        trang_thai = (f'<span class="xanh">{n} khung</span>' if n
                      else '<span class="mo">chưa chọn</span>')
        if not kho.da_co(q.qid):
            trang_thai += ' <span class="vang">· đang tính pool</span>'
        hang.append(
            f'<tr><td><a href="/q/{html.escape(q.qid)}">{html.escape(q.qid)}</a></td>'
            f'<td>{q.task.upper()}</td>'
            f'<td>{trang_thai}</td>'
            f'<td><span class="vang">{html.escape(ai)}</span></td>'
            f'<td class="mo">{html.escape(kho.de_text(q)[:95])}…</td></tr>')
    tong = sum(len(v) for v in kho.chon.values())
    return _trang("Máy xem", f"""
<h1>Máy xem — {len(kho.truy_van)} câu · pool sâu {kho.a.sau}</h1>
<div class="mo">Bấm vào câu để xem lưới ảnh. Mỗi cú bấm ghi thẳng vào
  <code>{html.escape(kho.a.ghim)}</code> — không phải dán JSON.<br>
  Đang có <b class="xanh">{tong}</b> khung người chọn.
  Nộp bằng: <code>nop_bai.py --ghim {html.escape(os.path.basename(kho.a.ghim))}</code></div>
<p><button onclick="localStorage.removeItem('ai');ten();location.reload()">Đổi tên</button>
   <button onclick="location.reload()">Làm mới</button></p>
<table><tr><th>câu</th><th>dạng</th><th>đã chọn</th><th>ai làm</th><th>đề</th></tr>
{''.join(hang)}</table>
<script>{JS_CHUNG}
window.addEventListener('DOMContentLoaded', ten);
setInterval(()=>location.reload(), 30000);
</script>""")


def _o_anh(kho, vid, frames, nhan_trai, nhan_phai, rong):
    """Một ô ảnh. `frames` nhiều phần tử = một hàng TRAKE."""
    khoa = "|".join([vid] + [str(f) for f in frames])
    imgs = []
    for f in frames:
        n = kho.n_cua(vid, f)
        if n is None:
            imgs.append('<div class="mat">không có map</div>')
        else:
            imgs.append(f'<img loading="lazy" src="/anh?v={vid}&n={n}&w={rong}">')
    than = ('<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:3px">'
            + "".join(imgs) + "</div>") if len(imgs) > 1 else imgs[0]
    return (f'<div class="o" data-k="{html.escape(khoa)}" onclick="bam(this)">'
            f'{than}<div class="nhan"><span>{html.escape(nhan_trai)}</span>'
            f'<span>{html.escape(nhan_phai)}</span></div></div>')


def trang_cau(kho, qid):
    q = kho.q(qid)
    rows = kho.rows(qid)
    rong = kho.a.rong
    o, giay_cua = [], {}
    for i, r in enumerate(rows, 1):
        vid = str(r["video_id"]).upper()
        frames = ([int(x) for x in r["frames"]] if q.task == "trake"
                  else [int(r["frame_idx"])])
        if vid not in giay_cua:
            giay_cua[vid] = {int(k.frame_idx): float(k.pts_time)
                             for k in kho.keyframes(vid)}
        giay = giay_cua[vid].get(frames[0])
        o.append(_o_anh(kho, vid, frames, f"#{i} {vid}",
                        f"{giay:.0f}s" if giay is not None else "", rong))
    da = json.dumps(kho.chon.get(qid, []))
    return _trang(qid, f"""
<div class="thanh">
  <h1>{html.escape(qid)} · {q.task.upper()} · {len(rows)} ứng viên
      <a href="/" style="font-size:13px;font-weight:400">← danh sách</a></h1>
  <span id="tt" class="mo"></span>
  <button class="chinh" onclick="nhan_viec()">Nhận câu này</button>
  <button onclick="xoa()">Bỏ chọn hết</button>
  <input id="vid" placeholder="mở cả video: L21_V014" size="14">
  <button onclick="mo_video()">Xem cả video →</button>
</div>
<div class="de">{html.escape(kho.de_text(q))}</div>
<div class="mo" style="margin-bottom:10px">
  Bấm ảnh ĐÚNG. Thứ tự bấm = thứ tự dòng nộp, nên bấm cái chắc nhất trước.
  {"Câu TRAKE: mỗi ô là một bộ " + str(kho.n_events(q)) + " khung." if q.task == "trake" else ""}
</div>
<div class="luoi">{''.join(o)}</div>
<script>{JS_CHUNG}
const QID = {json.dumps(qid)};
let CHON = {da};
function khoa_cua(k){{ return k.join('|'); }}
function ve(){{
  document.querySelectorAll('.thutu').forEach(e=>e.remove());
  document.querySelectorAll('.o').forEach(e=>e.classList.remove('chon'));
  CHON.forEach((k,i)=>{{
    const el = document.querySelector('[data-k="'+CSS.escape(khoa_cua(k))+'"]');
    if(!el) return;
    el.classList.add('chon');
    const b = document.createElement('div');
    b.className='thutu'; b.textContent=(i+1); el.prepend(b);
  }});
  document.getElementById('tt').textContent =
    CHON.length ? CHON.length + ' khung đã chọn — đã ghi vào file ghim'
                : 'chưa chọn khung nào';
}}
async function luu(){{ await gui('/chon', {{qid:QID, khung:CHON, ai:ten()}}); ve(); }}
function bam(el){{
  const k = el.dataset.k.split('|');
  k.slice(1).forEach((x,i)=>k[i+1]=parseInt(x));
  const i = CHON.findIndex(c=>khoa_cua(c)===el.dataset.k);
  if(i>=0) CHON.splice(i,1); else CHON.push(k);
  luu();
}}
function xoa(){{ CHON=[]; luu(); }}
async function nhan_viec(){{ await gui('/nhan', {{qid:QID, ai:ten()}});
  document.getElementById('tt').textContent = 'bạn đang giữ câu này'; }}
function mo_video(){{
  const v = document.getElementById('vid').value.trim().toUpperCase();
  if(v) location.href = '/v/' + v + '?q=' + encodeURIComponent(QID);
}}
window.addEventListener('DOMContentLoaded', ()=>{{ten();ve();}});
</script>""", f"--w:{rong}px")


def trang_video(kho, vid, qid):
    q = kho.q(qid)
    kfs = kho.keyframes(vid)
    if not kfs:
        return _trang(vid, f'<h1>{html.escape(vid)}</h1>'
                           f'<p class="mat">Không có map-keyframes cho video này.</p>'
                           f'<p><a href="/q/{html.escape(qid)}">← về câu</a></p>')
    rong = kho.a.rong
    o = [_o_anh(kho, vid, [int(k.frame_idx)], f"kf {k.n}", f"{k.pts_time:.0f}s", rong)
         for k in kfs]
    da = json.dumps(kho.chon.get(qid, []))
    ne = kho.n_events(q)
    return _trang(f"{vid} · {qid}", f"""
<div class="thanh">
  <h1>{html.escape(vid)} · toàn bộ {len(kfs)} keyframe · cho câu
      <a href="/q/{html.escape(qid)}">{html.escape(qid)}</a></h1>
  <span id="tt" class="mo"></span>
  <button onclick="xoa()">Bỏ chọn hết</button>
  <a href="/q/{html.escape(qid)}"><button>← lưới ứng viên</button></a>
</div>
<div class="de">{html.escape(kho.de_text(q))}</div>
<div class="mo" style="margin-bottom:10px">
  Đây là chế độ cho nhóm B (ĐÚNG video, SAI khung — 22% câu KIS, hạng trung vị
  358 nên máy không bao giờ tự đưa lên top).
  {"Câu TRAKE " + str(ne) + " sự kiện: bấm ĐÚNG THỨ TỰ E1→E" + str(ne) + ", đủ " + str(ne) + " khung thì tự gộp thành một dòng." if q.task == "trake" else ""}
</div>
<div class="luoi">{''.join(o)}</div>
<script>{JS_CHUNG}
const QID = {json.dumps(qid)}, VID = {json.dumps(vid)}, NE = {ne};
let CHON = {da};
// Trang này bấm từng KHUNG một. Với TRAKE thì `khung` phải là [video, f1..fN],
// nên gom NE cú bấm liên tiếp trong cùng video thành MỘT mục.
let dem = [];
function ve(){{
  document.querySelectorAll('.thutu').forEach(e=>e.remove());
  document.querySelectorAll('.o').forEach(e=>e.classList.remove('chon'));
  const phang = [];
  CHON.forEach(c=>{{ if(c[0]===VID) c.slice(1).forEach(f=>phang.push(f)); }});
  dem.forEach(f=>phang.push(f));
  phang.forEach((f,i)=>{{
    const el = document.querySelector('[data-k="'+CSS.escape(VID+'|'+f)+'"]');
    if(!el) return;
    el.classList.add('chon');
    const b=document.createElement('div'); b.className='thutu';
    b.textContent = NE>1 ? 'E'+((i%NE)+1) : (i+1);
    el.prepend(b);
  }});
  document.getElementById('tt').textContent =
    CHON.length + ' dòng đã ghi' + (dem.length ? ' · đang gom '+dem.length+'/'+NE : '');
}}
async function luu(){{ await gui('/chon', {{qid:QID, khung:CHON, ai:ten()}}); ve(); }}
function bam(el){{
  const f = parseInt(el.dataset.k.split('|')[1]);
  const i = CHON.findIndex(c=>c[0]===VID && c.length===2 && c[1]===f);
  if(NE===1){{
    if(i>=0) CHON.splice(i,1); else CHON.push([VID,f]);
    luu(); return;
  }}
  dem.push(f);
  if(dem.length>=NE){{ CHON.push([VID].concat(dem)); dem=[]; luu(); }}
  else ve();
}}
function xoa(){{ CHON=[]; dem=[]; luu(); }}
window.addEventListener('DOMContentLoaded', ()=>{{ten();ve();}});
</script>""", f"--w:{rong}px")


# --------------------------------------------------------------------------- #
# SERVER
# --------------------------------------------------------------------------- #
class Tay(BaseHTTPRequestHandler):
    kho: Kho = None

    def log_message(self, *a):                     # im lặng: 300 ảnh/trang
        pass

    def _tra(self, ma, kieu, than):
        if isinstance(than, str):
            than = than.encode("utf-8")
        self.send_response(ma)
        self.send_header("Content-Type", kieu)
        self.send_header("Content-Length", str(len(than)))
        self.end_headers()
        try:
            self.wfile.write(than)
        except (BrokenPipeError, ConnectionResetError):
            pass                                   # người xem đã sang trang khác

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        p, tv = u.path, urllib.parse.parse_qs(u.query)
        kho = self.kho
        try:
            if p == "/":
                return self._tra(200, "text/html; charset=utf-8", trang_chu(kho))
            if p == "/anh":
                vid = (tv.get("v") or [""])[0].upper()
                n = int((tv.get("n") or ["0"])[0])
                w = int((tv.get("w") or [str(kho.a.rong)])[0])
                data = kho.thumb(vid, n, w)
                if data is None:
                    return self._tra(404, "text/plain; charset=utf-8",
                                     f"không có ảnh {vid}#{n:05d}")
                return self._tra(200, "image/jpeg", data)
            if p.startswith("/q/"):
                qid = urllib.parse.unquote(p[3:])
                if kho.q(qid) is None:
                    return self._tra(404, "text/plain; charset=utf-8",
                                     f"không có câu {qid}")
                trang = trang_cau(kho, qid)
                q = kho.q(qid)
                kho.nong_anh([
                    (str(r["video_id"]).upper(),
                     kho.n_cua(str(r["video_id"]).upper(),
                               (r["frames"][0] if q.task == "trake"
                                else r["frame_idx"])))
                    for r in kho.rows(qid)])
                return self._tra(200, "text/html; charset=utf-8", trang)
            if p.startswith("/v/"):
                vid = urllib.parse.unquote(p[3:]).upper()
                qid = (tv.get("q") or [""])[0]
                if kho.q(qid) is None:
                    return self._tra(404, "text/plain; charset=utf-8",
                                     "thiếu ?q=<qid> hợp lệ")
                trang = trang_video(kho, vid, qid)
                kho.nong_anh([(vid, int(k.n)) for k in kho.keyframes(vid)])
                return self._tra(200, "text/html; charset=utf-8", trang)
            return self._tra(404, "text/plain; charset=utf-8", "không có trang này")
        except Exception as exc:                              # noqa: BLE001
            import traceback
            traceback.print_exc()
            return self._tra(500, "text/plain; charset=utf-8", f"lỗi: {exc}")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return self._tra(400, "application/json", '{"ok":false}')
        kho, u = self.kho, urllib.parse.urlparse(self.path)
        try:
            if u.path == "/chon":
                qid = str(body.get("qid") or "")
                khung = [[str(k[0]).upper()] + [int(x) for x in k[1:]]
                         for k in (body.get("khung") or [])]
                so = kho.luu(qid, khung, str(body.get("ai") or ""))
                print(f"  ✎ {qid}: {so} khung  ({body.get('ai') or '?'})",
                      flush=True)
                return self._tra(200, "application/json",
                                 json.dumps({"ok": True, "so": so}))
            if u.path == "/nhan":
                qid = str(body.get("qid") or "")
                kho.nhan[qid] = str(body.get("ai") or "")
                return self._tra(200, "application/json", '{"ok":true}')
            return self._tra(404, "application/json", '{"ok":false}')
        except Exception as exc:                              # noqa: BLE001
            import traceback
            traceback.print_exc()
            return self._tra(500, "application/json",
                             json.dumps({"ok": False, "loi": str(exc)}))


def _ip_lan():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def main():
    a = _args()
    print("=" * 74)
    print("  MÁY XEM — vòng lặp người–máy")
    print("=" * 74)
    kho = Kho(a)
    Tay.kho = kho
    srv = ThreadingHTTPServer((a.host, a.cong), Tay)
    print(f"\n  đề        : {a.de}  ({len(kho.truy_van)} câu)")
    print(f"  ghim ghi ra: {a.ghim}")
    print(f"  ảnh       : {', '.join(str(r) for r in kho.resolver.roots)}")
    print(f"\n  MỞ:  http://127.0.0.1:{a.cong}")
    if a.host not in ("127.0.0.1", "localhost"):
        print(f"  LAN: http://{_ip_lan()}:{a.cong}   (chia câu cho đồng đội)")
    print("\n  Ctrl+C để dừng. Mỗi cú bấm được ghi ngay, dừng server không mất gì.\n")
    threading.Thread(target=kho.nong, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  đã dừng.")


if __name__ == "__main__":
    main()
