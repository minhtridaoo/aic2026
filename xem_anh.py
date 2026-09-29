#!/usr/bin/env python
"""TRANG XEM ẢNH — cho MẮT NGƯỜI vào vòng lặp. Chạy trên Kaggle, xuất HTML tĩnh.

    python -X utf8 xem_anh.py out/bai_nop.zip --config config_kaggle.yaml \
        --de data/de_thunghiem.json --ra /kaggle/working/xem

Rồi tải thư mục `xem/` về, mở `index.html` bằng trình duyệt. Ảnh đã nhúng thẳng
vào HTML dưới dạng base64 nên **không cần mạng, không cần Kaggle sống**, mở
offline vẫn xem được.

VÌ SAO ĐÂY LÀ VIỆC ĐÁNG GIÁ NHẤT
--------------------------------
Số đo của chính hệ thống này, 120 câu KIS, đúng cấu hình thi:

    R@100 = 57,5%      khung đúng ĐÃ nằm trong 100 dòng máy trả về
    Final = 0,3450     nhưng chỉ ăn chừng này, vì nó không đứng hạng 1

Xác nhận trên đề thật ngày 20/08: nộp 5,6/12 = 0,467 điểm/câu. Toàn bộ phần
chênh giữa "đã tìm thấy" và "được tính điểm" nằm ở chỗ **không ai nhìn**.

Thang điểm khiến việc nhìn đáng giá đến mức khó tin:
`Final = (1/5)·Σ_{k∈{1,5,20,50,100}} R@k`, tức kéo khung đúng từ hạng 100 lên
hạng 1 biến 0,2 điểm thành 1,0 điểm. Cùng một khung, cùng một hệ thống, chỉ khác
thứ tự dòng.

Trần lý thuyết khi người nhìn hết 100 ảnh là **0,575/câu**; nhìn 500 ảnh
(`--sau 500`) là **~0,75/câu** — xem phân tích độ sâu pool trong
`chan_doan_qa_truot.py`.

CÁCH DÙNG KHI THI
-----------------
Mỗi trang là một truy vấn. Bấm vào ảnh đúng -> nó được chọn. Bảng bên dưới sinh
sẵn đoạn JSON để dán vào file ghim, rồi:

    python -X utf8 nop_bai.py ... --ghim data/ghim.json --ra out/ban_moi.zip

Bấm nhiều ảnh cũng được: thứ tự bấm chính là thứ tự dòng nộp, và vì `R@k` lấy
max trên tiền tố nên chắc chắn nhất phải bấm trước.
"""
import argparse
import base64
import csv
import html
import io
import json
import os
import sys
import zipfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as cau_hinh
from aic2026.core import dataset
from aic2026.data_plane.image_resolver import (EventImageResolver,
                                               ImageResolutionError)

GOC = os.path.dirname(os.path.abspath(__file__))


def _args():
    p = argparse.ArgumentParser(description="Sinh trang xem ảnh từ bài nộp")
    p.add_argument("zip", help="zip bài nộp đã dựng bằng nop_bai.py")
    p.add_argument("--config", default=None)
    p.add_argument("--de", default=os.path.join(GOC, "data", "de_thunghiem.json"))
    p.add_argument("--ra", default=os.path.join(GOC, "out", "xem"))
    p.add_argument("--sau", type=int, default=100,
                   help="số dòng đầu mỗi câu đem ra xem (mặc định 100)")
    p.add_argument("--rong", type=int, default=190, help="bề ngang ảnh thu nhỏ")
    p.add_argument("--rong-trake", type=int, default=320,
                   help="bề ngang MỖI ảnh con TRAKE (mặc định 320; cũ chỉ còn "
                        "khoảng rong/4 nên rất mờ)")
    p.add_argument("--chi", default="", help="chỉ làm các qid này, ngăn bằng phẩy")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# frame_idx  ->  event_id
# --------------------------------------------------------------------------- #
def bang_khung(goc_data, videos):
    """{(VIDEO, frame_idx): n} — chỉ nạp map-keyframes của video cần tới.

    Bài nộp ghi `frame_idx` (số khung trong video) còn ảnh keyframe đặt tên theo
    `n` (số thứ tự keyframe). Hai thứ khác nhau: `L26_V004` có khung 4803 nhưng
    ảnh của nó là `061.jpg`. Không có bảng này thì không mở được ảnh nào.
    """
    paths = dataset.scan(goc_data)
    ra, thieu = {}, []
    for vid in sorted(videos):
        vp = paths.get(vid)
        if vp is None or vp.mapping is None:
            thieu.append(vid)
            continue
        for r in dataset.load_map_keyframes(vp.mapping):
            ra[(vid, int(r.frame_idx))] = int(r.n)
    return ra, thieu


def thumb(path, rong):
    """Ảnh -> data URI base64. Trả None nếu không đọc được (không làm sập trang)."""
    try:
        from PIL import Image
        im = Image.open(path)
        im = im.convert("RGB")
        w, h = im.size
        im = im.resize((rong, max(1, round(h * rong / w))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=72)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:                                          # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
CSS = """
:root{color-scheme:dark}
body{background:#14161a;color:#e6e8eb;font:14px/1.5 system-ui,sans-serif;margin:0;padding:18px 20px 240px}
a{color:#7fb2ff}
h1{font-size:19px;margin:0 0 6px}
.dedai{background:#1d2026;border-left:3px solid #7fb2ff;padding:10px 14px;margin:10px 0 18px;
       border-radius:0 6px 6px 0;max-width:1100px}
.luoi{display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--w),1fr));gap:10px}
.o{background:#1d2026;border:2px solid #2b3038;border-radius:7px;padding:5px;cursor:pointer;
   transition:border-color .1s}
.o:hover{border-color:#4a5568}
.o.chon{border-color:#4ade80;background:#16281d}
.o img{width:100%;display:block;border-radius:4px}
.o .nhan{font:11px ui-monospace,monospace;color:#98a2b3;padding:4px 2px 1px;
         display:flex;justify-content:space-between;gap:6px}
.o.chon .nhan{color:#4ade80}
.o .thutu{position:absolute;background:#4ade80;color:#0b2016;font-weight:700;font-size:11px;
          border-radius:3px;padding:1px 5px;margin:4px}
.dai{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px}
.dai img{width:100%;height:auto}
.mat{background:#2a1d1d;color:#f0a0a0;font:11px monospace;padding:14px 6px;text-align:center;
     border-radius:4px}
#khay{position:fixed;left:0;right:0;bottom:0;background:#0f1115;border-top:2px solid #2b3038;
      padding:10px 20px;max-height:220px;overflow:auto}
#khay h2{font-size:13px;margin:0 0 6px;color:#98a2b3;font-weight:600}
pre{background:#1d2026;padding:10px 12px;border-radius:6px;margin:0;white-space:pre-wrap;
    font:12px/1.5 ui-monospace,monospace;color:#a5e8b8}
button{background:#2b3038;color:#e6e8eb;border:0;border-radius:5px;padding:6px 12px;
       cursor:pointer;font-size:12px;margin-right:8px}
button:hover{background:#3a4150}
"""

JS = """
const CHON = [];
function bam(el){
  const k = el.dataset.k;
  const i = CHON.indexOf(k);
  if (i >= 0) { CHON.splice(i,1); el.classList.remove('chon'); }
  else { CHON.push(k); el.classList.add('chon'); }
  ve();
}
function ve(){
  document.querySelectorAll('.thutu').forEach(e=>e.remove());
  CHON.forEach((k,i)=>{
    const el = document.querySelector('[data-k="'+CSS.escape(k)+'"]');
    if(!el) return;
    const b = document.createElement('div');
    b.className='thutu'; b.textContent=(i+1);
    el.style.position='relative'; el.prepend(b);
  });
  const p = document.getElementById('ra');
  if(!CHON.length){ p.textContent='(chưa chọn ảnh nào — bấm vào ảnh đúng)'; return; }
  const muc = TASK==='trake'
      ? {khung: CHON.map(k=>k.split('|'))}
      : {khung: CHON.map(k=>k.split('|'))};
  const o = {}; o[QID] = Object.assign(muc,
      {ghi_chu:'người chọn bằng mắt trên xem_anh.py'});
  p.textContent = JSON.stringify(o, null, 2).slice(1,-1).trim().replace(/^\\s{2}/gm,'');
}
function chep(){
  navigator.clipboard.writeText(document.getElementById('ra').textContent);
}
function xoa(){
  CHON.length=0;
  document.querySelectorAll('.o.chon').forEach(e=>e.classList.remove('chon'));
  ve();
}
window.addEventListener('DOMContentLoaded', ve);
"""


def trang(qid, task, de_text, o_html, rong, tieu_de_video):
    return f"""<!doctype html><meta charset="utf-8"><title>{html.escape(qid)}</title>
<style>{CSS}</style>
<body style="--w:{rong}px">
<h1>{html.escape(qid)} · {task.upper()} · {len(o_html)} dòng</h1>
<div class="dedai">{html.escape(de_text)}</div>
<div style="color:#98a2b3;font-size:12px;margin-bottom:12px">
  Bấm vào ảnh ĐÚNG. Bấm nhiều ảnh cũng được — thứ tự bấm là thứ tự dòng nộp,
  nên bấm cái chắc nhất trước. <a href="index.html">← danh sách câu</a>
</div>
<div class="luoi">
{''.join(o_html)}
</div>
<div id="khay">
  <h2>Dán đoạn này vào <code>data/ghim.json</code></h2>
  <button onclick="chep()">Sao chép</button><button onclick="xoa()">Bỏ chọn hết</button>
  <pre id="ra"></pre>
</div>
<script>const QID={json.dumps(qid)}; const TASK={json.dumps(task)};
{JS}</script>
"""


def main():
    a = _args()
    config = cau_hinh.load_config(a.config)
    goc_data = (config.get("offline") or {}).get("btc_data_root")
    resolver = EventImageResolver.from_config(config)
    if not resolver.roots:
        raise SystemExit(
            "❌ `qa.image_roots` rỗng — không có ảnh thì trang này vô nghĩa.\n"
            "   Chạy trên Kaggle với config_kaggle.yaml (xem notebook).")

    with open(a.de, "r", encoding="utf-8") as fh:
        de = {q["qid"]: q for q in json.load(fh)["queries"]}

    with zipfile.ZipFile(a.zip) as z:
        bai = {os.path.basename(n)[:-4]:
               list(csv.reader(io.StringIO(z.read(n).decode("utf-8"))))
               for n in z.namelist() if n.lower().endswith(".csv")}

    chi = {x.strip() for x in a.chi.split(",") if x.strip()}
    if chi:
        bai = {k: v for k, v in bai.items() if k in chi}

    videos = {r[0].strip() for rows in bai.values() for r in rows[:a.sau]}
    print(f"Nạp map-keyframes cho {len(videos)} video ...")
    khung2n, thieu = bang_khung(goc_data, videos)
    if thieu:
        print(f"  ⚠ {len(thieu)} video không có map-keyframes: {thieu[:5]}")
    print(f"  {len(khung2n):,d} cặp (video, frame_idx)\n")

    os.makedirs(a.ra, exist_ok=True)
    tom_tat, tong_mat = [], 0

    for qid in sorted(bai, key=lambda t: int(''.join(c for c in t.split('-')[2]
                                                     if c.isdigit()) or 0)):
        rows = bai[qid][:a.sau]
        q = de.get(qid, {})
        task = q.get("task") or ("trake" if "trake" in qid
                                 else "qa" if "qa" in qid else "kis")
        o_html, mat = [], 0
        for i, r in enumerate(rows, 1):
            vid = r[0].strip()
            frames = [int(x) for x in r[1:]] if task == "trake" else [int(r[1])]
            anh = []
            # TRAKE có bốn ảnh trong cùng một hàng. Bản cũ vừa resize ảnh xuống
            # `rong/4`, vừa để trình duyệt co tiếp còn 25% ô nên chữ/người đều
            # nhoè. Giữ 320 px cho MỖI ảnh và làm cả hàng rộng tương ứng.
            rong_anh = a.rong_trake if len(frames) > 1 else a.rong
            for f in frames:
                n = khung2n.get((vid, f))
                p = None
                if n is not None:
                    try:
                        p = resolver.resolve(f"{vid}#{n:05d}")
                    except ImageResolutionError:
                        p = None
                anh.append(thumb(p, rong_anh) if p else None)
            if all(x is None for x in anh):
                mat += 1
            khoa = "|".join([vid] + [str(f) for f in frames])
            than = ('<div class="dai">' + "".join(
                        f'<img src="{x}">' if x else '<div class="mat">?</div>'
                        for x in anh) + '</div>'
                    if len(anh) > 1 else
                    (f'<img src="{anh[0]}">' if anh[0]
                     else '<div class="mat">không có ảnh</div>'))
            phu = (r[2] if task == "qa" and len(r) > 2 else "")
            o_html.append(
                f'<div class="o" data-k="{html.escape(khoa)}" onclick="bam(this)">'
                f'{than}<div class="nhan"><span>#{i} {html.escape(vid)}</span>'
                f'<span>{",".join(str(f) for f in frames)}</span></div>'
                + (f'<div class="nhan" style="color:#c9a227">'
                   f'{html.escape(phu[:40])}</div>' if phu else "")
                + '</div>')

        de_text = q.get("text", "")
        with open(os.path.join(a.ra, f"{qid}.html"), "w", encoding="utf-8") as fh:
            rong_o = (a.rong_trake * 4 + 36) if task == "trake" else a.rong
            fh.write(trang(qid, task, de_text, o_html, rong_o, {}))
        tong_mat += mat
        tom_tat.append((qid, task, len(rows), mat))
        print(f"  {qid:22s} {task:6s} {len(rows):3d} ảnh"
              + (f"  ⚠ {mat} không mở được" if mat else ""))

    with open(os.path.join(a.ra, "index.html"), "w", encoding="utf-8") as fh:
        fh.write('<!doctype html><meta charset="utf-8"><title>Xem bài nộp</title>'
                 f'<style>{CSS}</style><body><h1>Xem bài nộp — '
                 f'{len(tom_tat)} truy vấn</h1><div class="dedai">'
                 'Mở từng câu, bấm ảnh đúng, dán JSON sinh ra vào file ghim.</div>'
                 '<table style="border-spacing:0 4px">'
                 + "".join(
                     f'<tr><td style="padding-right:16px"><a href="{q}.html">{q}</a></td>'
                     f'<td style="padding-right:16px;color:#98a2b3">{t}</td>'
                     f'<td style="color:#98a2b3">{n} ảnh</td>'
                     f'<td style="color:#f0a0a0;padding-left:12px">'
                     f'{"⚠ " + str(m) + " thiếu" if m else ""}</td></tr>'
                     for q, t, n, m in tom_tat)
                 + '</table>')

    print(f"\n-> {a.ra}/index.html")
    if tong_mat:
        print(f"⚠ tổng {tong_mat} dòng không mở được ảnh — kiểm `qa.image_roots`")


if __name__ == "__main__":
    main()
