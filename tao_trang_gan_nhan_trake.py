#!/usr/bin/env python
"""VIỆC 4.2 — Trang gán nhãn TRAKE: chọn 3 mốc liên tiếp trong cùng một video.

    python tao_trang_gan_nhan_trake.py

KHÔNG CẦN KAGGLE
----------------
Máy đã có `data cuộc thi\\keyframes(1 phần nhỏ)\\` — 7.800 keyframe của 29 video,
đặt tên `001.jpg` = keyframe `n=1`, khớp đúng số dòng của `map-keyframes`. Đủ để
gán 20 chuỗi TRAKE mà không phải xuất thêm ảnh từ Kaggle.

TRAKE KHÁC KIS Ở ĐÂU
--------------------
KIS: một câu -> một khung. TRAKE: một câu -> **N khung theo THỨ TỰ**, và
`scoring.r_score_trake` cho **điểm từng phần**: trúng 2/3 mốc được 0.667. Nên
mốc nào cũng đáng, không phải được ăn cả ngã về không.

Ràng buộc thật của hàm chấm:
  - `row["frames"][j]` phải nằm trong `gt.event_windows[j]` — ĐÚNG THỨ TỰ j.
    Chọn 3 mốc rồi mô tả ngược thứ tự là nhãn sai.
  - Cả N mốc phải trong CÙNG một video (`row["video_id"] != gt.video_id` -> 0).

CHỌN MỐC THẾ NÀO
----------------
Ba mốc phải là ba việc PHÂN BIỆT ĐƯỢC BẰNG MẮT và có thứ tự tự nhiên — kiểu
"mở cửa → bước vào → ngồi xuống". Đừng chọn ba khung gần giống nhau: hệ thống
không có cách nào biết khung nào là mốc 1 và khung nào là mốc 2, và nhãn đó chỉ
đo được độ may mắn.

Trang chỉ hiện MỘT PHẦN keyframe của mỗi video (cách đều nhau) — 290 ảnh một
video thì không ai xem hết. Chọn trong số hiện ra là đủ: cửa sổ đáp án tính từ
`map-keyframes` của đúng keyframe bạn chọn.
"""
import csv
import json
import os
import sys
from urllib.parse import quote

GOC = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.join(GOC, "dev_set")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

KEYFRAMES = r"D:\COMP\AIC2026\data cuộc thi\keyframes(1 phần nhỏ)"
MAP_GOC = r"D:\COMP\AIC2026\data cuộc thi\aic2026_data_nhe"

SO_CHUOI = 20        # việc 4.2 cần ~20 chuỗi
ANH_MOI_VIDEO = 48   # số keyframe hiện ra cho mỗi video (cách đều nhau)
SO_MOC = 3           # 3 mốc mỗi chuỗi

TRANG = """<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<title>Gán nhãn TRAKE — AIC2026</title>
<style>
 :root{color-scheme:light dark}
 body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;
      background:#12141a;color:#e8eaed;line-height:1.5}
 header{position:sticky;top:0;z-index:9;background:#1b1e26;border-bottom:1px solid #2c313c;
        padding:14px 22px;display:flex;gap:20px;align-items:center;flex-wrap:wrap}
 h1{font-size:17px;margin:0;font-weight:600}
 .tt{font-variant-numeric:tabular-nums;color:#9aa3b2;font-size:14px}
 .thanh{flex:1;min-width:160px;height:7px;background:#2c313c;border-radius:4px;overflow:hidden}
 .thanh i{display:block;height:100%;background:#5fd0a0;width:0;transition:width .2s}
 button{background:#5fd0a0;color:#12141a;border:0;padding:9px 16px;border-radius:7px;
        font-size:14px;font-weight:600;cursor:pointer}
 button:hover{background:#48b98a}
 button.phu{background:#2c313c;color:#c8cdd6}
 main{padding:22px;display:flex;flex-direction:column;gap:22px;max-width:1400px;margin:0 auto}
 .the{background:#1b1e26;border:1px solid #2c313c;border-radius:11px;padding:16px}
 .the.xong{border-color:#2f7d4f}
 .dau{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap;margin-bottom:10px}
 .dau b{font-size:15px}
 .dau .meta{font-size:12px;color:#7f8796}
 .luoi{display:grid;grid-template-columns:repeat(auto-fill,minmax(112px,1fr));gap:7px}
 .khung{position:relative;cursor:pointer;border:2px solid transparent;border-radius:6px;
        overflow:hidden;background:#000;aspect-ratio:16/9}
 .khung img{width:100%;height:100%;object-fit:cover;display:block}
 .khung .n{position:absolute;left:0;bottom:0;background:#000a;font-size:10px;
           padding:1px 4px;color:#c8cdd6}
 .khung.chon{border-color:#5fd0a0}
 .khung .badge{position:absolute;right:3px;top:3px;background:#5fd0a0;color:#12141a;
               font-size:12px;font-weight:700;width:19px;height:19px;border-radius:50%;
               display:flex;align-items:center;justify-content:center}
 .o{margin-top:12px;display:grid;grid-template-columns:1fr 1fr;gap:12px}
 .o label{display:block;font-size:12px;color:#9aa3b2;margin-bottom:4px;font-weight:600}
 .o input{width:100%;box-sizing:border-box;background:#12141a;border:1px solid #343a47;
          color:#e8eaed;border-radius:7px;padding:10px 12px;font-size:14px}
 .o input:focus{outline:0;border-color:#5fd0a0}
 .daChon{font-size:12px;color:#5fd0a0;margin-top:8px;min-height:17px;
         font-family:ui-monospace,monospace}
 .canh{font-size:12px;color:#e0a35c;min-height:17px;margin-top:2px}
 .tip{background:#1b1e26;border:1px solid #2c313c;border-left:3px solid #5fd0a0;
      border-radius:8px;padding:14px 18px;font-size:14px;color:#c8cdd6}
 .tip b{color:#e8eaed}
 dialog{border:0;background:#000;padding:0;max-width:96vw;max-height:96vh}
 dialog img{max-width:96vw;max-height:96vh;display:block}
</style></head><body>
<header>
  <h1>Gán nhãn TRAKE</h1>
  <span class="tt" id="tt">0 / 0</span>
  <span class="thanh"><i id="thanh"></i></span>
  <button onclick="xuat()">⬇ Xuất dan_nhan_trake.csv</button>
  <button class="phu" onclick="xoaHet()">Xoá hết</button>
</header>
<main>
  <div class="tip">
    <b>Bấm __SO_MOC__ khung theo ĐÚNG THỨ TỰ diễn ra.</b> Bấm lại để bỏ chọn.
    Số ①②③ hiện ở góc khung cho biết thứ tự bạn đã chọn.<br><br>
    <b>Ba mốc phải phân biệt được bằng mắt và có thứ tự tự nhiên</b> — kiểu
    "mở cửa → bước vào → ngồi xuống". Đừng chọn ba khung gần giống nhau: máy
    không có cách nào biết khung nào là mốc 1, và nhãn đó chỉ đo độ may mắn.<br><br>
    <b>Mô tả phải nêu rõ cả ba bước theo thứ tự</b>, dùng dấu <b>→</b> hoặc
    "sau đó" để tách. Máy tách câu ở đúng những dấu đó
    (<code>RuleBasedProvider.parse_query</code>).<br>
    ✅ &nbsp;"người đàn ông mở cửa → bước vào phòng → ngồi xuống ghế"<br>
    ❌ &nbsp;"người đàn ông đi vào phòng rồi làm việc"<br><br>
    Chấm có <b>điểm từng phần</b>: trúng 2/3 mốc được 0.667. Nên mốc nào cũng đáng.
    Tự lưu sau mỗi lần bấm. Ảnh không hiện thì xem ghi chú ở cuối trang.
  </div>
  <div id="ds"></div>
  <div class="tip">Ảnh không hiện? Trang này trỏ tới
    <code>__DUONG__</code> bằng đường dẫn tương đối. Nếu bạn di chuyển thư mục
    <code>keyframes(1 phần nhỏ)</code> thì sửa <code>KEYFRAMES</code> trong
    <code>tao_trang_gan_nhan_trake.py</code> rồi tạo lại trang.</div>
</main>
<dialog id="to" onclick="this.close()"><img id="toimg"></dialog>
<script>
const VIDEO = __DATA__;
const SO_MOC = __SO_MOC__;
const KHOA = "aic2026_gan_nhan_trake_v1";
let luu = JSON.parse(localStorage.getItem(KHOA) || "{}");

function g1(vid){ return Object.assign({moc:[],vi:"",en:""}, luu[vid]); }
function xong1(g){ return g.moc.length === SO_MOC && g.vi && g.en; }

function dem(){
  const n = VIDEO.filter(v => xong1(g1(v.video_id))).length;
  document.getElementById("tt").textContent = n + " / " + VIDEO.length + " xong";
  document.getElementById("thanh").style.width = (100*n/VIDEO.length) + "%";
}
function luuLai(vid, g){
  luu[vid] = g;
  localStorage.setItem(KHOA, JSON.stringify(luu));
  document.getElementById("the_"+vid).classList.toggle("xong", xong1(g));
  dem();
}
function bam(vid, n){
  const g = g1(vid);
  const i = g.moc.indexOf(n);
  if(i >= 0) g.moc.splice(i, 1);
  else if(g.moc.length < SO_MOC) g.moc.push(n);
  else { canh(vid, "Đã chọn đủ " + SO_MOC + " mốc — bấm lại một mốc để bỏ chọn"); return; }
  canh(vid, "");
  luuLai(vid, g);
  veLuoi(vid);
}
function canh(vid, s){ document.getElementById("canh_"+vid).textContent = s; }
function ghi(vid, truong, gt){ const g = g1(vid); g[truong] = gt; luuLai(vid, g); }
function phong(src){ document.getElementById("toimg").src = src;
                     document.getElementById("to").showModal(); }
function xoaHet(){
  if(!confirm("Xoá toàn bộ nhãn TRAKE đã gõ?")) return;
  luu = {}; localStorage.removeItem(KHOA); ve();
}
function veLuoi(vid){
  const v = VIDEO.find(x => x.video_id === vid);
  const g = g1(vid);
  document.getElementById("luoi_"+vid).innerHTML = v.khung.map(k => {
    const j = g.moc.indexOf(k.n);
    return `<div class="khung${j>=0?" chon":""}" onclick="bam('${vid}',${k.n})"
              ondblclick="phong('${v.goc}/${k.file}')">
      <img src="${v.goc}/${k.file}" loading="lazy">
      <span class="n">n=${k.n}</span>
      ${j>=0 ? `<span class="badge">${j+1}</span>` : ""}</div>`;
  }).join("");
  const ten = g.moc.map((n,i) => `${"①②③④⑤"[i]||i+1} n=${n}`).join("   ");
  document.getElementById("chon_"+vid).textContent =
    g.moc.length ? "đã chọn: " + ten : "chưa chọn mốc nào";
}
function csvO(s){ s = (s||"").replace(/"/g,'""');
                  return /[",\\n]/.test(s) ? '"'+s+'"' : s; }
function xuat(){
  const cot = ["stt","video_id","cac_n","mo_ta","mo_ta_en"];
  const dong = [cot.join(",")];
  let n = 0;
  for(const v of VIDEO){
    const g = g1(v.video_id);
    if(!xong1(g)) continue;          // bỏ dòng dở: nửa nhãn là nhãn sai
    n++;
    dong.push([n, v.video_id, csvO(g.moc.join(" ")),
               csvO(g.vi), csvO(g.en)].join(","));
  }
  if(!n){ alert("Chưa có chuỗi nào đủ " + SO_MOC + " mốc + cả hai ô mô tả."); return; }
  const blob = new Blob(["\\ufeff"+dong.join("\\n")+"\\n"],
                        {type:"text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "dan_nhan_trake.csv"; a.click();
  alert("Đã tải dan_nhan_trake.csv (" + n + " chuỗi).\\n\\n"
        + "Chép vào dev_set\\\\ rồi chạy:\\npython tao_dev_set.py");
}
function ve(){
  document.getElementById("ds").innerHTML = VIDEO.map(v => {
    const g = g1(v.video_id);
    const q = s => (s||"").replace(/"/g,'&quot;');
    return `<div class="the${xong1(g)?" xong":""}" id="the_${v.video_id}">
      <div class="dau"><b>${v.video_id}</b>
        <span class="meta">${v.tong_kf} keyframe · hiện ${v.khung.length} khung
          cách đều · ${(v.tieu_de||"").slice(0,80)}</span></div>
      <div class="luoi" id="luoi_${v.video_id}"></div>
      <div class="daChon" id="chon_${v.video_id}"></div>
      <div class="canh" id="canh_${v.video_id}"></div>
      <div class="o">
        <div><label>Mô tả chuỗi — tiếng Việt (dùng → giữa các bước)</label>
          <input value="${q(g.vi)}" placeholder="mở cửa → bước vào phòng → ngồi xuống ghế"
                 oninput="ghi('${v.video_id}','vi',this.value)"></div>
        <div><label>Bản tiếng Anh (bắt buộc)</label>
          <input value="${q(g.en)}" placeholder="opens the door → walks into the room → sits down"
                 oninput="ghi('${v.video_id}','en',this.value)"></div>
      </div></div>`;
  }).join("");
  for(const v of VIDEO) veLuoi(v.video_id);
  dem();
}
ve();
</script></body></html>
"""


def doc_map(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return [(int(r["n"]), float(r["pts_time"]), int(r["frame_idx"]))
                for r in csv.DictReader(fh)]


def tim_map():
    """{video_id: đường dẫn map-keyframes CSV}."""
    ra = {}
    for goc, _, files in os.walk(MAP_GOC):
        low = goc.lower().replace("\\", "/")
        if "map" not in low or "keyframe" not in low:
            continue
        for ten in files:
            if ten.lower().endswith(".csv"):
                ra[os.path.splitext(ten)[0].upper()] = os.path.join(goc, ten)
    return ra


def main():
    if not os.path.isdir(KEYFRAMES):
        raise SystemExit(f"Không thấy thư mục keyframe: {KEYFRAMES}")
    mp = tim_map()
    if not mp:
        raise SystemExit(f"Không thấy map-keyframes trong {MAP_GOC}")

    # Đường dẫn tương đối từ dev_set/ tới thư mục keyframe, đã URL-encode từng
    # đoạn: tên có khoảng trắng, dấu ngoặc và dấu tiếng Việt.
    rel = os.path.relpath(KEYFRAMES, DEV)
    rel_url = "/".join(quote(x) for x in rel.replace("\\", "/").split("/"))

    ctx = {}
    ctx_path = os.path.join(DEV, "ngu_canh_video.json")
    if os.path.exists(ctx_path):
        with open(ctx_path, "r", encoding="utf-8") as fh:
            ctx = json.load(fh)

    du = []
    for ten in sorted(os.listdir(KEYFRAMES)):
        d = os.path.join(KEYFRAMES, ten)
        if not os.path.isdir(d):
            continue
        vid = ten.upper()
        if vid not in mp:
            continue
        anh = sorted(f for f in os.listdir(d)
                     if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
        rows = doc_map(mp[vid])
        if len(anh) != len(rows):
            # Lệch thì bỏ video: gán nhãn trên ảnh lệch với map là nhãn sai
            # cửa sổ đáp án, và sai đó không có triệu chứng gì.
            print(f"  ⚠ bỏ {vid}: {len(anh)} ảnh nhưng {len(rows)} dòng map")
            continue
        if len(anh) < ANH_MOI_VIDEO:
            continue
        du.append((vid, d, anh, rows))

    if not du:
        raise SystemExit("Không video nào đủ điều kiện.")
    chon = du[:SO_CHUOI]

    data = []
    for vid, d, anh, rows in chon:
        # Lấy ANH_MOI_VIDEO khung CÁCH ĐỀU nhau, bỏ 10% đầu và 5% cuối (hình
        # hiệu, intro, credit) — cùng tinh thần với kaggle_chuan_bi_dev_set.
        lo, hi = int(len(anh) * 0.10), int(len(anh) * 0.95)
        chi_so = [lo + round(i * (hi - lo - 1) / (ANH_MOI_VIDEO - 1))
                  for i in range(ANH_MOI_VIDEO)]
        khung = [{"n": rows[i][0], "file": anh[i]} for i in sorted(set(chi_so))]
        data.append({
            "video_id": vid,
            "goc": f"{rel_url}/{quote(os.path.basename(d))}",
            "tong_kf": len(anh),
            "tieu_de": (ctx.get(vid, {}) or {}).get("title", ""),
            "khung": khung,
        })

    ra = os.path.join(DEV, "gan_nhan_trake.html")
    html = (TRANG.replace("__DATA__", json.dumps(data, ensure_ascii=False))
                 .replace("__SO_MOC__", str(SO_MOC))
                 .replace("__DUONG__", rel.replace("\\", "/")))
    with open(ra, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"✅ Đã tạo {ra}")
    print(f"   {len(data)} video · {ANH_MOI_VIDEO} khung mỗi video · {SO_MOC} mốc mỗi chuỗi")
    print("\n   Mở bằng trình duyệt. Bấm 3 khung THEO THỨ TỰ, gõ mô tả cả hai ô,")
    print("   bấm '⬇ Xuất dan_nhan_trake.csv', chép vào dev_set\\, rồi:")
    print("   python tao_dev_set.py")


if __name__ == "__main__":
    main()
