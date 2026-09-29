#!/usr/bin/env python
"""Tạo trang web gán nhãn 60 keyframe — mở bằng trình duyệt, gõ, bấm xuất CSV.

    python tao_trang_gan_nhan.py

VÌ SAO KHÔNG DÙNG EXCEL
-----------------------
Gán nhãn trong Excel nghĩa là mở thư mục ảnh ở một cửa sổ, Excel ở cửa sổ khác,
rồi tự dò xem dòng nào ứng với ảnh nào. 60 lần như vậy là 60 cơ hội gõ nhầm dòng
— mà gõ nhầm thì đáp án sai, và điểm đo được sẽ sai theo mà không ai biết.

Trang này đặt ảnh ngay cạnh ô nhập, tự lưu vào trình duyệt sau mỗi lần gõ (đóng
tab giữa chừng không mất), và xuất lại đúng khuôn CSV mà `tao_dev_set.py` đọc.

HAI Ô NHẬP
----------
  mo_ta      tiếng Việt — giống cách đề thi thật sẽ hỏi
  mo_ta_en   tiếng Anh  — BẮT BUỘC, vì CLIP ViT-B/32 không hiểu tiếng Việt
                          (đo được: truy vấn tiếng Việt chồng lấn 53% top-100)
"""
import csv
import json
import os
import sys

GOC = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.join(GOC, "dev_set")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TRANG = """<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<title>Gán nhãn dev set — AIC2026</title>
<style>
 :root{color-scheme:light dark}
 body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;
      background:#12141a;color:#e8eaed;line-height:1.5}
 header{position:sticky;top:0;z-index:9;background:#1b1e26;border-bottom:1px solid #2c313c;
        padding:14px 22px;display:flex;gap:20px;align-items:center;flex-wrap:wrap}
 h1{font-size:17px;margin:0;font-weight:600}
 .tt{font-variant-numeric:tabular-nums;color:#9aa3b2;font-size:14px}
 .thanh{flex:1;min-width:160px;height:7px;background:#2c313c;border-radius:4px;overflow:hidden}
 .thanh i{display:block;height:100%;background:#4c8bf5;width:0;transition:width .2s}
 button{background:#4c8bf5;color:#fff;border:0;padding:9px 16px;border-radius:7px;
        font-size:14px;font-weight:600;cursor:pointer}
 button:hover{background:#3a76d8}
 button.phu{background:#2c313c;color:#c8cdd6}
 main{padding:22px;display:flex;flex-direction:column;gap:18px;max-width:1180px;margin:0 auto}
 .the{display:grid;grid-template-columns:400px 1fr;gap:20px;background:#1b1e26;
      border:1px solid #2c313c;border-radius:11px;padding:16px}
 .the.xong{border-color:#2f7d4f}
 .the img{width:100%;border-radius:7px;display:block;background:#000;cursor:zoom-in}
 .meta{font-size:12px;color:#7f8796;margin-top:8px;word-break:break-all}
 .o label{display:block;font-size:12px;color:#9aa3b2;margin:10px 0 4px;font-weight:600}
 .o input{width:100%;box-sizing:border-box;background:#12141a;border:1px solid #343a47;
          color:#e8eaed;border-radius:7px;padding:10px 12px;font-size:15px}
 .o input:focus{outline:0;border-color:#4c8bf5}
 .so{font-size:13px;color:#7f8796;font-weight:600}
 .tip{background:#1b1e26;border:1px solid #2c313c;border-left:3px solid #4c8bf5;
      border-radius:8px;padding:14px 18px;font-size:14px;color:#c8cdd6}
 .tip b{color:#e8eaed}
 dialog{border:0;background:#000;padding:0;max-width:96vw;max-height:96vh}
 dialog img{max-width:96vw;max-height:96vh;display:block}
</style></head><body>
<header>
  <h1>Gán nhãn dev set</h1>
  <span class="tt" id="tt">0 / 0</span>
  <span class="thanh"><i id="thanh"></i></span>
  <button onclick="xuat()">⬇ Xuất dan_nhan.csv</button>
  <button class="phu" onclick="xoaHet()">Xoá hết</button>
</header>
<main>
  <div class="tip">
    <b>Cách mô tả:</b> tả <b>thứ nhìn thấy được</b> trong ảnh — người, vật, hành động,
    bối cảnh. Đừng tả thứ phải suy đoán (tên riêng, ngày tháng, cảm xúc).<br>
    ✅ &nbsp;"hai người đàn ông bắt tay trên sân khấu có hoa"<br>
    ❌ &nbsp;"lễ khai mạc hội nghị ngày 12/8"<br><br>
    <b>Ô tiếng Anh là bắt buộc</b> — CLIP không hiểu tiếng Việt (đã đo: các câu
    tiếng Việt khác nghĩa trả về top-100 chồng lấn tới 53%). Dịch thô qua Google
    Translate cũng được. Bấm vào ảnh để phóng to.<br><br>
    Tự lưu sau mỗi lần gõ. Đóng tab rồi mở lại vẫn còn.
  </div>
  <div id="ds"></div>
</main>
<dialog id="to" onclick="this.close()"><img id="toimg"></dialog>
<script>
const HANG = __DATA__;
const KHOA = "aic2026_gan_nhan_v1";
let luu = JSON.parse(localStorage.getItem(KHOA) || "{}");

function dem(){
  const n = HANG.filter(h => (luu[h.anh]||{}).vi && (luu[h.anh]||{}).en).length;
  document.getElementById("tt").textContent = n + " / " + HANG.length + " xong";
  document.getElementById("thanh").style.width = (100*n/HANG.length) + "%";
  return n;
}
function ghi(anh, truong, gt){
  luu[anh] = Object.assign({vi:"",en:""}, luu[anh]);
  luu[anh][truong] = gt;
  localStorage.setItem(KHOA, JSON.stringify(luu));
  const the = document.getElementById("the_"+anh);
  the.classList.toggle("xong", !!(luu[anh].vi && luu[anh].en));
  dem();
}
function phong(src){ document.getElementById("toimg").src = src;
                     document.getElementById("to").showModal(); }
function xoaHet(){
  if(!confirm("Xoá toàn bộ nhãn đã gõ?")) return;
  luu = {}; localStorage.removeItem(KHOA); ve();
}
function csvO(s){ s = (s||"").replace(/"/g,'""');
                  return /[",\\n]/.test(s) ? '"'+s+'"' : s; }
function xuat(){
  const cot = ["stt","video_id","n","pts_time","fps","frame_idx","anh","mo_ta","mo_ta_en"];
  const dong = [cot.join(",")];
  for(const h of HANG){
    const g = luu[h.anh] || {vi:"",en:""};
    dong.push([h.stt,h.video_id,h.n,h.pts_time,h.fps,h.frame_idx,h.anh,
               csvO(g.vi), csvO(g.en)].join(","));
  }
  const blob = new Blob(["\\ufeff"+dong.join("\\n")+"\\n"],
                        {type:"text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "dan_nhan.csv"; a.click();
  alert("Đã tải dan_nhan.csv.\\n\\nChép đè vào thư mục dev_set\\\\ rồi chạy:\\n"
        + "python tao_dev_set.py");
}
function ve(){
  document.getElementById("ds").innerHTML = HANG.map(h => {
    const g = luu[h.anh] || {vi:"",en:""};
    const xong = (g.vi && g.en) ? " xong" : "";
    return `<div class="the${xong}" id="the_${h.anh}">
      <div><img src="anh/${h.anh}" loading="lazy" onclick="phong(this.src)">
        <div class="meta">${h.video_id} · keyframe ${h.n} · ${h.pts_time}s
          · frame ${h.frame_idx}<br>${(h.tieu_de||"").slice(0,70)}</div></div>
      <div class="o"><div class="so">Câu ${h.stt} / ${HANG.length}</div>
        <label>Mô tả tiếng Việt</label>
        <input value="${g.vi.replace(/"/g,'&quot;')}" placeholder="ví dụ: hai người đàn ông bắt tay trên sân khấu"
               oninput="ghi('${h.anh}','vi',this.value)">
        <label>Bản tiếng Anh (bắt buộc)</label>
        <input value="${g.en.replace(/"/g,'&quot;')}" placeholder="e.g. two men shaking hands on a stage"
               oninput="ghi('${h.anh}','en',this.value)">
      </div></div>`;
  }).join("");
  dem();
}
ve();
</script></body></html>
"""


def main():
    csv_path = os.path.join(DEV, "dan_nhan.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"Không thấy {csv_path}")

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    ctx = {}
    ctx_path = os.path.join(DEV, "ngu_canh_video.json")
    if os.path.exists(ctx_path):
        with open(ctx_path, "r", encoding="utf-8") as fh:
            ctx = json.load(fh)

    for r in rows:
        r["tieu_de"] = (ctx.get(r["video_id"], {}) or {}).get("title", "")

    ra = os.path.join(DEV, "gan_nhan.html")
    with open(ra, "w", encoding="utf-8") as fh:
        fh.write(TRANG.replace("__DATA__", json.dumps(rows, ensure_ascii=False)))

    print(f"✅ Đã tạo {ra}  ({len(rows)} ảnh)")
    print("\n   Mở file đó bằng trình duyệt, gõ xong bấm '⬇ Xuất dan_nhan.csv',")
    print("   chép đè vào dev_set\\, rồi chạy:  python tao_dev_set.py")


if __name__ == "__main__":
    main()
