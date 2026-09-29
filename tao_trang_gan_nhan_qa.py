#!/usr/bin/env python
"""VIỆC 4.1 — Trang gán nhãn QA: nhìn ảnh, đặt câu hỏi + đáp án đúng.

    python tao_trang_gan_nhan_qa.py

Dùng lại ĐÚNG 120 keyframe đã gán nhãn KIS, nên cửa sổ đáp án
(`frame_start`/`frame_end`) lấy y nguyên từ dev set KIS — không phải bốc lại,
không phải tải thêm ảnh.

ĐÁP ÁN PHẢI NGẮN VÀ CHUẨN HOÁ — ĐỌC KỸ CHỖ NÀY
----------------------------------------------
`scoring.answers_match` so khớp **chuỗi CHÍNH XÁC** sau khi:
  - chuẩn hoá Unicode NFC, bỏ khoảng trắng đầu/cuối, hạ chữ thường
  - bỏ dấu ngoặc kép bọc ngoài, gộp mọi dãy khoảng trắng thành một

Nghĩa là `"Áo đỏ"` khớp `"áo  đỏ"`, nhưng `"màu đỏ"` KHÔNG khớp `"đỏ"`, và
`"đỏ."` KHÔNG khớp `"đỏ"`. Gõ đáp án dài dòng thì QA sẽ mãi mãi 0 điểm và bạn sẽ
tưởng hệ thống trả lời sai, trong khi nó trả lời đúng mà chỉ khác cách diễn đạt.

Vì vậy trang này CẢNH BÁO NGAY LÚC GÕ nếu đáp án quá dài hoặc có dấu câu cuối.
Nó không chặn — có câu đáp án đúng là một cụm 4-5 từ — nhưng bạn phải thấy mình
đang làm gì.

⚠ Đây là proxy offline. Ban tổ chức chấm bằng "semantic exact match" nên thực tế
có thể rộng hơn. Số đo được ở đây là GIỚI HẠN DƯỚI, đừng đọc thành điểm thi thật.

CÂU HỎI CẦN CẢ HAI NGÔN NGỮ
---------------------------
Đo 2026-08-12: truy vấn tiếng Việt thẳng kém tiếng Anh người dịch −0.1200 ±
0.0225 trên SigLIP2. Nên ô tiếng Anh vẫn bắt buộc, giống hệt phần KIS.
"""
import csv
import json
import os
import sys

GOC = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.join(GOC, "dev_set")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SO_CAU_QA = 30          # việc 4.1 cần ~30 câu; lấy 30 ảnh đầu của dev set

TRANG = """<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<title>Gán nhãn QA — AIC2026</title>
<style>
 :root{color-scheme:light dark}
 body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;
      background:#12141a;color:#e8eaed;line-height:1.5}
 header{position:sticky;top:0;z-index:9;background:#1b1e26;border-bottom:1px solid #2c313c;
        padding:14px 22px;display:flex;gap:20px;align-items:center;flex-wrap:wrap}
 h1{font-size:17px;margin:0;font-weight:600}
 .tt{font-variant-numeric:tabular-nums;color:#9aa3b2;font-size:14px}
 .thanh{flex:1;min-width:160px;height:7px;background:#2c313c;border-radius:4px;overflow:hidden}
 .thanh i{display:block;height:100%;background:#c58af9;width:0;transition:width .2s}
 button{background:#c58af9;color:#12141a;border:0;padding:9px 16px;border-radius:7px;
        font-size:14px;font-weight:600;cursor:pointer}
 button:hover{background:#b06ef0}
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
 .o input:focus{outline:0;border-color:#c58af9}
 .o input.dapan{font-weight:600;border-color:#4a5160}
 .so{font-size:13px;color:#7f8796;font-weight:600}
 .canh{font-size:12px;color:#e0a35c;margin-top:5px;min-height:16px}
 .khop{font-size:12px;color:#7f8796;margin-top:5px;font-family:ui-monospace,monospace}
 .tip{background:#1b1e26;border:1px solid #2c313c;border-left:3px solid #c58af9;
      border-radius:8px;padding:14px 18px;font-size:14px;color:#c8cdd6}
 .tip b{color:#e8eaed}
 .tip code{background:#12141a;padding:1px 5px;border-radius:4px;font-size:13px}
 dialog{border:0;background:#000;padding:0;max-width:96vw;max-height:96vh}
 dialog img{max-width:96vw;max-height:96vh;display:block}
</style></head><body>
<header>
  <h1>Gán nhãn QA</h1>
  <span class="tt" id="tt">0 / 0</span>
  <span class="thanh"><i id="thanh"></i></span>
  <button onclick="xuat()">⬇ Xuất dan_nhan_qa.csv</button>
  <button class="phu" onclick="xoaHet()">Xoá hết</button>
</header>
<main>
  <div class="tip">
    <b>Đáp án phải NGẮN và CHUẨN HOÁ.</b> Máy chấm so khớp <b>chuỗi chính xác</b>
    sau khi hạ chữ thường và gộp khoảng trắng — <code>"đỏ."</code> KHÔNG khớp
    <code>"đỏ"</code>, <code>"màu đỏ"</code> KHÔNG khớp <code>"đỏ"</code>.<br>
    ✅ &nbsp;hỏi "người đàn ông mặc áo màu gì?" → đáp <code>đỏ</code><br>
    ✅ &nbsp;hỏi "có mấy người trên sân khấu?" → đáp <code>3</code><br>
    ❌ &nbsp;đáp <code>Anh ấy mặc áo màu đỏ.</code> — dài, có dấu câu, sẽ luôn sai<br><br>
    <b>Câu hỏi phải trả lời được CHỈ BẰNG ẢNH này</b>, và đáp án phải là thứ nhìn
    thấy được. Đừng hỏi tên riêng, ngày tháng, hay thứ phải suy đoán.<br><br>
    <b>Ô tiếng Anh bắt buộc</b> — đo được tiếng Việt thẳng kém hơn
    −0.1200 ± 0.0225. Bấm ảnh để phóng to. Tự lưu sau mỗi lần gõ.
  </div>
  <div id="ds"></div>
</main>
<dialog id="to" onclick="this.close()"><img id="toimg"></dialog>
<script>
const HANG = __DATA__;
const KHOA = "aic2026_gan_nhan_qa_v1";
let luu = JSON.parse(localStorage.getItem(KHOA) || "{}");

// Bản sao của scoring.normalize_answer — để bạn thấy máy sẽ so cái gì.
function chuanHoa(s){
  return (s||"").normalize("NFC").trim().toLowerCase()
          .replace(/^"+|"+$/g, "").trim().replace(/\\s+/g, " ");
}
function xong1(g){ return !!(g.hoi && g.hoi_en && g.dap); }

function dem(){
  const n = HANG.filter(h => xong1(luu[h.anh]||{})).length;
  document.getElementById("tt").textContent = n + " / " + HANG.length + " xong";
  document.getElementById("thanh").style.width = (100*n/HANG.length) + "%";
  return n;
}
function canhBao(dap){
  const c = chuanHoa(dap);
  if(!c) return "";
  const tu = c.split(" ").length;
  if(/[.!?,;:]$/.test(c)) return "⚠ có dấu câu ở cuối — máy so chuỗi chính xác nên sẽ không khớp";
  if(tu > 5) return "⚠ " + tu + " từ, hơi dài. Đáp án càng dài càng khó khớp đúng";
  if(tu > 3) return "⚠ " + tu + " từ — cân nhắc rút gọn còn cụm ngắn nhất vẫn đúng";
  return "";
}
function ghi(anh, truong, gt){
  luu[anh] = Object.assign({hoi:"",hoi_en:"",dap:""}, luu[anh]);
  luu[anh][truong] = gt;
  localStorage.setItem(KHOA, JSON.stringify(luu));
  document.getElementById("the_"+anh).classList.toggle("xong", xong1(luu[anh]));
  if(truong === "dap"){
    document.getElementById("canh_"+anh).textContent = canhBao(gt);
    document.getElementById("khop_"+anh).textContent =
      gt ? "máy sẽ so: \\"" + chuanHoa(gt) + "\\"" : "";
  }
  dem();
}
function phong(src){ document.getElementById("toimg").src = src;
                     document.getElementById("to").showModal(); }
function xoaHet(){
  if(!confirm("Xoá toàn bộ nhãn QA đã gõ?")) return;
  luu = {}; localStorage.removeItem(KHOA); ve();
}
function csvO(s){ s = (s||"").replace(/"/g,'""');
                  return /[",\\n]/.test(s) ? '"'+s+'"' : s; }
function xuat(){
  const thieu = HANG.filter(h => {
    const g = luu[h.anh]||{};
    return (g.hoi||g.hoi_en||g.dap) && !xong1(g);
  });
  if(thieu.length && !confirm(thieu.length + " câu điền dở (thiếu ô) sẽ bị BỎ khi "
      + "tạo dev set. Xuất luôn?")) return;
  const cot = ["stt","video_id","n","pts_time","fps","frame_idx","anh",
               "cau_hoi","cau_hoi_en","dap_an"];
  const dong = [cot.join(",")];
  for(const h of HANG){
    const g = luu[h.anh] || {hoi:"",hoi_en:"",dap:""};
    dong.push([h.stt,h.video_id,h.n,h.pts_time,h.fps,h.frame_idx,h.anh,
               csvO(g.hoi), csvO(g.hoi_en), csvO(g.dap)].join(","));
  }
  const blob = new Blob(["\\ufeff"+dong.join("\\n")+"\\n"],
                        {type:"text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "dan_nhan_qa.csv"; a.click();
  alert("Đã tải dan_nhan_qa.csv.\\n\\nChép vào thư mục dev_set\\\\ rồi chạy:\\n"
        + "python tao_dev_set.py");
}
function ve(){
  document.getElementById("ds").innerHTML = HANG.map(h => {
    const g = Object.assign({hoi:"",hoi_en:"",dap:""}, luu[h.anh]);
    const q = s => (s||"").replace(/"/g,'&quot;');
    return `<div class="the${xong1(g)?" xong":""}" id="the_${h.anh}">
      <div><img src="anh/${h.anh}" loading="lazy" onclick="phong(this.src)">
        <div class="meta">${h.video_id} · keyframe ${h.n} · frame ${h.frame_idx}
          <br>${(h.mo_ta||"").slice(0,90)}</div></div>
      <div class="o"><div class="so">QA ${h.stt} / ${HANG.length}</div>
        <label>Câu hỏi tiếng Việt</label>
        <input value="${q(g.hoi)}" placeholder="ví dụ: người đàn ông mặc áo màu gì?"
               oninput="ghi('${h.anh}','hoi',this.value)">
        <label>Câu hỏi tiếng Anh (bắt buộc)</label>
        <input value="${q(g.hoi_en)}" placeholder="e.g. what colour is the man's shirt?"
               oninput="ghi('${h.anh}','hoi_en',this.value)">
        <label>Đáp án — NGẮN, không dấu câu cuối</label>
        <input class="dapan" value="${q(g.dap)}" placeholder="đỏ"
               oninput="ghi('${h.anh}','dap',this.value)">
        <div class="canh" id="canh_${h.anh}">${canhBao(g.dap)}</div>
        <div class="khop" id="khop_${h.anh}">${g.dap ? 'máy sẽ so: "'+chuanHoa(g.dap)+'"' : ''}</div>
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
        raise SystemExit(f"Không thấy {csv_path} — cần dev set KIS trước.")

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    # Chỉ lấy những ảnh ĐÃ có nhãn KIS: cửa sổ đáp án của QA lấy từ đó, và ảnh
    # chưa gán nhãn KIS thì cũng chưa chắc nhìn rõ nội dung.
    co_nhan = [r for r in rows if (r.get("mo_ta") or "").strip()]
    if not co_nhan:
        raise SystemExit("Chưa có nhãn KIS nào trong dan_nhan.csv.")
    chon = co_nhan[:SO_CAU_QA]

    ra = os.path.join(DEV, "gan_nhan_qa.html")
    with open(ra, "w", encoding="utf-8") as fh:
        fh.write(TRANG.replace("__DATA__", json.dumps(chon, ensure_ascii=False)))

    print(f"✅ Đã tạo {ra}  ({len(chon)}/{len(co_nhan)} ảnh có nhãn KIS)")
    print("\n   Mở bằng trình duyệt, gõ xong bấm '⬇ Xuất dan_nhan_qa.csv',")
    print("   chép vào dev_set\\, rồi chạy:  python tao_dev_set.py")
    print("\n   Đáp án NGẮN và không có dấu câu cuối — máy so chuỗi chính xác.")


if __name__ == "__main__":
    main()
