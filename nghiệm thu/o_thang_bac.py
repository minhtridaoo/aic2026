# ── LADDER / CELL: THANG BẬC NHÀ CUNG CẤP — THAY CHO BĂNG ĐẠN KEY ────────────
# Mục tiêu: đổi tầng API từ "một provider, xoay nhiều key" sang
#           "nhiều provider × nhiều model, xoay tuần tự".
# INPUT:    khoá trong Kaggle Secrets / biến môi trường / ô textarea cũ.
# OUTPUT:   ghi đè `_adapt_via_hybrid` và `_run_qa`; UI và server giữ nguyên.
# Pre-check: `in_thang()` in ra thang bậc trước khi chạy, 0 request.
# Post-check: `in_thang_trace()` cho biết BẬC NÀO đã trả lời từng câu.
# History v1.0.23: provider ladder thay key magazine.
# ──────────────────────────────────────────────────────────────────────────────
#
# ⚠ Ô NÀY PHẢI CHẠY SAU Ô UI **VÀ SAU** Ô BATCH.
#   Nó ghi đè `_adapt_via_hybrid`, `_run_qa`, `_tier_vua_roi` bằng cách định
#   nghĩa lại tên toàn cục; Python tra tên lúc GỌI nên định nghĩa sau thì thắng.
#   Chạy trước ô batch là bản cũ của `_tier_vua_roi` sẽ thắng và bảng tổng kết
#   mất phần tên model.
#
# VÌ SAO BỎ BĂNG ĐẠN KEY
# ----------------------
# Băng đạn xoay nhiều key CỦA CÙNG MỘT PROVIDER. Nó chỉ cứu được đúng một tình
# huống: key hỏng/bị thu hồi. Nó KHÔNG cứu được thứ hay xảy ra hơn nhiều:
#
#   * Google tính quota theo DỰ ÁN, không theo key. Nhiều key sinh trong cùng
#     một project dùng chung một túi quota — hết là hết cả loạt, xoay vô ích.
#   * Trần RPD gắn với TỪNG MODEL. Hết `gemini-3.6-flash` không có nghĩa là hết
#     `gemini-3.5-flash-lite`; đổi MODEL mới mở được túi quota mới.
#
# Nên thang bậc ở đây xoay theo (provider, model) trước, và vẫn xoay key BÊN
# TRONG mỗi bậc — giữ lại phần hữu ích của băng đạn, bỏ phần vô ích.
#
# Cộng dồn quota thật của Gemini free tier (số đọc từ dashboard 19/08):
#     bậc flash  RPM 5  · RPD 20  mỗi model   -> ~60 lượt/ngày cho 3 model
#     bậc lite   RPM 15 · RPD 500 mỗi model   -> ~1000 lượt/ngày cho 2 model
# Xếp chồng model = cộng dồn quota mà vẫn dùng chung một khoá.

import os as _os
import time as _time

# ── Cấu hình ─────────────────────────────────────────────────────────────────
CLOUDFLARE_ACCOUNT_ID = _os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()

# 🔴 QUYẾT ĐỊNH CÓ LÝ DO, ĐỔI ĐƯỢC BẰNG MỘT DÒNG.
# Thứ tự được yêu cầu là: Gemini → Qwen → Groq → Cloudflare → Local.
# Với KIS/TRAKE đó là thứ tự ĐÚNG: việc ở đó chỉ là viết lại tiếng Việt sang
# tiếng Anh — thuần chữ — nên Qwen3-4B chạy local làm được, miễn phí, không
# quota, và đặt nó trước Groq là tiết kiệm thật.
# Với QA thì khác: Qwen3-4B **không nhìn được ảnh**, nó chỉ đọc OCR/ASR. Đặt nó
# trước Groq/Cloudflare nghĩa là vừa hết Gemini đã tự bỏ khả năng nhìn ảnh,
# trong khi hai bậc vision vẫn còn nguyên quota. Vì vậy mặc định QA đi vision
# trước. Đặt False để về đúng thứ tự đã yêu cầu.
QA_VISION_TRUOC_QWEN = True

# Số model Gemini tối đa đưa vào thang. Mỗi model chết tốn đúng một request để
# phát hiện, và bậc chết được nhớ nên chỉ tốn một lần cho cả phiên.
TOI_DA_MODEL_GEMINI = 6

# Số lần thử lại CÙNG MỘT BẬC khi gặp 429. Mặc định 0, và đó là quyết định có
# lý do chứ không phải bỏ sót:
#   * 429 vì RPM  -> chờ vài giây thì qua, nhưng bậc kế tiếp là MODEL KHÁC nên
#     đi tiếp còn nhanh hơn chờ;
#   * 429 vì RPD  -> chờ bao lâu cũng vô ích, retry-after có khi 60s+;
# và ta không phân biệt được hai loại từ thông báo lỗi. Khi CHƯA có thang bậc
# thì chờ là lựa chọn duy nhất — đó là lý do bản băng-đạn cũ phải retry. Có
# thang rồi thì nhảy bậc luôn thắng: mỗi model là một túi quota riêng.
# Đặt >0 nếu chỉ còn đúng một bậc sống và buộc phải chờ nó.
SO_LAN_THU_LAI_RATE = 0

# Model Gemini dùng khi KHÔNG dò được (mạng lỗi / thiếu khoá).
GEMINI_MAC_DINH = [
    "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.5-flash",
    "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
]

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
CLOUDFLARE_MODEL = "@cf/meta/llama-3.2-11b-vision-instruct"

THANG = []                 # thang bậc đã dựng
THANG_TRACE = {}           # {query_id/label: [vết]}
_BAC_CHET = {}             # {nhãn bậc: lý do} — nhớ trong cả phiên
_BAC_VUA_ROI = {"nhan": None}


# ── Khoá ─────────────────────────────────────────────────────────────────────
def _keys_cua_buoc(buoc):
    """Danh sách key cho một bậc, theo thứ tự ưu tiên nguồn.

    Vẫn đọc ô textarea cũ khi provider của bậc trùng provider đang chọn trên
    dropdown — để ai quen cách cũ không phải đổi gì.
    """
    if buoc.get("keys"):
        return [str(x).strip() for x in buoc["keys"] if str(x).strip()]
    env = buoc.get("key_env")
    if env and str(_os.environ.get(env, "")).strip():
        return _parse_keys(_os.environ[env])
    try:
        if str(provider.value).strip() == buoc.get("provider"):
            return _parse_keys(api_keys.value)
    except Exception:                                        # noqa: BLE001
        pass
    return []


# ── Dò model Gemini ──────────────────────────────────────────────────────────
_LOAI_BO_MODEL = ("embedding", "embed", "aqa", "imagen", "veo", "tts",
                  "image-generation", "audio", "learnlm", "gemma")


def _xep_hang_gemini(ten):
    """Khoá sắp xếp: flash trước, flash-lite sau, còn lại cuối; mới trước cũ."""
    t = str(ten).lower()
    if "flash-lite" in t or "flash_lite" in t:
        bac = 1
    elif "flash" in t:
        bac = 0
    elif "pro" in t:
        bac = 3
    else:
        bac = 2
    so = re.findall(r"(\d+(?:\.\d+)?)", t)
    ver = -float(so[0]) if so else 0.0        # số lớn = mới = đứng trước
    return (bac, ver, t)


def do_model_gemini(api_key=None, im_lang=False):
    """Hỏi server danh sách model Gemini khả dụng cho khoá này. 1 request."""
    key = str(api_key or _os.environ.get("GEMINI_API_KEY", "")).strip()
    if not key:
        try:
            if str(provider.value).strip() == "gemini":
                ks = _parse_keys(api_keys.value)
                key = ks[0] if ks else ""
        except Exception:                                    # noqa: BLE001
            key = ""
    if not key:
        if not im_lang:
            print("  ⚠ không có GEMINI_API_KEY -> dùng danh sách mặc định")
        return list(GEMINI_MAC_DINH)
    try:
        resp = _be_post("/models", {"provider": "gemini", "api_key": key,
                                    "model": "", "base_url": ""}, timeout=60)
    except Exception as e:                                   # noqa: BLE001
        if not im_lang:
            print(f"  ⚠ dò model Gemini hỏng ({str(e)[:90]}) -> dùng mặc định")
        return list(GEMINI_MAC_DINH)
    ms = [str(m) for m in resp.get("models", []) if str(m).strip()]
    ms = [m for m in ms
          if "gemini" in m.lower()
          and not any(x in m.lower() for x in _LOAI_BO_MODEL)]
    if not ms:
        if not im_lang:
            print("  ⚠ /models không trả model Gemini nào -> dùng mặc định")
        return list(GEMINI_MAC_DINH)
    return sorted(set(ms), key=_xep_hang_gemini)[:TOI_DA_MODEL_GEMINI]


# ── Dựng thang ───────────────────────────────────────────────────────────────
def dung_thang(*, gemini_models=None, tu_dong_do=True, in_ra=True):
    """Dựng THANG cho cả phiên. Gọi lại bất cứ lúc nào để đổi cấu hình."""
    global THANG
    ms = list(gemini_models) if gemini_models else (
        do_model_gemini() if tu_dong_do else list(GEMINI_MAC_DINH))

    buoc = [{"kind": "api", "nhan": f"gemini/{m}", "provider": "gemini",
             "model": m, "key_env": "GEMINI_API_KEY", "base_url": ""}
            for m in ms]
    buoc.append({"kind": "qwen", "nhan": "qwen3-4b-kaggle"})
    buoc.append({"kind": "api", "nhan": f"groq/{GROQ_MODEL.split('/')[-1]}",
                 "provider": "groq", "model": GROQ_MODEL,
                 "key_env": "GROQ_API_KEY", "base_url": ""})
    if CLOUDFLARE_ACCOUNT_ID:
        buoc.append({
            "kind": "api", "nhan": "cloudflare/llama-3.2-11b-vision",
            "provider": "custom_openai_compatible", "model": CLOUDFLARE_MODEL,
            "key_env": "CLOUDFLARE_API_TOKEN",
            "base_url": (f"https://api.cloudflare.com/client/v4/accounts/"
                         f"{CLOUDFLARE_ACCOUNT_ID}/ai/v1")})
    buoc.append({"kind": "local", "nhan": "local"})

    THANG = buoc
    if in_ra:
        in_thang()
    return THANG


def _thang_cho(task):
    """Thang bậc theo task. QA đảo Qwen xuống sau vision — xem QA_VISION_TRUOC_QWEN."""
    t = list(THANG or dung_thang(in_ra=False))
    if str(task).upper() != "QA" or not QA_VISION_TRUOC_QWEN:
        return t
    qwen = [b for b in t if b["kind"] == "qwen"]
    if not qwen:
        return t
    con_lai = [b for b in t if b["kind"] != "qwen"]
    i = next((n for n, b in enumerate(con_lai) if b["kind"] == "local"),
             len(con_lai))
    return con_lai[:i] + qwen + con_lai[i:]


def in_thang():
    if not THANG:
        print("  (chưa dựng thang — gọi dung_thang())")
        return
    print(f"  THANG BẬC ({len(THANG)} bậc)")
    for i, b in enumerate(THANG, 1):
        n = len(_keys_cua_buoc(b)) if b["kind"] == "api" else 0
        gc = (f"{n} khoá" if b["kind"] == "api" else
              "model local trong Kaggle" if b["kind"] == "qwen" else
              "evidence-only, không API")
        chet = _BAC_CHET.get(b["nhan"])
        print(f"    {i}. {b['nhan']:44s} {gc}"
              + (f"   ☠ {chet}" if chet else "")
              + ("   ⚠ KHÔNG CÓ KHOÁ -> sẽ bị bỏ qua"
                 if b["kind"] == "api" and not n else ""))
    print(f"  QA: {'vision trước Qwen' if QA_VISION_TRUOC_QWEN else 'đúng thứ tự khai báo'}")


def reset_thang():
    """Quên các bậc đã đánh dấu chết. Gọi khi sang ngày mới / đổi khoá."""
    _BAC_CHET.clear()
    print("  đã xoá đánh dấu bậc chết")


# ── Gọi một bậc API ──────────────────────────────────────────────────────────
def _goi_bac_api(buoc, path, payload, nhan_pha=""):
    """Trả `(resp, None)` hoặc `(None, lý_do)`. Xoay key BÊN TRONG bậc."""
    if buoc["nhan"] in _BAC_CHET:
        return None, f"đã chết: {_BAC_CHET[buoc['nhan']]}"

    keys = _keys_cua_buoc(buoc)
    if not keys:
        _BAC_CHET[buoc["nhan"]] = "thiếu khoá"
        return None, "thiếu khoá"

    ly_do_cuoi = "?"
    for ki, key in enumerate(keys, 1):
        req = {**payload, "provider": buoc["provider"], "api_key": key,
               "model": buoc["model"], "base_url": buoc.get("base_url", "")}
        lan = 0
        while True:
            lan += 1
            try:
                return _be_post(path, req, timeout=360), None
            except Exception as e:                           # noqa: BLE001
                msg = str(e)
                cls = _err_class(msg)
                if cls == "RATE" and lan <= SO_LAN_THU_LAI_RATE:
                    cho = min(8, 2 ** lan)
                    try:
                        run_status.value = (
                            f"<b style='color:#ffc35b'>{nhan_pha} · "
                            f"{html.escape(buoc['nhan'])} rate-limit · "
                            f"chờ {cho}s</b>")
                    except Exception:                        # noqa: BLE001
                        pass
                    _time.sleep(cho)
                    continue
                ly_do_cuoi = f"{cls}: {msg[:130]}"
                break                                        # sang key kế
    # Hết mọi key của bậc này. Đánh dấu chết để 23 câu sau không gõ lại cửa đã
    # khoá — với 24 câu × 6 bậc thì đó là hơn trăm request vứt đi và vài phút.
    _BAC_CHET[buoc["nhan"]] = ly_do_cuoi
    return None, ly_do_cuoi


def _vet(nhan_cau, bac, ket_qua, chi_tiet=""):
    THANG_TRACE.setdefault(nhan_cau, []).append(
        {"bac": bac, "ket_qua": ket_qua, "chi_tiet": str(chi_tiet)[:200]})


def _chot(nhan_cau, bac):
    _BAC_VUA_ROI["nhan"] = bac
    _vet(nhan_cau, bac, "OK")


def _tier_vua_roi():
    """GHI ĐÈ bản trong ô batch: trả BẬC thật thay vì nhãn chung chung."""
    return _BAC_VUA_ROI["nhan"] or "?"


# ── GHI ĐÈ: KIS / TRAKE ──────────────────────────────────────────────────────
def _adapt_via_hybrid(task_name, query_text, events=None):
    """Đi hết thang bậc cho tầng viết lại truy vấn (KIS/TRAKE)."""
    nhan_cau = f"{task_name}:{str(query_text)[:40]}"
    THANG_TRACE.pop(nhan_cau, None)

    if str(provider.value).strip() == "local":
        return {"fallback_local": True, "reason": "FORCED_LOCAL_PROVIDER"}

    for buoc in _thang_cho(task_name):
        if buoc["kind"] == "local":
            break

        if buoc["kind"] == "qwen":
            q = _qwen_adapter(task_name, query_text, events=events)
            if q.get("fallback_local"):
                _vet(nhan_cau, buoc["nhan"], "HONG", q.get("reason"))
                continue
            if _adapter_result_is_degenerate(q, task_name):
                _vet(nhan_cau, buoc["nhan"], "DEGENERATE")
                continue
            q["tier"] = "KAGGLE_QWEN3_4B_TEXT"
            _chot(nhan_cau, buoc["nhan"])
            return q

        resp, ly_do = _goi_bac_api(
            buoc, "/query/adapt",
            {"task_type": str(task_name).upper(), "query": str(query_text),
             "events": events}, "ADAPT")
        if resp is None:
            _vet(nhan_cau, buoc["nhan"], "HONG", ly_do)
            continue
        if _adapter_result_is_degenerate(resp, task_name):
            # Trả lời được nhưng ra rác: KHÔNG đánh dấu bậc chết (lần sau có
            # thể ổn), chỉ bỏ qua câu này.
            _vet(nhan_cau, buoc["nhan"], "DEGENERATE")
            continue
        resp["tier"] = "REMOTE_API"
        _chot(nhan_cau, buoc["nhan"])
        return resp

    _BAC_VUA_ROI["nhan"] = "local"
    _vet(nhan_cau, "local", "OK")
    return {"fallback_local": True, "reason": "THANG_CAN_MOI_BAC"}


# ── GHI ĐÈ: QA ───────────────────────────────────────────────────────────────
def _run_qa(query_text):
    """Hai pha (answer + verify), mỗi pha đi thang bậc riêng.

    Giữ NGUYÊN hợp đồng của bản gốc, kể cả điều quan trọng nhất: khi không
    verify được thì trả local evidence-only và KHÔNG ghi CSV giả.
    """
    original = str(query_text).strip()
    nhan_cau = f"QA:{original[:40]}"
    THANG_TRACE.pop(nhan_cau, None)

    if str(provider.value).strip() == "local":
        return _use_local_fallback(
            "QA", original, reason="FORCED_LOCAL_PROVIDER_EVIDENCE_ONLY")

    thang = _thang_cho("QA")

    # ── PHA 1: ANSWER ────────────────────────────────────────────────────────
    answer_resp, i_answer = None, None
    for i, buoc in enumerate(thang):
        if buoc["kind"] == "local":
            break
        if buoc["kind"] == "qwen":
            q = _qwen_full_qa(original)
            if q.get("fallback_local"):
                _vet(nhan_cau, buoc["nhan"], "HONG", q.get("reason"))
                continue
            adapted_query.value = ("[QA · QWEN3-4B TEXT PASS]\n"
                                   "answer + verifier từ OCR/ASR.")
            _chot(nhan_cau, buoc["nhan"])
            return q

        resp, ly_do = _goi_bac_api(
            buoc, "/qa/phase/answer",
            {"task_type": "QA", "query": original}, "QA ANSWER")
        if resp is None:
            _vet(nhan_cau, buoc["nhan"], "HONG", ly_do)
            continue
        if not str(resp.get("phase_id", "")).strip() or \
           not str(resp.get("candidate_answer", "")).strip():
            _vet(nhan_cau, buoc["nhan"], "MALFORMED")
            continue
        answer_resp, i_answer = resp, i
        _vet(nhan_cau, buoc["nhan"], "ANSWER_OK")
        break

    if answer_resp is None:
        _BAC_VUA_ROI["nhan"] = "local"
        _vet(nhan_cau, "local", "OK")
        return _use_local_fallback(
            "QA", original, reason="THANG_CAN_MOI_BAC_TRUOC_ANSWER")

    phase_id = str(answer_resp["phase_id"]).strip()
    candidate = str(answer_resp["candidate_answer"]).strip()
    adapted_query.value = (f"[QA ANSWER · {thang[i_answer]['nhan']}]\n"
                           f"Candidate: {candidate}\nĐang verify…")

    # ── PHA 2: VERIFY ────────────────────────────────────────────────────────
    # Bắt đầu TỪ bậc vừa trả lời answer: nó vừa chứng minh là còn sống, còn các
    # bậc phía trên vừa chứng minh là đã chết. Đi lại từ đầu chỉ tốn thời gian.
    for buoc in thang[i_answer:]:
        if buoc["kind"] == "local":
            break
        if buoc["kind"] == "qwen":
            qv = _qwen_verify_phase(phase_id)
            if qv.get("fallback_local"):
                _vet(nhan_cau, buoc["nhan"], "VERIFY_HONG", qv.get("reason"))
                continue
            adapted_query.value = (
                f"[QA · answer {thang[i_answer]['nhan']} + verify "
                f"{buoc['nhan']}]\nAnswer: {candidate}")
            _chot(nhan_cau, f"{thang[i_answer]['nhan']}+{buoc['nhan']}")
            return qv

        resp, ly_do = _goi_bac_api(
            buoc, "/qa/phase/verify",
            {"task_type": "QA", "phase_id": phase_id}, "QA VERIFY")
        if resp is None:
            _vet(nhan_cau, buoc["nhan"], "VERIFY_HONG", ly_do)
            continue
        nhan = (thang[i_answer]["nhan"] if buoc is thang[i_answer]
                else f"{thang[i_answer]['nhan']}+{buoc['nhan']}")
        adapted_query.value = f"[QA · {nhan}]\nAnswer: {candidate}"
        _chot(nhan_cau, nhan)
        return resp

    # Có answer nhưng không verify nổi -> local evidence-only, không CSV giả.
    _BAC_VUA_ROI["nhan"] = "local(answer chưa verify)"
    _vet(nhan_cau, "local", "OK")
    local = _use_local_fallback("QA", original,
                                reason="THANG_CAN_MOI_BAC_O_PHA_VERIFY")
    adapted_query.value = (
        f"[QA · CÓ ANSWER · KHÔNG VERIFY ĐƯỢC]\n"
        f"Candidate chưa verify: {candidate}\n"
        f"Đã chuyển local evidence-only; không lưu CSV giả.")
    return local


# ── Vết ──────────────────────────────────────────────────────────────────────
def in_thang_trace(chi_tiet=False):
    """IN RA bậc nào đã trả lời. Gọi sau MỖI mẻ, không có ngoại lệ.

    Đây không phải tiện ích gỡ rối. Ngày 20/08 một bộ 30 câu QA đã tụt sạch
    xuống backend dự phòng mà kết quả không có một dấu hiệu nào; điểm đo được
    là điểm của backend dự phòng, và không ai biết cho tới khi mở log ra soi.
    """
    if not THANG_TRACE:
        print("  (chưa có lượt nào)")
        return
    dem = {}
    for nhan_cau, vet in THANG_TRACE.items():
        ok = next((v["bac"] for v in vet if v["ket_qua"] == "OK"), None)
        dem[ok or "(KHÔNG BẬC NÀO)"] = dem.get(ok or "(KHÔNG BẬC NÀO)", 0) + 1
        if chi_tiet:
            print(f"  {nhan_cau[:52]:52s} -> {ok}")
            for v in vet:
                if v["ket_qua"] == "OK":
                    continue
                # ANSWER_OK là bước thành công của pha 1, không phải lỗi. Đánh
                # dấu nó bằng ⨯ như thất bại là tự tạo báo động giả.
                dau = "·" if v["ket_qua"].endswith("_OK") else "⨯"
                print(f"      {dau} {v['bac']}: {v['ket_qua']} "
                      f"{v['chi_tiet'][:90]}")
    print("\n  BẬC ĐÃ TRẢ LỜI:")
    for b, n in sorted(dem.items(), key=lambda kv: -kv[1]):
        print(f"    {n:4d}  {b}")
    if _BAC_CHET:
        print("\n  BẬC ĐÃ CHẾT TRONG PHIÊN:")
        for b, r in _BAC_CHET.items():
            print(f"    ☠ {b:44s} {r[:90]}")
    tong = sum(dem.values())
    khong_api = sum(n for b, n in dem.items()
                    if b.startswith(("local", "qwen")) or "KHÔNG" in b)
    if khong_api:
        print(f"\n  ⚠ {khong_api}/{tong} câu KHÔNG do bậc API trả lời — "
              f"đừng đọc điểm như thể đó là điểm của API.")


# ── Tự kiểm bằng server giả: 0 request thật ─────────────────────────────────
def _tu_kiem_thang():
    that_be_post, that_qwen_a = _be_post, _qwen_adapter
    that_provider_value = provider.value
    goi = []

    def gia_be_post(path, payload, timeout=None):
        goi.append((path, payload.get("provider"), payload.get("model")))
        m = str(payload.get("model", ""))
        if "gemini" in m:
            raise RuntimeError("HTTP 429: rate limit exceeded")
        if "scout" in m:
            return {"ok": True, "search_query_en": "a chef plating spring rolls"}
        raise RuntimeError("HTTP 500: server error")

    def gia_qwen(task, q, events=None):
        goi.append(("QWEN", task, None))
        return {"fallback_local": True, "reason": "QWEN_MISSING"}

    try:
        globals()["_be_post"] = gia_be_post
        globals()["_qwen_adapter"] = gia_qwen
        provider.value = "groq"                 # khác "local" là được
        _BAC_CHET.clear()
        THANG_TRACE.clear()

        dung_thang(gemini_models=["gemini-a", "gemini-b"], in_ra=False)
        _os.environ.setdefault("GEMINI_API_KEY", "k-gemini")
        _os.environ.setdefault("GROQ_API_KEY", "k-groq")

        r = _adapt_via_hybrid("KIS", "một đầu bếp đang cuốn gỏi")
        thu_tu = [g for g in goi]
        assert r.get("search_query_en"), r
        assert _tier_vua_roi().startswith("groq/"), _tier_vua_roi()
        # đúng thứ tự: 2 model Gemini -> Qwen -> Groq
        assert thu_tu[0][2] == "gemini-a" and thu_tu[1][2] == "gemini-b"
        assert thu_tu[2][0] == "QWEN"
        assert "scout" in str(thu_tu[3][2])
        assert set(_BAC_CHET) == {"gemini/gemini-a", "gemini/gemini-b"}, _BAC_CHET

        # Bậc chết được NHỚ: lượt hai không gõ lại cửa đã khoá.
        goi.clear()
        _adapt_via_hybrid("KIS", "câu khác")
        assert not any("gemini" in str(g[2]) for g in goi), goi

        # QA đảo Qwen xuống sau vision.
        tq = [b["nhan"] for b in _thang_cho("QA")]
        assert tq.index("qwen3-4b-kaggle") > tq.index(
            f"groq/{GROQ_MODEL.split('/')[-1]}"), tq
        tk = [b["nhan"] for b in _thang_cho("KIS")]
        assert tk.index("qwen3-4b-kaggle") < tk.index(
            f"groq/{GROQ_MODEL.split('/')[-1]}"), tk

        # Sắp hạng model Gemini: flash trước lite, mới trước cũ.
        xep = sorted(["gemini-3.1-flash-lite", "gemini-3.6-flash",
                      "gemini-3.5-flash", "gemini-3.5-flash-lite",
                      "gemini-2.0-pro"], key=_xep_hang_gemini)
        assert xep[0] == "gemini-3.6-flash" and xep[1] == "gemini-3.5-flash"
        assert xep[-1] == "gemini-2.0-pro", xep
    finally:
        globals()["_be_post"] = that_be_post
        globals()["_qwen_adapter"] = that_qwen_a
        provider.value = that_provider_value
        _BAC_CHET.clear()
        THANG_TRACE.clear()
        _BAC_VUA_ROI["nhan"] = None


_tu_kiem_thang()
print("[PASS] thang bậc: đúng thứ tự · nhớ bậc chết · QA đảo Qwen sau vision "
      "· sắp hạng model")
print()
dung_thang()
print()
print("[READY] đường ra kết quả GIỜ đi theo thang bậc:")
print("  KIS/TRAKE : Gemini(mọi model) -> Qwen3-4B -> Groq -> Cloudflare -> Local")
print("  QA        : Gemini(mọi model) -> Groq -> Cloudflare -> Qwen3-4B -> Local")
print("  in_thang() · in_thang_trace(chi_tiet=True) · reset_thang()")
