# ── FINAL PATCH / CELL: FROZEN KIS + TRAKE/QA CONTRACT + BTC ADAPTER ──────────
# This cell intentionally runs AFTER the original online1 technical harness and
# BEFORE the public server cell.
#
# It changes the active KIS serving backend to the final promoted CLIP corpus,
# keeps the frozen TRAKE v3/raw-PTS engine, activates final-runtime QA provider
# selection, and enables the now-known AIC26 preliminary CSV wire format.
# ──────────────────────────────────────────────────────────────────────────────

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import io, base64, csv, zipfile, hashlib, json, math, os, re, time
import subprocess, sys, threading, socket, shutil
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
import requests
from transformers import CLIPModel, CLIPProcessor
from huggingface_hub import snapshot_download

# ---- final identity -----------------------------------------------------------
LEGACY_ONLINE_BUILD_VERSION=BUILD_VERSION
BUILD_VERSION="aic2026-final-competition-server-v1.0.21"
STAGE_ID="FINAL_competition_multitask_server"

FINAL_FROZEN_CONTRACTS={
    "KIS":{
        "source_stage":"50MEGArr",
        "handoff_sha256":"20050888c4ac7944c2e96c1effe2305b7053cc17a5471b44013eb98a32de93d2",
        "contract_sha256":"33e975018ac032e62ca7303d354f374f6e69507a20a020dd18b2d6796cc43139",
        "status":"FROZEN",
        "selected_run_name":"CLIP_ONLY",
        "selected_config_id":"CLIP_VIT_B32_ORGANIZER_SHADOW",
        "promotion_score_internal":0.388,
        "official_competition_score_claim":False,
    },
    "TRAKE":{
        "source_stage":"51TRAKEMEGArr",
        "handoff_sha256":"e225904954adac2b81f57e75c72c917bc678fc7ddd8882131339ea399cb755f1",
        "contract_sha256":"4735576b274f0e227545ea50259f4f274a7f90f17f9961fe305c36b58488cc61",
        "status":"FROZEN_TECHNICAL",
        "temporal_authority":"RAW_VIDEO_ACTUAL_PTS",
    },
    "QA":{
        "source_stage":"52QAMEGArr",
        "handoff_sha256":"dfdb18ceae08382cedc3b08be860e054fb5511487a3a5576ca98d5cea8467478",
        "contract_sha256":"8cd53b23b21920d45924198c31aefe96c3c1d7d7374023ba959a3eefa73583d6",
        "status":"READY_EXTERNAL_API_PENDING",
        "provider_lock":"NONE",
        "model_lock":"NONE",
    },
}

OFFICIAL_SUBMISSION_STATUS="READY_AIC26_PRELIMINARY_SPEC_PINNED"
BTC_SUBMISSION_ROOT=Path("/kaggle/working/submission")
BTC_SUBMISSION_ROOT.mkdir(parents=True,exist_ok=True)
BTC_ZIP_PATH=Path("/kaggle/working/submission.zip")

# KIS final runtime still requires exactly two GPUs.
if not torch.cuda.is_available() or torch.cuda.device_count()!=2:
    raise RuntimeError(("FINAL_EXACTLY_TWO_CUDA_GPUS_REQUIRED",
                        torch.cuda.device_count() if torch.cuda.is_available() else 0))

# Disable the old Stage03-active KIS asynchronous shadow, because KIS active is
# now the promoted CLIP backend.
O2_KIS_SHADOW_ASYNC=False

# ---- exact final Component06 CLIP corpus -------------------------------------
FINAL_CLIP_MODEL_ID="openai/clip-vit-base-patch32"
FINAL_CLIP_REVISION="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
FINAL_CLIP_WEIGHT_SHA="a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"
FINAL_COMP06_BUILD="aic2026-kis-rrf-shadow-v0.2.1"
FINAL_COMP06_HANDOFF_SHA="787ad4472a8ff0149a2338dcb3c0650d1f72b10446495d58126cafa1c595d5ca"
FINAL_CLIP_DIM=512
FINAL_CLIP_ROWS=177321
FINAL_KIS_OUTPUT_TOPK=100
FINAL_CLIP_PARITY_MIN=0.9999

FINAL_COMP06_ROOT=resolve_dataset("selaninguyen/6rrnote")
FINAL_H06_PATH=unique_under(FINAL_COMP06_ROOT,"component06_handoff.json")
if sha256_file(FINAL_H06_PATH)!=FINAL_COMP06_HANDOFF_SHA:
    raise RuntimeError(("FINAL_COMP06_HANDOFF_SHA",sha256_file(FINAL_H06_PATH)))
FINAL_H06=read_json(FINAL_H06_PATH)
if FINAL_H06.get("build_version")!=FINAL_COMP06_BUILD:
    raise RuntimeError(("FINAL_COMP06_BUILD",FINAL_H06.get("build_version")))

FINAL_R06_PATH=unique_under(FINAL_COMP06_ROOT,"component06_rrf_shadow_report.json")
FINAL_R06=read_json(FINAL_R06_PATH)
if FINAL_R06.get("canonical_identity")!="STAGE03_FRAME_REF_ID":
    raise RuntimeError(("FINAL_COMP06_CANONICAL_ID",FINAL_R06.get("canonical_identity")))
if FINAL_R06.get("pts_rounding_identity_used") is not False:
    raise RuntimeError("FINAL_COMP06_PTS_ROUNDING_FORBIDDEN")

FINAL_CLIP_MATRIX_PATH=unique_under(
    FINAL_COMP06_ROOT,"legacy_clip_aligned_stage03_order.fp16.npy"
)
FINAL_CLIP_NP=np.load(FINAL_CLIP_MATRIX_PATH,mmap_mode="r")
if FINAL_CLIP_NP.shape!=(FINAL_CLIP_ROWS,FINAL_CLIP_DIM):
    raise RuntimeError(("FINAL_CLIP_CORPUS_SHAPE",FINAL_CLIP_NP.shape))
if len(VISUAL_META)!=FINAL_CLIP_ROWS:
    raise RuntimeError(("FINAL_STAGE03_ROWS",len(VISUAL_META)))
if not np.array_equal(
    VISUAL_META.index_row.to_numpy(np.int64),
    np.arange(FINAL_CLIP_ROWS,dtype=np.int64),
):
    raise RuntimeError("FINAL_STAGE03_ORDER_DRIFT")

# ---- pinned CLIP model --------------------------------------------------------
if transformers.__version__!="5.0.0":
    raise RuntimeError(("FINAL_TRANSFORMERS_VERSION",
                        transformers.__version__,"5.0.0"))

clip_env=os.environ.get("AIC2026_CLIP_MODEL_DIR","").strip()
if clip_env:
    FINAL_CLIP_MODEL_DIR=Path(clip_env).resolve()
    if not (FINAL_CLIP_MODEL_DIR/"config.json").is_file():
        raise RuntimeError(("AIC2026_CLIP_MODEL_DIR_INVALID",str(FINAL_CLIP_MODEL_DIR)))
    FINAL_CLIP_MODEL_SOURCE="EXPLICIT_LOCAL_DIR"
else:
    FINAL_CLIP_MODEL_DIR=Path(snapshot_download(
        repo_id=FINAL_CLIP_MODEL_ID,
        revision=FINAL_CLIP_REVISION,
        allow_patterns=[
            "config.json","preprocessor_config.json","tokenizer.json",
            "tokenizer_config.json","vocab.json","merges.txt",
            "special_tokens_map.json","model.safetensors","pytorch_model.bin",
        ],
    )).resolve()
    FINAL_CLIP_MODEL_SOURCE="HF_PINNED_REVISION"

_weights=[
    p for p in (
        FINAL_CLIP_MODEL_DIR/"model.safetensors",
        FINAL_CLIP_MODEL_DIR/"pytorch_model.bin",
    ) if p.is_file()
]
if len(_weights)!=1:
    raise RuntimeError(("FINAL_CLIP_WEIGHT_COUNT",len(_weights),[str(x) for x in _weights]))
FINAL_CLIP_WEIGHT_PATH=_weights[0]
if sha256_file(FINAL_CLIP_WEIGHT_PATH)!=FINAL_CLIP_WEIGHT_SHA:
    raise RuntimeError(("FINAL_CLIP_WEIGHT_SHA",
                        sha256_file(FINAL_CLIP_WEIGHT_PATH),FINAL_CLIP_WEIGHT_SHA))

def _final_load_clip_worker(gpu_id):
    torch.cuda.set_device(gpu_id)
    dev=f"cuda:{gpu_id}"
    proc=CLIPProcessor.from_pretrained(
        str(FINAL_CLIP_MODEL_DIR),local_files_only=True,use_fast=True
    )
    model=CLIPModel.from_pretrained(
        str(FINAL_CLIP_MODEL_DIR),
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval().to(dev)

    # Preserve the executed promotion hotfix: CLIP text path is FP32 under
    # transformers 5.0 even though the rest of the model may be FP16.
    model.text_model.to(device=dev,dtype=torch.float32)
    model.text_projection.to(device=dev,dtype=torch.float32)

    if {p.dtype for p in model.text_model.parameters()}!={torch.float32}:
        raise RuntimeError(("FINAL_CLIP_TEXT_DTYPE",dev))
    if {p.dtype for p in model.text_projection.parameters()}!={torch.float32}:
        raise RuntimeError(("FINAL_CLIP_PROJ_DTYPE",dev))
    return {"gpu_id":gpu_id,"device":dev,"processor":proc,"model":model}

with ThreadPoolExecutor(max_workers=2) as ex:
    FINAL_CLIP_WORKERS=list(ex.map(_final_load_clip_worker,(0,1)))

_split=(FINAL_CLIP_ROWS+1)//2
_ranges=((0,_split),(_split,FINAL_CLIP_ROWS))

def _final_attach_clip_shard(worker,ab):
    a,b=ab
    torch.cuda.set_device(worker["gpu_id"])
    x=torch.from_numpy(
        np.asarray(FINAL_CLIP_NP[a:b],dtype=np.float32).copy()
    ).to(worker["device"])
    x=F.normalize(x,dim=1)
    worker["row_start"]=a
    worker["row_end"]=b
    worker["clip_shard"]=x
    return {"device":worker["device"],"rows":b-a,"shape":tuple(x.shape)}

with ThreadPoolExecutor(max_workers=2) as ex:
    FINAL_CLIP_SHARDS=list(ex.map(
        _final_attach_clip_shard,FINAL_CLIP_WORKERS,_ranges
    ))
if sum(x["rows"] for x in FINAL_CLIP_SHARDS)!=FINAL_CLIP_ROWS:
    raise RuntimeError("FINAL_CLIP_SHARD_TOTAL")

FINAL_CLIP_GPU_USAGE={
    "cuda:0":{"text_encodes":0,"searches":0},
    "cuda:1":{"text_encodes":0,"searches":0},
}

def _final_clip_encode_one(worker,text):
    torch.cuda.set_device(worker["gpu_id"])
    # OpenAI CLIP text encoder has max_position_embeddings=77.
    # Manual BTC queries can be much longer, so fail-safe truncate at the
    # tokenizer boundary instead of crashing with sequence_length > 77.
    inp=worker["processor"](
        text=[str(text)],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    inp={k:v.to(worker["device"]) for k,v in inp.items()}
    with torch.inference_mode():
        raw=worker["model"].get_text_features(**inp)
        if hasattr(raw,"pooler_output") and raw.pooler_output is not None:
            raw=raw.pooler_output
        if not torch.is_tensor(raw):
            raise TypeError(("FINAL_CLIP_TEXT_OUTPUT",type(raw)))
        v=F.normalize(raw.float(),dim=-1)[0]
        if not torch.isfinite(v).all():
            raise RuntimeError(("FINAL_CLIP_NONFINITE_QUERY",worker["device"]))
    FINAL_CLIP_GPU_USAGE[worker["device"]]["text_encodes"]+=1
    return v.detach().cpu().numpy().astype(np.float32)

def _final_clip_encode_replicated(text):
    with ThreadPoolExecutor(max_workers=2) as ex:
        vecs=list(ex.map(
            lambda w:_final_clip_encode_one(w,text),
            FINAL_CLIP_WORKERS
        ))
    a=vecs[0]/max(float(np.linalg.norm(vecs[0])),1e-12)
    b=vecs[1]/max(float(np.linalg.norm(vecs[1])),1e-12)
    cos=float(np.dot(a,b))
    if cos<FINAL_CLIP_PARITY_MIN:
        raise RuntimeError(("FINAL_CLIP_REPLICA_PARITY",cos,FINAL_CLIP_PARITY_MIN))
    return a.astype(np.float32),cos

def _final_clip_search_shard(worker,q,k):
    torch.cuda.set_device(worker["gpu_id"])
    qt=torch.from_numpy(np.asarray(q,dtype=np.float32)).to(worker["device"])
    with torch.inference_mode():
        scores=worker["clip_shard"]@qt
        kk=min(int(k),len(scores))
        val,idx=torch.topk(scores,k=kk,largest=True,sorted=True)
    FINAL_CLIP_GPU_USAGE[worker["device"]]["searches"]+=1
    return (
        val.detach().cpu().numpy().astype(np.float32),
        idx.detach().cpu().numpy().astype(np.int64)+int(worker["row_start"]),
    )

def _final_clip_search(q,k=100):
    with ThreadPoolExecutor(max_workers=2) as ex:
        got=list(ex.map(
            lambda w:_final_clip_search_shard(w,q,k),
            FINAL_CLIP_WORKERS
        ))
    vals=np.concatenate([x[0] for x in got])
    inds=np.concatenate([x[1] for x in got])
    kk=min(int(k),len(vals))
    sel=np.argpartition(-vals,kk-1)[:kk]
    order=np.lexsort((inds[sel],-vals[sel]))
    sel=sel[order]
    return vals[sel],inds[sel]

def _final_kis_rows(scores,indices):
    rows=[]
    for rank,(score,idx) in enumerate(zip(scores,indices),1):
        r=VISUAL_META.iloc[int(idx)]
        rows.append({
            "rank":rank,
            "frame_ref_id":str(r.frame_ref_id),
            "video_id":str(r.video_id),
            "pts_time":float(r.pts_time),
            "frame_idx_candidate":int(r.frame_idx_candidate),
            "image_artifact_ref":str(getattr(r,"image_artifact_ref","")),
            "raw_video_ref":str(getattr(r,"raw_video_ref","")),
            "clip_cosine":float(score),
            # Compatibility aliases consumed by the original UI walkers.
            "raw_cosine":float(score),
            "prompt_cosine":float(score),
            "dual_view_fused_cosine":float(score),
            "rrf_pool_score_diagnostic":None,
        })
    return rows

# Preserve the pre-promotion local KIS backend as an explicit emergency fallback.
# At this point `run_kis` is still the original Online1/Stage03 SigLIP2 path.
# The final CLIP override below remains the normal primary backend.
run_kis_stage03_local_fallback = run_kis

# FINAL ACTIVE KIS — override old Stage03 active backend.
def run_kis(query,query_id=None,include_multimodal_support=False):
    q=canonical_query_text(query)
    if query_id is None:
        query_id="kis_"+hashlib.sha256(q.encode()).hexdigest()[:16]
    t0=time.perf_counter()
    qvec,parity=_final_clip_encode_replicated(q)
    scores,indices=_final_clip_search(qvec,FINAL_KIS_OUTPUT_TOPK)
    rows=_final_kis_rows(scores,indices)
    if not rows:
        raise RuntimeError("FINAL_KIS_EMPTY")
    payload={
        "schema_version":"final-kis-clip-promoted-v1",
        "build_version":BUILD_VERSION,
        "query_id":query_id,
        "task_type":"KIS",
        "query_text":q,
        "active_policy":"CLIP_VIT_B32_ORGANIZER_SHADOW_PROMOTED_FINAL",
        "selected_config_id":"CLIP_VIT_B32_ORGANIZER_SHADOW",
        "hard_valid":True,
        "results":rows,
        "risk_diagnostics":{"seed_flags":[],"final_backend_frozen":True},
        "normalization_status":"L2_NORMALIZED_EXACT_COSINE",
        "probability_calibration_status":"NOT_CALIBRATED",
        "official_competition_score_claim":False,
        "timing_ms":{"total":(time.perf_counter()-t0)*1000.0},
        "dual_gpu_text_replica_cosine":parity,
    }
    RESULT_STORE.put_initial(query_id,payload)
    return RESULT_STORE.get(query_id)

# Update branch-plan semantics without changing TRAKE/QA retrieval engines.
_old_branch_execution_plan=branch_execution_plan
def branch_execution_plan(clean_req):
    if clean_req["task_type"]=="KIS":
        return {
            "task_type":"KIS",
            "primary":"FINAL_PROMOTED_CLIP_VIT_B32_STAGE03_ORDER",
            "selected_config_id":"CLIP_VIT_B32_ORGANIZER_SHADOW",
            "replacement_authority":"FROZEN_FINAL_KIS_BACKEND_CONTRACT",
        }
    return _old_branch_execution_plan(clean_req)

# ---- official Stage03 frame adapter ------------------------------------------
_FINAL_META_BY_VIDEO={
    vid:g.sort_values(["pts_time","frame_idx_candidate","frame_ref_id"]).reset_index(drop=True)
    for vid,g in VISUAL_META.groupby("video_id",sort=False)
}

def _final_same_video_nearest_frame(video_id,target_pts,after_frame=None):
    vid=str(video_id).strip()
    if vid not in _FINAL_META_BY_VIDEO:
        raise RuntimeError(("FINAL_VIDEO_NOT_IN_STAGE03",vid))
    t=float(target_pts)
    if not math.isfinite(t) or t<0:
        raise RuntimeError(("FINAL_INVALID_PTS",target_pts))
    g=_FINAL_META_BY_VIDEO[vid]
    if after_frame is not None:
        g=g[g.frame_idx_candidate.astype(np.int64)>int(after_frame)]
        if g.empty:
            raise RuntimeError(("FINAL_NO_LATER_FRAME",vid,after_frame,t))
    d=(g.pts_time.astype(float)-t).abs()
    x=g.loc[d.idxmin()]
    return {
        "video_id":vid,
        "frame_ref_id":str(x.frame_ref_id),
        "frame_idx":int(x.frame_idx_candidate),
        "stage03_pts_time":float(x.pts_time),
        "actual_pts_time":t,
        "abs_map_error_sec":abs(float(x.pts_time)-t),
    }

def _final_trake_btc(active_payload):
    if active_payload.get("task_type")!="TRAKE":
        raise RuntimeError("NOT_TRAKE")
    if active_payload.get("verifier_pass") is not True:
        raise RuntimeError("TRAKE_NOT_VERIFIED")
    if active_payload.get("final_temporal_status")!="RAW_PTS_VERIFIED":
        raise RuntimeError(("TRAKE_FINAL_TEMPORAL_STATUS",
                            active_payload.get("final_temporal_status")))
    vid=str(active_payload.get("video_id") or active_payload.get("recovered_video_id") or "").strip()
    pts=active_payload.get("actual_pts",[])
    if not isinstance(pts,list) or not (2<=len(pts)<=5):
        raise RuntimeError(("TRAKE_ACTUAL_PTS_COUNT",len(pts) if isinstance(pts,list) else None))
    mapped=[]; prev=-1
    for t in pts:
        m=_final_same_video_nearest_frame(vid,float(t),after_frame=prev)
        mapped.append(m); prev=m["frame_idx"]
    frames=[x["frame_idx"] for x in mapped]
    if not all(frames[i]<frames[i+1] for i in range(len(frames)-1)):
        raise RuntimeError(("TRAKE_BTC_FRAME_ORDER",frames))
    return [[vid,*frames]],mapped

# ---- QA agentic final-runtime helper -----------------------------------------
def _final_ev_time(x):
    fr=str(x.get("best_frame_ref_id") or "").strip()
    if fr:
        hit=VISUAL_META[VISUAL_META.frame_ref_id.astype(str).eq(fr)]
        if len(hit)==1:
            return float(hit.iloc[0].pts_time)
    for k in ("pts_time","actual_pts_time","start_pts_time_technical","end_pts_time_technical"):
        v=x.get(k)
        try:
            f=float(v)
            if math.isfinite(f):
                if k=="start_pts_time_technical":
                    try:
                        e=float(x.get("end_pts_time_technical"))
                        if math.isfinite(e): return 0.5*(f+e)
                    except Exception:
                        pass
                return f
        except Exception:
            pass
    return None

def _final_qa_evidence(result,limit=8):
    active=result.get("active_result") or {}
    rows=list(active.get("evidence_top20") or [])
    out=[]
    for i,x in enumerate(rows[:int(limit)]):
        vid=str(x.get("video_id","")).strip()
        pts=_final_ev_time(x)
        eid=str(
            x.get("evidence_pack_id")
            or x.get("interval_id")
            or x.get("candidate_id")
            or f"e{i}"
        )
        if not vid or pts is None:
            continue
        image_b64=None
        image_mime=None
        try:
            dec=decode_raw_first_at_or_after(
                vid,float(pts),
                request_id="final_qa_"+hashlib.sha256(
                    f"{vid}|{float(pts):.6f}|{i}".encode()
                ).hexdigest()[:18],
                persist_jpeg=False,
            )
            im=dec["_image"]
            buf=io.BytesIO()
            im.save(buf,format="JPEG",quality=86,optimize=True)
            image_b64=base64.b64encode(buf.getvalue()).decode("ascii")
            image_mime="image/jpeg"
            try: im.close()
            except Exception: pass
        except Exception:
            image_b64=None
            image_mime=None
        out.append({
            "evidence_index":i,
            "evidence_id":eid,
            "video_id":vid,
            "actual_pts_time":float(pts),
            "ocr_text_compact":str(x.get("ocr_text_compact",""))[:2500],
            "asr_text_compact":str(x.get("asr_text_compact",""))[:2500],
            "image_b64":image_b64,
            "image_mime":image_mime,
        })
    if not out:
        raise RuntimeError("FINAL_QA_NO_USABLE_EVIDENCE")
    if not any(x["image_b64"] for x in out):
        raise RuntimeError("FINAL_QA_NO_PREVIEW_IMAGES")
    return out



# ---- tier-2 Kaggle Qwen3-4B TEXT runtime ------------------------------------
#
# Temporary competition tier while Qwen3.5-4B is being prepared.
#
# IMPORTANT:
# - Qwen3-4B is TEXT-ONLY, not a vision-language model.
# - It is excellent for Vietnamese/English query adaptation and TRAKE event text.
# - For QA fallback it can only reason over OCR/ASR/text metadata retrieved locally.
# - It must never be described as having seen the evidence images.
#
# Exact Kaggle Model selected by the user:
# https://www.kaggle.com/models/qwen-lm/qwen-3/Transformers/4b/1

QWEN_LOCAL_MODEL_FAMILY="Qwen3-4B"
QWEN_LOCAL_VARIANT="TEXT_ONLY"
QWEN_KAGGLE_MODEL_URL=(
    "https://www.kaggle.com/models/qwen-lm/"
    "qwen-3/Transformers/4b/1"
)
QWEN_KAGGLE_CANONICAL_PATH=Path(
    "/kaggle/input/qwen-3/transformers/4b/1"
)

QWEN_LOCAL_WORKER_PORT=int(
    os.environ.get("AIC2026_QWEN_PORT","8877")
)
QWEN_LOCAL_WORKER_URL=(
    f"http://127.0.0.1:{QWEN_LOCAL_WORKER_PORT}"
)
QWEN_LOCAL_RUNTIME_DIR=Path(
    "/kaggle/working/qwen3_4b_runtime"
)
QWEN_LOCAL_RUNTIME_PKGS=(
    QWEN_LOCAL_RUNTIME_DIR/"pkgs"
)
QWEN_LOCAL_WORKER_SCRIPT=(
    QWEN_LOCAL_RUNTIME_DIR/"worker.py"
)
QWEN_LOCAL_WORKER_LOG=(
    QWEN_LOCAL_RUNTIME_DIR/"worker.log"
)
QWEN_LOCAL_RUNTIME_MARKER=(
    QWEN_LOCAL_RUNTIME_DIR/"runtime.json"
)
QWEN_LOCAL_TRANSFORMERS_VERSION="5.0.0"
QWEN_LOCAL_WORKER_PROC=globals().get(
    "QWEN_LOCAL_WORKER_PROC"
)
QWEN_LOCAL_LOCK=globals().get(
    "QWEN_LOCAL_LOCK",
    threading.RLock(),
)


def _qwen_validate_model_dir(path):
    p=Path(path)
    cfgp=p/"config.json"
    if not cfgp.is_file():
        return False,None

    try:
        cfg=json.loads(
            cfgp.read_text(encoding="utf-8")
        )
    except Exception:
        return False,None

    model_type=str(
        cfg.get("model_type","")
    ).lower()
    arch=" ".join(
        str(x)
        for x in cfg.get("architectures",[])
    ).lower()
    low=str(p).lower()

    # Official Qwen3-4B text model.
    ok=(
        model_type=="qwen3"
        and "base" not in low
        and (
            "qwen3forcausallm" in arch
            or not arch
        )
    )
    if not ok:
        return False,cfg

    has_weights=(
        any(p.glob("*.safetensors"))
        or (p/"model.safetensors.index.json").is_file()
        or (p/"pytorch_model.bin").is_file()
    )
    return bool(has_weights),cfg


def _qwen_resolve_kaggle_model_dir():
    """Resolve Qwen from known Kaggle mounts without scanning all inputs."""
    explicit=os.environ.get(
        "AIC2026_QWEN_MODEL_DIR",""
    ).strip()

    candidates=[]
    if explicit:
        candidates.append(Path(explicit))

    candidates.extend([
        Path("/kaggle/input/qwen-3/transformers/4b/1"),
        Path("/kaggle/input/qwen-3/Transformers/4b/1"),
        Path("/kaggle/input/models/qwen-lm/qwen-3/transformers/4b/1"),
        Path("/kaggle/input/models/qwen-lm/qwen-3/Transformers/4b/1"),
    ])

    seen=set()
    for path in candidates:
        try:
            path=path.resolve()
        except Exception:
            pass

        key=str(path)
        if key in seen:
            continue
        seen.add(key)

        try:
            ok,_=_qwen_validate_model_dir(path)
        except Exception:
            ok=False

        if ok:
            return path

    return None


def _qwen_model_status():
    p=_qwen_resolve_kaggle_model_dir()
    return {
        "found":p is not None,
        "state":"READY_LOCAL" if p else "MISSING_KAGGLE_INPUT",
        "source":"KAGGLE_MODEL_INPUT" if p else None,
        "model_dir":str(p) if p else None,
        "family":QWEN_LOCAL_MODEL_FAMILY,
        "variant":QWEN_LOCAL_VARIANT,
        "kaggle_model_url":QWEN_KAGGLE_MODEL_URL,
        "canonical_path":str(QWEN_KAGGLE_CANONICAL_PATH),
        "worker_url":QWEN_LOCAL_WORKER_URL,
        "main_transformers":
            transformers.__version__,
        "worker_transformers":
            QWEN_LOCAL_TRANSFORMERS_VERSION,
        "strict_dual_gpu":True,
        "vision_capable":False,
        "qa_fallback_scope":"OCR_ASR_TEXT_ONLY",
    }


def _qwen_prepare_isolated_runtime():
    """
    Qwen3 is supported by Transformers >=4.51.0.
    The final notebook already freezes Transformers 5.0.0.

    Run Qwen in a separate subprocess, but DO NOT pip-install another
    Transformers build during the competition.
    """
    if transformers.__version__!="5.0.0":
        raise RuntimeError(
            (
                "QWEN3_EXPECTS_FROZEN_TRANSFORMERS_5_0_0",
                transformers.__version__,
            )
        )

    QWEN_LOCAL_RUNTIME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    atomic_json(
        QWEN_LOCAL_RUNTIME_MARKER,
        {
            "transformers":"5.0.0",
            "source":"MAIN_ENV_IN_SEPARATE_PROCESS",
            "created_at_utc":
                pd.Timestamp.utcnow().isoformat(),
            "family":QWEN_LOCAL_MODEL_FAMILY,
        },
    )


_QWEN_WORKER_SOURCE=r"""
import os, sys, json, threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR=Path(os.environ["AIC_QWEN_MODEL_DIR"])
PORT=int(os.environ["AIC_QWEN_PORT"])

if not torch.cuda.is_available() or torch.cuda.device_count()!=2:
    raise RuntimeError(
        (
            "QWEN_EXACTLY_TWO_CUDA_GPUS_REQUIRED",
            torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0,
        )
    )

_tokenizer=None
_model=None
_load_error=None
_gen_lock=threading.RLock()


def _norm_device(x):
    if isinstance(x,int):
        return x
    s=str(x).lower()
    if s.startswith("cuda:"):
        try:
            return int(s.split(":",1)[1])
        except Exception:
            return s
    return s


def load_model():
    global _tokenizer,_model,_load_error

    if _model is not None:
        return
    if _load_error is not None:
        raise RuntimeError(_load_error)

    try:
        _tokenizer=AutoTokenizer.from_pretrained(
            str(MODEL_DIR),
            local_files_only=True,
        )

        kwargs=dict(
            local_files_only=True,
            device_map="balanced",
            low_cpu_mem_usage=True,
            # Force the 4B model to genuinely span both T4s while
            # leaving substantial VRAM headroom for retrieval.
            max_memory={
                0:"6GiB",
                1:"6GiB",
            },
        )

        try:
            _model=AutoModelForCausalLM.from_pretrained(
                str(MODEL_DIR),
                dtype=torch.float16,
                **kwargs,
            )
        except TypeError:
            _model=AutoModelForCausalLM.from_pretrained(
                str(MODEL_DIR),
                torch_dtype=torch.float16,
                **kwargs,
            )

        _model.eval()

        dmap=getattr(_model,"hf_device_map",{}) or {}
        used=set()
        bad=[]

        for d in dmap.values():
            nd=_norm_device(d)
            if nd in {0,1}:
                used.add(nd)
            else:
                bad.append(str(d))

        if bad:
            raise RuntimeError(
                (
                    "QWEN_CPU_OR_DISK_OFFLOAD_FORBIDDEN",
                    bad,
                )
            )

        if used!={0,1}:
            raise RuntimeError(
                (
                    "QWEN_MUST_USE_BOTH_GPUS",
                    sorted(used),
                )
            )

    except Exception as e:
        _load_error=(
            f"{type(e).__name__}: {e}"
        )
        raise


def generate(payload):
    load_model()

    system_prompt=str(
        payload.get("system_prompt","")
    ).strip()
    user_text=str(
        payload.get("user_text","")
    )

    max_new_tokens=max(
        32,
        min(
            700,
            int(payload.get(
                "max_new_tokens",400
            )),
        ),
    )

    messages=[]
    if system_prompt:
        messages.append({
            "role":"system",
            "content":system_prompt,
        })
    messages.append({
        "role":"user",
        "content":user_text,
    })

    rendered=_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs=_tokenizer(
        rendered,
        return_tensors="pt",
    )

    first_device=next(_model.parameters()).device
    inputs={
        k:v.to(first_device)
        for k,v in inputs.items()
    }
    input_len=int(
        inputs["input_ids"].shape[-1]
    )

    with _gen_lock, torch.inference_mode():
        out=_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    text=_tokenizer.decode(
        out[0][input_len:],
        skip_special_tokens=True,
    ).strip()

    return {
        "ok":True,
        "text":text,
        "model_dir":str(MODEL_DIR),
        "used_gpus":[0,1],
        "vision_capable":False,
    }


class H(BaseHTTPRequestHandler):
    def log_message(self,*args):
        return

    def sendj(self,code,obj):
        raw=json.dumps(
            obj,
            ensure_ascii=False,
        ).encode("utf-8")
        self.send_response(code)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        self.send_header(
            "Content-Length",
            str(len(raw)),
        )
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path=="/health":
            try:
                load_model()
                return self.sendj(
                    200,
                    {
                        "ok":True,
                        "model_dir":str(MODEL_DIR),
                        "cuda_count":
                            torch.cuda.device_count(),
                        "transformers":
                            __import__(
                                "transformers"
                            ).__version__,
                        "vision_capable":False,
                    },
                )
            except Exception as e:
                return self.sendj(
                    500,
                    {
                        "ok":False,
                        "error":
                            f"{type(e).__name__}: {e}",
                    },
                )

        self.sendj(404,{"ok":False})

    def do_POST(self):
        if self.path!="/generate":
            return self.sendj(
                404,
                {"ok":False},
            )

        try:
            n=int(self.headers.get(
                "Content-Length","0"
            ))
            body=json.loads(
                self.rfile.read(n).decode("utf-8")
            )
            return self.sendj(
                200,
                generate(body),
            )
        except Exception as e:
            return self.sendj(
                500,
                {
                    "ok":False,
                    "error":
                        f"{type(e).__name__}: {e}",
                },
            )


ThreadingHTTPServer(
    ("127.0.0.1",PORT),
    H,
).serve_forever()
"""


def _qwen_write_worker_script():
    QWEN_LOCAL_RUNTIME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    QWEN_LOCAL_WORKER_SCRIPT.write_text(
        _QWEN_WORKER_SOURCE,
        encoding="utf-8",
    )


def _qwen_worker_health(timeout=3):
    try:
        r=requests.get(
            QWEN_LOCAL_WORKER_URL+"/health",
            timeout=timeout,
        )
        if r.status_code==200:
            data=r.json()
            return bool(data.get("ok")),data

        try:
            return False,r.json()
        except Exception:
            return False,{
                "status":r.status_code
            }

    except Exception as e:
        return False,{
            "error":
                f"{type(e).__name__}: {e}"
        }


def _qwen_ensure_worker():
    global QWEN_LOCAL_WORKER_PROC

    with QWEN_LOCAL_LOCK:
        ok,info=_qwen_worker_health(
            timeout=1
        )
        if ok:
            return info

        model_dir=_qwen_resolve_kaggle_model_dir()
        if model_dir is None:
            raise RuntimeError(
                "QWEN3_4B_KAGGLE_INPUT_NOT_FOUND"
            )

        _qwen_prepare_isolated_runtime()
        _qwen_write_worker_script()

        p=QWEN_LOCAL_WORKER_PROC
        if p is not None:
            try:
                if p.poll() is None:
                    p.terminate()
                    p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass

        logf=open(
            QWEN_LOCAL_WORKER_LOG,
            "a",
            encoding="utf-8",
        )

        env=dict(os.environ)
        env.update({
            "AIC_QWEN_MODEL_DIR":
                str(model_dir),
            "AIC_QWEN_PORT":
                str(QWEN_LOCAL_WORKER_PORT),
            "CUDA_VISIBLE_DEVICES":"0,1",
            "TOKENIZERS_PARALLELISM":"false",
        })

        QWEN_LOCAL_WORKER_PROC=subprocess.Popen(
            [
                sys.executable,
                str(QWEN_LOCAL_WORKER_SCRIPT),
            ],
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
        )

        globals()["QWEN_LOCAL_WORKER_PROC"]=(
            QWEN_LOCAL_WORKER_PROC
        )

        deadline=time.time()+240
        last={}

        while time.time()<deadline:
            if QWEN_LOCAL_WORKER_PROC.poll() is not None:
                break

            ok,last=_qwen_worker_health(
                timeout=2
            )
            if ok:
                return last

            time.sleep(2)

        tail=""
        try:
            tail=QWEN_LOCAL_WORKER_LOG.read_text(
                encoding="utf-8",
                errors="ignore",
            )[-6000:]
        except Exception:
            pass

        raise RuntimeError(
            "QWEN_WORKER_START_FAIL: "
            +str(last)
            +" LOG="+tail
        )


def _qwen_local_raw(
    system_prompt,
    user_text,
    evidence=None,
    max_new_tokens=400,
):
    """
    Text-only Qwen3-4B.
    `evidence` images are intentionally ignored.
    QA receives OCR/ASR through user_obj instead.
    """
    _qwen_ensure_worker()

    r=requests.post(
        QWEN_LOCAL_WORKER_URL+"/generate",
        json={
            "system_prompt":
                str(system_prompt),
            "user_text":
                str(user_text),
            "max_new_tokens":
                int(max_new_tokens),
        },
        timeout=360,
    )

    if r.status_code>=400:
        try:
            detail=r.json()
        except Exception:
            detail=r.text[-2000:]
        raise RuntimeError(
            (
                "QWEN_LOCAL_HTTP",
                r.status_code,
                detail,
            )
        )

    data=r.json()
    if not data.get("ok"):
        raise RuntimeError(
            ("QWEN_LOCAL_FAIL",data)
        )

    text=str(
        data.get("text","")
    ).strip()

    if not text:
        raise RuntimeError(
            "QWEN_LOCAL_EMPTY_OUTPUT"
        )

    return text


def _qwen_local_json(
    system_prompt,
    user_obj,
    evidence=None,
):
    user_text=(
        user_obj
        if isinstance(user_obj,str)
        else json.dumps(
            user_obj,
            ensure_ascii=False,
        )
    )

    raw=_qwen_local_raw(
        system_prompt,
        user_text,
        evidence=None,
        max_new_tokens=500,
    )

    return _extract_json_object(raw)


class _QwenLocalCfg:
    provider="qwen_local"
    model="Qwen3-4B@Kaggle"
    kind="qwen3_text_local"
    base_url=QWEN_LOCAL_WORKER_URL
    api_key=None
    timeout_s=360


class QwenLocalClient:
    def __init__(self):
        self.cfg=_QwenLocalCfg()

    def generate_json(
        self,
        system_prompt,
        user_text,
    ):
        return _qwen_local_json(
            system_prompt,
            user_text,
            evidence=None,
        )


def _qwen_local_client():
    return QwenLocalClient()


def _qwen_adapter_output_sane(
    task,
    original_query,
    adapted,
):
    t=str(task).upper()

    bad={
        "","...","…",".","-",
        "n/a","none","null",
    }

    if t=="KIS":
        x=str(
            adapted.get(
                "search_query_en",""
            )
        ).strip()

        words=re.findall(
            r"[A-Za-z0-9]+",
            x,
        )

        if (
            x.lower() in bad
            or not (3<=len(words)<=50)
        ):
            return False,"KIS_LENGTH_OR_DEGENERATE"

        viet_chars=set(
            "ăâđêôơưĂÂĐÊÔƠƯ"
            "áàảãạấầẩẫậắằẳẵặ"
            "éèẻẽẹếềểễệ"
            "íìỉĩịóòỏõọốồổỗộớờởỡợ"
            "úùủũụứừửữựýỳỷỹỵ"
        )

        if any(
            ch in viet_chars
            for ch in x
        ):
            return False,"KIS_NOT_ENGLISH"

        src_nums=set(
            re.findall(
                r"\d+",
                str(original_query),
            )
        )
        out_nums=set(
            re.findall(r"\d+",x)
        )

        if (
            src_nums
            and not src_nums.issubset(
                out_nums
            )
        ):
            return False,"KIS_NUMERIC_ANCHOR_LOSS"

        acronyms={
            z.lower()
            for z in re.findall(
                r"\b[A-Z][A-Z0-9_-]{2,}\b",
                str(original_query),
            )
        }

        if acronyms:
            low=x.lower()
            missing=[
                z
                for z in acronyms
                if z not in low
            ]
            if missing:
                return False,(
                    "KIS_NAMED_ANCHOR_LOSS:"
                    +",".join(missing)
                )

        return True,"PASS"

    if t=="TRAKE":
        arr=adapted.get("events_en")

        if (
            not isinstance(arr,list)
            or len(arr)<2
        ):
            return False,"TRAKE_EVENT_LIST_INVALID"

        for i,x in enumerate(arr):
            z=str(x).strip()
            wc=len(
                re.findall(
                    r"[A-Za-z0-9]+",
                    z,
                )
            )

            if (
                z.lower() in bad
                or not (2<=wc<=50)
            ):
                return False,(
                    f"TRAKE_EVENT_{i}_DEGENERATE"
                )

        return True,"PASS"

    return False,"UNSUPPORTED_TASK"


QWEN_LOCAL_DISCOVERY=_qwen_model_status()


def _final_provider_multimodal_json(client,system_prompt,user_obj,evidence):
    """
    Uses the request-only provider/model/key already parsed by the online1 server.
    No secret is persisted.
    """
    cfg=client.cfg
    user_text=json.dumps(user_obj,ensure_ascii=False)
    imgs=[x for x in evidence if x.get("image_b64")]
    if str(cfg.provider).lower()=="groq":
        imgs=imgs[:3]

    if cfg.kind=="qwen3_text_local":
        return _qwen_local_json(
            system_prompt,
            user_obj,
            evidence=None,
        )

    if cfg.kind=="openai_compatible":
        content=[{"type":"text","text":user_text}]
        for x in imgs:
            content.append({
                "type":"image_url",
                "image_url":{"url":f"data:{x['image_mime']};base64,{x['image_b64']}"},
            })
        payload={
            "model":cfg.model,
            "messages":[
                {"role":"system","content":system_prompt},
                {"role":"user","content":content},
            ],
            "temperature":0,
            "max_tokens":600,
            "response_format":{"type":"json_object"},
        }
        r=requests.post(
            cfg.base_url.rstrip("/")+"/chat/completions",
            headers=client._openai_headers(),
            json=payload,
            timeout=cfg.timeout_s,
        )
        if r.status_code>=400:
            payload.pop("response_format",None)
            r=requests.post(
                cfg.base_url.rstrip("/")+"/chat/completions",
                headers=client._openai_headers(),
                json=payload,
                timeout=cfg.timeout_s,
            )
        if r.status_code>=400:
            raise RuntimeError(
                f"{cfg.provider} HTTP {r.status_code}: {_redact_secret_text(r.text)}"
            )
        body=r.json()
        raw=body["choices"][0]["message"]["content"]
        if isinstance(raw,list):
            raw="\n".join(
                str(z.get("text","")) for z in raw if isinstance(z,dict)
            )
        return _extract_json_object(raw)

    if cfg.kind=="gemini":
        parts=[{"text":user_text}]
        for x in imgs:
            parts.append({
                "inlineData":{
                    "mimeType":x["image_mime"],
                    "data":x["image_b64"],
                }
            })
        payload={
            "systemInstruction":{"parts":[{"text":system_prompt}]},
            "contents":[{"role":"user","parts":parts}],
            "generationConfig":{
                "responseMimeType":"application/json",
                "maxOutputTokens":600,
                "temperature":0,
            },
        }
        r=requests.post(
            cfg.base_url.rstrip("/")+"/models/"+cfg.model+":generateContent",
            headers={
                "Content-Type":"application/json",
                "x-goog-api-key":cfg.api_key,
            },
            json=payload,
            timeout=cfg.timeout_s,
        )
        if r.status_code>=400:
            raise RuntimeError(
                f"gemini HTTP {r.status_code}: {_redact_secret_text(r.text)}"
            )
        data=r.json()
        raw="\n".join(
            str(z.get("text",""))
            for z in data["candidates"][0]["content"]["parts"]
            if isinstance(z,dict)
        )
        return _extract_json_object(raw)

    if cfg.kind=="anthropic":
        content=[{"type":"text","text":user_text}]
        for x in imgs:
            content.append({
                "type":"image",
                "source":{
                    "type":"base64",
                    "media_type":x["image_mime"],
                    "data":x["image_b64"],
                },
            })
        payload={
            "model":cfg.model,
            "max_tokens":600,
            "temperature":0,
            "system":system_prompt,
            "messages":[{"role":"user","content":content}],
        }
        r=requests.post(
            cfg.base_url.rstrip("/")+"/messages",
            headers={
                "Content-Type":"application/json",
                "x-api-key":cfg.api_key,
                "anthropic-version":"2023-06-01",
            },
            json=payload,
            timeout=cfg.timeout_s,
        )
        if r.status_code>=400:
            raise RuntimeError(
                f"anthropic HTTP {r.status_code}: {_redact_secret_text(r.text)}"
            )
        data=r.json()
        raw="\n".join(
            str(z.get("text",""))
            for z in data.get("content",[])
            if isinstance(z,dict) and z.get("type")=="text"
        )
        return _extract_json_object(raw)

    raise RuntimeError(("FINAL_QA_UNSUPPORTED_PROVIDER_KIND",cfg.kind))

def _final_valid_evidence_ids(evidence):
    return {str(x["evidence_id"]) for x in evidence}

def _final_qa_prepare_answer(original_query,retrieval_response,client):
    """
    Phase 1: retrieve a grounded candidate answer once.

    IMPORTANT:
    - no verifier call happens here;
    - no API key is stored in the returned phase context;
    - the evidence pack is kept in server RAM for phase 2.
    """
    evidence_limit=3 if str(client.cfg.provider).lower()=="groq" else 8
    evidence=_final_qa_evidence(
        retrieval_response,
        limit=evidence_limit,
    )

    compact=[
        {
            "evidence_index":x["evidence_index"],
            "evidence_id":x["evidence_id"],
            "video_id":x["video_id"],
            "actual_pts_time":x["actual_pts_time"],
            "ocr_text_compact":x["ocr_text_compact"],
            "asr_text_compact":x["asr_text_compact"],
        }
        for x in evidence
    ]
    allowed=_final_valid_evidence_ids(evidence)

    is_qwen3_text=(
        getattr(client.cfg,"kind","")
        =="qwen3_text_local"
    )

    if is_qwen3_text:
        text_signal=[
            (
                str(x.get("ocr_text_compact","")).strip()
                +" "
                +str(x.get("asr_text_compact","")).strip()
            ).strip()
            for x in compact
        ]
        if not any(text_signal):
            raise RuntimeError(
                "QWEN3_TEXT_QA_NO_OCR_ASR_EVIDENCE"
            )

    answer_prompt=(
        "Answer ONLY from the supplied OCR/ASR/text evidence. "
        "You are TEXT-ONLY and have NOT seen any image. "
        "Return JSON only with keys answer, cited_evidence_ids, abstain. "
        "answer must be <=100 characters. Cite only supplied evidence_id values. "
        "If the text evidence is insufficient, abstain=true."
        if is_qwen3_text
        else
        "Answer ONLY from the supplied video evidence and images. "
        "Return JSON only with keys answer, cited_evidence_ids, abstain. "
        "answer must be <=100 characters. Cite only supplied evidence_id values. "
        "If insufficient, abstain=true."
    )

    answer=_final_provider_multimodal_json(
        client,
        answer_prompt,
        {
            "task_type":"QA",
            "question":original_query,
            "evidence":compact,
            "required_response_schema":{
                "answer":"string <=100 chars",
                "cited_evidence_ids":["supplied evidence_id"],
                "abstain":"boolean",
            },
        },
        evidence,
    )

    ans=str(answer.get("answer","")).strip()
    ac=[
        str(x).strip()
        for x in answer.get("cited_evidence_ids",[])
        if str(x).strip()
    ]

    if bool(answer.get("abstain",False)):
        raise RuntimeError("FINAL_QA_PROVIDER_ABSTAIN")
    if not ans or len(ans)>100:
        raise RuntimeError(("FINAL_QA_ANSWER_LENGTH",len(ans)))
    if not ac or set(ac)-allowed:
        raise RuntimeError(
            ("FINAL_QA_BAD_ANSWER_CITATIONS",ac,sorted(allowed))
        )

    return {
        "original_query":str(original_query),
        "retrieval_response":retrieval_response,
        "evidence":evidence,
        "compact_evidence":compact,
        "allowed_evidence_ids":sorted(allowed),
        "candidate_answer":ans,
        "answer_evidence_ids":ac,
        "answer_provider":client.cfg.provider,
        "answer_model":client.cfg.model,
        "provider_api_key_persisted":False,
    }


def _final_qa_verify_prepared(prepared,client):
    """
    Phase 2: verify an already-created answer.

    This can use a DIFFERENT API key from phase 1. Therefore verifier
    rate-limit/key rotation never recomputes the answer.
    """
    original_query=str(prepared["original_query"])
    evidence=prepared["evidence"]
    compact=prepared["compact_evidence"]
    allowed=set(prepared["allowed_evidence_ids"])
    ans=str(prepared["candidate_answer"])
    ac=[str(x) for x in prepared["answer_evidence_ids"]]

    is_qwen3_text=(
        getattr(client.cfg,"kind","")
        =="qwen3_text_local"
    )
    verifier_prompt=(
        "Verify the candidate answer ONLY against the supplied OCR/ASR/text evidence. "
        "You are TEXT-ONLY and have NOT seen any image. "
        "Return JSON only with keys supported, cited_evidence_ids, abstain, reason. "
        "If the text evidence cannot prove the answer, abstain=true."
        if is_qwen3_text
        else
        "Verify the candidate answer ONLY against the same supplied evidence/images. "
        "Return JSON only with keys supported, cited_evidence_ids, abstain, reason. "
        "Cite only supplied evidence_id values."
    )

    verifier=_final_provider_multimodal_json(
        client,
        verifier_prompt,
        {
            "task_type":"QA_VERIFY",
            "question":original_query,
            "candidate_answer":ans,
            "evidence":compact,
            "required_response_schema":{
                "supported":"boolean",
                "cited_evidence_ids":["supplied evidence_id"],
                "abstain":"boolean",
                "reason":"short string",
            },
        },
        evidence,
    )

    vc=[
        str(x).strip()
        for x in verifier.get("cited_evidence_ids",[])
        if str(x).strip()
    ]

    if (
        bool(verifier.get("abstain",False))
        or verifier.get("supported") is not True
    ):
        raise RuntimeError(
            ("FINAL_QA_VERIFIER_REJECT",verifier.get("reason"))
        )

    if not vc or set(vc)-allowed:
        raise RuntimeError(
            ("FINAL_QA_BAD_VERIFIER_CITATIONS",vc)
        )

    common=[x for x in ac if x in set(vc)]
    if not common:
        raise RuntimeError("FINAL_QA_CITATION_NO_OVERLAP")

    byid={str(x["evidence_id"]):x for x in evidence}
    rep=byid[common[0]]

    return {
        "answer":ans,
        "recommended_result_index":int(rep["evidence_index"]),
        "confidence":None,
        "reason":str(verifier.get("reason",""))[:500],
        "evidence_ids":ac,
        "verifier_evidence_ids":vc,
        "abstain":False,
        "grounded":True,
        "verifier_pass":True,
        "representative_evidence_id":rep["evidence_id"],
        "representative_video_id":rep["video_id"],
        "representative_actual_pts_time":float(
            rep["actual_pts_time"]
        ),
        "answer_provider":prepared.get("answer_provider"),
        "answer_model":prepared.get("answer_model"),
        "verifier_provider":client.cfg.provider,
        "verifier_model":client.cfg.model,
        "provider_api_key_persisted":False,
    }


def _final_qa_agent_verify(original_query,retrieval_response,client):
    """
    Compatibility wrapper for legacy callers.
    New manual UI uses the two phase endpoints below instead.
    """
    prepared=_final_qa_prepare_answer(
        original_query,
        retrieval_response,
        client,
    )
    return _final_qa_verify_prepared(prepared,client)


# ---- exact AIC26 BTC row generation ------------------------------------------
def _final_clean_video(v):
    s=str(v).strip()
    if s.lower().endswith(".mp4"):
        s=s[:-4]
    if not re.fullmatch(r"[A-Za-z0-9_]+",s):
        raise RuntimeError(("BTC_VIDEO_ID",v))
    return s

def _final_int_frame(v):
    x=float(v)
    if not math.isfinite(x) or x<0 or abs(x-round(x))>1e-9:
        raise RuntimeError(("BTC_FRAME_ID",v))
    return int(round(x))

def _final_btc_from_result(task,result,verification=None):
    task=str(task).upper()
    p=result.get("active_result") or {}

    if task=="KIS":
        rows=[]
        seen=set()
        for x in p.get("results",[])[:100]:
            row=[_final_clean_video(x["video_id"]),_final_int_frame(x["frame_idx_candidate"])]
            k=tuple(row)
            if k not in seen:
                seen.add(k); rows.append(row)
        if not rows:
            return {"ready":False,"reason":"KIS_NO_ROWS","rows":[]}
        return {"ready":True,"reason":"PASS","rows":rows}

    if task=="TRAKE":
        try:
            rows,mapped=_final_trake_btc(p)
            return {"ready":True,"reason":"PASS","rows":rows,"mapped_events":mapped}
        except Exception as e:
            return {"ready":False,"reason":f"TRAKE_NOT_FINAL: {e}","rows":[]}

    if task=="QA":
        v=verification or {}
        if v.get("verifier_pass") is not True or v.get("grounded") is not True:
            return {"ready":False,"reason":"QA_EXTERNAL_GROUNDED_VERIFIER_REQUIRED","rows":[]}
        ans=str(v.get("answer","")).strip()
        if not ans or len(ans)>100:
            return {"ready":False,"reason":"QA_ANSWER_INVALID","rows":[]}
        try:
            m=_final_same_video_nearest_frame(
                v["representative_video_id"],
                float(v["representative_actual_pts_time"]),
            )
            row=[_final_clean_video(m["video_id"]),_final_int_frame(m["frame_idx"]),ans]
            return {"ready":True,"reason":"PASS","rows":[row],"mapped_evidence":m}
        except Exception as e:
            return {"ready":False,"reason":f"QA_FRAME_MAP_FAIL: {e}","rows":[]}

    return {"ready":False,"reason":"UNKNOWN_TASK","rows":[]}

def _final_btc_csv_text(rows):
    sio=io.StringIO(newline="")
    csv.writer(sio,delimiter=",",lineterminator="\n",quoting=csv.QUOTE_MINIMAL).writerows(rows)
    return sio.getvalue()

def _final_validate_query_filename(name,task):
    base=Path(str(name).strip()).name
    if base.lower().endswith(".txt"):
        base=base[:-4]
    elif base.lower().endswith(".csv"):
        base=base[:-4]
    if not base:
        raise ValueError("query filename empty")
    suffix={"KIS":"-kis","QA":"-qa","TRAKE":"-trake"}[str(task).upper()]
    if not base.lower().endswith(suffix):
        raise ValueError(f"query filename must end with {suffix}")
    if not re.fullmatch(r"[A-Za-z0-9._-]+",base):
        raise ValueError("unsafe query filename")
    return base+".csv"

def _final_validate_btc_rows(task,rows):
    task=str(task).upper()
    if not isinstance(rows,list) or not (1<=len(rows)<=100):
        raise ValueError("BTC rows must contain 1..100 rows")
    out=[]
    for row in rows:
        if not isinstance(row,list):
            raise ValueError("BTC row must be list")
        if task=="KIS":
            if len(row)!=2: raise ValueError("KIS BTC row must have 2 columns")
            out.append([_final_clean_video(row[0]),_final_int_frame(row[1])])
        elif task=="QA":
            if len(row)!=3: raise ValueError("QA BTC row must have 3 columns")
            ans=str(row[2])
            if not ans or len(ans)>100: raise ValueError("QA answer empty or >100 chars")
            out.append([_final_clean_video(row[0]),_final_int_frame(row[1]),ans])
        elif task=="TRAKE":
            if len(row)<3: raise ValueError("TRAKE needs video + >=2 frames")
            frames=[_final_int_frame(x) for x in row[1:]]
            if not all(frames[i]<frames[i+1] for i in range(len(frames)-1)):
                raise ValueError("TRAKE frame IDs must be strictly increasing")
            out.append([_final_clean_video(row[0]),*frames])
        else:
            raise ValueError("unknown task")
    return out

def _final_save_btc(query_file,task,rows):
    rows=_final_validate_btc_rows(task,rows)
    filename=_final_validate_query_filename(query_file,task)
    path=BTC_SUBMISSION_ROOT/filename
    with path.open("w",encoding="utf-8",newline="") as f:
        csv.writer(f,delimiter=",",lineterminator="\n",quoting=csv.QUOTE_MINIMAL).writerows(rows)
    # Reparse strict no-header structure.
    with path.open("r",encoding="utf-8",newline="") as f:
        parsed=list(csv.reader(f))
    _final_validate_btc_rows(task,parsed)
    return {
        "file":str(path),
        "filename":filename,
        "rows":len(rows),
        "sha256":sha256_file(path),
    }

def _final_build_submission_zip():
    files=sorted(BTC_SUBMISSION_ROOT.glob("*.csv"),key=lambda p:p.name)
    if not files:
        raise RuntimeError("NO_BTC_CSV_FILES_SAVED")
    if BTC_ZIP_PATH.exists(): BTC_ZIP_PATH.unlink()
    with zipfile.ZipFile(BTC_ZIP_PATH,"w",zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p,arcname=f"submission/{p.name}")
    with zipfile.ZipFile(BTC_ZIP_PATH) as z:
        bad=z.testzip()
        if bad is not None: raise RuntimeError(("BTC_ZIP_CRC",bad))
        names=z.namelist()
        if any(not n.startswith("submission/") for n in names):
            raise RuntimeError(("BTC_ZIP_ROOT",names))
    return {
        "zip_path":str(BTC_ZIP_PATH),
        "sha256":sha256_file(BTC_ZIP_PATH),
        "file_count":len(files),
        "files":[p.name for p in files],
    }

FINAL_COMPETITION_CONTRACT={
    "build_version":BUILD_VERSION,
    "source_online1_build":LEGACY_ONLINE_BUILD_VERSION,
    "frozen_backends":FINAL_FROZEN_CONTRACTS,
    "official_submission_status":OFFICIAL_SUBMISSION_STATUS,
    "btc":{
        "encoding":"UTF-8",
        "delimiter":",",
        "header":False,
        "max_rows_per_query":100,
        "KIS":"<video_id>,<frame_id>",
        "QA":"<video_id>,<frame_id>,<answer>",
        "TRAKE":"<video_id>,<frame_1>,...,<frame_N>",
        "zip_root":"submission/",
    },
    "strict_dual_gpu":True,
    "provider_model_runtime_selected":True,
    "api_key_persisted":False,
    "groq_qa_max_images_per_request":3,
    "emergency_local_fallback":"STAGE03_SIGLIP2_FOR_KIS",
    "hybrid_fallback_order":[
        "REMOTE_API_MAGAZINE",
        "KAGGLE_QWEN3_4B_TEXT",
        "FROZEN_LOCAL_BACKEND",
    ],
    "qwen_local_model_discovery":QWEN_LOCAL_DISCOVERY,
    "qwen_hf_repo_id":globals().get("QWEN_HF_REPO_ID"),
    "qwen_hf_revision":globals().get("QWEN_HF_REVISION"),
    "qwen3_kaggle_input_only":True,
    "qwen_hf_preflight":(
        globals()["_qwen_hf_preflight"]()
        if callable(globals().get("_qwen_hf_preflight"))
        else {
            "status":"NOT_CONFIGURED",
            "source":"KAGGLE_MODEL_INPUT_ONLY",
        }
    ),
    "qwen_process_isolation":True,
    "qwen_uses_existing_transformers_5_0_0":True,
}
atomic_json(
    Path("/kaggle/working/final_competition_contract.json"),
    FINAL_COMPETITION_CONTRACT,
)

print("[PASS] FINAL frozen backends + AIC26 wire adapter",{
    "build":BUILD_VERSION,
    "KIS":"CLIP_VIT_B32_ORGANIZER_SHADOW",
    "TRAKE":"RAW_VIDEO_ACTUAL_PTS",
    "QA":"RUNTIME_PROVIDER_MODEL",
    "btc_status":OFFICIAL_SUBMISSION_STATUS,
    "clip_model_source":FINAL_CLIP_MODEL_SOURCE,
    "clip_gpu_shards":FINAL_CLIP_SHARDS,
})
