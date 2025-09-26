# applio_api_server.py
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

# Applio's core helpers (동일 컨테이너에서 import)
from core import run_infer_script

app = FastAPI(title="Applio VC API", version="0.1.0")

# 경로 기본값 (compose에서 마운트한다고 가정)
SHARED_DIR = Path(os.getenv("SHARED_DIR", "/workspace/shared_data"))
VOICE_DIR  = Path(os.getenv("VOICE_DIR",  "/workspace/voice_model"))
LOGS_DIR   = Path(os.getenv("LOGS_DIR",   "/app/Applio/logs"))

INPUT_AUDIO_DIR = SHARED_DIR / "input_audio"
WARMUP_DIR = SHARED_DIR / "warmup"
APPLIO_OUT_DIR = SHARED_DIR / "applio_output_queue"

SHARED_DIR.mkdir(parents=True, exist_ok=True)
(LOGS_DIR / "api").mkdir(parents=True, exist_ok=True)
WARMUP_DIR.mkdir(parents=True, exist_ok=True)
INPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
APPLIO_OUT_DIR.mkdir(parents=True, exist_ok=True)

class HealthResp(BaseModel):
    status: str = "ok"

def _pick_latest(directory: Path, patterns: List[str]) -> Optional[Path]:
    cand = []
    for pat in patterns:
        cand.extend(directory.glob(pat))
    if not cand:
        return None
    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0]

@app.get("/health", response_model=HealthResp)
def health():
    return HealthResp()

@app.post("/warmup")
def warmup(
    pth_path: Optional[str] = Form(None),
    index_path: Optional[str] = Form(None),
    embedder_model: str = Form("korean-hubert-base"),
):
    """
    모델/커널 워밍업: 사용자가 준비한 warmup_in.wav를 변환하여 예열
    - 입력 고정: {WARMUP_DIR}/warmup_in.wav (없으면 400 반환)
    - 출력 고정: {WARMUP_DIR}/warmup_out.wav (덮어쓰기)
    """
    # 기본 모델 경로(환경변수나 마운트된 디렉토리 기준)
    pth = Path(pth_path) if pth_path else (VOICE_DIR / "garen"/ "garen.pth")
    idx = Path(index_path) if index_path else (VOICE_DIR / "garen"/ "garen.index")
    if not pth.exists() or not idx.exists():
        raise HTTPException(status_code=400, detail=f"Model/index not found: {pth} / {idx}")

    # 입력/출력 경로
    in_path  = WARMUP_DIR / "warmup_in.wav"
    out_path = WARMUP_DIR / "warmup_out.wav"

    # 입력 파일 필수 존재
    if not in_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Warmup input not found: {in_path}. "
                   f"한 단어짜리 짧은 wav를 이 경로에 미리 배치해 주세요."
        )

    # (선택) 간단한 형식 검증: 길이/샘플레이트 확인
    try:
        import soundfile as sf
        info = sf.info(str(in_path))
        # 너무 긴 파일 방지: 5초 초과 시 경고 (필요 시 조건 완화/삭제)
        if info.duration and info.duration > 5.0:
            raise HTTPException(status_code=400, detail=f"Warmup wav too long ({info.duration:.2f}s). 5초 이하 권장.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or unreadable wav: {e}")

    # 실제 변환 실행
    try:
        msg, final_out = run_infer_script(
            pitch=0,
            index_rate=0.3,
            volume_envelope=1.0,
            protect=0.33,
            f0_method="rmvpe",
            input_path=str(in_path),
            output_path=str(out_path),
            pth_path=str(pth),
            index_path=str(idx),
            split_audio=False,
            f0_autotune=False,
            f0_autotune_strength=1.0,
            proposed_pitch=False,
            proposed_pitch_threshold=155.0,
            clean_audio=False,
            clean_strength=0.7,
            export_format="WAV",
            embedder_model=embedder_model,
            embedder_model_custom=None,
            # 효과 비활성
            formant_shifting=False,
            formant_qfrency=1.0,
            formant_timbre=1.0,
            post_process=False,
            reverb=False,
            pitch_shift=False,
            limiter=False,
            gain=False,
            distortion=False,
            chorus=False,
            bitcrush=False,
            clipping=False,
            compressor=False,
            delay=False,
            sid=0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {e}")

    return JSONResponse({
        "status": "ok",
        "message": "warmed with user-provided wav",
        "in": str(in_path),
        "out": final_out,
    })

@app.post("/infer")
async def infer(
    # 업로드 또는 경로 둘 중 하나 제공
    input_wav: Optional[UploadFile] = File(None),
    input_wav_path: Optional[str] = Form(None),

    # 출력 파일명(없으면 자동)
    output_basename: Optional[str] = Form(None),

    # 모델 경로
    pth_path: str = Form(...),
    index_path: str = Form(...),

    # 변환 파라미터(필요한 만큼 노출)
    embedder_model: str = Form("korean-hubert-base"),
    pitch: int = Form(0),
    index_rate: float = Form(0.3),
    protect: float = Form(0.33),
    f0_method: str = Form("rmvpe"),
    split_audio: bool = Form(False),
    clean_audio: bool = Form(False),
    export_format: str = Form("WAV"),
):
    """
    업로드 wav 또는 경로 기반 wav → RVC 변환
    """
    # 입력 확보 (우선순위: 업로드 > 경로지정 > input_audio 최신 자동선택)
    if input_wav is not None:
        in_name = f"vc_in_{int(time.time())}.wav"
        in_path = INPUT_AUDIO_DIR / in_name
        with open(in_path, "wb") as f:
            f.write(await input_wav.read())
    elif input_wav_path:
        in_path = Path(input_wav_path)
        if not in_path.exists():
            raise HTTPException(status_code=400, detail=f"input_wav_path not found: {in_path}")
    else:
        # 자동선택: input_audio에서 가장 최근 파일(wav/mp3 등) 선택
        auto = _pick_latest(INPUT_AUDIO_DIR, ["*.wav", "*.mp3", "*.m4a", "*.flac"])
        if not auto:
            raise HTTPException(status_code=400, detail="No audio found to use. Upload a file or place one under input_audio/.")
        in_path = auto

    # 출력 경로
    bn = output_basename or f"vc_out_{int(time.time())}"
    out_path = APPLIO_OUT_DIR / f"{bn}.wav"

    # 모델 경로 확인
    pth = Path(pth_path)
    idx = Path(index_path)
    if not pth.exists() or not idx.exists():
        raise HTTPException(status_code=400, detail=f"Model/index not found: {pth} / {idx}")

    try:
        msg, final_out = run_infer_script(
            pitch=pitch,
            index_rate=index_rate,
            volume_envelope=1.0,
            protect=protect,
            f0_method=f0_method,
            input_path=str(in_path),
            output_path=str(out_path),
            pth_path=str(pth),
            index_path=str(idx),
            split_audio=split_audio,
            f0_autotune=False,
            f0_autotune_strength=1.0,
            proposed_pitch=False,
            proposed_pitch_threshold=155.0,
            clean_audio=clean_audio,
            clean_strength=0.7,
            export_format=export_format,
            embedder_model=embedder_model,
            embedder_model_custom=None,
            # 효과 비활성 기본
            formant_shifting=False,
            formant_qfrency=1.0,
            formant_timbre=1.0,
            post_process=False,
            reverb=False,
            pitch_shift=False,
            limiter=False,
            gain=False,
            distortion=False,
            chorus=False,
            bitcrush=False,
            clipping=False,
            compressor=False,
            delay=False,
            sid=0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return JSONResponse({
        "status": "ok",
        "message": msg,
        "input": str(in_path),
        "output": final_out,
        "params": {
            "embedder_model": embedder_model,
            "pitch": pitch,
            "index_rate": index_rate,
            "protect": protect,
            "f0_method": f0_method,
            "export_format": export_format,
        }
    })
