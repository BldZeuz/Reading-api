from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "Wav2Vec2 Reading Analysis API"

# Wav2Vec2 expects 16 kHz audio
SAMPLE_RATE = 16_000

# Default English Wav2Vec2 model
MODEL_ID = os.getenv(
    "WAV2VEC2_MODEL",
    "facebook/wav2vec2-base-960h"
)

# Upload limits
MAX_UPLOAD_MB = float(
    os.getenv("MAX_UPLOAD_MB", "25")
)

MAX_AUDIO_SECONDS = float(
    os.getenv("MAX_AUDIO_SECONDS", "300")
)

# Speaking speed target
TARGET_WPM_MIN = float(
    os.getenv("TARGET_WPM_MIN", "110")
)

TARGET_WPM_MAX = float(
    os.getenv("TARGET_WPM_MAX", "180")
)


# ============================================================
# GLOBAL MODEL
# ============================================================

_ASR_PIPELINE: Any | None = None

_MODEL_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

allow_origins = [
    x.strip()
    for x in os.getenv("ALLOW_ORIGINS", "*").split(",")
    if x.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False if allow_origins == ["*"] else True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def clamp(
    value: float,
    low: float = 0.0,
    high: float = 100.0
) -> float:
    return max(low, min(high, value))


def round_or_none(
    value: float | None,
    digits: int = 2
) -> float | None:

    if value is None:
        return None

    if not math.isfinite(value):
        return None

    return round(float(value), digits)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:

    text = text.lower()

    # Normalize curly apostrophe
    text = text.replace("’", "'")

    # Remove punctuation
    text = re.sub(
        r"[^a-z0-9']+",
        " ",
        text
    )

    # Normalize spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# WORD ACCURACY / WER
# ============================================================

def align_words(
    reference_text: str,
    spoken_text: str
) -> dict[str, Any]:

    reference_words = normalize_text(
        reference_text
    ).split()

    spoken_words = normalize_text(
        spoken_text
    ).split()

    n = len(reference_words)
    m = len(spoken_words)

    # Dynamic programming table
    dp = [
        [0] * (m + 1)
        for _ in range(n + 1)
    ]

    back: list[list[str | None]] = [
        [None] * (m + 1)
        for _ in range(n + 1)
    ]

    # Initial deletion costs
    for i in range(1, n + 1):

        dp[i][0] = i
        back[i][0] = "deletion"

    # Initial insertion costs
    for j in range(1, m + 1):

        dp[0][j] = j
        back[0][j] = "insertion"

    priority = {
        "correct": 0,
        "substitution": 1,
        "deletion": 2,
        "insertion": 3,
    }

    # Build alignment table
    for i in range(1, n + 1):

        for j in range(1, m + 1):

            if (
                reference_words[i - 1]
                == spoken_words[j - 1]
            ):

                candidates = [
                    (
                        dp[i - 1][j - 1],
                        "correct"
                    )
                ]

            else:

                candidates = [
                    (
                        dp[i - 1][j - 1] + 1,
                        "substitution"
                    )
                ]

            candidates.extend(
                [
                    (
                        dp[i - 1][j] + 1,
                        "deletion"
                    ),
                    (
                        dp[i][j - 1] + 1,
                        "insertion"
                    ),
                ]
            )

            best_cost, best_operation = min(
                candidates,
                key=lambda x: (
                    x[0],
                    priority[x[1]]
                )
            )

            dp[i][j] = best_cost
            back[i][j] = best_operation


    # ========================================================
    # BACKTRACK ALIGNMENT
    # ========================================================

    i = n
    j = m

    operations: list[
        dict[str, str | None]
    ] = []

    counts = {
        "correct": 0,
        "substitution": 0,
        "deletion": 0,
        "insertion": 0,
    }

    while i > 0 or j > 0:

        operation = back[i][j]

        if operation == "correct":

            operations.append(
                {
                    "reference":
                        reference_words[i - 1],

                    "spoken":
                        spoken_words[j - 1],

                    "status":
                        operation,
                }
            )

            counts[operation] += 1

            i -= 1
            j -= 1


        elif operation == "substitution":

            operations.append(
                {
                    "reference":
                        reference_words[i - 1],

                    "spoken":
                        spoken_words[j - 1],

                    "status":
                        operation,
                }
            )

            counts[operation] += 1

            i -= 1
            j -= 1


        elif operation == "deletion":

            operations.append(
                {
                    "reference":
                        reference_words[i - 1],

                    "spoken":
                        None,

                    "status":
                        operation,
                }
            )

            counts[operation] += 1

            i -= 1


        elif operation == "insertion":

            operations.append(
                {
                    "reference":
                        None,

                    "spoken":
                        spoken_words[j - 1],

                    "status":
                        operation,
                }
            )

            counts[operation] += 1

            j -= 1

        else:
            break


    operations.reverse()


    # ========================================================
    # CALCULATE WER
    # ========================================================

    errors = (
        counts["substitution"]
        + counts["deletion"]
        + counts["insertion"]
    )

    if n > 0:

        wer = errors / n

        accuracy = clamp(
            100.0 * (
                1.0 - wer
            )
        )

    else:

        wer = None
        accuracy = None


    return {

        "reference_word_count":
            n,

        "spoken_word_count":
            m,

        "correct":
            counts["correct"],

        "substitutions":
            counts["substitution"],

        "deletions":
            counts["deletion"],

        "insertions":
            counts["insertion"],

        "wer":
            round_or_none(
                wer,
                4
            ),

        "accuracy_score":
            round_or_none(
                accuracy
            ),

        "word_feedback":
            operations,
    }


# ============================================================
# GENERIC RANGE SCORER
# ============================================================

def _range_score(
    value: float,
    ideal_low: float,
    ideal_high: float,
    outer_low: float,
    outer_high: float
) -> float:

    # Perfect range
    if ideal_low <= value <= ideal_high:
        return 100.0

    # Too low
    if value < ideal_low:

        if value <= outer_low:
            return 0.0

        return (
            100.0
            * (value - outer_low)
            / (ideal_low - outer_low)
        )

    # Too high
    if value >= outer_high:
        return 0.0

    return (
        100.0
        * (outer_high - value)
        / (outer_high - ideal_high)
    )


# ============================================================
# SPEAKING SPEED
# ============================================================

def calculate_speed(
    word_count: int,
    total_seconds: float,
    active_speech_seconds: float
) -> dict[str, Any]:

    if total_seconds > 0:

        gross_wpm = (
            word_count
            * 60.0
            / total_seconds
        )

    else:

        gross_wpm = 0.0


    if active_speech_seconds > 0:

        articulation_wpm = (
            word_count
            * 60.0
            / active_speech_seconds
        )

    else:

        articulation_wpm = 0.0


    score = _range_score(
        gross_wpm,

        TARGET_WPM_MIN,
        TARGET_WPM_MAX,

        50.0,
        260.0
    )


    if gross_wpm < TARGET_WPM_MIN:

        label = "slow"

    elif gross_wpm > TARGET_WPM_MAX:

        label = "fast"

    else:

        label = "good"


    return {

        "word_count":
            int(word_count),

        "gross_wpm":
            round_or_none(
                gross_wpm
            ),

        "articulation_wpm":
            round_or_none(
                articulation_wpm
            ),

        "speed_score":
            round_or_none(
                score
            ),

        "pace_label":
            label,

        "target_wpm_range": [
            TARGET_WPM_MIN,
            TARGET_WPM_MAX
        ],
    }


# ============================================================
# PROSODY ANALYSIS
# ============================================================

def calculate_acoustics(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE
) -> dict[str, Any]:

    # Convert stereo -> mono if somehow needed
    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1
        )

    audio = np.asarray(
        audio,
        dtype=np.float32
    )


    if audio.size == 0:

        raise ValueError(
            "Audio contains no samples."
        )


    total_seconds = (
        len(audio)
        / float(sr)
    )


    if total_seconds <= 0.05:

        raise ValueError(
            "Audio is too short to analyze."
        )


    # ========================================================
    # SILENCE CHECK
    # ========================================================

    peak = float(
        np.max(
            np.abs(audio)
        )
    )

    if peak < 1e-5:

        raise ValueError(
            "Audio is silent or nearly silent."
        )


    # ========================================================
    # SPEECH / SILENCE SEGMENTATION
    # ========================================================

    intervals = librosa.effects.split(

        audio,

        top_db=35,

        # Smaller window helps detect normal speech pauses
        frame_length=1024,

        hop_length=256
    )


    active_samples = int(
        sum(
            int(end) - int(start)
            for start, end in intervals
        )
    )


    active_seconds = (
        active_samples
        / float(sr)
    )


    # ========================================================
    # PAUSES
    # ========================================================

    pause_durations: list[float] = []


    if len(intervals) > 1:

        for k in range(
            len(intervals) - 1
        ):

            gap = (
                int(
                    intervals[k + 1][0]
                )

                - int(
                    intervals[k][1]
                )
            ) / float(sr)


            # Ignore extremely tiny gaps
            if gap >= 0.15:

                pause_durations.append(
                    gap
                )


    internal_pause_seconds = float(
        sum(
            pause_durations
        )
    )


    if total_seconds:

        pause_ratio = (
            internal_pause_seconds
            / total_seconds
        )

    else:

        pause_ratio = 0.0


    # ========================================================
    # ENERGY / LOUDNESS
    # ========================================================

    rms = librosa.feature.rms(

        y=audio,

        frame_length=2048,

        hop_length=512

    )[0]


    if rms.size:

        threshold = max(

            float(
                np.max(rms)
            ) * 0.08,

            1e-6
        )

        voiced_rms = rms[
            rms >= threshold
        ]

    else:

        voiced_rms = np.array(
            [],
            dtype=np.float32
        )


    if (
        voiced_rms.size >= 2

        and float(
            np.mean(
                voiced_rms
            )
        ) > 0
    ):

        energy_cv = float(

            np.std(
                voiced_rms
            )

            / np.mean(
                voiced_rms
            )
        )


        rms_db = librosa.amplitude_to_db(

            voiced_rms,

            ref=np.max
        )


        energy_db_std = float(

            np.std(
                rms_db
            )
        )

    else:

        energy_cv = 0.0
        energy_db_std = 0.0


    # ========================================================
    # PITCH / INTONATION
    # ========================================================

    try:

        f0, voiced_flag, _ = librosa.pyin(

            audio,

            # Human speech range
            fmin=65.0,

            fmax=min(
                700.0,
                sr / 2.0 - 1.0
            ),

            sr=sr,

            frame_length=2048,

            hop_length=512,
        )


        if f0 is not None:

            valid_f0 = f0[
                np.isfinite(f0)
            ]

        else:

            valid_f0 = np.array(
                [],
                dtype=np.float32
            )


        if (
            valid_f0.size >= 3

            and float(
                np.mean(
                    valid_f0
                )
            ) > 0
        ):

            pitch_mean = float(
                np.mean(
                    valid_f0
                )
            )

            pitch_std = float(
                np.std(
                    valid_f0
                )
            )

            pitch_cv = (
                pitch_std
                / pitch_mean
            )

            pitch_range = float(

                np.percentile(
                    valid_f0,
                    90
                )

                - np.percentile(
                    valid_f0,
                    10
                )
            )


        else:

            pitch_mean = 0.0
            pitch_std = 0.0
            pitch_cv = 0.0
            pitch_range = 0.0


        if (
            voiced_flag is not None

            and len(
                voiced_flag
            )
        ):

            voiced_ratio = float(
                np.mean(
                    voiced_flag
                )
            )

        else:

            voiced_ratio = 0.0


    except Exception:

        # Still return pause/energy values if
        # pitch extraction fails.

        pitch_mean = 0.0
        pitch_std = 0.0
        pitch_cv = 0.0
        pitch_range = 0.0
        voiced_ratio = 0.0


    # ========================================================
    # PROSODY SCORING
    # ========================================================

    pitch_score = _range_score(

        pitch_cv,

        ideal_low=0.08,
        ideal_high=0.30,

        outer_low=0.01,
        outer_high=0.60
    )


    energy_score = _range_score(

        energy_cv,

        ideal_low=0.12,
        ideal_high=0.50,

        outer_low=0.01,
        outer_high=1.00
    )


    pause_score = _range_score(

        pause_ratio,

        ideal_low=0.04,
        ideal_high=0.22,

        outer_low=0.0,
        outer_high=0.55
    )


    # Weighted prosody score
    prosody_score = (

        0.45
        * pitch_score

        + 0.30
        * energy_score

        + 0.25
        * pause_score
    )


    return {

        "duration_seconds":
            round_or_none(
                total_seconds
            ),

        "active_speech_seconds":
            round_or_none(
                active_seconds
            ),

        "internal_pause_seconds":
            round_or_none(
                internal_pause_seconds
            ),

        "pause_count":
            len(
                pause_durations
            ),

        "pause_ratio":
            round_or_none(
                pause_ratio,
                4
            ),

        "pitch_mean_hz":
            round_or_none(
                pitch_mean
            ),

        "pitch_std_hz":
            round_or_none(
                pitch_std
            ),

        "pitch_range_hz_p10_p90":
            round_or_none(
                pitch_range
            ),

        "pitch_variation_cv":
            round_or_none(
                pitch_cv,
                4
            ),

        "voiced_ratio":
            round_or_none(
                voiced_ratio,
                4
            ),

        "energy_variation_cv":
            round_or_none(
                energy_cv,
                4
            ),

        "energy_db_std":
            round_or_none(
                energy_db_std
            ),

        "prosody_score":
            round_or_none(
                clamp(
                    prosody_score
                )
            ),

        "prosody_components": {

            "pitch_variation_score":
                round_or_none(
                    pitch_score
                ),

            "energy_variation_score":
                round_or_none(
                    energy_score
                ),

            "pause_score":
                round_or_none(
                    pause_score
                ),
        },
    }


# ============================================================
# FFMPEG
# ============================================================

def ensure_ffmpeg() -> None:

    if shutil.which("ffmpeg") is None:

        raise RuntimeError(
            "FFmpeg was not found. "
            "Install FFmpeg or use the Dockerfile."
        )


def convert_to_wav(
    input_path: str | Path,
    output_path: str | Path
) -> None:

    """
    Convert virtually any common audio format to:

    WAV
    PCM 16-bit
    Mono
    16 kHz
    """

    ensure_ffmpeg()


    command = [

        "ffmpeg",

        "-hide_banner",

        "-loglevel",
        "error",

        "-y",

        "-i",
        str(input_path),

        # Ignore video if MP4/WebM contains it
        "-vn",

        # Mono
        "-ac",
        "1",

        # Wav2Vec2 sample rate
        "-ar",
        str(SAMPLE_RATE),

        # PCM WAV
        "-c:a",
        "pcm_s16le",

        str(output_path),
    ]


    result = subprocess.run(

        command,

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE,

        text=True,

        timeout=90
    )


    if result.returncode != 0:

        message = (

            result.stderr

            or

            "FFmpeg could not decode the uploaded file."
        ).strip()


        raise ValueError(
            message[-1200:]
        )


# ============================================================
# LOAD WAV
# ============================================================

def load_normalized_wav(
    wav_path: str | Path
) -> np.ndarray:

    audio, sr = sf.read(

        str(wav_path),

        dtype="float32",

        always_2d=False
    )


    if sr != SAMPLE_RATE:

        raise ValueError(

            f"Expected {SAMPLE_RATE} Hz after conversion, "
            f"got {sr} Hz."
        )


    # Stereo -> mono
    if np.ndim(audio) > 1:

        audio = np.mean(
            audio,
            axis=1
        ).astype(
            np.float32
        )


    return np.asarray(
        audio,
        dtype=np.float32
    )


# ============================================================
# WAV2VEC2 MODEL
# ============================================================

def get_asr_pipeline() -> Any:

    global _ASR_PIPELINE


    # Already loaded
    if _ASR_PIPELINE is not None:

        return _ASR_PIPELINE


    # Prevent multiple simultaneous model loads
    with _MODEL_LOCK:

        if _ASR_PIPELINE is not None:

            return _ASR_PIPELINE


        try:

            import torch

            from transformers import pipeline

        except ImportError as exc:

            raise RuntimeError(

                "Missing Wav2Vec2 dependencies. "
                "Install requirements.txt first."

            ) from exc


        # GPU automatically used if available
        device = (
            0
            if torch.cuda.is_available()
            else -1
        )


        _ASR_PIPELINE = pipeline(

            task=
                "automatic-speech-recognition",

            model=
                MODEL_ID,

            device=
                device,

            # Helps longer audio recordings
            chunk_length_s=
                20,

            stride_length_s=
                (4, 2),
        )


        return _ASR_PIPELINE


# ============================================================
# TRANSCRIPTION
# ============================================================

def transcribe(
    audio: np.ndarray
) -> dict[str, Any]:

    asr = get_asr_pipeline()


    # Avoid overlapping inference calls when CPU constrained
    with _INFERENCE_LOCK:

        result = asr(

            {
                "raw":
                    audio,

                "sampling_rate":
                    SAMPLE_RATE
            },

            return_timestamps="word",
        )


    text = str(
        result.get(
            "text",
            ""
        )
    ).strip()


    chunks = []


    for chunk in (
        result.get(
            "chunks",
            []
        )

        or []
    ):

        timestamp = (
            chunk.get(
                "timestamp"
            )

            or (
                None,
                None
            )
        )


        chunks.append(

            {
                "word":
                    str(
                        chunk.get(
                            "text",
                            ""
                        )
                    ).strip(),

                "start":
                    round_or_none(
                        timestamp[0]
                    ),

                "end":
                    round_or_none(
                        timestamp[1]
                    ),
            }
        )


    return {

        "text":
            text,

        "words":
            chunks
    }


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def analyze_wav(
    wav_path: str | Path,
    reference_text: str | None = None
) -> dict[str, Any]:

    # Load converted WAV
    audio = load_normalized_wav(
        wav_path
    )


    duration = (
        len(audio)
        / float(SAMPLE_RATE)
    )


    # Prevent very long requests
    if duration > MAX_AUDIO_SECONDS:

        raise ValueError(

            f"Audio is {duration:.1f}s; "
            f"maximum allowed is "
            f"{MAX_AUDIO_SECONDS:.0f}s."
        )


    # ========================================================
    # PROSODY
    # ========================================================

    acoustics = calculate_acoustics(

        audio,

        SAMPLE_RATE
    )


    # ========================================================
    # WAV2VEC2 TRANSCRIPTION
    # ========================================================

    transcript = transcribe(
        audio
    )


    spoken_text = transcript[
        "text"
    ]


    recognized_word_count = len(

        normalize_text(
            spoken_text
        ).split()
    )


    # ========================================================
    # SPEED
    # ========================================================

    speed = calculate_speed(

        recognized_word_count,

        float(
            acoustics[
                "duration_seconds"
            ]

            or duration
        ),

        float(
            acoustics[
                "active_speech_seconds"
            ]

            or duration
        ),
    )


    # ========================================================
    # ACCURACY
    # ========================================================

    accuracy = None


    if (
        reference_text

        and normalize_text(
            reference_text
        )
    ):

        accuracy = align_words(

            reference_text,

            spoken_text
        )


    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "model":
            MODEL_ID,

        "transcript":
            spoken_text,

        "word_timestamps":
            transcript[
                "words"
            ],

        "accuracy":
            accuracy,

        "speed":
            speed,

        "prosody":
            acoustics,

        "notes": {

            "accuracy":

                "ASR-based reading accuracy using WER "
                "against reference_text. "
                "It is not yet phoneme-level "
                "pronunciation scoring.",

            "prosody":

                "Prosody is a heuristic based on "
                "pitch variation, energy variation "
                "and pauses."
        },
    }


# ============================================================
# UPLOAD HANDLER
# ============================================================

async def save_upload_limited(
    upload: UploadFile,
    destination: Path
) -> int:

    max_bytes = int(
        MAX_UPLOAD_MB
        * 1024
        * 1024
    )


    written = 0


    with destination.open(
        "wb"
    ) as output_file:

        while True:

            chunk = await upload.read(
                1024 * 1024
            )


            if not chunk:
                break


            written += len(
                chunk
            )


            if written > max_bytes:

                raise HTTPException(

                    status_code=413,

                    detail=(
                        f"File exceeds "
                        f"{MAX_UPLOAD_MB:g} MB limit."
                    )
                )


            output_file.write(
                chunk
            )


    if written == 0:

        raise HTTPException(

            status_code=400,

            detail=
                "Uploaded file is empty."
        )


    return written


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root() -> dict[str, Any]:

    return {

        "name":
            APP_NAME,

        "status":
            "ok",

        "model":
            MODEL_ID,

        "endpoint":
            "POST /analyze",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health() -> dict[str, Any]:

    return {

        "status":
            "ok",

        "ffmpeg":
            shutil.which(
                "ffmpeg"
            ) is not None,

        "model_loaded":
            _ASR_PIPELINE is not None,
    }


# ============================================================
# ANALYZE ENDPOINT
# ============================================================

@app.post("/analyze")
async def analyze_endpoint(

    file: UploadFile = File(
        ...,
        description=(
            "WAV, MP3, M4A, AAC, FLAC, "
            "OGG, OPUS, WebM, MP4, etc."
        )
    ),

    reference_text: str | None = Form(
        default=None
    ),

) -> dict[str, Any]:

    filename = (
        file.filename
        or "upload.bin"
    )


    suffix = (
        Path(
            filename
        ).suffix[:12]

        or ".bin"
    )


    try:

        with tempfile.TemporaryDirectory(
            prefix="reading-audio-"
        ) as temp_directory:

            temp_directory = Path(
                temp_directory
            )


            # Original upload
            original_path = (

                temp_directory

                / f"input{suffix}"
            )


            # Normalized WAV
            wav_path = (

                temp_directory

                / "normalized.wav"
            )


            # Save upload
            await save_upload_limited(

                file,

                original_path
            )


            # Convert audio using FFmpeg
            await run_in_threadpool(

                convert_to_wav,

                original_path,

                wav_path
            )


            # Run Wav2Vec2 + scoring
            result = await run_in_threadpool(

                analyze_wav,

                wav_path,

                reference_text
            )


            result["input"] = {

                "filename":
                    filename,

                "content_type":
                    file.content_type,

                "normalized_format":
                    "wav/pcm_s16le/mono/16000Hz",
            }


            return result


    except HTTPException:

        raise


    except subprocess.TimeoutExpired as exc:

        raise HTTPException(

            status_code=422,

            detail=
                "Audio conversion timed out."

        ) from exc


    except ValueError as exc:

        raise HTTPException(

            status_code=422,

            detail=str(
                exc
            )

        ) from exc


    except RuntimeError as exc:

        raise HTTPException(

            status_code=503,

            detail=str(
                exc
            )

        ) from exc


    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(

                "Analysis failed: "

                f"{type(exc).__name__}: "

                f"{exc}"
            )

        ) from exc


    finally:

        await file.close()