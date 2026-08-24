from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "Vosk Reading Analysis API"

SAMPLE_RATE = 16_000

MODEL_PATH = Path(
    os.getenv(
        "VOSK_MODEL_PATH",
        "/opt/vosk-model"
    )
)

MAX_UPLOAD_MB = float(
    os.getenv(
        "MAX_UPLOAD_MB",
        "15"
    )
)

MAX_AUDIO_SECONDS = float(
    os.getenv(
        "MAX_AUDIO_SECONDS",
        "60"
    )
)

TARGET_WPM_MIN = float(
    os.getenv(
        "TARGET_WPM_MIN",
        "110"
    )
)

TARGET_WPM_MAX = float(
    os.getenv(
        "TARGET_WPM_MAX",
        "180"
    )
)

PRELOAD_MODEL = (
    os.getenv(
        "PRELOAD_MODEL",
        "1"
    ).lower()
    not in {
        "0",
        "false",
        "no"
    }
)


# ============================================================
# GLOBAL MODEL
# ============================================================

_VOSK_MODEL: Any | None = None
_MODEL_LOCK = threading.Lock()


# ============================================================
# GENERAL HELPERS
# ============================================================

def clamp(
    value: float,
    low: float = 0.0,
    high: float = 100.0
) -> float:

    return max(
        low,
        min(
            high,
            value
        )
    )


def round_or_none(
    value: float | None,
    digits: int = 2
) -> float | None:

    if value is None:
        return None

    if not math.isfinite(
        float(value)
    ):
        return None

    return round(
        float(value),
        digits
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(
    text: str
) -> str:

    text = (
        text
        .lower()
        .replace(
            "’",
            "'"
        )
    )

    text = re.sub(
        r"[^a-z0-9']+",
        " ",
        text
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


# ============================================================
# WORD ALIGNMENT / ACCURACY
# ============================================================

def align_words(
    reference_text: str,
    spoken_text: str
) -> dict[str, Any]:

    reference_words = (
        normalize_text(
            reference_text
        ).split()
    )

    spoken_words = (
        normalize_text(
            spoken_text
        ).split()
    )

    n = len(
        reference_words
    )

    m = len(
        spoken_words
    )


    dp = [
        [0] * (m + 1)
        for _ in range(n + 1)
    ]

    back: list[
        list[str | None]
    ] = [
        [None] * (m + 1)
        for _ in range(n + 1)
    ]


    for i in range(
        1,
        n + 1
    ):

        dp[i][0] = i

        back[i][0] = (
            "deletion"
        )


    for j in range(
        1,
        m + 1
    ):

        dp[0][j] = j

        back[0][j] = (
            "insertion"
        )


    priority = {

        "correct": 0,

        "substitution": 1,

        "deletion": 2,

        "insertion": 3,

    }


    # ========================================================
    # BUILD EDIT DISTANCE TABLE
    # ========================================================

    for i in range(
        1,
        n + 1
    ):

        for j in range(
            1,
            m + 1
        ):

            candidates: list[
                tuple[int, str]
            ] = []


            if (
                reference_words[i - 1]
                ==
                spoken_words[j - 1]
            ):

                candidates.append(
                    (
                        dp[i - 1][j - 1],
                        "correct"
                    )
                )

            else:

                candidates.append(
                    (
                        dp[i - 1][j - 1] + 1,
                        "substitution"
                    )
                )


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


            cost, operation = min(

                candidates,

                key=lambda x: (
                    x[0],
                    priority[x[1]]
                )
            )


            dp[i][j] = cost

            back[i][j] = operation


    # ========================================================
    # BACKTRACK
    # ========================================================

    i = n
    j = m


    counts = {

        "correct": 0,

        "substitution": 0,

        "deletion": 0,

        "insertion": 0,

    }


    operations: list[
        dict[
            str,
            str | None
        ]
    ] = []


    while (
        i > 0
        or
        j > 0
    ):

        operation = (
            back[i][j]
        )


        if operation == "correct":

            operations.append(
                {
                    "reference":
                        reference_words[
                            i - 1
                        ],

                    "spoken":
                        spoken_words[
                            j - 1
                        ],

                    "status":
                        operation,
                }
            )

            counts[
                operation
            ] += 1

            i -= 1
            j -= 1


        elif (
            operation
            ==
            "substitution"
        ):

            operations.append(
                {
                    "reference":
                        reference_words[
                            i - 1
                        ],

                    "spoken":
                        spoken_words[
                            j - 1
                        ],

                    "status":
                        operation,
                }
            )

            counts[
                operation
            ] += 1

            i -= 1
            j -= 1


        elif (
            operation
            ==
            "deletion"
        ):

            operations.append(
                {
                    "reference":
                        reference_words[
                            i - 1
                        ],

                    "spoken":
                        None,

                    "status":
                        operation,
                }
            )

            counts[
                operation
            ] += 1

            i -= 1


        elif (
            operation
            ==
            "insertion"
        ):

            operations.append(
                {
                    "reference":
                        None,

                    "spoken":
                        spoken_words[
                            j - 1
                        ],

                    "status":
                        operation,
                }
            )

            counts[
                operation
            ] += 1

            j -= 1

        else:

            break


    operations.reverse()


    errors = (
        counts["substitution"]
        +
        counts["deletion"]
        +
        counts["insertion"]
    )


    if n:

        wer = (
            errors
            /
            n
        )

        accuracy = clamp(
            100.0
            *
            (
                1.0
                -
                wer
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
            counts[
                "correct"
            ],

        "substitutions":
            counts[
                "substitution"
            ],

        "deletions":
            counts[
                "deletion"
            ],

        "insertions":
            counts[
                "insertion"
            ],

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
# GENERIC SCORE
# ============================================================

def _range_score(
    value: float,
    ideal_low: float,
    ideal_high: float,
    outer_low: float,
    outer_high: float
) -> float:

    if (
        ideal_low
        <=
        value
        <=
        ideal_high
    ):

        return 100.0


    if value < ideal_low:

        if value <= outer_low:
            return 0.0

        return (
            100.0
            *
            (
                value
                -
                outer_low
            )
            /
            max(
                ideal_low
                -
                outer_low,
                1e-9
            )
        )


    if value >= outer_high:

        return 0.0


    return (
        100.0
        *
        (
            outer_high
            -
            value
        )
        /
        max(
            outer_high
            -
            ideal_high,
            1e-9
        )
    )


# ============================================================
# SPEED
# ============================================================

def calculate_speed(
    word_count: int,
    total_seconds: float,
    active_speech_seconds: float
) -> dict[str, Any]:

    if total_seconds > 0:

        gross_wpm = (
            word_count
            *
            60.0
            /
            total_seconds
        )

    else:

        gross_wpm = 0.0


    if (
        active_speech_seconds
        >
        0
    ):

        articulation_wpm = (
            word_count
            *
            60.0
            /
            active_speech_seconds
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


    if (
        gross_wpm
        <
        TARGET_WPM_MIN
    ):

        label = "slow"


    elif (
        gross_wpm
        >
        TARGET_WPM_MAX
    ):

        label = "fast"


    else:

        label = "good"


    return {

        "word_count":
            int(
                word_count
            ),

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
# FFMPEG
# ============================================================

def ensure_ffmpeg() -> None:

    if (
        shutil.which(
            "ffmpeg"
        )
        is None
    ):

        raise RuntimeError(

            "FFmpeg was not found. "
            "Use the included Dockerfile "
            "or install FFmpeg."

        )


def convert_to_wav(
    input_path: str | Path,
    output_path: str | Path
) -> None:

    ensure_ffmpeg()


    command = [

        "ffmpeg",

        "-hide_banner",

        "-loglevel",
        "error",

        "-y",

        "-i",
        str(
            input_path
        ),

        # Ignore video streams
        "-vn",

        # Mono
        "-ac",
        "1",

        # 16 kHz
        "-ar",
        str(
            SAMPLE_RATE
        ),

        # 16-bit PCM WAV
        "-c:a",
        "pcm_s16le",

        str(
            output_path
        ),
    ]


    result = subprocess.run(

        command,

        stdout=
            subprocess.PIPE,

        stderr=
            subprocess.PIPE,

        text=True,

        timeout=45

    )


    if (
        result.returncode
        !=
        0
    ):

        message = (

            result.stderr

            or

            "FFmpeg could not decode "
            "the uploaded audio."

        ).strip()


        raise ValueError(
            message[-1200:]
        )


# ============================================================
# WAV READER
# ============================================================

def load_pcm16_wav(
    wav_path: str | Path
) -> tuple[
    np.ndarray,
    bytes,
    float
]:

    with wave.open(
        str(
            wav_path
        ),
        "rb"
    ) as wf:

        if (
            wf.getnchannels()
            !=
            1

            or

            wf.getsampwidth()
            !=
            2

            or

            wf.getcomptype()
            !=
            "NONE"
        ):

            raise ValueError(

                "Normalized WAV must be "
                "mono 16-bit PCM."

            )


        if (
            wf.getframerate()
            !=
            SAMPLE_RATE
        ):

            raise ValueError(

                f"Expected "
                f"{SAMPLE_RATE} Hz WAV, "
                f"got "
                f"{wf.getframerate()} Hz."

            )


        frames = wf.readframes(
            wf.getnframes()
        )


    samples_i16 = np.frombuffer(

        frames,

        dtype="<i2"

    )


    audio = (

        samples_i16
        .astype(
            np.float32
        )

        /

        32768.0

    )


    duration = (

        len(
            audio
        )

        /

        float(
            SAMPLE_RATE
        )

    )


    if duration <= 0.05:

        raise ValueError(
            "Audio is too short to analyze."
        )


    if (
        duration
        >
        MAX_AUDIO_SECONDS
    ):

        raise ValueError(

            f"Audio is "
            f"{duration:.1f}s; "
            f"maximum allowed is "
            f"{MAX_AUDIO_SECONDS:.0f}s."

        )


    if (

        audio.size == 0

        or

        float(
            np.max(
                np.abs(
                    audio
                )
            )
        )
        <
        1e-5

    ):

        raise ValueError(

            "Audio is silent "
            "or nearly silent."

        )


    return (
        audio,
        frames,
        duration
    )


# ============================================================
# FAST RMS
# ============================================================

def _frame_rms(
    audio: np.ndarray,
    frame_length: int = 320,
    hop_length: int = 160
) -> tuple[
    np.ndarray,
    np.ndarray
]:

    if (
        len(audio)
        <
        frame_length
    ):

        padded = np.pad(

            audio,

            (
                0,
                frame_length
                -
                len(audio)
            )

        )


        return (

            np.array(
                [
                    float(
                        np.sqrt(
                            np.mean(
                                padded
                                *
                                padded
                            )
                            +
                            1e-12
                        )
                    )
                ]
            ),

            np.array(
                [0],
                dtype=np.int64
            )

        )


    starts = np.arange(

        0,

        len(audio)
        -
        frame_length
        +
        1,

        hop_length,

        dtype=np.int64

    )


    rms = np.empty(

        len(
            starts
        ),

        dtype=np.float32

    )


    for (
        idx,
        start
    ) in enumerate(
        starts
    ):

        frame = audio[

            start
            :
            start
            +
            frame_length

        ]


        rms[idx] = float(

            np.sqrt(

                np.mean(
                    frame
                    *
                    frame
                )

                +

                1e-12

            )

        )


    return (
        rms,
        starts
    )


# ============================================================
# LIGHTWEIGHT PITCH ESTIMATOR
# ============================================================

def _estimate_pitch_hz(
    frame: np.ndarray,
    sr: int = SAMPLE_RATE
) -> float | None:

    # Downsample from 16kHz -> 8kHz.
    # This makes autocorrelation much cheaper.

    x = np.asarray(

        frame[::2],

        dtype=np.float32

    )


    ds_sr = (
        sr
        //
        2
    )


    if len(x) < 160:

        return None


    # Remove DC offset

    x = (
        x
        -
        float(
            np.mean(
                x
            )
        )
    )


    peak = float(

        np.max(
            np.abs(
                x
            )
        )

    )


    if peak < 1e-4:

        return None


    # Window signal

    x *= (
        np.hanning(
            len(
                x
            )
        )
        .astype(
            np.float32
        )
    )


    # Autocorrelation

    corr = np.correlate(

        x,

        x,

        mode="full"

    )[

        len(x)
        -
        1
        :

    ]


    if (

        corr.size == 0

        or

        corr[0] <= 1e-9

    ):

        return None


    min_hz = 65.0

    max_hz = 400.0


    min_lag = max(

        1,

        int(
            ds_sr
            /
            max_hz
        )

    )


    max_lag = min(

        len(corr)
        -
        1,

        int(
            ds_sr
            /
            min_hz
        )

    )


    if (
        max_lag
        <=
        min_lag
    ):

        return None


    segment = corr[

        min_lag
        :
        max_lag
        +
        1

    ]


    lag = (

        min_lag

        +

        int(
            np.argmax(
                segment
            )
        )

    )


    confidence = float(

        corr[
            lag
        ]

        /

        max(
            corr[0],
            1e-9
        )

    )


    if confidence < 0.28:

        return None


    return float(

        ds_sr
        /
        lag

    )


# ============================================================
# LIGHTWEIGHT PROSODY
# ============================================================

def calculate_acoustics(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE
) -> dict[str, Any]:

    total_seconds = (

        len(
            audio
        )

        /

        float(
            sr
        )

    )


    rms, starts = (
        _frame_rms(
            audio
        )
    )


    max_rms = (

        float(
            np.max(
                rms
            )
        )

        if rms.size

        else 0.0

    )


    if max_rms <= 1e-7:

        raise ValueError(

            "Audio is silent "
            "or nearly silent."

        )


    # About -24dB relative to
    # loudest short-time RMS frame.

    threshold = max(

        max_rms
        *
        0.063,

        0.002

    )


    speech_mask = (
        rms
        >=
        threshold
    )


    if not np.any(
        speech_mask
    ):

        raise ValueError(

            "No speech-like audio "
            "was detected."

        )


    hop_seconds = (

        160.0
        /
        sr

    )


    first_speech = int(

        np.argmax(
            speech_mask
        )

    )


    last_speech = (

        len(
            speech_mask
        )

        -
        1

        -

        int(
            np.argmax(
                speech_mask[::-1]
            )
        )

    )


    active_speech_seconds = float(

        np.sum(
            speech_mask
        )

        *
        hop_seconds

    )


    # ========================================================
    # PAUSES
    # ========================================================

    pause_count = 0

    pause_seconds = 0.0


    min_pause_frames = max(

        1,

        int(
            round(
                0.15
                /
                hop_seconds
            )
        )

    )


    run = 0


    for flag in speech_mask[

        first_speech
        :
        last_speech
        +
        1

    ]:

        if not flag:

            run += 1


        else:

            if (
                run
                >=
                min_pause_frames
            ):

                pause_count += 1

                pause_seconds += (

                    run
                    *
                    hop_seconds

                )


            run = 0


    if (
        run
        >=
        min_pause_frames
    ):

        pause_count += 1

        pause_seconds += (

            run
            *
            hop_seconds

        )


    pause_ratio = (

        pause_seconds
        /
        total_seconds

        if total_seconds > 0

        else 0.0

    )


    # ========================================================
    # ENERGY VARIATION
    # ========================================================

    voiced_rms = (
        rms[
            speech_mask
        ]
    )


    energy_mean = (

        float(
            np.mean(
                voiced_rms
            )
        )

        if voiced_rms.size

        else 0.0

    )


    energy_cv = (

        float(
            np.std(
                voiced_rms
            )
            /
            energy_mean
        )

        if energy_mean > 1e-9

        else 0.0

    )


    # ========================================================
    # PITCH
    # ========================================================

    pitch_frame_length = int(

        0.04
        *
        sr

    )


    candidate_starts = (

        starts[
            speech_mask
        ]

    )


    candidate_starts = (

        candidate_starts[

            candidate_starts
            +
            pitch_frame_length
            <=
            len(
                audio
            )

        ]

    )


    # Prevent long audio from
    # creating excessive work.

    if (
        len(
            candidate_starts
        )
        >
        120
    ):

        indices = np.linspace(

            0,

            len(
                candidate_starts
            )
            -
            1,

            120,

            dtype=np.int64

        )


        candidate_starts = (

            candidate_starts[
                indices
            ]

        )


    pitches: list[
        float
    ] = []


    for start in candidate_starts:

        pitch = _estimate_pitch_hz(

            audio[

                int(
                    start
                )
                :
                int(
                    start
                )
                +
                pitch_frame_length

            ],

            sr

        )


        if (

            pitch is not None

            and

            math.isfinite(
                pitch
            )

        ):

            pitches.append(
                pitch
            )


    if pitches:

        pitch_values = np.asarray(

            pitches,

            dtype=np.float32

        )


        pitch_mean = float(

            np.mean(
                pitch_values
            )

        )


        pitch_std = float(

            np.std(
                pitch_values
            )

        )


        pitch_cv = (

            pitch_std
            /
            max(
                pitch_mean,
                1e-9
            )

        )


        pitch_range = float(

            np.percentile(
                pitch_values,
                90
            )

            -

            np.percentile(
                pitch_values,
                10
            )

        )


        voiced_ratio = (

            len(
                pitches
            )

            /

            max(
                len(
                    candidate_starts
                ),
                1
            )

        )


    else:

        pitch_mean = 0.0

        pitch_std = 0.0

        pitch_cv = 0.0

        pitch_range = 0.0

        voiced_ratio = 0.0


    # ========================================================
    # PROSODY SCORE
    # ========================================================

    pitch_score = _range_score(

        pitch_cv,

        0.05,

        0.30,

        0.0,

        0.60

    )


    energy_score = _range_score(

        energy_cv,

        0.10,

        0.55,

        0.0,

        1.10

    )


    # No pause is okay for a
    # short reading sentence.

    pause_score = _range_score(

        pause_ratio,

        0.0,

        0.22,

        0.0,

        0.55

    )


    prosody_score = (

        0.45
        *
        pitch_score

        +

        0.30
        *
        energy_score

        +

        0.25
        *
        pause_score

    )


    return {

        "duration_seconds":
            round_or_none(
                total_seconds
            ),

        "active_speech_seconds":
            round_or_none(
                active_speech_seconds
            ),

        "internal_pause_seconds":
            round_or_none(
                pause_seconds
            ),

        "pause_count":
            pause_count,

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
# VOSK MODEL
# ============================================================

def get_vosk_model() -> Any:

    global _VOSK_MODEL


    if (
        _VOSK_MODEL
        is not None
    ):

        return (
            _VOSK_MODEL
        )


    with _MODEL_LOCK:

        if (
            _VOSK_MODEL
            is not None
        ):

            return (
                _VOSK_MODEL
            )


        try:

            from vosk import (
                Model,
                SetLogLevel,
            )

        except ImportError as exc:

            raise RuntimeError(

                "Vosk is not installed. "
                "Install requirements.txt "
                "or use the Dockerfile."

            ) from exc


        if not MODEL_PATH.exists():

            raise RuntimeError(

                f"Vosk model not found "
                f"at {MODEL_PATH}. "
                f"The included Dockerfile "
                f"downloads it automatically."

            )


        SetLogLevel(
            -1
        )


        started = (
            time.perf_counter()
        )


        _VOSK_MODEL = Model(

            str(
                MODEL_PATH
            )

        )


        elapsed = (

            time.perf_counter()
            -
            started

        )


        print(

            f"[STARTUP] "
            f"Vosk model loaded "
            f"in {elapsed:.2f}s "
            f"from {MODEL_PATH}",

            flush=True

        )


        return (
            _VOSK_MODEL
        )


# ============================================================
# OPTIONAL REFERENCE GRAMMAR
# ============================================================

def _build_reference_grammar(
    reference_text: str
) -> str | None:

    words = (
        normalize_text(
            reference_text
        ).split()
    )


    if not words:

        return None


    # Remove duplicates while
    # preserving order.

    unique_words = list(

        dict.fromkeys(
            words
        )

    )


    # Allow unknown words so the
    # recognizer is not completely
    # forced to output reference words.

    unique_words.append(
        "[unk]"
    )


    return json.dumps(
        unique_words
    )


# ============================================================
# VOSK TRANSCRIPTION
# ============================================================

def transcribe_vosk(
    pcm_bytes: bytes,
    reference_text: str | None = None,
    constrain_vocabulary: bool = False
) -> dict[str, Any]:

    try:

        from vosk import (
            KaldiRecognizer,
        )

    except ImportError as exc:

        raise RuntimeError(

            "Vosk is not installed."

        ) from exc


    model = (
        get_vosk_model()
    )


    grammar = (

        _build_reference_grammar(
            reference_text
            or
            ""
        )

        if constrain_vocabulary

        else None

    )


    if grammar:

        recognizer = (
            KaldiRecognizer(

                model,

                SAMPLE_RATE,

                grammar

            )
        )


    else:

        recognizer = (
            KaldiRecognizer(

                model,

                SAMPLE_RATE

            )
        )


    recognizer.SetWords(
        True
    )


    all_words: list[
        dict[str, Any]
    ] = []


    text_parts: list[
        str
    ] = []


    # 8000 bytes:
    # 0.25 seconds of 16kHz PCM16.

    chunk_bytes = 8000


    for offset in range(

        0,

        len(
            pcm_bytes
        ),

        chunk_bytes

    ):

        chunk = pcm_bytes[

            offset
            :
            offset
            +
            chunk_bytes

        ]


        if (
            recognizer
            .AcceptWaveform(
                chunk
            )
        ):

            payload = json.loads(

                recognizer.Result()
                or
                "{}"

            )


            if payload.get(
                "text"
            ):

                text_parts.append(

                    str(
                        payload[
                            "text"
                        ]
                    )

                )


            if isinstance(

                payload.get(
                    "result"
                ),

                list

            ):

                all_words.extend(

                    payload[
                        "result"
                    ]

                )


    # Final segment

    payload = json.loads(

        recognizer.FinalResult()
        or
        "{}"

    )


    if payload.get(
        "text"
    ):

        text_parts.append(

            str(
                payload[
                    "text"
                ]
            )

        )


    if isinstance(

        payload.get(
            "result"
        ),

        list

    ):

        all_words.extend(

            payload[
                "result"
            ]

        )


    # ========================================================
    # NORMALIZE WORD TIMESTAMPS
    # ========================================================

    normalized_words = []


    for item in all_words:

        normalized_words.append(
            {
                "word":
                    str(
                        item.get(
                            "word",
                            ""
                        )
                    ).strip(),

                "start":
                    round_or_none(
                        item.get(
                            "start"
                        )
                    ),

                "end":
                    round_or_none(
                        item.get(
                            "end"
                        )
                    ),

                "confidence":
                    round_or_none(
                        item.get(
                            "conf"
                        ),
                        4
                    ),
            }
        )


    transcript = " ".join(

        part.strip()

        for part in text_parts

        if part.strip()

    ).strip()


    if (
        not transcript
        and
        normalized_words
    ):

        transcript = " ".join(

            item["word"]

            for item
            in normalized_words

            if item[
                "word"
            ]

        )


    return {

        "text":
            transcript,

        "words":
            normalized_words,

        "constrained_vocabulary":
            bool(
                grammar
            ),
    }


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def analyze_wav(
    wav_path: str | Path,
    reference_text: str | None = None,
    constrain_vocabulary: bool = False
) -> dict[str, Any]:

    total_start = (
        time.perf_counter()
    )


    # ========================================================
    # LOAD
    # ========================================================

    load_start = (
        time.perf_counter()
    )


    (
        audio,
        pcm_bytes,
        duration
    ) = load_pcm16_wav(
        wav_path
    )


    load_ms = (

        time.perf_counter()
        -
        load_start

    ) * 1000


    # ========================================================
    # PROSODY
    # ========================================================

    prosody_start = (
        time.perf_counter()
    )


    acoustics = calculate_acoustics(

        audio,

        SAMPLE_RATE

    )


    prosody_ms = (

        time.perf_counter()
        -
        prosody_start

    ) * 1000


    # ========================================================
    # VOSK
    # ========================================================

    asr_start = (
        time.perf_counter()
    )


    transcript = transcribe_vosk(

        pcm_bytes,

        reference_text,

        constrain_vocabulary

    )


    asr_ms = (

        time.perf_counter()
        -
        asr_start

    ) * 1000


    spoken_text = (
        transcript[
            "text"
        ]
    )


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

            or

            duration

        ),

        float(

            acoustics[
                "active_speech_seconds"
            ]

            or

            duration

        ),

    )


    # ========================================================
    # ACCURACY
    # ========================================================

    accuracy = None


    if (

        reference_text

        and

        normalize_text(
            reference_text
        )

    ):

        accuracy = align_words(

            reference_text,

            spoken_text

        )


    total_ms = (

        time.perf_counter()
        -
        total_start

    ) * 1000


    print(

        f"[TIMING] "
        f"audio={duration:.2f}s "
        f"load={load_ms:.0f}ms "
        f"prosody={prosody_ms:.0f}ms "
        f"vosk={asr_ms:.0f}ms "
        f"total={total_ms:.0f}ms",

        flush=True

    )


    return {

        "model":
            "vosk-model-small-en-us-0.15",

        "engine":
            "vosk",

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

        "timing": {

            "audio_duration_seconds":
                round(
                    duration,
                    2
                ),

            "audio_load_ms":
                round(
                    load_ms
                ),

            "prosody_ms":
                round(
                    prosody_ms
                ),

            "asr_ms":
                round(
                    asr_ms
                ),

            "total_analysis_ms":
                round(
                    total_ms
                ),
        },

        "notes": {

            "accuracy":

                "Word-level reading accuracy "
                "based on Vosk transcription "
                "vs reference_text; not "
                "phoneme-level pronunciation "
                "scoring.",

            "prosody":

                "Lightweight heuristic from "
                "pitch variation, energy "
                "variation and pauses; "
                "no Librosa/PyTorch model "
                "is used.",

            "constrained_vocabulary":

                transcript[
                    "constrained_vocabulary"
                ],
        },
    }


# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI
):

    if PRELOAD_MODEL:

        await run_in_threadpool(
            get_vosk_model
        )

    yield


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(

    title=
        APP_NAME,

    version=
        "2.0.0",

    lifespan=
        lifespan

)


# ============================================================
# CORS
# ============================================================

allow_origins = [

    x.strip()

    for x in os.getenv(
        "ALLOW_ORIGINS",
        "*"
    ).split(",")

    if x.strip()

]


app.add_middleware(

    CORSMiddleware,

    allow_origins=
        allow_origins,

    allow_credentials=
        False
        if allow_origins == ["*"]
        else True,

    allow_methods=[
        "GET",
        "POST"
    ],

    allow_headers=[
        "*"
    ],

)


# ============================================================
# UPLOAD
# ============================================================

async def save_upload_limited(
    upload: UploadFile,
    destination: Path
) -> int:

    max_bytes = int(

        MAX_UPLOAD_MB
        *
        1024
        *
        1024

    )


    written = 0


    with destination.open(
        "wb"
    ) as output_file:

        while True:

            chunk = await upload.read(
                1024
                *
                1024
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
                        f"{MAX_UPLOAD_MB:g} "
                        f"MB limit."

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

        "engine":
            "vosk",

        "model":
            "vosk-model-small-en-us-0.15",

        "endpoint":
            "POST /analyze",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health() -> dict[str, Any]:

    return {

        "status":
            "ok",

        "ffmpeg":
            shutil.which(
                "ffmpeg"
            )
            is not None,

        "model_path_exists":
            MODEL_PATH.exists(),

        "model_loaded":
            _VOSK_MODEL
            is not None,
    }


# ============================================================
# ANALYZE
# ============================================================

@app.post("/analyze")
async def analyze_endpoint(

    file: UploadFile = File(

        ...,

        description=(
            "WAV, MP3, M4A, AAC, "
            "FLAC, OGG, OPUS, "
            "WebM, MP4, etc."
        )
    ),

    reference_text: str | None = Form(
        default=None
    ),

    constrain_vocabulary: bool = Form(

        default=False,

        description=(
            "Optional. Restrict Vosk "
            "to expected words. "
            "Can improve recognition, "
            "but may inflate accuracy."
        )

    ),

) -> dict[str, Any]:

    request_start = (
        time.perf_counter()
    )


    filename = (

        file.filename

        or

        "upload.bin"

    )


    suffix = (

        Path(
            filename
        ).suffix[:12]

        or

        ".bin"

    )


    try:

        with tempfile.TemporaryDirectory(

            prefix=
                "reading-audio-"

        ) as temp_directory:


            temp_directory = Path(
                temp_directory
            )


            original_path = (

                temp_directory

                /

                f"input{suffix}"

            )


            wav_path = (

                temp_directory

                /

                "normalized.wav"

            )


            # =================================================
            # SAVE UPLOAD
            # =================================================

            await save_upload_limited(

                file,

                original_path

            )


            # =================================================
            # FFMPEG
            # =================================================

            conversion_start = (
                time.perf_counter()
            )


            await run_in_threadpool(

                convert_to_wav,

                original_path,

                wav_path

            )


            conversion_ms = (

                time.perf_counter()
                -
                conversion_start

            ) * 1000


            # =================================================
            # ANALYZE
            # =================================================

            result = await run_in_threadpool(

                analyze_wav,

                wav_path,

                reference_text,

                constrain_vocabulary

            )


            request_ms = (

                time.perf_counter()
                -
                request_start

            ) * 1000


            # Add conversion timing

            result[
                "timing"
            ][
                "audio_conversion_ms"
            ] = round(
                conversion_ms
            )


            result[
                "timing"
            ][
                "total_server_ms"
            ] = round(
                request_ms
            )


            # Input info

            result[
                "input"
            ] = {

                "filename":
                    filename,

                "content_type":
                    file.content_type,

                "normalized_format":
                    "wav/pcm_s16le/"
                    "mono/16000Hz",
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

            detail=
                str(
                    exc
                )

        ) from exc


    except RuntimeError as exc:

        raise HTTPException(

            status_code=503,

            detail=
                str(
                    exc
                )

        ) from exc


    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(

                f"Analysis failed: "
                f"{type(exc).__name__}: "
                f"{exc}"

            )

        ) from exc


    finally:

        await file.close()