# ADR: Use `opencv-python-headless` for Webcam Entropy Capture

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-05 |
| **Decision** | Adopt `opencv-python-headless` as a required core dependency for the webcam entropy source. |
| **Scope** | New capture backend (`low_level/capture.py`); does not alter the existing RAW-file path. |

## Context

`photorand` is a True Random Number Generator that extracts physical entropy from CMOS
sensor noise. The core pipeline is source-agnostic and dependency-injected:

```
ingest  →  sample  →  estimate (NIST SP 800-90B)  →  hash (SHA3-512)  →  expand (ChaCha20)
```

`low_level/generate.py::generate_with_assessment()` accepts pluggable `ingest_fn`,
`sample_fn`, and `hash_fn` callables. A webcam is therefore **a new ingest function**
that returns the same `np.ndarray` the sampler already consumes — the estimator,
conditioner, and expander all work unchanged.

Choosing a capture library reduces to one question: *can it deliver uncompressed sensor
noise into the pipeline?* Everything else is secondary.

### The make-or-break constraint: uncompressed pixel data

A cryptographic hash is a whitener — it distributes entropy but cannot create it. A TRNG
must therefore feed on **real sensor noise**, not on artifacts of lossy compression.

Commodity UVC webcams (laptop cameras, USB webcams) expose only two pixel formats:

- **`YUYV`** — uncompressed, post-demosaic YUV 4:2:2 (8-bit).
- **`MJPEG`** — lossy, DCT-quantized.

**MJPEG destroys the fine sensor noise** that `photorand` relies on; the LSBs of a JPEG
frame encode quantization residuals, not quantum fluctuations. The capture library
**must let us force an uncompressed (`YUYV`) format**, or we are harvesting compression
artifacts rather than entropy.

> **Raw Bayer is not an option here.** True sensor Bayer data is essentially never exposed
> by commodity UVC webcams — it requires machine-vision cameras (Basler/FLIR), Raspberry Pi
> cameras (via `libcamera`/`v4l2`), or custom drivers. The `picamera` documentation
> explicitly notes that "raw" YUV/RGB formats are post-GPU, *not* sensor Bayer. The
> realistic best case on commodity hardware is therefore **uncompressed YUYV**, and this
> decision targets that.

### The Python 3.14 constraint

`photorand` pins `requires-python = ">=3.14"`. Python 3.14 is recent, and any library
with a compiled C extension must ship a compatible wheel (or a forward-compatible
`abi3` wheel), or it will fail to install without a from-source build. This is a hard,
non-negotiable filter that eliminated candidates that would otherwise look reasonable.

## Decision

We will use **`opencv-python-headless`** as a required core dependency:

```toml
[project]
dependencies = [
    "rawpy>=0.26.1",
    "numpy>=2.0.0",
    "cryptography>=46.0.0",
    "opencv-python-headless>=5.0.0",
]
```

The capture code will force an uncompressed format and use frame differencing to cancel
static scene content and fixed-pattern noise, leaving the temporal noise floor. Rather
than differencing a single pair of frames, it accumulates the absolute difference between
each consecutive frame pair over the full capture duration — more frames yield more noise
data, and the resulting accumulation is a 2-D array the sampler consumes directly. The
existing NIST SP 800-90B `estimate_entropy()` then gates the quality exactly as it does
for RAW files.

## Rationale

`opencv-python-headless` wins on every criterion that matters for this project:

1. **NumPy-native return.** `VideoCapture.read()` returns an `np.ndarray` directly — the
   exact type `sample_entropy_grid` and `reduce_fixed_pattern_noise` already operate on.
   Every other candidate requires a conversion step.
2. **Cross-platform.** It is the only candidate with real device control that also honors
   the project's `Operating System :: OS Independent` classifier (Linux, macOS, Windows).
3. **Forward-compatible `abi3` wheels** — see Verified Evidence below. This is the most
   Python-version-future-proof option available.
4. **Forces uncompressed format** via `cap.set(cv2.CAP_PROP_FOURCC, fourcc("YUYV"))`.
5. **License:** Apache-2.0, fully compatible with the project's MIT license.
6. **`headless` variant** avoids pulling in GUI/Qt dependencies that a CLI/server TRNG does
   not need.

### The one weakness — and why it is acceptable

`CAP_PROP_FOURCC` is backend-dependent and can be silently ignored on some platforms
(notably Windows `CAP_DSHOW`). In isolation this would be a serious defect for a TRNG.
It is acceptable here because **the project's own machinery is the safety net**: if
OpenCV quietly hands back MJPEG frames, `estimate_entropy()` reports a `LOW`/`FAIL`
status and the startup health checks (Repetition Count, Adaptive Proportion) fail. The
system refuses to emit weak entropy. A weak source cannot pass undetected.

## Verified Evidence

All facts below were gathered **first-hand** against this repository's interpreter
(Python 3.14.2) — both from the PyPI JSON API (exact wheel tags) and from real
`uv run --with <pkg>` install + import smoke tests. This was necessary because the
obvious assumptions were wrong in ways that mattered.

### Install + import on Python 3.14.2 (smoke tests)

| Library | Result | Wheel mechanism |
|---|---|---|
| **opencv-python-headless** | ✅ `import cv2` → 5.0.0 | `cp37-abi3` (forward-compatible to all Python ≥ 3.7) |
| PyAV (`av`) | ✅ `import av` → 18.0.0 | regular `cp314` wheels **only in 15.1.0–16.1.0**; 17.x/18.x ship only free-threaded `cp314t` |
| pygame-ce | ✅ `import pygame` → 2.5.7 | regular `cp314` wheels |
| v4l2py | ✅ → 3.0.0, **but deprecated** | pure-Python `py3-none-any` |

Two near-misses that justified the verification rather than trusting metadata:

- **OpenCV** initially appeared to have "zero cp314 wheels" — until the actual filenames
  were inspected (`opencv_python-5.0.0.93-cp37-abi3-*.whl`). The stable ABI is
  forward-compatible *by design*, so it installs cleanly on 3.14 with no `cp314`-specific
  wheel at all.
- **PyAV** initially appeared clean on 3.14 — but its newest releases (17.x, 18.x) ship
  **only free-threaded (`cp314t`) wheels**; a regular (GIL) Python 3.14 install is only
  guaranteed by pinning to 15.1.0–16.1.0.

### Feature comparison

| Criterion | OpenCV (headless) | PyAV | pygame-ce | linuxpy.video |
|---|---|---|---|---|
| Force uncompressed YUYV | ✅ | ✅ | ⚠️ limited | ✅ direct ioctl |
| Raw Bayer (capable HW) | ❌ | ❌ | ❌ | ✅ (Linux) |
| Cross-platform | ✅ | ✅ | ✅ | ❌ Linux only |
| Python 3.14 wheels | ✅ `abi3` | ⚠️ ≤16.1.0 only | ✅ `cp314` | ✅ pure-Python |
| Returns `np.ndarray` | ✅ directly | ⚠️ `to_ndarray()` | ❌ Surface | ❌ bytes |
| Maintenance | Massive | Active | Active | Maintained |
| License | Apache-2 | BSD | LGPL | MIT/BSD |

## Alternatives Considered

- **PyAV (`av`)** — strong FFmpeg bindings and good raw-frame read performance, but its
  webcam input-device handling is less portable for forcing raw `YUYV` across platforms,
  its wheel situation on regular Python 3.14 is unsettled (free-threaded-only in current
  releases), and it requires a `VideoFrame → ndarray` conversion that OpenCV avoids.
  Rejected.
- **pygame-ce** — does ship real `cp314` wheels and has a camera module, but camera
  support is a side feature with limited format control, and it returns `Surface` objects
  that must be converted to arrays. Rejected.
- **vidgear** — a high-level wrapper *on top of* OpenCV/FFmpeg. This project needs *more*
  low-level control over the pixel format, not an abstraction that hides it. It adds a
  dependency stack and provides nothing OpenCV does not. Rejected.
- **v4l2py** — would have been a strong Linux choice, but it now emits a deprecation
  warning directing users to `linuxpy.video`. Rejected (see Tier-2 below).
- **imageio-ffmpeg** — file/stream oriented, not designed for live device capture.
  Rejected.
- **GStreamer (PyGObject)** — powerful (`bayer2rgb` caps), but heavy, has install pain on
  Windows/macOS, and PyGObject wheels chronically lag new Python versions. Overkill.

## Consequences

- **Positive:** A single, well-understood, cross-platform dependency that slots natively
  into the existing pipeline with minimal glue. No compiled-wheel risk on Python 3.14 or
  future versions thanks to `abi3`. The existing NIST estimator provides a hard safety net
  against silently-compressed input.
- **Negative:** Reliance on OpenCV's backend-dependent `CAP_PROP_FOURCC`, which must be
  verified per platform. Mitigated by entropy estimation gating.
- **Tier-2 option (not part of this decision):** For a hardened Linux-only entropy rig or
  embedded appliance where byte-exact pixel-format control (or raw Bayer on capable
  hardware) is required, `linuxpy.video` — the maintained successor to `v4l2py` — is the
  recommended power-user backend. It is pure-Python (no ABI constraints) and speaks V4L2
  ioctl directly. It can be added behind a `--backend linuxpy` flag later without
  disturbing the OpenCV default.

## References

- `low_level/generate.py` — dependency-injected pipeline (`ingest_fn`, `sample_fn`, `hash_fn`).
- `low_level/entropy.py` — NIST SP 800-90B estimation and startup health checks.
- `low_level/sample.py` — `sample_entropy_grid`, `reduce_fixed_pattern_noise`.
- `docs/entropy-extraction.md` — sensor-noise sources and FPN reduction rationale.
- PyPI JSON API and `uv run --with` import smoke tests against Python 3.14.2 (2026-08-05).