r"""
all_camera_recording_tests.py

One-shot “dump everything” script to diagnose why OpenCV/LeRobot recordings are
short/sped-up (i.e., youre capturing fewer frames than expected, then encoding at 30fps).

It runs:
  A) System info (Python/OpenCV)
  B) OpenCV camera index scan (DSHOW + MSMF)
  C) OpenCV timed FPS tests across common modes (DSHOW + MSMF)
     - baseline (no sets)
     - 640x480 @ 30 and @ 15 with MJPG + YUY2 requests
     - 320x240 @ 30
     - 1280x720 @ 30
  D) (If ffmpeg exists) ffmpeg DirectShow device list + mode list for your camera name
  E) (If ffmpeg exists) ffmpeg capture-throughput tests for 8s at 640x480@30 using:
     - mjpeg
     - yuyv422

How to run (PowerShell):
  python all_camera_recording_tests.py

Optional args:
  python all_camera_recording_tests.py --index 1
  python all_camera_recording_tests.py --camera_name "PC Camera"
  python all_camera_recording_tests.py --camera_input "@device_pnp_\\?\usb#vid_....\global"
  python all_camera_recording_tests.py --duration 8 --max_index 6

Notes:
- If ffmpeg shows multiple devices with the same display name ("PC Camera"), use --camera_input with the
  Alternative name printed by ffmpeg (the @device_pnp... string). That uniquely identifies the device.
- This script prints a lot; copy/paste the whole output back and Ill tell you exactly whats wrong.

"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is not installed. Install with: pip install opencv-python")
    raise


# -------------------------
# helpers
# -------------------------
def header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def run_cmd(cmd: List[str], timeout_s: int = 60) -> Tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        out = (p.stdout or "") + (p.stderr or "")
        return p.returncode, out
    except FileNotFoundError:
        return 127, "Command not found"
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        return 124, f"TIMEOUT after {timeout_s}s\n{out}"


def fourcc_to_str(fourcc_val: float) -> str:
    try:
        v = int(fourcc_val)
    except Exception:
        return "????"
    chars = [chr((v >> 8 * i) & 0xFF) for i in range(4)]
    return "".join(chars)


def str_to_fourcc(code: str) -> int:
    code = (code or "").strip().upper()
    if len(code) != 4:
        raise ValueError(f"FOURCC must be 4 chars, got '{code}'")
    return cv2.VideoWriter_fourcc(*code)


def backend_id(name: str) -> int:
    name = name.upper()
    if name == "DSHOW":
        return cv2.CAP_DSHOW
    if name == "MSMF":
        return cv2.CAP_MSMF
    if name == "ANY":
        return 0
    raise ValueError(f"Unknown backend {name}")


# -------------------------
# OpenCV tests
# -------------------------
@dataclass
class CaptureResult:
    ok: bool
    frames: int
    duration_s: float
    measured_fps: float
    reported_fps: float
    width: float
    height: float
    fourcc_actual: str
    notes: str


def timed_capture(
    index: int,
    backend: int,
    duration_s: float,
    warmup_s: float,
    width: Optional[int] = None,
    height: Optional[int] = None,
    target_fps: Optional[float] = None,
    fourcc: Optional[str] = None,
) -> CaptureResult:
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return CaptureResult(False, 0, duration_s, 0.0, 0.0, 0.0, 0.0, "????", "could_not_open")

    notes = []

    # Setting FOURCC early often matters.
    if fourcc:
        try:
            cap.set(cv2.CAP_PROP_FOURCC, str_to_fourcc(fourcc))
            notes.append(f"set_fourcc={fourcc}")
        except Exception as e:
            notes.append(f"set_fourcc_failed={e}")

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        notes.append(f"set_width={width}")
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        notes.append(f"set_height={height}")
    if target_fps is not None:
        cap.set(cv2.CAP_PROP_FPS, float(target_fps))
        notes.append(f"set_fps={target_fps}")

    # Warmup
    t0 = time.time()
    while time.time() - t0 < warmup_s:
        cap.read()

    # Timed loop
    frames = 0
    t0 = time.time()
    while time.time() - t0 < duration_s:
        ok, _ = cap.read()
        if ok:
            frames += 1

    measured_fps = frames / max(duration_s, 1e-9)
    reported_fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fcc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))

    cap.release()

    return CaptureResult(
        True,
        frames,
        duration_s,
        measured_fps,
        reported_fps,
        w,
        h,
        fcc,
        ";".join(notes) if notes else "",
    )


def print_capture(label: str, r: CaptureResult) -> None:
    if not r.ok:
        print(f"{label:40s}  ERROR  notes={r.notes}")
        return
    print(
        f"{label:40s}  frames={r.frames:4d}  meas_fps={r.measured_fps:6.3f}  "
        f"rep_fps={r.reported_fps:8.3f}  wh=({int(r.width)}x{int(r.height)})  "
        f"fourcc={r.fourcc_actual}  notes=[{r.notes}]"
    )


def scan_indices(max_index: int, backend: int, open_timeout_s: float = 1.0) -> List[int]:
    found = []
    for idx in range(max_index):
        t0 = time.time()
        cap = cv2.VideoCapture(idx, backend)
        while time.time() - t0 < open_timeout_s and not cap.isOpened():
            time.sleep(0.05)
        if cap.isOpened():
            ok, _ = cap.read()
            found.append(idx)
            print(f"  index={idx}  opened=True  read_ok={ok}")
        cap.release()
    return found


# -------------------------
# ffmpeg tests
# -------------------------
@dataclass
class FFMpegThroughput:
    ok: bool
    label: str
    frames_reported: Optional[int]
    raw_tail: str


def parse_ffmpeg_frames(output: str) -> Optional[int]:
    # ffmpeg progress lines contain "frame=  240"
    # We'll take the last occurrence.
    matches = re.findall(r"frame=\s*([0-9]+)", output)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except Exception:
        return None


def ffmpeg_devices() -> str:
    code, out = run_cmd(["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"], timeout_s=60)
    return out.strip()


def ffmpeg_options(camera_input: str) -> str:
    # camera_input should be a name like "PC Camera" or an @device_pnp... alternative name
    code, out = run_cmd(
        ["ffmpeg", "-hide_banner", "-f", "dshow", "-list_options", "true", "-i", f"video={camera_input}"],
        timeout_s=60,
    )
    return out.strip()


def ffmpeg_capture_test(camera_input: str, duration_s: int, mode: str) -> FFMpegThroughput:
    # mode in {"mjpeg", "yuyv422"}
    if mode == "mjpeg":
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-f",
            "dshow",
            "-video_size",
            "640x480",
            "-framerate",
            "30",
            "-vcodec",
            "mjpeg",
            "-i",
            f"video={camera_input}",
            "-t",
            str(duration_s),
            "-an",
            "-f",
            "null",
            "-",
        ]
        label = "ffmpeg_capture mjpeg 640x480@30"
    elif mode == "yuyv422":
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-f",
            "dshow",
            "-video_size",
            "640x480",
            "-framerate",
            "30",
            "-pixel_format",
            "yuyv422",
            "-i",
            f"video={camera_input}",
            "-t",
            str(duration_s),
            "-an",
            "-f",
            "null",
            "-",
        ]
        label = "ffmpeg_capture yuyv422 640x480@30"
    else:
        return FFMpegThroughput(False, f"unknown_mode_{mode}", None, "")

    code, out = run_cmd(cmd, timeout_s=max(60, duration_s + 20))
    frames = parse_ffmpeg_frames(out)

    # Keep tail for paste-friendliness
    tail_lines = "\n".join(out.splitlines()[-25:])
    ok = (code == 0 or code == 1) and (frames is not None)  # ffmpeg often returns 1 on device close quirks
    return FFMpegThroughput(ok, label, frames, tail_lines)


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None, help="Preferred camera index to focus on (optional)")
    ap.add_argument("--max_index", type=int, default=6, help="How many OpenCV indices to scan (0..max_index-1)")
    ap.add_argument("--duration", type=float, default=8.0, help="Timed capture duration (OpenCV + ffmpeg tests)")
    ap.add_argument("--warmup", type=float, default=1.0, help="Warmup seconds for OpenCV tests")
    ap.add_argument("--camera_name", type=str, default="PC Camera", help='ffmpeg display name (e.g. "PC Camera")')
    ap.add_argument(
        "--camera_input",
        type=str,
        default=None,
        help="ffmpeg camera input override (use the Alternative name like @device_pnp_\\\\?\\usb#...\\global)",
    )
    args = ap.parse_args()

    header("A) System info")
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("OpenCV:", cv2.__version__)
    print("ffmpeg_on_path:", have_ffmpeg())

    header("B) OpenCV scan: DSHOW indices")
    dshow_found = scan_indices(args.max_index, backend_id("DSHOW"))
    print("Found (DSHOW):", dshow_found)

    header("C) OpenCV scan: MSMF indices")
    msmf_found = scan_indices(args.max_index, backend_id("MSMF"))
    print("Found (MSMF):", msmf_found)

    # Choose candidate indices to test
    candidates: List[int] = []
    if args.index is not None:
        candidates = [args.index]
    else:
        # Prefer DSHOW indices discovered, but test a couple.
        candidates = dshow_found[:2] if dshow_found else msmf_found[:2]

    if not candidates:
        header("OpenCV: No cameras opened")
        print("No OpenCV camera indices opened. Try increasing --max_index or check device permissions.")
        return

    # OpenCV test matrix
    test_modes = [
        ("baseline(no sets)", None, None, None, None),
        ("640x480 req MJPG@30", 640, 480, 30.0, "MJPG"),
        ("640x480 req YUY2@30", 640, 480, 30.0, "YUY2"),
        ("640x480 req MJPG@15", 640, 480, 15.0, "MJPG"),
        ("320x240 req MJPG@30", 320, 240, 30.0, "MJPG"),
        ("1280x720 req MJPG@30", 1280, 720, 30.0, "MJPG"),
    ]

    for idx in candidates:
        header(f"D) OpenCV timed tests (index={idx}) — duration={args.duration}s warmup={args.warmup}s")

        for backend_name in ["DSHOW", "MSMF"]:
            b = backend_id(backend_name)
            print(f"\n--- Backend: {backend_name} ---")
            for label, w, h, fps, fcc in test_modes:
                r = timed_capture(
                    index=idx,
                    backend=b,
                    duration_s=args.duration,
                    warmup_s=args.warmup,
                    width=w,
                    height=h,
                    target_fps=fps,
                    fourcc=fcc,
                )
                print_capture(label, r)

            print("\n(Interpretation tip) If fourcc stays YUY2 even when requesting MJPG, OpenCV is not switching formats.")

    # ffmpeg section
    header("E) ffmpeg device listing (DirectShow)")
    if not have_ffmpeg():
        print("ffmpeg not found on PATH. Skipping ffmpeg tests.")
        return

    devs = ffmpeg_devices()
    print(devs)

    # Camera input string for ffmpeg tests
    cam_input = args.camera_input if args.camera_input else args.camera_name
    header(f"F) ffmpeg list_options for camera_input={cam_input!r}")
    opts = ffmpeg_options(cam_input)
    print(opts)

    header("G) ffmpeg capture-throughput tests (8s @ 640x480@30)")
    dur_i = int(round(args.duration))
    for mode in ["mjpeg", "yuyv422"]:
        res = ffmpeg_capture_test(cam_input, duration_s=dur_i, mode=mode)
        print(f"\n{res.label}")
        print("-" * 80)
        print("ok:", res.ok, "frames_reported:", res.frames_reported)
        # Expected ~240 frames if truly 30fps for 8 seconds
        if res.frames_reported is not None:
            print(f"measured_fps≈ {res.frames_reported / max(dur_i,1):.3f} (frames/duration)")
        print("\n(last ~25 lines of ffmpeg output)")
        print(res.raw_tail)

    header("DONE")
    print(
        "Copy/paste this whole output back. If ffmpeg shows ~240 frames but OpenCV shows ~80, "
        "then OpenCV/DSHOW negotiation is the issue. If both show ~80, the camera is actually running ~10fps "
        "(often exposure/lighting or driver settings)."
    )


if __name__ == "__main__":
    main()
