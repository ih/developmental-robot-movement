"""
camera_fps_diagnostics.py

Runs a bundle of diagnostics to figure out what FPS your Windows webcam
is *actually* delivering under OpenCV (DSHOW), whether MJPG “sticks”,
and (optionally) prints ffmpeg DirectShow device/mode listings.

What it does:
1) Enumerate camera indices (0..N-1) that OpenCV can open (DSHOW).
2) For a chosen index, run timed capture tests and report measured FPS.
3) Tries multiple FOURCC / resolution / target-FPS combinations.
4) (Optional) Tries to disable auto-exposure and set exposure values.
5) If ffmpeg is installed, prints:
   - DirectShow devices
   - Supported modes for a selected camera name

Usage examples (PowerShell):
  python camera_fps_diagnostics.py --scan
  python camera_fps_diagnostics.py --index 1 --duration 8
  python camera_fps_diagnostics.py --index 1 --duration 8 --try_mjpg --try_yuy2
  python camera_fps_diagnostics.py --index 1 --duration 8 --width 640 --height 480 --fps 30 --fourcc MJPG
  python camera_fps_diagnostics.py --index 1 --duration 8 --exposure_sweep
  python camera_fps_diagnostics.py --ffmpeg_devices
  python camera_fps_diagnostics.py --ffmpeg_options --ffmpeg_camera_name "Integrated Camera"

Notes:
- On Windows, OpenCV CAP_PROP_FPS is often “best effort” and may not apply.
- The measured FPS (frames/duration) is the truth.
"""

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is not installed. Install with: pip install opencv-python")
    sys.exit(1)


def fourcc_to_str(fourcc_val: float) -> str:
    # OpenCV returns FOURCC as float sometimes
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


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def run_cmd(cmd: List[str]) -> Tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = (p.stdout or "") + (p.stderr or "")
        return p.returncode, out
    except FileNotFoundError:
        return 127, "Command not found"


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def scan_indices(max_index: int = 10, backend=cv2.CAP_DSHOW, open_timeout_s: float = 1.0) -> List[int]:
    found = []
    print_header(f"Scanning camera indices 0..{max_index-1} (backend=DSHOW)")
    for idx in range(max_index):
        t0 = time.time()
        cap = cv2.VideoCapture(idx, backend)
        # Give it a moment
        while time.time() - t0 < open_timeout_s and not cap.isOpened():
            time.sleep(0.05)
        if cap.isOpened():
            found.append(idx)
            # Try a single read
            ok, _ = cap.read()
            print(f"Index {idx}: OPENED (read_ok={ok})")
        else:
            print(f"Index {idx}: not opened")
        cap.release()
    return found


@dataclass
class CaptureResult:
    ok: bool
    frames: int
    measured_fps: float
    reported_fps: float
    width: float
    height: float
    fourcc: str
    notes: str


def capture_test(
    index: int,
    duration_s: float,
    backend=cv2.CAP_DSHOW,
    warmup_s: float = 1.0,
    width: Optional[int] = None,
    height: Optional[int] = None,
    target_fps: Optional[float] = None,
    fourcc: Optional[str] = None,
    try_exposure: Optional[Tuple[Optional[float], Optional[float], Optional[float]]] = None,
) -> CaptureResult:
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return CaptureResult(False, 0, 0.0, 0.0, 0.0, 0.0, "????", "Could not open camera")

    notes = []
    # Order can matter: many drivers require FOURCC first.
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

    # Optional exposure tweaks: (auto_exposure, exposure, gain)
    if try_exposure is not None:
        auto_exp, exp, gain = try_exposure
        if auto_exp is not None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(auto_exp))
            notes.append(f"set_auto_exposure={auto_exp}")
        if exp is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, float(exp))
            notes.append(f"set_exposure={exp}")
        if gain is not None:
            cap.set(cv2.CAP_PROP_GAIN, float(gain))
            notes.append(f"set_gain={gain}")

    # Warmup
    t0 = time.time()
    while time.time() - t0 < warmup_s:
        cap.read()

    # Timed capture
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
    fourcc_actual = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))

    cap.release()
    return CaptureResult(
        True,
        frames,
        measured_fps,
        reported_fps,
        w,
        h,
        fourcc_actual,
        ";".join(notes) if notes else "",
    )


def print_result(label: str, r: CaptureResult) -> None:
    if not r.ok:
        print(f"{label}: ERROR: {r.notes}")
        return
    print(
        f"{label}: frames={r.frames}, measured_fps={r.measured_fps:.3f}, "
        f"reported_fps={r.reported_fps:.3f}, actual_wh=({int(r.width)}x{int(r.height)}), "
        f"fourcc_actual={r.fourcc}, notes=[{r.notes}]"
    )


def recommend_dataset_fps(measured_fps: float) -> int:
    # Round to nearest common rate
    common = [5, 10, 12, 15, 20, 24, 25, 30, 60]
    best = min(common, key=lambda x: abs(x - measured_fps))
    return int(best)


def ffmpeg_list_devices() -> None:
    print_header("ffmpeg DirectShow devices")
    if not have_ffmpeg():
        print("ffmpeg not found on PATH. Install ffmpeg or add it to PATH.")
        return
    code, out = run_cmd(["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"])
    print(out.strip())


def ffmpeg_list_options(camera_name: str) -> None:
    print_header(f"ffmpeg DirectShow options for camera: {camera_name!r}")
    if not have_ffmpeg():
        print("ffmpeg not found on PATH. Install ffmpeg or add it to PATH.")
        return
    code, out = run_cmd(["ffmpeg", "-f", "dshow", "-list_options", "true", "-i", f"video={camera_name}"])
    print(out.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true", help="Scan indices 0..--max_index-1")
    ap.add_argument("--max_index", type=int, default=10, help="Max indices to scan (exclusive)")
    ap.add_argument("--index", type=int, default=None, help="Camera index to test")
    ap.add_argument("--duration", type=float, default=8.0, help="Capture test duration seconds")
    ap.add_argument("--warmup", type=float, default=1.0, help="Warmup seconds")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--fps", type=float, default=None, help="Requested FPS (best-effort)")
    ap.add_argument("--fourcc", type=str, default=None, help="Requested FOURCC (e.g. MJPG, YUY2)")
    ap.add_argument("--try_mjpg", action="store_true", help="Try MJPG and default")
    ap.add_argument("--try_yuy2", action="store_true", help="Try YUY2 and default")
    ap.add_argument("--grid", action="store_true", help="Try a grid of common modes (MJPG/YUY2, 640x480/1280x720, fps 30/15)")
    ap.add_argument("--exposure_sweep", action="store_true", help="Try manual exposure settings sweep (common DSHOW values)")
    ap.add_argument("--ffmpeg_devices", action="store_true", help="Print ffmpeg DirectShow devices")
    ap.add_argument("--ffmpeg_options", action="store_true", help="Print ffmpeg DirectShow options for a camera name")
    ap.add_argument("--ffmpeg_camera_name", type=str, default=None, help="Camera name as shown by ffmpeg devices listing")

    args = ap.parse_args()

    if args.ffmpeg_devices:
        ffmpeg_list_devices()

    if args.ffmpeg_options:
        if not args.ffmpeg_camera_name:
            print("ERROR: --ffmpeg_options requires --ffmpeg_camera_name \"...\"")
        else:
            ffmpeg_list_options(args.ffmpeg_camera_name)

    if args.scan:
        found = scan_indices(max_index=args.max_index)
        print("\nFound indices:", found)

    if args.index is None:
        # If they only wanted scan/ffmpeg listings, that’s fine.
        if not (args.scan or args.ffmpeg_devices or args.ffmpeg_options):
            print("Nothing to do. Provide --scan and/or --index <n>.")
        return

    print_header(f"OpenCV capture tests (index={args.index}, duration={args.duration}s, warmup={args.warmup}s)")

    # 0) Baseline: no forced settings
    r0 = capture_test(args.index, args.duration, warmup_s=args.warmup)
    print_result("baseline(no sets)", r0)

    # 1) Single requested mode (if provided)
    if any(v is not None for v in [args.width, args.height, args.fps, args.fourcc]):
        r1 = capture_test(
            args.index,
            args.duration,
            warmup_s=args.warmup,
            width=args.width,
            height=args.height,
            target_fps=args.fps,
            fourcc=args.fourcc,
        )
        print_result("requested(mode)", r1)

    # 2) Try MJPG/YUY2 toggles
    if args.try_mjpg:
        r = capture_test(args.index, args.duration, warmup_s=args.warmup, width=args.width or 640, height=args.height or 480, target_fps=args.fps or 30, fourcc="MJPG")
        print_result("try(MJPG)", r)

    if args.try_yuy2:
        r = capture_test(args.index, args.duration, warmup_s=args.warmup, width=args.width or 640, height=args.height or 480, target_fps=args.fps or 30, fourcc="YUY2")
        print_result("try(YUY2)", r)

    # 3) Grid search
    grid_results: List[Tuple[str, CaptureResult]] = []
    if args.grid:
        modes = []
        for fcc in ["MJPG", "YUY2"]:
            for (w, h) in [(640, 480), (1280, 720)]:
                for fps in [30, 15]:
                    modes.append((fcc, w, h, fps))
        for fcc, w, h, fps in modes:
            label = f"grid({fcc},{w}x{h},{fps})"
            r = capture_test(args.index, args.duration, warmup_s=args.warmup, width=w, height=h, target_fps=fps, fourcc=fcc)
            grid_results.append((label, r))
            print_result(label, r)

        # Best measured fps
        ok_results = [(lbl, rr) for (lbl, rr) in grid_results if rr.ok]
        if ok_results:
            best = max(ok_results, key=lambda t: t[1].measured_fps)
            print("\nBest grid result:", best[0])
            print_result("best_grid", best[1])

    # 4) Exposure sweep
    if args.exposure_sweep:
        print_header("Exposure sweep (manual exposure tries)")
        # Common patterns for DSHOW:
        # auto_exposure=0.25 often means manual; 0.75 often means auto (varies).
        # exposure values are device-specific; negative values often used.
        trials = [
            (0.75, None, None),  # try auto
            (0.25, -4, None),
            (0.25, -6, None),
            (0.25, -8, None),
            (0.25, -10, None),
            (0.25, -12, None),
        ]
        base_mode = dict(width=args.width or 640, height=args.height or 480, target_fps=args.fps or 30, fourcc=args.fourcc or "MJPG")
        sweep_results = []
        for auto_exp, exp, gain in trials:
            label = f"exposure(auto={auto_exp},exp={exp})"
            r = capture_test(
                args.index,
                args.duration,
                warmup_s=args.warmup,
                width=base_mode["width"],
                height=base_mode["height"],
                target_fps=base_mode["target_fps"],
                fourcc=base_mode["fourcc"],
                try_exposure=(auto_exp, exp, gain),
            )
            sweep_results.append((label, r))
            print_result(label, r)

        ok = [(lbl, rr) for (lbl, rr) in sweep_results if rr.ok]
        if ok:
            best = max(ok, key=lambda t: t[1].measured_fps)
            print("\nBest exposure result:", best[0])
            print_result("best_exposure", best[1])

    # Recommendation for LeRobot playback correctness
    print_header("Recommendation")
    measured = r0.measured_fps if r0.ok else 0.0
    if measured > 0:
        rec = recommend_dataset_fps(measured)
        print(f"Measured baseline FPS ≈ {measured:.2f}.")
        print(f"If LeRobot encodes at 30fps, videos will look sped up if capture < 30.")
        print(f"Suggested temporary LeRobot setting: --dataset.fps={rec} (until you get true 30).")
        if measured < 20:
            print("Since measured FPS is low (<20), likely causes: camera mode not supporting 30fps, driver ignoring FPS, "
                  "auto-exposure in low light, or pixel format not switching to MJPG.")
            print("Next: run --ffmpeg_devices then --ffmpeg_options with the exact camera name to see supported 30fps modes.")
    else:
        print("Could not measure baseline FPS (camera did not open). Run with --scan to find a working index.")


if __name__ == "__main__":
    main()

