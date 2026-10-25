"""
Vtagger - AI Artist Tagger

Tags image/video files as "[artist] - original name" and learns each artist's style
so it can suggest artists for new files and hunt down every file by a given artist.

Engine
  * Backbone: SigLIP2 So400m (384px) on CUDA in fp16 with CUDA graphs. WD-EVA02-Large,
    a fused WD+SigLIP2 mode and ConvNeXt are selectable in Settings.
  * Index: every labelled embedding lives in one fp16 matrix in VRAM. Suggestions are a
    temperature-weighted kNN vote over the top-K neighbours, with human rejections vetoed.
  * Train Brain: trains a linear metric projection (SupCon + proxy loss) on the library and
    only keeps it if held-out accuracy improves.
  * Focus: trains a per-artist detector (linear probe + nearest-neighbour margin, with the
    user's rejections as hard negatives), Platt-calibrated on held-out files.
  * Confidence is self-calibrated, so "80%" means roughly 80% of such suggestions are right.
  * Files are identified by a content fingerprint, so renames and moves keep their memory.
"""
import os
import sys
import subprocess
import importlib.util

# =================================================================================
#  DEPENDENCY BOOTSTRAP
# =================================================================================
_REQUIRED = [
    ("customtkinter", "customtkinter"),
    ("PIL", "Pillow"),
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("timm", "timm"),
    ("safetensors", "safetensors"),
    ("huggingface_hub", "huggingface_hub"),
    ("transformers", "transformers"),
]
_TORCH_INDEX = "https://download.pytorch.org/whl/cu130"


def _pip_install(*args) -> bool:
    if importlib.util.find_spec("pip") is None:
        subprocess.call([sys.executable, "-m", "ensurepip", "--upgrade"])
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *args]
    print("[setup]", " ".join(args))
    return subprocess.call(cmd) == 0


def _torch_cuda_ok() -> bool:
    """Checks CUDA torch in a child process so a reinstall never fights a loaded DLL."""
    probe = "import torch,sys; t=torch.zeros(4,device='cuda')+1; torch.cuda.synchronize(); sys.exit(0 if float(t.sum())==4 else 1)"
    try:
        return subprocess.call([sys.executable, "-c", probe], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=180) == 0
    except Exception:
        return False


def _has_nvidia_gpu() -> bool:
    try:
        return subprocess.call(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30) == 0
    except Exception:
        return False


def bootstrap_environment() -> None:
    for mod, pkg in _REQUIRED:
        if importlib.util.find_spec(mod) is None:
            _pip_install(pkg)

    # The CUDA probe costs a few seconds, so remember a verified torch build.
    from importlib import metadata
    try:
        torch_ver = metadata.version("torch")
    except metadata.PackageNotFoundError:
        torch_ver = None
    marker_dir = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "Vtagger")
    os.makedirs(marker_dir, exist_ok=True)
    marker = os.path.join(marker_dir, ".cuda_ok")
    stamp = f"{sys.executable}|{torch_ver}"
    try:
        if torch_ver and open(marker, encoding="utf-8").read() == stamp:
            return
    except OSError:
        pass

    if torch_ver and _torch_cuda_ok():
        with open(marker, "w", encoding="utf-8") as f:
            f.write(stamp)
        return

    if os.environ.get("VTAGGER_BOOTSTRAPPED") == "1" or os.environ.get("VTAGGER_NO_TORCH_INSTALL") == "1":
        print("[setup] CUDA PyTorch still unavailable; continuing (the app will report the problem).")
        return
    if not _has_nvidia_gpu():
        if not torch_ver:
            print("[setup] No NVIDIA GPU found. Installing the CPU build of PyTorch (works, but slowly)...")
            _pip_install("torch", "torchvision")
        return
    print("[setup] CUDA-enabled PyTorch not found. Installing the CUDA 13.0 build (one time)...")
    if _pip_install("--upgrade", "torch", "torchvision", "--index-url", _TORCH_INDEX):
        env = dict(os.environ, VTAGGER_BOOTSTRAPPED="1")
        sys.exit(subprocess.call([sys.executable] + sys.argv, env=env))


if __name__ == "__main__" and not getattr(sys, "frozen", False):
    bootstrap_environment()

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")   # AV_LOG_QUIET: broken videos are handled, not printed
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")

import ctypes
import hashlib
import io
import json
import logging
import math
import queue
import random
import re
import sqlite3
import threading
import time
import traceback
import difflib
import tkinter as tk
import tkinter.font as tkfont
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from ctypes import wintypes
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from tkinter import filedialog, messagebox
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import customtkinter as ctk
from PIL import Image, ImageOps, ImageDraw

Image.MAX_IMAGE_PIXELS = None

try:
    import cv2
    cv2.setNumThreads(1)  # decode workers are already parallel
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(max(2, min(4, (os.cpu_count() or 4) // 4)))
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


# =================================================================================
#  PATHS & CONFIG
# =================================================================================
APP_NAME = "Vtagger"
APP_VERSION = "50.0"
APP_DIR = os.path.join(os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config"), APP_NAME)
os.makedirs(APP_DIR, exist_ok=True)


class Cfg:
    DB_FILE = os.path.join(APP_DIR, "vtagger_brain_v2.db")
    LEGACY_DB_FILE = os.path.join(APP_DIR, "artist_features_convnext.db")
    SETTINGS_FILE = os.path.join(APP_DIR, "settings.json")
    COUNTS_FILE = os.path.join(APP_DIR, "tagger_counts.json")
    HISTORY_FILE = os.path.join(APP_DIR, "artist_history.json")
    SESSION_FILE = os.path.join(APP_DIR, "focus_session.json")
    LOG_FILE = os.path.join(APP_DIR, "vtagger.log")

    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".jfif", ".bmp", ".gif", ".webp", ".avif", ".tif", ".tiff"}
    VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"} if CV2_AVAILABLE else set()
    PIL_ONLY_EXTS = {".gif", ".avif", ".tif", ".tiff"}

    TAG_RE = re.compile(r"^\[(.+?)\] - (.+)$", re.S)
    ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\[\]\x00-\x1f]')

    PREVIEW_MAX = (1400, 1000)
    THUMB = 168
    HISTORY_MAX_LEN = 200
    PREFETCH_AHEAD = 6


DEFAULT_SETTINGS = {
    "backbone": "siglip2-so400m",
    "tta": False,  # flip TTA: about +0.25 pts accuracy for 2x the GPU time
    "knn_k": 50,
    "knn_tau": 0.15,
    "batch_size": 32,
    "io_workers": min(12, max(4, (os.cpu_count() or 8) // 2 + 2)),
    "focus_min_conf": 0.35,
    "recursive": False,
    "last_folder": "",
}


class Settings(dict):
    def __init__(self):
        super().__init__(DEFAULT_SETTINGS)
        try:
            with open(Cfg.SETTINGS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.update({k: v for k, v in data.items() if k in DEFAULT_SETTINGS})
        except (OSError, ValueError):
            pass

    def save(self):
        atomic_write_json(Cfg.SETTINGS_FILE, dict(self))


logger = logging.getLogger("vtagger")


def setup_logging() -> None:
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(message)s")
    fh = RotatingFileHandler(Cfg.LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)


# =================================================================================
#  SMALL UTILITIES
# =================================================================================
def atomic_write_json(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except OSError as e:
        logger.error(f"Could not write {path}: {e}")


def read_json(path: str, default: Any) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def split_artists(artist_str: str) -> List[str]:
    """'a, b & c' -> ['a', 'b', 'c'] (deduplicated case-insensitively, order kept)."""
    if not artist_str:
        return []
    out, seen = [], set()
    for part in re.split(r"[,&;+]", artist_str):
        p = part.strip()
        if p and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return out


def sanitize_artist(name: str) -> str:
    """Makes a tag safe for Windows filenames and for the [tag] - name format."""
    cleaned = Cfg.ILLEGAL_CHARS.sub("", name or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip().rstrip(". ")
    return ", ".join(split_artists(cleaned)) if cleaned else ""


def normalize_artist_name(name: str) -> str:
    return re.sub(r"[\s_\-\.\+\&\,\;\:\'\"/\\\[\]\(\)]+", "", (name or "").lower())


def parse_tagged_name(basename: str) -> Optional[Tuple[str, str]]:
    m = Cfg.TAG_RE.match(basename)
    if not m:
        return None
    return m.group(1).strip(), m.group(2)


def media_kind(path: str) -> Optional[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext in Cfg.IMAGE_EXTS:
        return "image"
    if ext in Cfg.VIDEO_EXTS:
        return "video"
    return None


def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    i = 2
    while os.path.exists(f"{root} ({i}){ext}"):
        i += 1
    return f"{root} ({i}){ext}"


def safe_rename(src: str, dst: str, retries: int = 6) -> str:
    """Renames without ever overwriting another file. Returns the final path."""
    if os.path.normcase(os.path.abspath(src)) == os.path.normcase(os.path.abspath(dst)):
        return src
    dst = unique_path(dst)
    last_err = None
    for i in range(retries):
        try:
            os.rename(src, dst)
            return dst
        except PermissionError as e:  # usually a viewer/AV holding the file briefly
            last_err = e
            time.sleep(0.15 * (i + 1))
    raise last_err


def format_time(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0 or not math.isfinite(seconds):
        return "--:--"
    s = int(seconds)
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def list_media_stat(folder: str) -> Dict[str, Tuple[int, int]]:
    """Recursive listing: path -> (size, mtime_ns), straight from the directory entries."""
    out: Dict[str, Tuple[int, int]] = {}
    stack = [folder]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        elif media_kind(e.name) and e.is_file():
                            st = e.stat()
                            out[e.path] = (st.st_size, st.st_mtime_ns)
                    except OSError:
                        continue
        except OSError:
            continue
    return out


def list_media(folder: str, recursive: bool) -> List[str]:
    out = []
    if recursive:
        for root, _, files in os.walk(folder):
            out.extend(os.path.join(root, n) for n in files if media_kind(n))
    else:
        with os.scandir(folder) as it:
            out.extend(e.path for e in it if e.is_file() and media_kind(e.name))
    return out


def open_external(path: str) -> None:
    if not path or not os.path.exists(path):
        return
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        logger.error(f"Could not open {path}: {e}")


def reveal_in_explorer(path: str) -> None:
    if sys.platform == "win32" and os.path.exists(path):
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    else:
        open_external(os.path.dirname(path))


# =================================================================================
#  WINDOWS INTEGRATION (recycle bin, clipboard, taskbar progress)
# =================================================================================
if sys.platform == "win32":
    _shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
                    ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_uint16), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]

    _shell32.SHFileOperationW.argtypes = [ctypes.POINTER(_SHFILEOPSTRUCTW)]
    _shell32.SHFileOperationW.restype = ctypes.c_int
    _kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    _kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    _kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    _kernel32.GlobalLock.restype = ctypes.c_void_p
    _kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    _kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
    _kernel32.GlobalFree.restype = wintypes.HGLOBAL
    _user32.OpenClipboard.argtypes = [wintypes.HWND]
    _user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    _user32.SetClipboardData.restype = wintypes.HANDLE


def send_to_recycle_bin(path: str) -> bool:
    """Moves a file to the Recycle Bin. Never silently falls back to a permanent delete."""
    if not os.path.exists(path):
        return False
    if sys.platform != "win32":
        try:
            from send2trash import send2trash
            send2trash(path)
            return True
        except Exception as e:
            logger.error(f"Trash failed for {path}: {e}")
            return False
    FO_DELETE, FOF_SILENT, FOF_NOCONFIRMATION, FOF_ALLOWUNDO, FOF_NOERRORUI = 3, 0x4, 0x10, 0x40, 0x400
    op = _SHFILEOPSTRUCTW(None, FO_DELETE, os.path.abspath(path) + "\0", None,
                          FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI, False, None, None)
    rc = _shell32.SHFileOperationW(ctypes.byref(op))
    ok = rc == 0 and not op.fAnyOperationsAborted and not os.path.exists(path)
    if not ok:
        logger.error(f"Recycle Bin delete failed for {path} (code {rc})")
    return ok


def copy_image_to_clipboard(pil_image: Image.Image) -> bool:
    if sys.platform != "win32" or pil_image is None:
        return False
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, "BMP")
    data = buf.getvalue()[14:]  # strip BITMAPFILEHEADER -> CF_DIB payload
    GMEM_MOVEABLE, CF_DIB = 0x0002, 8
    if not _user32.OpenClipboard(None):
        return False
    try:
        _user32.EmptyClipboard()
        h = _kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
        if not h:
            return False
        p = _kernel32.GlobalLock(h)
        if not p:
            _kernel32.GlobalFree(h)
            return False
        ctypes.memmove(p, data, len(data))
        _kernel32.GlobalUnlock(h)
        if not _user32.SetClipboardData(CF_DIB, h):
            _kernel32.GlobalFree(h)
            return False
        return True  # the clipboard owns the memory now
    finally:
        _user32.CloseClipboard()


class TaskbarProgress:
    """ITaskbarList3 progress on the app's taskbar button (main thread only)."""
    TBPF_NOPROGRESS, TBPF_INDETERMINATE, TBPF_NORMAL = 0, 0x1, 0x2

    def __init__(self):
        self.ptr = None
        if sys.platform != "win32":
            return
        try:
            class GUID(ctypes.Structure):
                _fields_ = [("Data1", ctypes.c_ulong), ("Data2", ctypes.c_ushort),
                            ("Data3", ctypes.c_ushort), ("Data4", ctypes.c_ubyte * 8)]
            HRESULT = ctypes.c_long
            proto = ctypes.WINFUNCTYPE

            class Vtbl(ctypes.Structure):
                _fields_ = [("QueryInterface", ctypes.c_void_p), ("AddRef", ctypes.c_void_p), ("Release", ctypes.c_void_p),
                            ("HrInit", proto(HRESULT, ctypes.c_void_p)), ("AddTab", ctypes.c_void_p),
                            ("DeleteTab", ctypes.c_void_p), ("ActivateTab", ctypes.c_void_p),
                            ("SetActiveAlt", ctypes.c_void_p), ("MarkFullscreenWindow", ctypes.c_void_p),
                            ("SetProgressValue", proto(HRESULT, ctypes.c_void_p, wintypes.HWND, ctypes.c_ulonglong, ctypes.c_ulonglong)),
                            ("SetProgressState", proto(HRESULT, ctypes.c_void_p, wintypes.HWND, ctypes.c_int))]

            class Obj(ctypes.Structure):
                _fields_ = [("lpVtbl", ctypes.POINTER(Vtbl))]

            ole32 = ctypes.WinDLL("ole32")
            ole32.CoInitialize(None)
            clsid = GUID(0x56FDF344, 0xFD6D, 0x11D0, (ctypes.c_ubyte * 8)(0x95, 0x8A, 0x00, 0x60, 0x97, 0xC9, 0xA0, 0x90))
            iid = GUID(0xEA1AFB91, 0x9E28, 0x4B86, (ctypes.c_ubyte * 8)(0x90, 0xE9, 0x9E, 0x9F, 0x8A, 0x5E, 0xEF, 0xAF))
            ptr = ctypes.POINTER(Obj)()
            if ole32.CoCreateInstance(ctypes.byref(clsid), None, 1, ctypes.byref(iid), ctypes.byref(ptr)) == 0:
                ptr.contents.lpVtbl.contents.HrInit(ptr)
                self.ptr = ptr
        except Exception as e:
            logger.debug(f"Taskbar progress unavailable: {e}")

    def set(self, hwnd: int, done: int, total: int) -> None:
        if not self.ptr or not hwnd:
            return
        try:
            vt = self.ptr.contents.lpVtbl.contents
            if total <= 0:
                vt.SetProgressState(self.ptr, hwnd, self.TBPF_NOPROGRESS)
            else:
                vt.SetProgressState(self.ptr, hwnd, self.TBPF_NORMAL)
                vt.SetProgressValue(self.ptr, hwnd, max(0, int(done)), int(total))
        except Exception:
            pass

    def clear(self, hwnd: int) -> None:
        self.set(hwnd, 0, 0)


# =================================================================================
#  MEDIA DECODING
# =================================================================================
FP_HEAD, FP_MID, FP_TAIL = 256 * 1024, 64 * 1024, 256 * 1024


def _fp_digest(size: int, head: bytes, mid: bytes, tail: bytes) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(size.to_bytes(8, "little"))
    h.update(head)
    h.update(mid)
    h.update(tail)
    return h.hexdigest()


def fingerprint_bytes(data: bytes) -> str:
    n = len(data)
    if n <= FP_HEAD + FP_TAIL + FP_MID:
        return _fp_digest(n, data, b"", b"")
    mid = n // 2
    return _fp_digest(n, data[:FP_HEAD], data[mid:mid + FP_MID], data[-FP_TAIL:])


def fingerprint_file(path: str) -> str:
    """Content fingerprint (size + head/middle/tail). Survives renames and moves."""
    n = os.path.getsize(path)
    with open(path, "rb") as f:
        if n <= FP_HEAD + FP_TAIL + FP_MID:
            return _fp_digest(n, f.read(), b"", b"")
        head = f.read(FP_HEAD)
        f.seek(n // 2)
        mid = f.read(FP_MID)
        f.seek(n - FP_TAIL)
        tail = f.read(FP_TAIL)
    return _fp_digest(n, head, mid, tail)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img >> 8).astype(np.uint8)
    img = img.astype(np.float32)
    if img.max() <= 1.0:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def decode_image(path: str, data: Optional[bytes] = None) -> np.ndarray:
    """Decodes to RGB or RGBA uint8 (H, W, 3|4). OpenCV fast path, Pillow fallback."""
    ext = os.path.splitext(path)[1].lower()
    if CV2_AVAILABLE and ext not in Cfg.PIL_ONLY_EXTS:
        buf = np.frombuffer(data, np.uint8) if data is not None else np.fromfile(path, np.uint8)
        flag = cv2.IMREAD_COLOR if ext in (".jpg", ".jpeg", ".jfif") else cv2.IMREAD_UNCHANGED
        img = cv2.imdecode(buf, flag)
        if img is not None:
            img = _to_uint8(img)
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    src = io.BytesIO(data) if data is not None else path
    with Image.open(src) as im:
        try:
            im.seek(0)
        except EOFError:
            pass
        im = ImageOps.exif_transpose(im)
        has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
        return np.asarray(im.convert("RGBA" if has_alpha else "RGB"))


def flatten_alpha(img: np.ndarray, bg: Tuple[int, int, int]) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[..., 3:4].astype(np.float32) / 255.0
        out = img[..., :3].astype(np.float32) * a + np.array(bg, np.float32) * (1.0 - a)
        return out.astype(np.uint8)
    return img


def _run_with_timeout(fn: Callable, timeout: float, *args):
    """OpenCV can hang forever on corrupt videos; run it on a sacrificial thread."""
    box: Dict[str, Any] = {}

    def runner():
        try:
            box["v"] = fn(*args)
        except Exception as e:
            box["e"] = e

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"timed out after {timeout:.0f}s")
    if "e" in box:
        raise box["e"]
    return box.get("v")


VIDEO_SCAN_FRAMES = 150  # only look at the first ~5 s: sequential decoding is fast, seeking is not


def _video_frames(path: str, fractions: Sequence[float]) -> Tuple[List[np.ndarray], str]:
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise IOError("cannot open video")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        span = min(n, VIDEO_SCAN_FRAMES) if n > 0 else VIDEO_SCAN_FRAMES
        targets = sorted({max(0, min(span - 1, int(span * f))) for f in fractions}) if span > 10 else [0]
        frames, last = [], None
        for i in range(targets[-1] + 1):
            if i in targets:
                ok, fr = cap.read()
                if ok and fr is not None:
                    last = fr
                    frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            elif not cap.grab():
                break
        if not frames and last is None:
            ok, fr = cap.read()
            if ok and fr is not None:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        if not frames:
            raise IOError("no decodable frames")
        return frames, f"{w}x{h}"
    finally:
        cap.release()


def decode_video_frames(path: str, fractions=(0.15, 0.5, 0.9), timeout: float = 8.0) -> Tuple[List[np.ndarray], str]:
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for video")
    return _run_with_timeout(_video_frames, timeout, path, fractions)


def load_media_frames(path: str, data: Optional[bytes] = None) -> Tuple[List[np.ndarray], str]:
    """All frames the model should see for this file, plus a dimension string."""
    if media_kind(path) == "video":
        frames, dims = decode_video_frames(path)
        return frames, f"VIDEO  {dims}"
    img = decode_image(path, data)
    return [img], f"IMAGE  {img.shape[1]}x{img.shape[0]}"


def square_letterbox(img: np.ndarray, size: int, fill: Tuple[int, int, int]) -> np.ndarray:
    """Aspect-preserving resize onto a square canvas (area interpolation keeps line work clean)."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    if CV2_AVAILABLE:
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
        resized = cv2.resize(img, (nw, nh), interpolation=interp)
    else:
        resized = np.asarray(Image.fromarray(img).resize((nw, nh), Image.Resampling.BICUBIC))
    resized = flatten_alpha(resized, fill)
    canvas = np.empty((size, size, 3), np.uint8)
    canvas[:] = fill
    y, x = (size - nh) // 2, (size - nw) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def make_thumbnail(img: np.ndarray, box: Tuple[int, int], bg=(10, 10, 10)) -> Image.Image:
    h, w = img.shape[:2]
    s = min(box[0] / w, box[1] / h, 1.0)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    if CV2_AVAILABLE and s < 1:
        return Image.fromarray(flatten_alpha(cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), bg))
    pil = Image.fromarray(flatten_alpha(img, bg))
    return pil.resize((nw, nh), Image.Resampling.LANCZOS) if s < 1 else pil.copy()


# =================================================================================
#  DATABASE (thread-local connections, WAL, fp16 embeddings keyed by content)
# =================================================================================
class BrainDB:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
    CREATE TABLE IF NOT EXISTS paths (path TEXT PRIMARY KEY, size INTEGER, mtime_ns INTEGER, fkey TEXT);
    CREATE INDEX IF NOT EXISTS idx_paths_key ON paths(fkey);
    CREATE TABLE IF NOT EXISTS emb (fkey TEXT, model TEXT, vec BLOB, PRIMARY KEY (fkey, model)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS labels (fkey TEXT, artist_l TEXT, artist TEXT, verified INTEGER DEFAULT 0,
                                       ts REAL, PRIMARY KEY (fkey, artist_l)) WITHOUT ROWID;
    CREATE INDEX IF NOT EXISTS idx_labels_artist ON labels(artist_l);
    CREATE TABLE IF NOT EXISTS negatives (fkey TEXT, artist_l TEXT, ts REAL, PRIMARY KEY (fkey, artist_l)) WITHOUT ROWID;
    """

    def __init__(self, path: str):
        self.path = path
        self._local = threading.local()
        self.write_lock = threading.RLock()
        with self.write_lock:
            self.conn().executescript(self.SCHEMA)

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "c", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA temp_store=MEMORY")
            c.execute("PRAGMA cache_size=-262144")
            c.execute("PRAGMA mmap_size=1073741824")
            self._local.c = c
        return c

    def write(self, fn: Callable[[sqlite3.Connection], Any]) -> Any:
        with self.write_lock:
            c = self.conn()
            with c:
                return fn(c)

    @staticmethod
    def _chunks(seq: Sequence, n: int = 900) -> Iterator[Sequence]:
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    # --- meta ---
    def get_meta(self, k: str, default: Optional[str] = None) -> Optional[str]:
        r = self.conn().execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return r[0] if r else default

    def set_meta(self, k: str, v: str) -> None:
        self.write(lambda c: c.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (k, v)))

    # --- path -> content key cache ---
    def load_path_cache(self) -> Dict[str, Tuple[int, int, str]]:
        return {p: (s, m, k) for p, s, m, k in self.conn().execute("SELECT path, size, mtime_ns, fkey FROM paths")}

    def lookup_path(self, path: str) -> Optional[Tuple[int, int, str]]:
        r = self.conn().execute("SELECT size, mtime_ns, fkey FROM paths WHERE path=?", (os.path.normcase(path),)).fetchone()
        return tuple(r) if r else None

    def put_paths(self, rows: List[Tuple[str, int, int, str]]) -> None:
        if rows:
            rows = [(os.path.normcase(p), s, m, k) for p, s, m, k in rows]
            self.write(lambda c: c.executemany("INSERT OR REPLACE INTO paths VALUES (?, ?, ?, ?)", rows))

    def move_path(self, old: str, new: str) -> None:
        def fn(c):
            r = c.execute("SELECT fkey FROM paths WHERE path=?", (os.path.normcase(old),)).fetchone()
            c.execute("DELETE FROM paths WHERE path=?", (os.path.normcase(old),))
            if r:
                st = os.stat(new)
                c.execute("INSERT OR REPLACE INTO paths VALUES (?, ?, ?, ?)",
                          (os.path.normcase(new), st.st_size, st.st_mtime_ns, r[0]))
        try:
            self.write(fn)
        except OSError:
            pass

    # --- embeddings ---
    def get_embeddings(self, keys: Sequence[str], model: str) -> Dict[str, np.ndarray]:
        out = {}
        c = self.conn()
        for chunk in self._chunks(list(keys)):
            q = f"SELECT fkey, vec FROM emb WHERE model=? AND fkey IN ({','.join('?' * len(chunk))})"
            for k, blob in c.execute(q, (model, *chunk)):
                out[k] = np.frombuffer(blob, dtype=np.float16)
        return out

    def keys_with_embedding(self, model: str) -> Set[str]:
        return {r[0] for r in self.conn().execute("SELECT fkey FROM emb WHERE model=?", (model,))}

    def put_embeddings(self, rows: List[Tuple[str, str, bytes]]) -> None:
        if rows:
            self.write(lambda c: c.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?, ?)", rows))

    # --- labels ---
    def labels_for(self, key: str) -> List[Tuple[str, int]]:
        return list(self.conn().execute("SELECT artist, verified FROM labels WHERE fkey=?", (key,)))

    def all_label_keys(self) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = defaultdict(set)
        for k, a in self.conn().execute("SELECT fkey, artist_l FROM labels"):
            out[k].add(a)
        return out

    def set_labels_many(self, items: List[Tuple[str, List[str], bool]]) -> None:
        """Replaces the full label set of each key. Human-verified flags are kept for kept artists."""
        now = time.time()

        def fn(c):
            for key, artists, verified in items:
                prev = {a: v for a, v in c.execute("SELECT artist_l, verified FROM labels WHERE fkey=?", (key,))}
                c.execute("DELETE FROM labels WHERE fkey=?", (key,))
                for a in artists:
                    v = 1 if verified or prev.get(a.lower(), 0) else 0
                    c.execute("INSERT OR REPLACE INTO labels VALUES (?, ?, ?, ?, ?)", (key, a.lower(), a, v, now))
        if items:
            self.write(fn)

    def iter_labeled(self, model: str, batch: int = 20000) -> Iterator[List[Tuple[str, str, int, bytes]]]:
        cur = self.conn().execute(
            "SELECT l.fkey, l.artist, l.verified, e.vec FROM labels l JOIN emb e ON e.fkey=l.fkey AND e.model=? "
            "ORDER BY l.fkey", (model,))
        while True:
            rows = cur.fetchmany(batch)
            if not rows:
                break
            yield rows

    def count_labeled(self, model: str) -> int:
        return self.conn().execute(
            "SELECT COUNT(*) FROM labels l JOIN emb e ON e.fkey=l.fkey AND e.model=?", (model,)).fetchone()[0]

    def artist_counts(self) -> Dict[str, int]:
        """Display name -> number of files, merged case-insensitively."""
        counts: Dict[str, int] = {}
        best: Dict[str, Tuple[int, str]] = {}
        for a_l, a, n in self.conn().execute("SELECT artist_l, artist, COUNT(*) FROM labels GROUP BY artist_l, artist"):
            counts[a_l] = counts.get(a_l, 0) + n
            if a_l not in best or n > best[a_l][0]:
                best[a_l] = (n, a)
        return {best[k][1]: v for k, v in counts.items()}

    def rename_artists(self, olds: List[str], target: str) -> int:
        t_l = target.lower()

        def fn(c):
            n = 0
            for old in olds:
                o_l = old.lower()
                if o_l == t_l:
                    n += c.execute("UPDATE labels SET artist=? WHERE artist_l=?", (target, t_l)).rowcount
                    continue
                c.execute("UPDATE labels SET verified=1 WHERE artist_l=? AND fkey IN "
                          "(SELECT fkey FROM labels WHERE artist_l=? AND verified=1)", (t_l, o_l))
                c.execute("DELETE FROM labels WHERE artist_l=? AND fkey IN (SELECT fkey FROM labels WHERE artist_l=?)", (o_l, t_l))
                n += c.execute("UPDATE labels SET artist_l=?, artist=? WHERE artist_l=?", (t_l, target, o_l)).rowcount
                c.execute("DELETE FROM negatives WHERE artist_l=? AND fkey IN (SELECT fkey FROM negatives WHERE artist_l=?)", (o_l, t_l))
                c.execute("UPDATE negatives SET artist_l=? WHERE artist_l=?", (t_l, o_l))
            c.execute("UPDATE labels SET artist=? WHERE artist_l=?", (target, t_l))
            return n
        return self.write(fn)

    def delete_artists(self, names: List[str]) -> int:
        lows = [n.lower() for n in names]

        def fn(c):
            n = 0
            for a in lows:
                n += c.execute("DELETE FROM labels WHERE artist_l=?", (a,)).rowcount
                c.execute("DELETE FROM negatives WHERE artist_l=?", (a,))
            return n
        return self.write(fn)

    # --- negatives (human rejections) ---
    def add_negative(self, key: str, artist: str) -> None:
        self.write(lambda c: c.execute("INSERT OR REPLACE INTO negatives VALUES (?, ?, ?)", (key, artist.lower(), time.time())))

    def all_negatives(self) -> List[Tuple[str, str]]:
        return list(self.conn().execute("SELECT fkey, artist_l FROM negatives"))

    def reset_brain(self) -> None:
        self.write(lambda c: (c.execute("DELETE FROM labels"), c.execute("DELETE FROM negatives")))

    def stats(self) -> Dict[str, int]:
        c = self.conn()
        return {"labels": c.execute("SELECT COUNT(*) FROM labels").fetchone()[0],
                "artists": c.execute("SELECT COUNT(DISTINCT artist_l) FROM labels").fetchone()[0],
                "embeddings": c.execute("SELECT COUNT(*) FROM emb").fetchone()[0],
                "negatives": c.execute("SELECT COUNT(*) FROM negatives").fetchone()[0]}


# =================================================================================
#  VISION BACKBONES
# =================================================================================
class Backbone:
    key = ""            # settings key
    model_id = ""       # stored with every embedding; change it and vectors are recomputed
    label = ""
    dim = 0
    size = 224
    fill = (255, 255, 255)
    squash = False

    def load(self, device: torch.device) -> None:
        raise NotImplementedError

    def prep_one(self, img: np.ndarray) -> np.ndarray:
        if self.squash:
            if CV2_AVAILABLE:
                interp = cv2.INTER_AREA if max(img.shape[:2]) > self.size else cv2.INTER_CUBIC
                return flatten_alpha(cv2.resize(img, (self.size, self.size), interpolation=interp), self.fill)
            img = flatten_alpha(img, self.fill)
            return np.asarray(Image.fromarray(img).resize((self.size, self.size), Image.Resampling.BICUBIC))
        return square_letterbox(img, self.size, self.fill)

    def prep(self, img: np.ndarray) -> List[np.ndarray]:
        """One uint8 HWC array per model input."""
        return [self.prep_one(img)]

    def features(self, xs: List[torch.Tensor]) -> torch.Tensor:
        return self.forward(xs[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def dummy(self) -> List[torch.Tensor]:
        return [torch.zeros(3, self.size, self.size, dtype=torch.uint8)]


class WDEva02Backbone(Backbone):
    key = "wd-eva02-large"
    model_id = "wd-eva02-large-v3@448"
    label = "WD EVA02-Large (Danbooru)"
    dim = 1024
    size = 448

    def load(self, device):
        import timm
        m = timm.create_model("hf-hub:SmilingWolf/wd-eva02-large-tagger-v3", pretrained=True)
        self.m = m.eval().half().to(device)

    def forward(self, x):
        x = x.half().div_(127.5).sub_(1.0)[:, [2, 1, 0]]  # WD taggers expect BGR in [-1, 1]
        return self.m.forward_head(self.m.forward_features(x), pre_logits=True)


class SigLIP2Backbone(Backbone):
    key = "siglip2-so400m"
    model_id = "siglip2-so400m-p14-384"
    label = "SigLIP2 So400m"
    dim = 1152
    size = 384
    squash = True

    def load(self, device):
        from transformers import SiglipVisionModel
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()  # the checkpoint also holds the (unused) text tower
        self.m = SiglipVisionModel.from_pretrained("google/siglip2-so400m-patch14-384", dtype=torch.float16).eval().to(device)

    def forward(self, x):
        x = x.half().div_(127.5).sub_(1.0)
        return self.m(pixel_values=x).pooler_output


class _GeM(nn.Module):
    def forward(self, x):
        return x.float().clamp(min=1e-6).pow(3).mean(dim=(2, 3)).pow(1.0 / 3)


class ConvNeXtBackbone(Backbone):
    key = "convnext-base"
    model_id = "convnext-base-imagenet-gem@224"
    label = "ConvNeXt-Base ImageNet (legacy)"
    dim = 1024
    size = 224
    fill = (0, 0, 0)

    def load(self, device):
        from torchvision import models
        self.m = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT).features.eval().half().to(device)
        self.pool = _GeM()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1) * 255
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1) * 255

    def forward(self, x):
        return self.pool(self.m(((x.float() - self.mean) / self.std).half()))


class FusedBackbone(Backbone):
    """WD-EVA02 + SigLIP2 side by side: the most accurate option (about half the speed of either alone)."""
    key = "fused-wd-siglip2"
    model_id = "fuse(wd-eva02-large-v3@448,siglip2-so400m-p14-384)"
    label = "WD-EVA02 + SigLIP2 (max accuracy)"
    dim = WDEva02Backbone.dim + SigLIP2Backbone.dim
    size = WDEva02Backbone.size

    def __init__(self):
        self.parts = [WDEva02Backbone(), SigLIP2Backbone()]

    def load(self, device):
        for p in self.parts:
            p.load(device)

    def prep(self, img):
        return [p.prep_one(img) for p in self.parts]

    def features(self, xs):
        feats = [F.normalize(p.forward(x).float(), dim=1) for p, x in zip(self.parts, xs)]
        return torch.cat(feats, dim=1) / math.sqrt(len(feats))

    def dummy(self):
        return [torch.zeros(3, p.size, p.size, dtype=torch.uint8) for p in self.parts]


BACKBONES: Dict[str, type] = {b.key: b for b in (SigLIP2Backbone, FusedBackbone, WDEva02Backbone, ConvNeXtBackbone)}


# =================================================================================
#  EMBEDDING PIPELINE (parallel decode -> batched fp16 GPU inference)
# =================================================================================
@dataclass
class EmbedResult:
    path: str
    key: Optional[str] = None
    vec: Optional[np.ndarray] = None      # float16 [D], L2-normalised
    thumb: Optional[Image.Image] = None
    dims: str = ""
    stat: Optional[Tuple[int, int]] = None
    error: str = ""


class Embedder:
    def __init__(self, backbone: Backbone, device: torch.device, settings: Settings):
        self.bb = backbone
        self.device = device
        self.tta = bool(settings.get("tta", False))
        self.batch_size = max(1, int(settings.get("batch_size", 32)))
        self.workers = max(2, int(settings.get("io_workers", 8)))
        self.gpu_lock = threading.Lock()

    def to_tensors(self, frames: List[np.ndarray]) -> List[List[torch.Tensor]]:
        """Per frame: one uint8 CHW tensor per model input."""
        return [[torch.from_numpy(a).permute(2, 0, 1).contiguous() for a in self.bb.prep(f)] for f in frames]

    def _forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        try:
            f = F.normalize(self.bb.features(xs).float(), dim=1)
            if self.tta:
                f = F.normalize(f + F.normalize(self.bb.features([x.flip(3) for x in xs]).float(), dim=1), dim=1)
            return f
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            n = len(xs[0])
            if n <= 1:
                raise
            self.batch_size = max(1, n // 2)
            logger.warning(f"GPU out of memory; batch size lowered to {self.batch_size}")
            h = n // 2
            return torch.cat([self._forward([x[:h] for x in xs]), self._forward([x[h:] for x in xs])])

    # ----- CUDA graphs: the whole forward pass becomes one launch, so decode threads holding the
    # GIL can no longer starve the GPU (measured: eager inference stalls badly under thread load).
    def _buckets(self) -> List[int]:
        return sorted({1, 8, self.batch_size})

    def _graph_for(self, n: int):
        if self.device.type != "cuda" or getattr(self, "_graphs_disabled", False):
            return None
        size = next((b for b in self._buckets() if b >= n), None)
        if size is None:
            return None
        cache = self.__dict__.setdefault("_graphs", {})
        if size in cache:
            return cache[size]
        try:
            static = [torch.zeros(size, *t.shape, dtype=torch.uint8, device=self.device) for t in self.bb.dummy()]
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._forward(static)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            # thread_local: other threads (tagging, focus scoring) may use the GPU while this records
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                out = self._forward(static)
            cache[size] = (g, static, out)
            logger.info(f"CUDA graph captured for batch {size}")
            return cache[size]
        except Exception as e:
            logger.warning(f"CUDA graph capture failed ({e}); using eager inference")
            self._graphs_disabled = True
            torch.cuda.synchronize()
            return None

    def warmup(self):
        with torch.inference_mode(), self.gpu_lock:
            for b in self._buckets():  # record every size up front: nothing gets recorded later, mid-session
                self._graph_for(b)

    def _run(self, xs: List[torch.Tensor]) -> torch.Tensor:
        n = xs[0].shape[0]
        entry = self._graph_for(n)
        if entry is None:
            return self._forward(xs)
        g, static, out = entry
        for st, x in zip(static, xs):
            st[:n].copy_(x, non_blocking=True)
            if n < st.shape[0]:
                st[n:].zero_()
        g.replay()
        return out[:n].clone()

    @torch.inference_mode()
    def embed_tensors(self, items: List[List[torch.Tensor]]) -> torch.Tensor:
        outs = []
        for i in range(0, len(items), self.batch_size):
            chunk = items[i:i + self.batch_size]
            xs = []
            for j in range(len(chunk[0])):
                x = torch.stack([it[j] for it in chunk])
                if self.device.type == "cuda":
                    x = x.pin_memory().to(self.device, non_blocking=True)
                xs.append(x)
            with self.gpu_lock:
                outs.append(self._run(xs))
        return torch.cat(outs) if outs else torch.empty(0, self.bb.dim, device=self.device)

    def embed_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        f = self.embed_tensors(self.to_tensors(frames))
        return F.normalize(f.mean(0, keepdim=True), dim=1)[0].half().cpu().numpy()

    def _load_one(self, path: str, key: Optional[str], want_thumb: bool):
        try:
            st = os.stat(path)
            data = None
            if media_kind(path) == "image":
                with open(path, "rb") as f:
                    data = f.read()
                key = key or fingerprint_bytes(data)
            else:
                key = key or fingerprint_file(path)
            frames, dims = load_media_frames(path, data)
            tensors = self.to_tensors(frames)
            thumb = make_thumbnail(frames[0], (Cfg.THUMB, Cfg.THUMB)) if want_thumb else None
            return EmbedResult(path, key, None, thumb, dims, (st.st_size, st.st_mtime_ns)), tensors
        except Exception as e:
            return EmbedResult(path, key, error=f"{type(e).__name__}: {e}"), None

    def iter_files(self, items: Iterable[Tuple[str, Optional[str]]], stop: Callable[[], bool],
                   thumb_every: float = 0.0, workers: Optional[int] = None,
                   vid_workers: int = 3) -> Iterator[EmbedResult]:
        """Streams (path, known_key) pairs through decode workers and the GPU, yielding as they finish."""
        it = iter(items)
        last_thumb = [0.0]
        workers = workers or self.workers
        max_inflight = workers * 6

        def want_thumb():
            if thumb_every <= 0:
                return False
            now = time.time()
            if now - last_thumb[0] >= thumb_every:
                last_thumb[0] = now
                return True
            return False

        pending_batch: List[Tuple[EmbedResult, List[List[torch.Tensor]]]] = []
        frames_in_batch = 0

        def flush():
            nonlocal pending_batch, frames_in_batch
            batch, pending_batch, frames_in_batch = pending_batch, [], 0
            if not batch:
                return []
            flat = [t for _, ts in batch for t in ts]
            try:
                feats = self.embed_tensors(flat)
            except Exception as e:
                logger.error(f"GPU batch failed: {e}")
                for r, _ in batch:
                    r.error = f"GPU: {e}"
                return [r for r, _ in batch]
            out, i = [], 0
            for r, ts in batch:
                f = feats[i:i + len(ts)]
                i += len(ts)
                r.vec = F.normalize(f.mean(0, keepdim=True), dim=1)[0].half().cpu().numpy()
                out.append(r)
            return out

        with ThreadPoolExecutor(workers, thread_name_prefix="decode") as ex, \
                ThreadPoolExecutor(vid_workers, thread_name_prefix="video") as vex:
            inflight = set()
            vid_inflight = set()
            parked: Deque[Tuple[str, Optional[str]]] = deque()  # videos waiting for a video slot

            def submit_more():
                vid_inflight.intersection_update(inflight)
                while parked and len(vid_inflight) < vid_workers * 2 and not stop():
                    p, k = parked.popleft()
                    f = vex.submit(self._load_one, p, k, want_thumb())
                    inflight.add(f)
                    vid_inflight.add(f)
                while len(inflight) - len(vid_inflight) < max_inflight and len(parked) < 5000 and not stop():
                    try:
                        p, k = next(it)
                    except StopIteration:
                        return
                    if media_kind(p) == "video":
                        if len(vid_inflight) < vid_workers * 2:
                            f = vex.submit(self._load_one, p, k, want_thumb())
                            inflight.add(f)
                            vid_inflight.add(f)
                        else:
                            parked.append((p, k))
                    else:
                        inflight.add(ex.submit(self._load_one, p, k, want_thumb()))

            submit_more()
            while inflight or (parked and not stop()):
                if not inflight:
                    submit_more()
                    continue
                done, _ = wait(inflight, timeout=0.05, return_when=FIRST_COMPLETED)
                if not done:
                    # Keep the GPU busy instead of waiting on slow files.
                    if pending_batch and frames_in_batch >= max(1, self.batch_size // 2):
                        yield from flush()
                    continue
                for fut in done:
                    inflight.discard(fut)
                    res, tensors = fut.result()
                    if tensors is None:
                        yield res
                        continue
                    pending_batch.append((res, tensors))
                    frames_in_batch += len(tensors)
                if stop():
                    for fut in inflight:
                        fut.cancel()
                    break
                submit_more()
                if frames_in_batch >= self.batch_size:
                    yield from flush()
            yield from flush()


# =================================================================================
#  STYLE INDEX (VRAM-resident kNN with calibrated confidence)
# =================================================================================
@dataclass
class Suggestion:
    artist: str
    conf: float      # calibrated probability that this artist is correct
    share: float     # share of the neighbour vote
    sim: float       # best cosine similarity to one of the artist's files


@dataclass
class Prediction:
    suggestions: List[Suggestion] = field(default_factory=list)
    best_sim: float = 0.0

    @property
    def top(self) -> Optional[Suggestion]:
        return self.suggestions[0] if self.suggestions else None


class StyleIndex:
    MIN_CENTER_ROWS = 200
    VERIFIED_WEIGHT = 1.3
    NEG_DUP_SIM = 0.95

    def __init__(self, device: torch.device, dim: int, k: int, tau: float):
        self.device, self.dim, self.k, self.tau = device, dim, int(k), float(tau)
        self.lock = threading.RLock()
        self.clear()

    # ----- state -----
    def clear(self):
        with self.lock:
            self.base = None          # fp16 [cap, D]: centred + normalised (projector input)
            self.vecs = None          # fp16 [cap, D]: search space (== base without a projector)
            self.lab = None           # int32 [cap]: artist id, 0 = deleted row
            self.ver = None           # bool  [cap]: human-verified label
            self.n = 0
            self.row_key: List[str] = []
            self.key_rows: Dict[str, List[int]] = {}
            self.names: List[str] = [""]
            self.aid: Dict[str, int] = {"": 0}
            self.count: List[int] = [0]
            self.mean: Optional[torch.Tensor] = None
            self.W: Optional[torch.Tensor] = None
            self.neg_by_key: Dict[str, Set[int]] = {}
            self.neg_base: Optional[torch.Tensor] = None
            self.neg_vecs: Optional[torch.Tensor] = None
            self.neg_aid: Optional[torch.Tensor] = None
            self.calib: Optional[torch.Tensor] = None
            self.val_keys: List[str] = []  # files the projector never trained on
            self.metrics: Dict[str, float] = {}

    @property
    def size(self) -> int:
        return sum(self.count)

    def artist_id(self, name: str, create: bool = True) -> int:
        k = name.strip().lower()
        i = self.aid.get(k)
        if i is None and create:
            i = len(self.names)
            self.aid[k] = i
            self.names.append(name.strip())
            self.count.append(0)
        return i or 0

    def has_artist(self, name: str) -> bool:
        i = self.aid.get(name.strip().lower())
        return bool(i) and self.count[i] > 0

    def known_artists(self) -> List[str]:
        return [self.names[i] for i in range(1, len(self.names)) if self.count[i] > 0]

    # ----- transforms -----
    def _center(self, raw: torch.Tensor) -> torch.Tensor:
        raw = F.normalize(raw.float(), dim=1)
        if self.mean is not None:
            raw = F.normalize(raw - self.mean, dim=1)
        return raw

    def _project(self, base: torch.Tensor, W: Optional[torch.Tensor] = None) -> torch.Tensor:
        W = self.W if W is None else W
        if W is None:
            return base
        return F.normalize(base.float() @ W.T, dim=1)

    def transform(self, raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = self._center(raw)
        return b.half(), self._project(b).half()

    def _alloc(self, cap: int):
        dev = self.device
        base = torch.zeros(cap, self.dim, dtype=torch.float16, device=dev)
        lab = torch.zeros(cap, dtype=torch.int32, device=dev)
        ver = torch.zeros(cap, dtype=torch.bool, device=dev)
        if self.base is not None and self.n:
            base[:self.n] = self.base[:self.n]
            lab[:self.n] = self.lab[:self.n]
            ver[:self.n] = self.ver[:self.n]
        if self.W is not None:
            vecs = torch.zeros(cap, self.dim, dtype=torch.float16, device=dev)
            if self.vecs is not None and self.n:
                vecs[:self.n] = self.vecs[:self.n]
        else:
            vecs = base
        self.base, self.vecs, self.lab, self.ver = base, vecs, lab, ver

    def _ensure_capacity(self, extra: int):
        cap = 0 if self.base is None else self.base.shape[0]
        if self.n + extra > cap:
            self._alloc(max(int(cap * 1.25), self.n + extra + 4096))

    def _rebuild_vecs(self):
        if self.base is None:
            return
        if self.W is None:
            self.vecs = self.base
            return
        vecs = torch.empty_like(self.base)
        for i in range(0, self.n, 65536):
            vecs[i:min(i + 65536, self.n)] = self._project(self.base[i:min(i + 65536, self.n)]).half()
        self.vecs = vecs

    # ----- building -----
    def build(self, db: BrainDB, model_id: str, projector: Optional[dict], progress: Callable = None):
        with self.lock:
            self.clear()
            total = db.count_labeled(model_id)
            if progress:
                progress(0, max(total, 1), "Loading style memory...")
            cpu = np.empty((max(total, 1), self.dim), np.float16)
            labels, verified = [], []
            n = 0
            for rows in db.iter_labeled(model_id):
                for key, artist, ver, blob in rows:
                    v = np.frombuffer(blob, np.float16)
                    if v.shape[0] != self.dim or n >= total:
                        continue
                    cpu[n] = v
                    labels.append(artist)
                    verified.append(bool(ver))
                    self.row_key.append(key)
                    n += 1
                if progress:
                    progress(n, max(total, 1), f"Loading style memory... {n:,}/{total:,}")
            if projector is not None:
                self.mean = projector["mean"].to(self.device).float().view(1, -1)
                self.W = projector["W"].to(self.device).float()
                self.val_keys = list(projector.get("val_keys", []))
                self.metrics.update(projector.get("metrics", {}))
            self.n = 0
            self._alloc(n + 8192)
            if n:
                X = torch.from_numpy(cpu[:n])
                if self.mean is None and n >= self.MIN_CENTER_ROWS:
                    acc = torch.zeros(self.dim, dtype=torch.float64, device=self.device)
                    for i in range(0, n, 65536):
                        acc += F.normalize(X[i:i + 65536].to(self.device).float(), dim=1).sum(0).double()
                    self.mean = (acc / n).float().view(1, -1)
                for i in range(0, n, 65536):
                    chunk = self._center(X[i:i + 65536].to(self.device))
                    self.base[i:i + len(chunk)] = chunk.half()
                    if self.W is not None:
                        self.vecs[i:i + len(chunk)] = self._project(chunk).half()
                del X
            self.n = n
            ids = [self.artist_id(a) for a in labels]
            for i in ids:
                self.count[i] += 1
            if n:
                self.lab[:n] = torch.tensor(ids, dtype=torch.int32, device=self.device)
                self.ver[:n] = torch.tensor(verified, dtype=torch.bool, device=self.device)
            for r, k in enumerate(self.row_key):
                self.key_rows.setdefault(k, []).append(r)
            self._load_negatives(db, model_id)
            del cpu
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            logger.info(f"Index ready: {n:,} rows, {len(self.known_artists()):,} artists, "
                        f"projector={'on' if self.W is not None else 'off'}")

    def _load_negatives(self, db: BrainDB, model_id: str):
        neg = db.all_negatives()
        self.neg_by_key = {}
        for key, a_l in neg:
            self.neg_by_key.setdefault(key, set()).add(self.artist_id(a_l))
        embs = db.get_embeddings(list(self.neg_by_key), model_id)
        vecs, aids = [], []
        for key, aset in self.neg_by_key.items():
            if key in embs and embs[key].shape[0] == self.dim:
                for a in aset:
                    vecs.append(embs[key])
                    aids.append(a)
        if vecs:
            raw = torch.from_numpy(np.stack(vecs)).to(self.device)
            self.neg_base, self.neg_vecs = self.transform(raw)
            self.neg_aid = torch.tensor(aids, dtype=torch.long, device=self.device)
        else:
            self.neg_base = self.neg_vecs = self.neg_aid = None

    # ----- live edits -----
    def remove_key(self, key: str):
        with self.lock:
            rows = self.key_rows.pop(key, None)
            if not rows:
                return
            ids = self.lab[rows].tolist()
            for i in ids:
                if i:
                    self.count[i] -= 1
            self.lab[rows] = 0

    def add(self, key: str, artists: List[str], raw_vec: np.ndarray, verified: bool):
        with self.lock:
            self.remove_key(key)
            if not artists:
                return
            self._ensure_capacity(len(artists))
            raw = torch.from_numpy(np.array(raw_vec, np.float16)).to(self.device).view(1, -1)
            b, q = self.transform(raw)
            rows = []
            for a in artists:
                r = self.n
                aid = self.artist_id(a)
                self.base[r] = b[0]
                if self.W is not None:
                    self.vecs[r] = q[0]
                self.lab[r] = aid
                self.ver[r] = verified
                self.count[aid] += 1
                self.row_key.append(key)
                rows.append(r)
                self.n += 1
            self.key_rows[key] = rows
            if self.mean is None and self.n >= self.MIN_CENTER_ROWS:
                self._recenter()

    def _recenter(self):
        live = self.base[:self.n].float()
        self.mean = F.normalize(live, dim=1).mean(0, keepdim=True)
        for i in range(0, self.n, 65536):
            self.base[i:min(i + 65536, self.n)] = F.normalize(self.base[i:min(i + 65536, self.n)].float() - self.mean, dim=1).half()
        self._rebuild_vecs()

    def add_negative(self, key: str, artist: str, raw_vec: Optional[np.ndarray]):
        with self.lock:
            aid = self.artist_id(artist)
            self.neg_by_key.setdefault(key, set()).add(aid)
            if raw_vec is not None:
                b, q = self.transform(torch.from_numpy(np.array(raw_vec, np.float16)).to(self.device).view(1, -1))
                a = torch.tensor([aid], dtype=torch.long, device=self.device)
                self.neg_base = b if self.neg_base is None else torch.cat([self.neg_base, b])
                self.neg_vecs = q if self.neg_vecs is None else torch.cat([self.neg_vecs, q])
                self.neg_aid = a if self.neg_aid is None else torch.cat([self.neg_aid, a])

    def _remap(self, lut: torch.Tensor):
        if self.n:
            self.lab[:self.n] = lut[self.lab[:self.n].long()].int()
        counts = torch.bincount(self.lab[:self.n].long(), minlength=len(self.names)).tolist() if self.n else [0] * len(self.names)
        self.count = counts[:len(self.names)] + [0] * max(0, len(self.names) - len(counts))
        if self.neg_aid is not None:
            self.neg_aid = lut[self.neg_aid]
        lut_l = lut.tolist()
        self.neg_by_key = {k: {lut_l[a] for a in s if lut_l[a]} for k, s in self.neg_by_key.items()}

    def merge(self, olds: List[str], target: str):
        with self.lock:
            tid = self.artist_id(target)
            self.names[tid] = target.strip()
            lut = torch.arange(len(self.names), device=self.device)
            for o in olds:
                oid = self.aid.get(o.strip().lower())
                if oid and oid != tid:
                    lut[oid] = tid
            self._remap(lut)

    def delete(self, names: List[str]):
        with self.lock:
            lut = torch.arange(len(self.names), device=self.device)
            for o in names:
                oid = self.aid.get(o.strip().lower())
                if oid:
                    lut[oid] = 0
            self._remap(lut)

    # ----- scoring -----
    def _vote(self, q: torch.Tensor, keys: Sequence[Optional[str]], vecs: Optional[torch.Tensor] = None):
        vecs = self.vecs if vecs is None else vecs
        n = self.n
        sims = q @ vecs[:n].T
        for i, k in enumerate(keys):
            rows = self.key_rows.get(k) if k else None
            if rows:
                sims[i, rows] = -4.0  # never let a file vote for itself
        k = min(self.k, n)
        vals, idx = sims.topk(k, dim=1)
        vals = vals.float()
        labs = self.lab[idx].long()
        w = torch.exp((vals - vals[:, :1]) / self.tau)
        w = w * torch.where(self.ver[idx], self.VERIFIED_WEIGHT, 1.0)
        w = w * ((labs > 0) & (vals > -2)).float()
        A = len(self.names)
        scores = torch.zeros(len(q), A, device=self.device).scatter_add_(1, labs, w)
        best = torch.full((len(q), A), -1.0, device=self.device).scatter_reduce_(1, labs, vals, reduce="amax")
        for i, key in enumerate(keys):
            for a in self.neg_by_key.get(key, ()) if key else ():
                scores[i, a] = 0.0
        if self.neg_vecs is not None and len(self.neg_vecs):
            ns = q @ self.neg_vecs.T
            hit = (ns.float() >= self.NEG_DUP_SIM).nonzero()
            if len(hit):
                scores[hit[:, 0], self.neg_aid[hit[:, 1]]] = 0.0
        scores[:, 0] = 0.0
        return scores, best

    def _features(self, share: torch.Tensor, sim: torch.Tensor, margin: torch.Tensor, aids: torch.Tensor) -> torch.Tensor:
        cnt = torch.tensor(self.count, device=self.device, dtype=torch.float32)[aids]
        return torch.stack([share, sim, margin, torch.log1p(cnt), torch.ones_like(share)], dim=-1)

    def _conf(self, feats: torch.Tensor) -> torch.Tensor:
        if self.calib is None:
            return feats[..., 0] * torch.clamp((feats[..., 1] + 0.1) * 2, 0, 1)
        return torch.sigmoid(feats @ self.calib)

    def _ranked(self, scores, best, topn):
        """Top-n artists per query with their calibration features."""
        share = scores / scores.sum(1, keepdim=True).clamp_min(1e-9)
        tv, ta = share.topk(min(topn + 1, share.shape[1]), dim=1)
        if tv.shape[1] < 2:
            tv = torch.cat([tv, torch.zeros_like(tv)], 1)
            ta = torch.cat([ta, torch.zeros_like(ta)], 1)
        sims = best.gather(1, ta)
        first = torch.arange(tv.shape[1], device=self.device) == 0
        margin = torch.where(first, tv[:, :1] - tv[:, 1:2], tv - tv[:, :1])
        feats = self._features(tv, sims, margin, ta)
        return tv[:, :topn], ta[:, :topn], sims[:, :topn], feats[:, :topn]

    @torch.inference_mode()
    def predict(self, raw: np.ndarray, keys: Sequence[Optional[str]], topn: int = 5) -> List[Prediction]:
        raw = np.atleast_2d(raw)
        with self.lock:
            if self.n == 0 or self.size == 0:
                return [Prediction() for _ in range(len(raw))]
            out = []
            for s in range(0, len(raw), 256):
                q = self.transform(torch.from_numpy(np.array(raw[s:s + 256], np.float16)).to(self.device))[1]
                scores, best = self._vote(q, keys[s:s + 256])
                tv, ta, ts, feats = self._ranked(scores, best, topn)
                tc = self._conf(feats)
                tv, ta, ts, tc = (t.cpu().numpy() for t in (tv, ta, ts, tc))
                bmax = best.max(1).values.cpu().numpy()
                for i in range(len(q)):
                    sug = [Suggestion(self.names[a], float(c), float(v), float(sm))
                           for v, a, sm, c in zip(tv[i], ta[i], ts[i], tc[i]) if v > 0 and a > 0]
                    sug.sort(key=lambda x: x.conf, reverse=True)
                    out.append(Prediction(sug, float(bmax[i])))
            return out

    @torch.inference_mode()
    def score_target(self, raw: np.ndarray, keys: Sequence[Optional[str]], target: str) -> Tuple[np.ndarray, np.ndarray]:
        """Calibrated confidence that each file is by `target`, and whether target is the top vote."""
        raw = np.atleast_2d(raw)
        with self.lock:
            tid = self.aid.get(target.strip().lower(), 0)
            if not tid or self.n == 0:
                return np.zeros(len(raw)), np.zeros(len(raw), bool)
            confs, tops = [], []
            for s in range(0, len(raw), 256):
                q = self.transform(torch.from_numpy(np.array(raw[s:s + 256], np.float16)).to(self.device))[1]
                scores, best = self._vote(q, keys[s:s + 256])
                share = scores / scores.sum(1, keepdim=True).clamp_min(1e-9)
                st = share[:, tid]
                other = share.clone()
                other[:, tid] = 0
                om = other.max(1).values
                aids = torch.full_like(st, tid, dtype=torch.long)
                conf = self._conf(self._features(st, best[:, tid], st - om, aids))
                conf = torch.where(st > 0, conf, torch.zeros_like(conf))
                confs.append(conf.cpu().numpy())
                tops.append((st > om).cpu().numpy())
            return np.concatenate(confs), np.concatenate(tops)

    # ----- focus mode: "find every file by this artist" -----
    FOCUS_MARGIN_W = 10.0

    @torch.inference_mode(False)
    def focus_model(self, target: str, usable: Optional[torch.Tensor] = None, seed: int = 0) -> Optional[dict]:
        """Builds a one-artist detector: a linear probe (artist vs everyone else, plus the user's rejections
        as hard negatives) combined with the nearest-neighbour margin, then Platt-calibrated on held-out files.
        Ranks an artist's files far better than the k-NN vote alone."""
        with self.lock:
            tid = self.aid.get(target.strip().lower(), 0)
            if not tid or self.n == 0:
                return None
            n, dev = self.n, self.device
            g = torch.Generator(device="cpu").manual_seed(seed)
            ok = (self.lab[:n] > 0) if usable is None else (usable[:n] & (self.lab[:n] > 0))
            is_t = (self.lab[:n] == tid) & ok
            t_keys = {self.row_key[r] for r in is_t.nonzero().squeeze(1).tolist()}
            same_file = torch.zeros(n, dtype=torch.bool, device=dev)  # other-artist rows of the target's collab files
            same_file[[r for k in t_keys for r in self.key_rows.get(k, [])]] = True
            other = ok & ~same_file
            t_rows = is_t.nonzero().squeeze(1)
            o_rows = other.nonzero().squeeze(1)
            if len(t_rows) < 2 or len(o_rows) < 50:
                return None
            # one row per target file
            seen, uniq = set(), []
            for r in t_rows.tolist():
                k = self.row_key[r]
                if k not in seen:
                    seen.add(k)
                    uniq.append(r)
            t_rows = torch.tensor(uniq, device=dev)
            perm = t_rows[torch.randperm(len(t_rows), generator=g).to(dev)]
            n_hold = max(1, min(len(perm) // 4, 300))
            hold_p, train_p = perm[:n_hold], perm[n_hold:]
            o_perm = o_rows[torch.randperm(len(o_rows), generator=g).to(dev)]
            hold_n, train_n = o_perm[:6000], o_perm[6000:66000]
            V = self.vecs
            Xp = V[train_p].float().clone()
            hard = None
            if self.neg_vecs is not None and len(self.neg_vecs):
                rej = self.neg_aid == tid
                if rej.any():
                    hard = self.neg_vecs[rej].float().clone()
            Xn = V[train_n].float().clone()
            w = torch.zeros(self.dim, device=dev, requires_grad=True)
            b = torch.zeros(1, device=dev, requires_grad=True)
            opt = torch.optim.Adam([w, b], lr=0.05)
            with torch.enable_grad():
                for _ in range(150):
                    loss = F.softplus(-(Xp @ w + b)).mean() + F.softplus(Xn @ w + b).mean() + 1e-3 * (w ** 2).sum()
                    if hard is not None:
                        loss = loss + 2.0 * F.softplus(hard @ w + b).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            model = {"tid": tid, "w": w.detach(), "b": b.detach(), "usable": ok, "is_t": is_t, "other": other,
                     "a": 1.0, "c": 0.0}
            # Platt calibration on files the probe never saw, weighted to the library's base rate.
            hp = self._focus_raw(V[hold_p], [self.row_key[r] for r in hold_p.tolist()], model)
            hn = self._focus_raw(V[hold_n], [self.row_key[r] for r in hold_n.tolist()], model)
            s = torch.cat([hp, hn])
            y = torch.cat([torch.ones_like(hp), torch.zeros_like(hn)])
            prior = len(t_rows) / max(1, len(t_rows) + len(o_rows))
            wts = torch.where(y > 0, prior / len(hp), (1 - prior) / len(hn))
            a = torch.ones(1, device=dev, requires_grad=True)
            c = torch.zeros(1, device=dev, requires_grad=True)
            opt = torch.optim.LBFGS([a, c], lr=0.5, max_iter=100)

            def closure():
                opt.zero_grad()
                z = a * s + c
                l = (wts * F.binary_cross_entropy_with_logits(z, y, reduction="none")).sum() + 1e-4 * (a - 1) ** 2
                l.backward()
                return l
            with torch.enable_grad():
                opt.step(closure)
            model["a"], model["c"] = float(a.detach().clamp_min(1e-3)), float(c.detach())
            # Self-check: how a held-out file of this artist ranks against the held-out other files.
            model["heldout_recall_at_1pct"] = float((hp > hn.quantile(0.99)).float().mean())
            return model

    def _focus_raw(self, q: torch.Tensor, keys: Sequence[Optional[str]], model: dict) -> torch.Tensor:
        if len(q) > 256:
            return torch.cat([self._focus_raw(q[i:i + 256], keys[i:i + 256], model) for i in range(0, len(q), 256)])
        sims = (q @ self.vecs[:self.n].T).float()
        for i, k in enumerate(keys):
            rows = self.key_rows.get(k) if k else None
            if rows:
                sims[i, rows] = -4.0  # a file never matches itself
        st = torch.where(model["is_t"], sims, torch.full_like(sims, -4.0)).max(1).values
        so = torch.where(model["other"], sims, torch.full_like(sims, -4.0)).max(1).values
        probe = q.float() @ model["w"] + model["b"]
        return probe + self.FOCUS_MARGIN_W * (st - so)

    @torch.inference_mode()
    def score_focus(self, raw: np.ndarray, keys: Sequence[Optional[str]], model: dict, projected: bool = False) -> np.ndarray:
        """Calibrated probability that each file is by the focus artist (0 for files the user rejected)."""
        raw = np.atleast_2d(raw)
        with self.lock:
            out = []
            tid = model["tid"]
            for s in range(0, len(raw), 256):
                chunk = torch.from_numpy(np.array(raw[s:s + 256], np.float16)).to(self.device)
                q = chunk if projected else self.transform(chunk)[1]
                ks = list(keys[s:s + 256])
                p = torch.sigmoid(model["a"] * self._focus_raw(q, ks, model) + model["c"])
                for i, k in enumerate(ks):
                    if k and tid in self.neg_by_key.get(k, ()):
                        p[i] = 0.0
                if self.neg_vecs is not None and len(self.neg_vecs):
                    rej = self.neg_aid == tid
                    if rej.any():
                        dup = (q @ self.neg_vecs[rej].T).float().max(1).values >= self.NEG_DUP_SIM
                        p = torch.where(dup, torch.zeros_like(p), p)
                out.append(p.cpu().numpy())
            return np.concatenate(out) if out else np.zeros(0)

    # ----- self-evaluation & calibration -----
    def _sample_queries(self, n_queries: int, seed: int = 0, keys_pool: Optional[List[str]] = None):
        rng = random.Random(seed)
        pool = keys_pool if keys_pool is not None else [k for k, rows in self.key_rows.items() if rows]
        keys = rng.sample(pool, min(n_queries, len(pool)))
        rows = [self.key_rows[k][0] for k in keys]
        truth = [set(int(a) for a in self.lab[self.key_rows[k]].tolist() if a) for k in keys]
        return keys, rows, truth

    @torch.inference_mode()
    def _loo(self, keys, rows, truth, vecs: torch.Tensor, topn: int = 5):
        """Leave-one-out evaluation: each sampled file is predicted with itself removed."""
        feats, ys, top1, top5 = [], [], 0, 0
        for s in range(0, len(rows), 256):
            scores, best = self._vote(vecs[rows[s:s + 256]], keys[s:s + 256], vecs=vecs)
            tv, ta, _, f = self._ranked(scores, best, topn)
            for t, a_row, v_row in zip(truth[s:s + 256], ta.tolist(), tv.tolist()):
                y = [1.0 if (a in t and v > 0) else 0.0 for a, v in zip(a_row, v_row)]
                top1 += y[0] > 0
                top5 += any(y)
                ys.append(y)
            feats.append(f)
        n = max(1, len(rows))
        return (torch.cat(feats) if feats else None), torch.tensor(ys, device=self.device), top1 / n, top5 / n

    def calibrate(self, n_queries: int = 4000) -> Dict[str, float]:
        with self.lock:
            if self.n < 50:
                self.calib = None
                return {}
            pool = None
            if self.W is not None:
                # After training, only files the projector never saw give honest confidence.
                pool = [k for k in self.val_keys if self.key_rows.get(k)]
                if len(pool) < 200:
                    pool = None
            keys, rows, truth = self._sample_queries(n_queries, seed=1234, keys_pool=pool)
            feats, ys, top1, top5 = self._loo(keys, rows, truth, self.vecs)
            X = feats.reshape(-1, feats.shape[-1]).float()
            y = ys.reshape(-1).float()
            keep = X[:, 0] > 0
            X, y = X[keep], y[keep]
        if len(y) < 50 or y.sum() < 5 or (1 - y).sum() < 5:
            return {"top1": top1, "top5": top5}
        w = torch.zeros(X.shape[1], device=self.device, requires_grad=True)
        opt = torch.optim.LBFGS([w], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(X @ w, y) + 1e-3 * (w[:-1] ** 2).sum()
            loss.backward()
            return loss
        with torch.enable_grad():
            opt.step(closure)
        with self.lock:
            self.calib = w.detach()
            self.metrics.update({"top1": top1, "top5": top5, "eval_n": len(rows)})
        logger.info(f"Calibrated on {len(rows)} leave-one-out queries: top-1 {top1:.1%}, top-5 {top5:.1%}")
        return dict(self.metrics)

    # ----- metric-learning projector -----
    def train_projector(self, epochs: int = 40, progress: Callable = None, stop: Callable = lambda: False) -> Dict[str, Any]:
        with self.lock:
            if self.n < 500:
                raise ValueError("Need at least 500 tagged files to train.")
            keys_multi = [k for k, rows in self.key_rows.items() if rows and self.count[int(self.lab[rows[0]])] >= 3]
            if len(keys_multi) < 200:
                raise ValueError("Need more artists with 3+ files to train.")
            vkeys, vrows, vtruth = self._sample_queries(3000, seed=99, keys_pool=keys_multi)
            val_rows = set(r for k in vkeys for r in self.key_rows[k])
            mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
            mask[list(val_rows)] = False
            mask &= self.lab[:self.n] > 0
            train_idx = mask.nonzero().flatten()
            X = self.base[:self.n]
            y_all = self.lab[:self.n].long()
            _, _, before, before5 = self._loo(vkeys, vrows, vtruth, self.base)
        A = len(self.names)
        D = self.dim
        Xt = X[train_idx]
        yt = y_all[train_idx]
        # Identity-initialised linear map, random 512 batches,
        # supervised-contrastive + proxy (normalised softmax) loss.
        n_train = len(train_idx)
        if len(set(yt.tolist())) < 32:
            raise ValueError("Not enough distinct artists to train.")
        lin = nn.Linear(D, D, bias=False).to(self.device)
        nn.init.eye_(lin.weight)
        proxies = nn.Parameter(torch.randn(A, D, device=self.device) * 0.01)
        opt = torch.optim.AdamW([{"params": lin.parameters(), "weight_decay": 1e-3},
                                 {"params": [proxies], "weight_decay": 0.0}], lr=1e-3)
        per_epoch = max(1, math.ceil(n_train / 512))
        steps = int(min(20000, per_epoch * max(epochs, 60 if n_train < 50000 else epochs)))
        t0 = time.time()
        perm = torch.randperm(n_train, device=self.device)
        for step in range(steps):
            if stop():
                return {"accepted": False, "reason": "cancelled"}
            j = step % per_epoch
            if j == 0 and step:
                perm = torch.randperm(n_train, device=self.device)
            b = perm[j * 512:(j + 1) * 512]
            if len(b) < 8:
                continue
            z = F.normalize(lin(Xt[b].float()), dim=1)
            yb = yt[b]
            s = z @ z.T / 0.07
            eye = torch.eye(len(b), dtype=torch.bool, device=self.device)
            pos = (yb[:, None] == yb[None]) & ~eye
            s = s.masked_fill(eye, -1e9)
            logp = s - s.logsumexp(1, keepdim=True)
            has = pos.any(1)
            loss = F.cross_entropy(z @ F.normalize(proxies, dim=1).T / 0.05, yb)
            if has.any():
                loss = loss + (-(logp * pos).sum(1)[has] / pos.sum(1)[has]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if progress and (step % 50 == 0 or step == steps - 1):
                el = time.time() - t0
                progress(step + 1, steps, f"Step {step + 1:,}/{steps:,}  loss {loss.item():.3f}", el / (step + 1) * (steps - step - 1))
        W = lin.weight.detach().float()
        with self.lock:
            if progress:
                progress(steps, steps, "Validating on held-out files...", 0)
            cand = torch.empty_like(self.base[:self.n])
            for i in range(0, self.n, 65536):
                cand[i:min(i + 65536, self.n)] = self._project(self.base[i:min(i + 65536, self.n)], W).half()
            _, _, after, after5 = self._loo(vkeys, vrows, vtruth, cand)
            del cand
            result = {"before_top1": before, "after_top1": after, "before_top5": before5, "after_top5": after5,
                      "accepted": after >= before + 0.005 and after5 >= before5 - 0.005, "val_keys": list(vkeys)}
            if result["accepted"]:
                self.W = W
                self.val_keys = list(vkeys)
                self._rebuild_vecs()
                if self.neg_base is not None:
                    self.neg_vecs = self._project(self.neg_base.float()).half()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        logger.info(f"Projector training: top-1 {before:.1%} -> {after:.1%} ({'kept' if result['accepted'] else 'discarded'})")
        return result


# =================================================================================
#  BRAIN (database + backbone + index, used by the UI)
# =================================================================================
class Brain:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.gpu_name = torch.cuda.get_device_name(0) if self.cuda else "CPU"
        bb_cls = BACKBONES.get(settings.get("backbone"), SigLIP2Backbone)
        self.backbone: Backbone = bb_cls()
        self.model_id = self.backbone.model_id
        self.db = BrainDB(Cfg.DB_FILE)
        self.index = StyleIndex(self.device, self.backbone.dim, settings["knn_k"], settings["knn_tau"])
        self.embedder: Optional[Embedder] = None
        self.ready = False
        self.load_error = ""
        self._key_cache: Dict[str, Tuple[int, int, str]] = {}
        self._emb_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._emb_cache_lock = threading.Lock()

    # ----- startup -----
    @property
    def projector_file(self) -> str:
        return os.path.join(APP_DIR, f"projector_{re.sub(r'[^A-Za-z0-9_.-]', '_', self.model_id)}.pt")

    def _load_projector(self) -> Optional[dict]:
        if not os.path.exists(self.projector_file):
            return None
        try:
            p = torch.load(self.projector_file, map_location="cpu", weights_only=True)
            if p.get("model") == self.model_id and tuple(p["W"].shape) == (self.backbone.dim, self.backbone.dim):
                return p
            logger.warning("Projector belongs to a different backbone; ignoring it.")
        except Exception as e:
            logger.error(f"Could not load projector: {e}")
        return None

    def load(self, progress: Callable[[int, int, str], None]):
        if not self.cuda:
            logger.warning("CUDA unavailable - running on CPU (slow).")
        progress(0, 3, f"Loading {self.backbone.label} on {self.gpu_name}...")
        t = time.time()
        self.backbone.load(self.device)
        self.embedder = Embedder(self.backbone, self.device, self.settings)
        self.embedder.warmup()  # captures CUDA graphs so the first real file is fast
        logger.info(f"Backbone {self.model_id} ready in {time.time() - t:.1f}s on {self.gpu_name}")
        self.reload_index(progress)
        self.ready = True

    def reload_index(self, progress: Optional[Callable] = None):
        self.index.build(self.db, self.model_id, self._load_projector(), progress)
        if progress:
            progress(1, 1, "Calibrating confidence...")
        self.index.calibrate()

    # ----- file identity -----
    def key_for(self, path: str, st: Optional[os.stat_result] = None) -> str:
        st = st or os.stat(path)
        nc = os.path.normcase(path)
        hit = self._key_cache.get(nc) or self.db.lookup_path(path)
        if hit and hit[0] == st.st_size and hit[1] == st.st_mtime_ns:
            self._key_cache[nc] = hit
            return hit[2]
        key = fingerprint_file(path)
        self._key_cache[nc] = (st.st_size, st.st_mtime_ns, key)
        self.db.put_paths([(path, st.st_size, st.st_mtime_ns, key)])
        return key

    def note_move(self, old: str, new: str):
        hit = self._key_cache.pop(os.path.normcase(old), None)
        self.db.move_path(old, new)
        if hit:
            try:
                st = os.stat(new)
                self._key_cache[os.path.normcase(new)] = (st.st_size, st.st_mtime_ns, hit[2])
            except OSError:
                pass

    # ----- embeddings -----
    def _cache_put(self, key: str, vec: np.ndarray):
        with self._emb_cache_lock:
            self._emb_cache[key] = vec
            self._emb_cache.move_to_end(key)
            while len(self._emb_cache) > 512:
                self._emb_cache.popitem(last=False)

    def cached_vector(self, key: str) -> Optional[np.ndarray]:
        with self._emb_cache_lock:
            v = self._emb_cache.get(key)
        if v is not None:
            return v
        v = self.db.get_embeddings([key], self.model_id).get(key)
        if v is not None and v.shape[0] == self.backbone.dim:
            self._cache_put(key, v)
            return v
        return None

    def embed_file(self, path: str, frames: Optional[List[np.ndarray]] = None) -> Tuple[str, np.ndarray]:
        """Key + embedding for one file, from cache when possible."""
        key = self.key_for(path)
        vec = self.cached_vector(key)
        if vec is None:
            if frames is None:
                frames, _ = load_media_frames(path)
            vec = self.embedder.embed_frames(frames)
            self.db.put_embeddings([(key, self.model_id, vec.tobytes())])
            self._cache_put(key, vec)
        return key, vec

    def predict(self, key: str, vec: np.ndarray, topn: int = 5) -> Prediction:
        return self.index.predict(vec[None], [key], topn)[0]

    # ----- learning from the human -----
    def learn(self, key: str, artist_str: str, vec: Optional[np.ndarray], verified: bool = True,
              rejected: Optional[str] = None):
        artists = split_artists(artist_str)
        self.db.set_labels_many([(key, artists, verified)])
        if vec is not None:
            self.index.add(key, artists, vec, verified)
        if rejected and rejected.lower() not in {a.lower() for a in artists}:
            self.reject(key, rejected, vec)

    def reject(self, key: str, artist: str, vec: Optional[np.ndarray]):
        self.db.add_negative(key, artist)
        self.index.add_negative(key, artist, vec)

    def forget_labels(self, key: str, rejected: Optional[str] = None, vec: Optional[np.ndarray] = None):
        self.db.set_labels_many([(key, [], False)])
        self.index.remove_key(key)
        if rejected:
            self.reject(key, rejected, vec)

    def merge_artists(self, olds: List[str], target: str) -> int:
        n = self.db.rename_artists(olds, target)
        self.index.merge(olds, target)
        return n

    def delete_artists(self, names: List[str]) -> int:
        n = self.db.delete_artists(names)
        self.index.delete(names)
        return n

    def reset(self):
        self.db.reset_brain()
        if os.path.exists(self.projector_file):
            os.remove(self.projector_file)
        self.index.clear()

    def artist_counts(self) -> Dict[str, int]:
        return self.db.artist_counts()

    # ----- bulk operations -----
    def resolve_keys(self, paths: List[str], stop: Callable[[], bool], progress: Callable = None,
                     stats: Optional[Dict[str, Tuple[int, int]]] = None) -> Dict[str, Optional[str]]:
        """path -> content key using the path cache; unknown files are left as None (keyed while decoding)."""
        cache = self.db.load_path_cache()
        out: Dict[str, Optional[str]] = {}
        for i, p in enumerate(paths):
            if i % 2000 == 0:
                if stop():
                    break
                if progress:
                    progress(i, len(paths))
            st = stats.get(p) if stats is not None else None
            if st is None:
                try:
                    s = os.stat(p)
                    st = (s.st_size, s.st_mtime_ns)
                except OSError:
                    continue
            hit = cache.get(os.path.normcase(p))
            out[p] = hit[2] if hit and hit[0] == st[0] and hit[1] == st[1] else None
        return out

    def embed_many(self, path_keys: Dict[str, Optional[str]], stop: Callable[[], bool],
                   on_item: Callable[[EmbedResult], None], thumb_every: float = 0.25,
                   workers: Optional[int] = None, vid_workers: int = 3) -> Dict[str, Tuple[str, np.ndarray]]:
        """Ensures every path has an embedding. Returns path -> (key, vec)."""
        result: Dict[str, Tuple[str, np.ndarray]] = {}
        known = [k for k in path_keys.values() if k]
        have = self.db.get_embeddings(known, self.model_id) if known else {}
        todo = []
        for p, k in path_keys.items():
            if k and k in have and have[k].shape[0] == self.backbone.dim:
                result[p] = (k, have[k])
                on_item(EmbedResult(p, k, have[k]))
            else:
                todo.append((p, k))
        emb_rows, path_rows = [], []
        last_flush = time.time()

        def flush():
            nonlocal emb_rows, path_rows, last_flush
            self.db.put_embeddings(emb_rows)
            self.db.put_paths(path_rows)
            emb_rows, path_rows, last_flush = [], [], time.time()

        for r in self.embedder.iter_files(todo, stop, thumb_every, workers, vid_workers):
            if r.vec is not None:
                result[r.path] = (r.key, r.vec)
                emb_rows.append((r.key, self.model_id, r.vec.tobytes()))
                if r.stat:
                    path_rows.append((r.path, r.stat[0], r.stat[1], r.key))
                if len(emb_rows) >= 512 or time.time() - last_flush > 3:
                    flush()
            else:
                logger.warning(f"Skipped {r.path}: {r.error}")
            on_item(r)
        flush()
        return result

    def sync_library(self, folder: str, stop: Callable[[], bool], progress: Callable) -> Dict[str, int]:
        """Learns every "[artist] - name" file under folder and forgets labels of files that were untagged."""
        progress(0, 1, "Scanning folder...", None, None)
        stats = list_media_stat(folder)
        files = list(stats)
        tagged = {}
        untagged = []
        for p in files:
            parsed = parse_tagged_name(os.path.basename(p))
            if parsed and split_artists(parsed[0]):
                tagged[p] = split_artists(parsed[0])
            else:
                untagged.append(p)
        total = len(tagged)
        progress(0, max(total, 1), f"Found {total:,} tagged files - checking which are already learned...", None, None)
        keys = self.resolve_keys(list(tagged), stop, stats=stats)
        n_known = sum(1 for k in keys.values() if k)
        progress(0, max(total, 1), f"{n_known:,} already known, {total - n_known:,} new - starting GPU...", None, None)
        done = [0]
        t0 = time.time()
        errors = [0]
        # Labels (filenames are the ground truth) are saved as we go, so a cancelled sync keeps what it learned.
        existing = self.db.all_label_keys()
        pending: List[Tuple[str, List[str], bool]] = []
        n_changed = [0]

        def save_labels():
            if pending:
                self.db.set_labels_many(list(pending))
                n_changed[0] += len(pending)
                pending.clear()

        def on_item(r: EmbedResult):
            done[0] += 1
            if r.vec is None:
                errors[0] += 1
            elif r.key:
                want = tagged.get(r.path, [])
                if {a.lower() for a in want} != existing.get(r.key, set()):
                    pending.append((r.key, want, False))
                    existing[r.key] = {a.lower() for a in want}
                    if len(pending) >= 2000:
                        save_labels()
            if done[0] % 25 == 0 or r.thumb is not None or done[0] == total:
                progress(done[0], total, os.path.basename(r.path), r.thumb, time.time() - t0)

        embedded = self.embed_many(keys, stop, on_item)
        save_labels()
        if stop():
            self.reload_index()
            return {"tagged": total, "learned": len(embedded), "errors": errors[0], "cancelled": 1}
        changes = []
        # Files that were untagged outside the app lose their stale labels.
        ukeys = self.resolve_keys(untagged, stop, stats=stats)
        for p, k in ukeys.items():
            if k and k in existing and k not in {v[0] for v in embedded.values()}:
                changes.append((k, [], False))
        progress(total, max(total, 1), f"Saving {len(changes):,} label changes...", None, None)
        self.db.set_labels_many(changes)
        progress(total, max(total, 1), "Rebuilding style memory...", None, None)
        self.reload_index()
        return {"tagged": total, "learned": len(embedded), "changed": n_changed[0] + len(changes), "errors": errors[0]}

    def train(self, progress: Callable, stop: Callable[[], bool]) -> Dict[str, Any]:
        res = self.index.train_projector(progress=progress, stop=stop)
        if res.get("accepted"):
            torch.save({"W": self.index.W.cpu(), "mean": self.index.mean.cpu(), "model": self.model_id,
                        "val_keys": res["val_keys"],
                        "metrics": {"trained_top1": float(res["after_top1"]), "trained_at": time.time()}},
                       self.projector_file)
            self.index.calibrate()
        return res


# =================================================================================
#  OLED THEME & WIDGET HELPERS
# =================================================================================
class T:
    BG = "#000000"
    SURFACE = "#080808"
    SURFACE_2 = "#0f0f0f"
    SURFACE_3 = "#171717"
    HOVER = "#1f1f1f"
    BORDER = "#1c1c1c"
    BORDER_HI = "#2c2c2c"
    TEXT = "#f4f4f5"
    TEXT_2 = "#a1a1aa"
    TEXT_3 = "#52525b"
    ACCENT = "#8b5cf6"
    ACCENT_HI = "#a78bfa"
    ACCENT_DIM = "#1e1638"
    OK = "#22c55e"
    OK_HI = "#4ade80"
    OK_DIM = "#0c2416"
    WARN = "#f59e0b"
    DANGER = "#ef4444"
    DANGER_HI = "#f87171"
    DANGER_DIM = "#2a0d0d"
    CYAN = "#22d3ee"
    FAMILY = "Segoe UI"
    MONO = "Cascadia Mono"

    @classmethod
    def init_fonts(cls, root):
        fams = set(tkfont.families(root))
        for f in ("Segoe UI Variable Text", "Segoe UI", "Inter", "Helvetica"):
            if f in fams:
                cls.FAMILY = f
                break
        for f in ("Cascadia Mono", "Cascadia Code", "Consolas", "Courier New"):
            if f in fams:
                cls.MONO = f
                break

    @classmethod
    def font(cls, size=13, weight="normal", mono=False):
        return ctk.CTkFont(family=cls.MONO if mono else cls.FAMILY, size=size, weight=weight)


def conf_color(c: float) -> str:
    return T.OK if c >= 0.75 else (T.ACCENT_HI if c >= 0.5 else (T.WARN if c >= 0.25 else T.TEXT_3))


def btn(master, text, command=None, kind="ghost", width=0, height=34, size=13, **kw) -> ctk.CTkButton:
    styles = {
        "primary": dict(fg_color=T.ACCENT, hover_color=T.ACCENT_HI, text_color="#ffffff", border_width=0),
        "ghost": dict(fg_color=T.SURFACE_2, hover_color=T.HOVER, text_color=T.TEXT, border_width=1, border_color=T.BORDER_HI),
        "subtle": dict(fg_color="transparent", hover_color=T.SURFACE_3, text_color=T.TEXT_2, border_width=0),
        "danger": dict(fg_color=T.DANGER_DIM, hover_color="#3d1212", text_color=T.DANGER_HI, border_width=1, border_color="#4a1616"),
        "ok": dict(fg_color=T.OK_DIM, hover_color="#123320", text_color=T.OK_HI, border_width=1, border_color="#164d2a"),
        "accent": dict(fg_color=T.ACCENT_DIM, hover_color="#2a1f4d", text_color=T.ACCENT_HI, border_width=1, border_color="#3b2a70"),
    }
    st = dict(styles[kind])
    st.update(kw)
    return ctk.CTkButton(master, text=text, command=command, width=width, height=height, corner_radius=8,
                         font=T.font(size, "bold" if kind == "primary" else "normal"), **st)


def card(master, **kw) -> ctk.CTkFrame:
    opts = dict(fg_color=T.SURFACE, corner_radius=14, border_width=1, border_color=T.BORDER)
    opts.update(kw)
    return ctk.CTkFrame(master, **opts)


def label(master, text="", size=13, color=None, weight="normal", mono=False, **kw) -> ctk.CTkLabel:
    return ctk.CTkLabel(master, text=text, font=T.font(size, weight, mono), text_color=color or T.TEXT, **kw)


def section_title(master, text) -> ctk.CTkLabel:
    return label(master, text.upper(), 11, T.TEXT_3, "bold")


def ctk_image(pil: Image.Image, size: Optional[Tuple[int, int]] = None) -> ctk.CTkImage:
    size = size or pil.size
    return ctk.CTkImage(light_image=pil, dark_image=pil, size=size)


def dark_menu(master) -> tk.Menu:
    return tk.Menu(master, tearoff=0, bg=T.SURFACE_2, fg=T.TEXT, activebackground=T.ACCENT_DIM,
                   activeforeground=T.TEXT, bd=0, relief="flat", font=(T.FAMILY, 10))


def placeholder_tile(size: int, text: str = "") -> Image.Image:
    im = Image.new("RGB", (size, size), (12, 12, 12))
    if text:
        d = ImageDraw.Draw(im)
        d.text((size // 2, size // 2), text, fill=(90, 90, 98), anchor="mm")
    return im


class GhostEntry(ctk.CTkFrame):
    """Large entry with inline ghost-text completion and a live suggestion list.

    Typing shows the best match as grey ghost text plus a list of matches (most-used first).
    Tab or Right (at the end) accepts.  Up/Down walk the list.  Enter after walking the list accepts the row;
    otherwise Enter is left to the owner (e.g. to tag).  Esc closes the list.
    """
    ROWS = 7

    def __init__(self, master, placeholder: str, complete: Callable[[str], Optional[str]], height=50, size=17,
                 options: Optional[Callable[[str, int], List[Tuple[str, str, int]]]] = None, **kw):
        super().__init__(master, fg_color="transparent")
        self.complete = complete
        self.options_fn = options
        self.pending: Optional[str] = None
        self.opts: List[Tuple[str, str, int]] = []
        self.sel = 0
        self.navigated = False
        self.entry = ctk.CTkEntry(self, placeholder_text=placeholder, height=height, corner_radius=10,
                                  font=T.font(size), fg_color=T.SURFACE_2, border_color=T.BORDER_HI,
                                  border_width=1, text_color=T.TEXT, placeholder_text_color=T.TEXT_3, **kw)
        self.entry.pack(fill="x")
        inner = self.entry._entry
        self.ghost = tk.Label(self.entry, text="", bg=T.SURFACE_2, fg=T.TEXT_3, bd=0, padx=0, pady=0,
                              font=inner.cget("font"))
        self._measure = tkfont.Font(font=inner.cget("font"))
        self.ghost.bind("<Button-1>", lambda e: (inner.focus_set(), self.accept()))
        self._drop: Optional[tk.Frame] = None
        self._rows: List[Tuple[tk.Frame, tk.Label, tk.Label]] = []
        inner.bind("<KeyRelease>", self._on_key, add="+")
        inner.bind("<Down>", lambda e: self._move(1), add="+")
        inner.bind("<Up>", lambda e: self._move(-1), add="+")
        inner.bind("<Right>", self._on_right, add="+")
        inner.bind("<Escape>", self._on_escape, add="+")
        inner.bind("<Return>", self._on_return, add="+")
        inner.bind("<FocusIn>", lambda e: self.entry.configure(border_color=T.ACCENT), add="+")
        inner.bind("<FocusOut>", self._on_focus_out, add="+")
        self.bind("<Destroy>", lambda e: self._destroy_drop() if e.widget is self else None, add="+")

    # ----- public -----
    def bind_key(self, seq, fn):
        self.entry._entry.bind(seq, fn, add="+")

    def get(self) -> str:
        return self.entry.get()

    def set(self, text: str):
        self.entry.delete(0, tk.END)
        if text:
            self.entry.insert(0, text)
        try:
            self.entry._entry.icursor(tk.END)
            self.entry._entry.xview_moveto(1.0)
        except tk.TclError:
            pass
        self.hide()

    def focus(self):
        self.entry._entry.focus_set()

    def hide(self):
        self.pending = None
        self.opts = []
        self.navigated = False
        self.ghost.place_forget()
        if self._drop is not None:
            self._drop.place_forget()

    def accept(self) -> bool:
        if self.opts and 0 <= self.sel < len(self.opts):
            self.set(self.opts[self.sel][0])
            return True
        if self.pending:
            self.set(self.pending)
            return True
        return False

    def refresh(self):
        txt = self.entry.get()
        if not txt.strip():
            self.hide()
            return
        if self.options_fn is not None:
            self.opts = self.options_fn(txt, self.ROWS)
        else:
            c = self.complete(txt)
            self.opts = [(c, c, 0)] if c else []
        if len(self.opts) == 1 and self.opts[0][0].lower() == txt.lower():
            self.opts = []  # already fully typed
        self.sel = 0
        self.navigated = False
        self._render()

    # ----- internals -----
    def _render(self):
        self._show_ghost()
        self._show_drop()

    def _show_ghost(self):
        txt = self.entry.get()
        cand = self.opts[self.sel][0] if self.opts else None
        if not cand or len(cand) <= len(txt) or not cand.lower().startswith(txt.lower()):
            self.pending = None
            self.ghost.place_forget()
            return
        self.pending = cand
        inner = self.entry._entry
        try:
            if inner.index(tk.INSERT) != len(txt):
                self.ghost.place_forget()
                return
            # Right edge of the last character, as the entry actually draws it (handles scrolling too).
            bx = inner.tk.call(inner._w, "bbox", len(txt) - 1)
            end_x = int(bx[0]) + int(bx[2])
        except (tk.TclError, IndexError, ValueError):
            end_x = self._measure.measure(txt)
        if inner.winfo_width() - end_x < 24:
            self.ghost.place_forget()
            return
        self.ghost.configure(text=cand[len(txt):])
        self.ghost.place(in_=inner, x=end_x + 1, rely=0.5, anchor="w")

    def _scale(self) -> float:
        try:
            return self.entry._get_widget_scaling()
        except Exception:
            return 1.0

    def _ensure_drop(self):
        if self._drop is not None:
            return
        top = self.winfo_toplevel()
        s = self._scale()
        f_name = tkfont.Font(family=T.FAMILY, size=-int(15 * s))
        f_count = tkfont.Font(family=T.MONO, size=-int(12 * s))
        self._drop = tk.Frame(top, bg=T.BORDER_HI, bd=0, highlightthickness=0)
        self._drop._fonts = (f_name, f_count)
        body = tk.Frame(self._drop, bg=T.SURFACE_2, bd=0)
        body.pack(fill="both", expand=True, padx=1, pady=1)
        for i in range(self.ROWS):
            r = tk.Frame(body, bg=T.SURFACE_2, bd=0, cursor="hand2")
            n = tk.Label(r, text="", bg=T.SURFACE_2, fg=T.TEXT, font=f_name, anchor="w", bd=0,
                         padx=int(12 * s), pady=int(5 * s))
            c = tk.Label(r, text="", bg=T.SURFACE_2, fg=T.TEXT_3, font=f_count, anchor="e", bd=0, padx=int(12 * s))
            c.pack(side="right")
            n.pack(side="left", fill="x", expand=True)
            for w in (r, n, c):
                w.bind("<Button-1>", lambda e, k=i: self._click(k))
                w.bind("<Enter>", lambda e, k=i: self._hover(k))
            self._rows.append((r, n, c))

    def _show_drop(self):
        if not self.opts or not self.entry._entry.winfo_ismapped():
            if self._drop is not None:
                self._drop.place_forget()
            return
        self._ensure_drop()
        top = self.winfo_toplevel()
        for i, (r, n, c) in enumerate(self._rows):
            if i < len(self.opts):
                _, name, cnt = self.opts[i]
                on = i == self.sel
                bg = T.ACCENT_DIM if on else T.SURFACE_2
                for w in (r, n, c):
                    w.configure(bg=bg)
                n.configure(text=name, fg=T.ACCENT_HI if on else T.TEXT)
                c.configure(text=f"{cnt:,}" if cnt else "")
                r.pack(fill="x")
            else:
                r.pack_forget()
        self._drop.update_idletasks()
        s = self._scale()
        ex = self.entry.winfo_rootx() - top.winfo_rootx()
        ey = self.entry.winfo_rooty() - top.winfo_rooty()
        w = max(int(320 * s), min(self.entry.winfo_width(), int(560 * s)))
        h = self._drop.winfo_reqheight()
        gap = int(6 * s)
        if ey - h - gap >= 0:  # open upwards: the composer sits at the bottom of the window
            self._drop.place(x=ex, y=ey - gap, width=w, anchor="sw")
        else:
            self._drop.place(x=ex, y=ey + self.entry.winfo_height() + gap, width=w, anchor="nw")
        self._drop.lift()

    def _destroy_drop(self):
        if self._drop is not None:
            try:
                self._drop.destroy()
            except tk.TclError:
                pass
            self._drop = None
            self._rows = []

    def _move(self, d: int):
        if not self.opts:
            return None
        self.sel = (self.sel + d) % len(self.opts)
        self.navigated = True
        self._render()
        return "break"

    def _hover(self, k: int):
        if k < len(self.opts) and k != self.sel:
            self.sel = k
            self._render()

    def _click(self, k: int):
        if k < len(self.opts):
            self.set(self.opts[k][0])
            self.focus()
            self.entry._entry.event_generate("<<Accepted>>")

    def _on_right(self, e):
        inner = self.entry._entry
        if self.pending and inner.index(tk.INSERT) == len(self.entry.get()):
            self.accept()
            inner.event_generate("<<Accepted>>")
            return "break"
        return None

    def _on_escape(self, e):
        if self.opts:
            self.hide()
            return "break"
        return None

    def _on_return(self, e):
        # Enter after walking the list picks that row; a plain Enter is left to the owner (tag).
        if self.navigated and self.opts:
            self.accept()
            self.entry._entry.event_generate("<<Accepted>>")
            return "break"
        self.hide()
        return None

    def _on_focus_out(self, e):
        self.entry.configure(border_color=T.BORDER_HI)
        self.after(120, self._hide_if_unfocused)

    def _hide_if_unfocused(self):
        try:
            if self.focus_get() is not self.entry._entry:
                self.hide()
        except (tk.TclError, KeyError):
            self.hide()

    def _on_key(self, e):
        if e.keysym in ("Tab", "Return", "KP_Enter", "Escape", "Up", "Down", "Right", "Left", "Shift_L", "Shift_R",
                        "Control_L", "Control_R", "Alt_L", "Alt_R", "Home", "End"):
            return
        if e.state & 0x4 and e.keysym.lower() not in ("v", "x", "backspace"):  # other Ctrl shortcuts
            return
        self.refresh()


class ArtistCompleter:
    """Prefix autocomplete over known artists, preferring the most-used artist."""

    def __init__(self):
        self.names: List[str] = []
        self.lower: List[str] = []
        self.counts: Dict[str, int] = {}

    def update(self, counts: Dict[str, int], extra: Iterable[str] = ()):
        merged: Dict[str, Tuple[str, int]] = {}
        for n in extra:
            if n and n.lower() not in merged:
                merged[n.lower()] = (n, 0)
        for n, c in counts.items():
            if n:
                merged[n.lower()] = (n, c)
        items = sorted(merged.items())
        self.lower = [k for k, _ in items]
        self.names = [v[0] for _, v in items]
        self.counts = {v[0]: v[1] for _, v in items}

    def complete_one(self, prefix: str, exclude: Set[str] = frozenset()) -> Optional[str]:
        import bisect
        p = prefix.strip().lower()
        if not p:
            return None
        i = bisect.bisect_left(self.lower, p)
        best, best_c = None, -1
        while i < len(self.lower) and self.lower[i].startswith(p):
            n = self.names[i]
            c = self.counts.get(n, 0)
            if n.lower() not in exclude and (c > best_c or (c == best_c and best and len(n) < len(best))):
                best, best_c = n, c
            i += 1
            if i > 5000 + bisect.bisect_left(self.lower, p):
                break
        return best

    def options(self, text: str, n: int = 7) -> List[Tuple[str, str, int]]:
        """Ranked matches for the last artist in 'a, b, partial': (full text, name, files).
        Prefix matches first (most-used first), then names that merely contain the text."""
        import bisect
        parts = re.split(r"([,&;+])", text)
        last = parts[-1]
        core = last.lstrip()
        p = core.strip().lower()
        if not p:
            return []
        done = {x.strip().lower() for x in parts[:-1:2] if x.strip()}
        head = "".join(parts[:-1]) + last[:len(last) - len(core)]
        lo = bisect.bisect_left(self.lower, p)
        hi = bisect.bisect_left(self.lower, p + "\uffff")
        pref = [i for i in range(lo, min(hi, lo + 20000)) if self.lower[i] not in done]
        pref.sort(key=lambda i: (-self.counts.get(self.names[i], 0), len(self.names[i]), self.lower[i]))
        picked = pref[:n]
        if len(picked) < n and len(p) >= 2:
            seen = set(picked)
            more = [i for i, l in enumerate(self.lower) if p in l and i not in seen and l not in done]
            more.sort(key=lambda i: (-self.counts.get(self.names[i], 0), len(self.names[i])))
            picked += more[:n - len(picked)]
        return [(head + self.names[i], self.names[i], self.counts.get(self.names[i], 0)) for i in picked]

    def bump(self, names: Iterable[str]):
        """Counts a just-tagged artist immediately (no database round trip)."""
        import bisect
        for nm in names:
            nm = nm.strip()
            if not nm:
                continue
            k = nm.lower()
            i = bisect.bisect_left(self.lower, k)
            if i < len(self.lower) and self.lower[i] == k:
                known = self.names[i]
                self.counts[known] = self.counts.get(known, 0) + 1
            else:
                self.lower.insert(i, k)
                self.names.insert(i, nm)
                self.counts[nm] = 1

    def complete_text(self, text: str) -> Optional[str]:
        """Completes the last artist in 'a, b, partial'."""
        parts = re.split(r"([,&;+])", text)
        last = parts[-1]
        core = last.lstrip()
        if not core.strip():
            return None
        done = {p.strip().lower() for p in parts[:-1:2] if p.strip()}
        s = self.complete_one(core, done)
        if not s:
            return None
        lead = last[:len(last) - len(core)]
        return "".join(parts[:-1]) + lead + s

    def canonical(self, name: str) -> str:
        """Known spelling for a case-insensitive match, else the input."""
        import bisect
        k = name.strip().lower()
        i = bisect.bisect_left(self.lower, k)
        if i < len(self.lower) and self.lower[i] == k:
            return self.names[i]
        return name.strip()


# =================================================================================
#  TASK OVERLAY (long jobs: library sync, training, focus scan)
# =================================================================================
class TaskOverlay(ctk.CTkFrame):
    STRIP = 8

    def __init__(self, app: "VtaggerApp", title: str, subtitle: str, on_cancel: Callable, accent: str = T.ACCENT):
        super().__init__(app, fg_color=T.BG, corner_radius=0)
        self.app = app
        self.t0 = time.time()
        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.lift()
        wrap = ctk.CTkFrame(self, fg_color="transparent")
        wrap.place(relx=0.5, rely=0.46, anchor="center")
        label(wrap, title, 28, weight="bold").pack(anchor="w")
        self.sub = label(wrap, subtitle, 14, T.TEXT_2, anchor="w", justify="left")
        self.sub.pack(anchor="w", pady=(2, 22))
        self.bar = ctk.CTkProgressBar(wrap, width=760, height=6, corner_radius=3, progress_color=accent, fg_color=T.SURFACE_3)
        self.bar.pack(fill="x")
        self.bar.set(0)
        stats = ctk.CTkFrame(wrap, fg_color="transparent")
        stats.pack(fill="x", pady=(18, 6))
        self.stat = {}
        for i, name in enumerate(("Progress", "Speed", "Elapsed", "Remaining")):
            box = card(stats, fg_color=T.SURFACE, corner_radius=12)
            box.grid(row=0, column=i, padx=(0 if i == 0 else 10, 0), sticky="ew")
            stats.grid_columnconfigure(i, weight=1)
            section_title(box, name).pack(anchor="w", padx=14, pady=(10, 0))
            v = label(box, "--", 20, weight="bold", mono=True)
            v.pack(anchor="w", padx=14, pady=(0, 10))
            self.stat[name] = v
        self.extra = label(wrap, "", 14, accent, weight="bold")
        self.extra.pack(anchor="w", pady=(8, 0))
        self.detail = label(wrap, "", 12, T.TEXT_3, mono=True, anchor="w")
        self.detail.pack(anchor="w", fill="x", pady=(2, 14))
        strip = ctk.CTkFrame(wrap, fg_color="transparent", height=Cfg.THUMB // 1.4)
        strip.pack(fill="x")
        self._thumb_lbls = []
        self._thumb_imgs: Deque[ctk.CTkImage] = deque(maxlen=self.STRIP)
        ts = int(Cfg.THUMB / 1.4)
        for i in range(self.STRIP):
            l = ctk.CTkLabel(strip, text="", width=ts, height=ts, fg_color=T.SURFACE, corner_radius=10)
            l.grid(row=0, column=i, padx=(0 if i == 0 else 8, 0))
            self._thumb_lbls.append(l)
        self.cancel_btn = btn(wrap, "Cancel", on_cancel, "danger", width=140, height=38)
        self.cancel_btn.pack(anchor="w", pady=(24, 0))
        self._samples: Deque[Tuple[float, int]] = deque()
        self._tick()

    def _tick(self):
        if not self.winfo_exists():
            return
        self.stat["Elapsed"].configure(text=format_time(time.time() - self.t0))
        self.after(1000, self._tick)

    def update_progress(self, done: int, total: int, detail: str = "", eta: Optional[float] = None,
                        thumb: Optional[Image.Image] = None, extra: Optional[str] = None):
        now = time.time()
        el = now - self.t0
        frac = done / total if total else 0
        self.bar.set(min(1.0, frac))
        self._samples.append((now, done))
        while len(self._samples) > 2 and now - self._samples[0][0] > 15:
            self._samples.popleft()
        t_a, d_a = self._samples[0]
        speed = (done - d_a) / (now - t_a) if now - t_a > 1.0 and done > d_a else 0
        if eta is None and speed > 0 and total:
            eta = (total - done) / speed
        self.stat["Progress"].configure(text=f"{done:,} / {total:,}" if total else f"{done:,}")
        self.stat["Speed"].configure(text=f"{speed:,.1f}/s" if speed else "--")
        self.stat["Elapsed"].configure(text=format_time(el))
        self.stat["Remaining"].configure(text=format_time(eta) if eta is not None else "--:--")
        if detail:
            self.detail.configure(text=detail[:120])
        if extra is not None:
            self.extra.configure(text=extra)
        if thumb is not None:
            ts = int(Cfg.THUMB / 1.4)
            t = thumb.copy()
            t.thumbnail((ts, ts))
            self._retired = list(self._thumb_imgs)
            self._thumb_imgs.appendleft(ctk_image(t))
            for lbl, img in zip(self._thumb_lbls, self._thumb_imgs):
                lbl.configure(image=img)
        self.app.set_taskbar(done, total)

    def set_status(self, text: str):
        self.detail.configure(text=text)


@dataclass
class Prepared:
    path: str
    preview: Optional[Image.Image] = None
    dims: str = ""
    key: Optional[str] = None
    vec: Optional[np.ndarray] = None
    error: str = ""


# =================================================================================
#  MAIN APPLICATION
# =================================================================================
class VtaggerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        T.init_fonts(self)
        self.configure(fg_color=T.BG)
        self.title(f"Vtagger {APP_VERSION}")
        self.geometry("1380x900")
        self.minsize(1080, 720)
        self.after(50, lambda: oled_titlebar(self))

        self.settings = Settings()
        self.brain = Brain(self.settings)
        self.completer = ArtistCompleter()
        self.counts = {"TOTAL_NAMED": 0, "TOTAL_SKIPPED": 0, "TOTAL_DELETED": 0}
        self.counts.update({k: v for k, v in read_json(Cfg.COUNTS_FILE, {}).items() if k in self.counts})
        hist = read_json(Cfg.HISTORY_FILE, [])
        self.history_names: List[str] = hist if isinstance(hist, list) else []
        self.taskbar = TaskbarProgress()

        self._ui_q: "queue.Queue[Tuple[Callable, tuple]]" = queue.Queue()
        self._latest: Dict[str, Tuple[Callable, tuple]] = {}
        self._latest_lock = threading.Lock()
        self._closing = False

        self.folder: Optional[str] = None
        self.files: List[str] = []
        self.idx = 0
        self.undo: Deque[tuple] = deque(maxlen=Cfg.HISTORY_MAX_LEN)
        self.busy = False
        self.tag_pool = ThreadPoolExecutor(1, thread_name_prefix="tag")  # renames + learning, strictly in order
        self.session_tagged = 0
        self.focus_artist: Optional[str] = None
        self.focus_candidates: List[Tuple[str, float]] = []
        self.focus_vectors: Dict[str, Tuple[str, np.ndarray]] = {}
        self.job_stop = threading.Event()
        self.job_active = False
        self.overlay: Optional[TaskOverlay] = None
        self.gallery: Optional["FocusGallery"] = None
        self.manager_win = None
        self.suggestions: List[Suggestion] = []
        self.prepared: Dict[str, Prepared] = {}
        self.prep_cv = threading.Condition()
        self.predict_pool = ThreadPoolExecutor(1, thread_name_prefix="predict")
        self._preview_pil: Optional[Image.Image] = None
        self._preview_img = None
        self._resize_job = None

        self._build_ui()
        self._bind_keys()
        self._set_controls(False)
        self.after(30, self._pump)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        threading.Thread(target=self._boot, daemon=True, name="boot").start()
        threading.Thread(target=self._prefetch_loop, daemon=True, name="prefetch").start()
        threading.Thread(target=self._index_loop, daemon=True, name="indexer").start()

    # ------------------------------------------------------------------ plumbing
    def post(self, fn: Callable, *args):
        """Run fn on the UI thread."""
        self._ui_q.put((fn, args))

    def post_latest(self, channel: str, fn: Callable, *args):
        """Like post, but only the newest call per channel survives (for progress updates)."""
        with self._latest_lock:
            self._latest[channel] = (fn, args)

    def _pump(self):
        try:
            with self._latest_lock:
                latest, self._latest = self._latest, {}
            for fn, args in latest.values():
                self._safe_call(fn, args)
            deadline = time.time() + 0.03
            while time.time() < deadline:
                try:
                    fn, args = self._ui_q.get_nowait()
                except queue.Empty:
                    break
                self._safe_call(fn, args)
        finally:
            if not self._closing:
                self.after(16, self._pump)

    def _safe_call(self, fn, args):
        try:
            fn(*args)
        except Exception:
            logger.error("UI callback failed:\n" + traceback.format_exc())

    def hwnd(self) -> int:
        try:
            return int(self.wm_frame(), 16)
        except Exception:
            return 0

    def set_taskbar(self, done: int, total: int):
        self.taskbar.set(self.hwnd(), done, total)

    def status(self, msg: str, ms: int = 0, color: str = None):
        self.status_lbl.configure(text=msg, text_color=color or T.TEXT_2)
        if ms:
            token = object()
            self._status_token = token
            self.after(ms, lambda: getattr(self, "_status_token", None) is token and self.status_lbl.configure(text=""))

    def run_bg(self, fn: Callable, done: Optional[Callable] = None, error_title: str = "Error",
               on_error: Optional[Callable] = None):
        def runner():
            try:
                res = fn()
                if done:
                    self.post(done, res)
            except Exception as e:
                logger.error(traceback.format_exc())
                if on_error:
                    self.post(on_error)
                self.post(messagebox.showerror, error_title, str(e))
        threading.Thread(target=runner, daemon=True).start()

    # ------------------------------------------------------------------ boot
    def _boot(self):
        try:
            self.brain.load(lambda d, t, m: self.post_latest("boot", self._boot_progress, d, t, m))
            self.post(self._on_ready)
        except Exception as e:
            logger.error(traceback.format_exc())
            self.brain.load_error = str(e)
            self.post(self._on_boot_failed, str(e))

    def _boot_progress(self, d, t, m):
        self.status(m)
        self.progress.set(d / t if t else 0)

    def _on_boot_failed(self, err: str):
        self.status(f"AI failed to load: {err}", color=T.DANGER)
        messagebox.showerror("AI engine failed", f"The vision model could not be loaded:\n\n{err}\n\n"
                             "Check your internet connection (first run downloads the model) and the log:\n" + Cfg.LOG_FILE)

    def _on_ready(self):
        self.refresh_artists()
        self._refresh_brain_card()
        dev = self.brain.gpu_name if self.brain.cuda else "CPU (no CUDA!)"
        self.gpu_chip.configure(text=f"  {dev}  ", text_color=T.OK_HI if self.brain.cuda else T.DANGER_HI,
                                fg_color=T.OK_DIM if self.brain.cuda else T.DANGER_DIM)
        self.title(f"Vtagger {APP_VERSION}  -  {self.brain.backbone.label}  -  {dev}")
        self._update_progress()
        self.status("Ready." + ("" if self.brain.index.size else "  Brain is empty: open your library and run Sync Library."), 8000)
        with self.prep_cv:
            self.prep_cv.notify_all()
        if self.files:
            self._show_current()
        self._check_session()

    def refresh_artists(self):
        counts = self.brain.artist_counts() if self.brain.ready else {}
        self.completer.update(counts, self.history_names)

    # ------------------------------------------------------------------ layout
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Top bar
        top = ctk.CTkFrame(self, fg_color=T.BG, corner_radius=0, height=64)
        top.grid(row=0, column=0, sticky="ew", padx=20, pady=(14, 8))
        top.grid_columnconfigure(3, weight=1)
        brand = ctk.CTkFrame(top, fg_color="transparent")
        brand.grid(row=0, column=0, padx=(0, 18))
        ctk.CTkLabel(brand, text="", width=12, height=12, corner_radius=6, fg_color=T.ACCENT).pack(side="left", padx=(0, 10))
        label(brand, "Vtagger", 20, weight="bold").pack(side="left")
        self.folder_btn = btn(top, "Open folder", self.select_folder, "ghost", width=150, height=36)
        self.folder_btn.grid(row=0, column=1)
        self.recursive_var = ctk.BooleanVar(value=bool(self.settings.get("recursive")))
        self.recursive_sw = ctk.CTkSwitch(top, text="Subfolders", variable=self.recursive_var, font=T.font(12),
                                          text_color=T.TEXT_2, progress_color=T.ACCENT, button_color=T.TEXT,
                                          button_hover_color="#ffffff", fg_color=T.SURFACE_3,
                                          command=self._on_recursive_toggle)
        self.recursive_sw.grid(row=0, column=2, padx=14)
        self.folder_lbl = label(top, "No folder open", 12, T.TEXT_3, anchor="w", mono=True)
        self.folder_lbl.grid(row=0, column=3, sticky="ew", padx=4)
        self.focus_chip = btn(top, "", self._open_gallery, "accent", height=32, size=12)
        self.gpu_chip = ctk.CTkLabel(top, text="  Loading AI...  ", font=T.font(11, "bold"), text_color=T.TEXT_2,
                                     fg_color=T.SURFACE_3, corner_radius=12, height=26)
        self.gpu_chip.grid(row=0, column=5, padx=10)
        acts = ctk.CTkFrame(top, fg_color="transparent")
        acts.grid(row=0, column=6)
        self.btn_focus = btn(acts, "Focus", self._open_focus_dialog, "ghost", width=86, height=36)
        self.btn_focus.pack(side="left", padx=3)
        self.btn_artists = btn(acts, "Artists", self.open_artist_manager, "ghost", width=86, height=36)
        self.btn_artists.pack(side="left", padx=3)
        self.btn_sync = btn(acts, "Sync library", self.start_sync, "ghost", width=110, height=36)
        self.btn_sync.pack(side="left", padx=3)
        self.btn_train = btn(acts, "Train", self.start_training, "ghost", width=76, height=36)
        self.btn_train.pack(side="left", padx=3)
        self.btn_more = btn(acts, "\u22ef", self._show_more_menu, "ghost", width=40, height=36)
        self.btn_more.pack(side="left", padx=(3, 0))

        # Body: preview + side panel
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=20)
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self.preview_card = card(body, fg_color=T.BG, border_color=T.BORDER)
        self.preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self.preview_lbl = ctk.CTkLabel(self.preview_card, text="Open a folder to start tagging", font=T.font(15),
                                        text_color=T.TEXT_3)
        self.preview_lbl.place(relx=0.5, rely=0.5, anchor="center")
        self.preview_lbl.bind("<Button-1>", lambda e: open_external(self.current_path()))
        self.preview_lbl.bind("<Button-3>", self._preview_menu)
        self.preview_card.bind("<Configure>", self._on_preview_resize)
        self.badge = ctk.CTkLabel(self.preview_card, text="", font=T.font(11, "bold"), fg_color=T.SURFACE_3,
                                  text_color=T.TEXT_2, corner_radius=8, height=24)

        side = ctk.CTkFrame(body, fg_color="transparent", width=360)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)
        side.grid_columnconfigure(0, weight=1)

        sug = card(side)
        sug.grid(row=0, column=0, sticky="ew")
        head = ctk.CTkFrame(sug, fg_color="transparent")
        head.pack(fill="x", padx=16, pady=(14, 6))
        section_title(head, "AI suggestions").pack(side="left")
        self.ai_state = label(head, "", 11, T.TEXT_3)
        self.ai_state.pack(side="right")
        self.sug_rows = []
        for i in range(5):
            row = ctk.CTkFrame(sug, fg_color=T.SURFACE_2, corner_radius=10, height=46)
            row.pack(fill="x", padx=12, pady=3)
            row.pack_propagate(False)
            num = label(row, f"{i + 1}", 11, T.TEXT_3, mono=True, width=18)
            num.pack(side="left", padx=(10, 4))
            box = ctk.CTkFrame(row, fg_color="transparent")
            box.pack(side="left", fill="both", expand=True, pady=6)
            name = label(box, "", 14, T.TEXT, weight="bold", anchor="w")
            name.pack(fill="x")
            bar = ctk.CTkProgressBar(box, height=3, corner_radius=2, fg_color=T.SURFACE_3, progress_color=T.ACCENT)
            bar.pack(fill="x", pady=(3, 0), padx=(0, 6))
            bar.set(0)
            pct = label(row, "", 13, T.TEXT_2, mono=True, width=56, anchor="e")
            pct.pack(side="right", padx=(0, 12))
            for w in (row, num, box, name, pct):
                w.bind("<Button-1>", lambda e, k=i: self._use_suggestion(k))
                w.bind("<Double-Button-1>", lambda e, k=i: self._use_suggestion(k, commit=True))
                w.bind("<Enter>", lambda e, r=row: r.configure(fg_color=T.SURFACE_3))
                w.bind("<Leave>", lambda e, r=row: r.configure(fg_color=T.SURFACE_2))
            self.sug_rows.append((row, name, bar, pct))
        label(sug, "Click to fill  -  double-click to tag  -  Alt+1..5", 11, T.TEXT_3).pack(anchor="w", padx=16, pady=(4, 12))

        info = card(side)
        info.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        section_title(info, "File").pack(anchor="w", padx=16, pady=(14, 2))
        self.file_name = label(info, "-", 13, T.TEXT, anchor="w", justify="left", wraplength=320)
        self.file_name.pack(anchor="w", fill="x", padx=16)
        self.file_meta = label(info, "", 12, T.TEXT_3, mono=True, anchor="w")
        self.file_meta.pack(anchor="w", padx=16, pady=(2, 14))

        brain_card = card(side)
        brain_card.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        section_title(brain_card, "Brain").pack(anchor="w", padx=16, pady=(14, 4))
        self.brain_lines = label(brain_card, "Loading...", 12, T.TEXT_2, anchor="w", justify="left", mono=True)
        self.brain_lines.pack(anchor="w", padx=16, pady=(0, 14))

        # Composer
        comp = card(self, corner_radius=16)
        comp.grid(row=2, column=0, sticky="ew", padx=20, pady=(14, 0))
        comp.grid_columnconfigure(0, weight=1)
        self.entry = GhostEntry(comp, "Artist name  (a, b for collabs)", self.completer.complete_text,
                                options=self.completer.options)
        self.entry.grid(row=0, column=0, sticky="ew", padx=(14, 10), pady=(14, 6))
        self.btn_tag = btn(comp, "Tag  ⏎", self.confirm, "primary", width=130, height=50, size=15)
        self.btn_tag.grid(row=0, column=1, padx=(0, 14), pady=(14, 6))
        self.rename_lbl = label(comp, "", 12, T.TEXT_3, mono=True, anchor="w")
        self.rename_lbl.grid(row=1, column=0, columnspan=2, sticky="ew", padx=18)
        row = ctk.CTkFrame(comp, fg_color="transparent")
        row.grid(row=2, column=0, columnspan=2, sticky="ew", padx=14, pady=(8, 14))
        self.btn_undo = btn(row, "Undo", self.undo_last, "ghost", width=90)
        self.btn_undo.pack(side="left")
        self.btn_skip = btn(row, "Skip", self.skip, "ghost", width=90)
        self.btn_skip.pack(side="left", padx=8)
        self.btn_delete = btn(row, "Delete", self.delete_current, "danger", width=90)
        self.btn_delete.pack(side="left")
        self.btn_tag_focus = btn(row, "Tag + find all by artist", lambda: self.confirm(focus=True), "accent", width=200)
        self.btn_tag_focus.pack(side="right")

        # Footer
        foot = ctk.CTkFrame(self, fg_color="transparent")
        foot.grid(row=3, column=0, sticky="ew", padx=22, pady=(10, 12))
        foot.grid_columnconfigure(1, weight=1)
        self.count_lbl = label(foot, "", 12, T.TEXT_2, mono=True)
        self.count_lbl.grid(row=0, column=0, sticky="w")
        self.progress = ctk.CTkProgressBar(foot, height=3, corner_radius=2, fg_color=T.SURFACE_3, progress_color=T.ACCENT)
        self.progress.grid(row=0, column=1, sticky="ew", padx=16)
        self.progress.set(0)
        self.status_lbl = label(foot, "Starting...", 12, T.TEXT_2, anchor="e")
        self.status_lbl.grid(row=0, column=2, sticky="e")
        self.index_lbl = label(foot, "", 11, T.TEXT_3, mono=True, anchor="e")
        self.index_lbl.grid(row=1, column=2, sticky="e", pady=(6, 0))
        label(foot, "Enter tag/skip   Tab complete   Alt+1-5 pick   Ctrl+Enter tag+focus   Ctrl+S skip   Ctrl+Z undo   "
                    "Ctrl+D delete   Ctrl+O open   Ctrl+C copy image", 11, T.TEXT_3).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.entry.bind_key("<KeyRelease>", lambda e: self._update_rename_preview())
        self.entry.bind_key("<<Accepted>>", lambda e: self._update_rename_preview())

    def _bind_keys(self):
        self.entry.bind_key("<Return>", lambda e: (self.confirm(), "break")[1])
        self.entry.bind_key("<Control-Return>", lambda e: (self.confirm(focus=True), "break")[1])
        self.entry.bind_key("<Tab>", self._on_tab)
        self.entry.bind_key("<KP_Enter>", lambda e: (self.confirm(), "break")[1])
        self.entry.bind_key("<Escape>", lambda e: (self.entry.set(""), self._update_rename_preview()))
        for i in range(5):
            self.bind_all(f"<Alt-Key-{i + 1}>", lambda e, k=i: self._use_suggestion(k))
        for letter, fn in (("s", self.skip), ("z", self.undo_last), ("d", self.delete_current),
                           ("o", self.select_folder), ("c", self._copy_preview)):
            for l in (letter, letter.upper()):
                self.bind_all(f"<Control-{l}>", lambda e, f=fn: self._shortcut(f, e))

    def _shortcut(self, fn, event=None):
        if self.overlay or self.gallery or (self.manager_win and self._manager_focused()):
            return
        w = getattr(event, "widget", None)
        if fn == self._copy_preview and isinstance(w, tk.Entry) and w.selection_present():
            return  # normal text copy
        try:
            if w is not None and w.winfo_toplevel() is not self:
                return  # keystroke belongs to another window (dialog, artist manager)
        except (AttributeError, tk.TclError):
            pass
        fn()
        return "break"

    def _manager_focused(self) -> bool:
        try:
            return bool(self.manager_win.winfo_exists()) and str(self.focus_get()).startswith(str(self.manager_win))
        except Exception:
            return False

    def _show_more_menu(self):
        m = dark_menu(self)
        m.add_command(label="Settings...", command=self.open_settings)
        m.add_command(label="Rescan folder", command=lambda: self.folder and self.load_folder(self.folder))
        m.add_separator()
        m.add_command(label="Recalibrate confidence", command=self._recalibrate)
        m.add_command(label="Open data folder", command=lambda: open_external(APP_DIR))
        m.add_separator()
        m.add_command(label="Reset brain (forget all artists)...", command=self.reset_brain)
        x, y = self.btn_more.winfo_rootx(), self.btn_more.winfo_rooty() + self.btn_more.winfo_height() + 4
        m.tk_popup(x - 180, y)

    def _set_controls(self, on: bool):
        s = "normal" if on else "disabled"
        for w in (self.btn_tag, self.btn_skip, self.btn_delete, self.btn_tag_focus):
            w.configure(state=s)
        self.btn_undo.configure(state="normal" if self.undo and not self.busy else "disabled")

    def _refresh_brain_card(self):
        idx = self.brain.index
        m = idx.metrics
        lines = [f"{idx.size:,} files   {len(idx.known_artists()):,} artists"]
        if "top1" in m:
            lines.append(f"self-test  {m['top1']:.0%} top-1  {m['top5']:.0%} top-5")
        lines.append(f"projector  {'trained' if idx.W is not None else 'not trained'}")
        self.brain_lines.configure(text="\n".join(lines))

    # ------------------------------------------------------------------ folder & queue
    def _on_recursive_toggle(self):
        self.settings["recursive"] = bool(self.recursive_var.get())
        self.settings.save()
        if self.folder and not self.focus_artist:
            self.load_folder(self.folder)

    def select_folder(self):
        if self.job_active:
            return
        path = filedialog.askdirectory(initialdir=self.folder or self.settings.get("last_folder") or None)
        if path:
            self.exit_focus(silent=True)
            self.load_folder(os.path.normpath(path))

    def load_folder(self, path: str):
        self.folder = path
        self.settings["last_folder"] = path
        self.settings.save()
        self.folder_lbl.configure(text=path)
        self.status("Scanning folder...")
        self._set_controls(False)
        rec = bool(self.recursive_var.get())
        self._scan_token = token = object()

        def scan():
            files = [p for p in list_media(path, rec) if not parse_tagged_name(os.path.basename(p))]
            random.shuffle(files)
            return files
        self.run_bg(scan, lambda files: self._on_scanned(files, token), "Scan failed")

    def _on_scanned(self, files: List[str], token=None):
        if token is not None and token is not getattr(self, "_scan_token", None):
            return  # a newer folder was opened while this scan ran
        self.files = files
        self.idx = 0
        self.undo.clear()
        self.prepared.clear()
        self._update_progress()
        if not files:
            self._show_empty("No untagged files in this folder.")
            self.status("No untagged files found.", 5000)
            return
        self.status(f"{len(files):,} untagged files", 4000)
        self._show_current()

    def current_path(self) -> Optional[str]:
        return self.files[self.idx] if 0 <= self.idx < len(self.files) else None

    def _show_empty(self, text: str):
        self._preview_pil = None
        self.preview_lbl.configure(image=None, text=text)
        self.badge.place_forget()
        self.file_name.configure(text="-")
        self.file_meta.configure(text="")
        self.rename_lbl.configure(text="")
        self._show_suggestions([], "")
        self._set_controls(False)

    def _update_progress(self):
        total = len(self.files)
        self.progress.set((self.idx / total) if total else 0)
        named = self.counts["TOTAL_NAMED"]
        self.count_lbl.configure(text=f"{min(self.idx + 1, total):,}/{total:,} in queue   {self.session_tagged:,} tagged this session   "
                                      f"{named:,} all-time")

    def _show_current(self):
        path = self.current_path()
        if path is None:
            if self.focus_artist and self.files == []:
                self._show_empty(f"All candidates for {self.focus_artist} reviewed.")
            else:
                self._show_empty("Queue finished.  Nice work.")
            self._update_progress()
            return
        if not os.path.exists(path):
            self.files.pop(self.idx)
            self.after(1, self._show_current)
            return
        if path != getattr(self, "_shown_path", None):  # a refresh of the same file keeps what you typed
            self.entry.set("")
        self._shown_path = path
        self.entry.focus()
        self.suggestions = []
        self.file_name.configure(text=os.path.basename(path))
        self._update_rename_preview()
        self._update_progress()
        self._set_controls(True)
        item = self.prepared.get(path)
        if item and item.preview is not None:
            self._render_prepared(item)
        else:
            self._preview_pil = None
            self.preview_lbl.configure(image=None, text="Loading...")
            self.badge.place_forget()
            self._show_suggestions([], "Reading file...")
        with self.prep_cv:
            self.prep_cv.notify_all()

    # ------------------------------------------------------------------ prefetch pipeline
    def _wanted(self) -> List[str]:
        if not self.files:
            return []
        return [p for p in self.files[self.idx:self.idx + Cfg.PREFETCH_AHEAD + 1]]

    def _prefetch_loop(self):
        while not self._closing:
            with self.prep_cv:
                want = self._wanted()
                keep = set(want) | set(self.files[max(0, self.idx - 2):self.idx])
                for p in list(self.prepared):
                    if p not in keep:
                        self.prepared.pop(p, None)
                target = None
                for p in want:
                    it = self.prepared.get(p)
                    if it is None or (it.vec is None and not it.error and self.brain.ready):
                        target = (p, it)
                        break
                if target is None:
                    self.prep_cv.wait(0.5)
                    continue
            self._prepare(*target)

    def _prepare(self, path: str, item: Optional[Prepared]):
        item = item or Prepared(path)
        frames = None
        try:
            if item.preview is None:
                frames, dims = load_media_frames(path)
                item.preview = make_thumbnail(frames[len(frames) // 2], Cfg.PREVIEW_MAX, bg=(0, 0, 0))
                st = os.stat(path)
                size = f"{st.st_size / 1048576:.1f} MB" if st.st_size >= 1048576 else f"{st.st_size // 1024} KB"
                item.dims = f"{dims}   {size}   {os.path.splitext(path)[1][1:].upper()}"
                self.prepared[path] = item
                if path == self.current_path():
                    self.post(self._on_prepared, path)
            if self.brain.ready and item.vec is None:
                item.key, item.vec = self.brain.embed_file(path, frames)
                if path == self.current_path():
                    self.post(self._on_prepared, path)
        except Exception as e:
            item.error = f"{type(e).__name__}: {e}"
            logger.warning(f"Could not prepare {path}: {item.error}")
            self.prepared[path] = item
            if path == self.current_path():
                self.post(self._on_prepared, path)

    def _on_prepared(self, path: str):
        if path == self.current_path() and path in self.prepared:
            self._render_prepared(self.prepared[path])

    def _render_prepared(self, item: Prepared):
        if item.preview is not None and self._preview_pil is not item.preview:
            self._preview_pil = item.preview
            self._fit_preview()
            kind = media_kind(item.path)
            self.badge.configure(text=f"  {'VIDEO' if kind == 'video' else 'IMAGE'}  ")
            self.badge.place(x=14, y=14)
        self.file_meta.configure(text=item.dims)
        if item.error and item.preview is None:
            self.preview_lbl.configure(image=None, text=f"Cannot read this file\n{item.error}")
            self._show_suggestions([], "Unreadable")
            return
        if item.vec is None:
            self._show_suggestions([], "Analyzing..." if self.brain.ready else "AI loading...")
            return
        path, key, vec = item.path, item.key, item.vec
        self.ai_state.configure(text="Thinking...")

        def work():
            pred = self.brain.predict(key, vec)
            self.post(self._on_prediction, path, pred)
        self.predict_pool.submit(work)

    def _on_prediction(self, path: str, pred: Prediction):
        if path != self.current_path():
            return
        self.suggestions = pred.suggestions
        if not pred.suggestions:
            self._show_suggestions([], "No match - new artist?" if self.brain.index.size else "Brain empty")
        else:
            top = pred.suggestions[0]
            state = "Confident" if top.conf >= 0.75 else ("Likely" if top.conf >= 0.5 else ("Unsure" if top.conf >= 0.25 else "Probably unknown"))
            self._show_suggestions(pred.suggestions, state)

    def _show_suggestions(self, sugs: List[Suggestion], state: str):
        self.ai_state.configure(text=state)
        for i, (row, name, bar, pct) in enumerate(self.sug_rows):
            if i < len(sugs):
                s = sugs[i]
                name.configure(text=s.artist, text_color=T.TEXT)
                bar.configure(progress_color=conf_color(s.conf))
                bar.set(max(0.02, s.conf))
                pct.configure(text=f"{s.conf:.0%}", text_color=conf_color(s.conf))
            else:
                name.configure(text="", text_color=T.TEXT_3)
                bar.set(0)
                pct.configure(text="")

    def _on_preview_resize(self, _e=None):
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(60, self._fit_preview)

    def _fit_preview(self):
        self._resize_job = None
        pil = self._preview_pil
        if pil is None:
            return
        scale = self.preview_card._get_widget_scaling()
        bw = max(50, int(self.preview_card.winfo_width() / scale) - 24)
        bh = max(50, int(self.preview_card.winfo_height() / scale) - 24)
        s = min(bw / pil.width, bh / pil.height)
        size = (max(1, int(pil.width * s)), max(1, int(pil.height * s)))
        old = self._preview_img
        self._preview_img = ctk_image(pil, size)
        self.preview_lbl.configure(image=self._preview_img, text="")
        del old

    def _preview_menu(self, e):
        p = self.current_path()
        if not p:
            return
        m = dark_menu(self)
        m.add_command(label="Open", command=lambda: open_external(p))
        m.add_command(label="Show in Explorer", command=lambda: reveal_in_explorer(p))
        m.add_command(label="Copy image", command=self._copy_preview)
        m.add_command(label="Copy path", command=lambda: (self.clipboard_clear(), self.clipboard_append(p)))
        m.tk_popup(e.x_root, e.y_root)

    def _copy_preview(self):
        p = self.current_path()
        if not p:
            return

        def work():
            frames, _ = load_media_frames(p)
            return copy_image_to_clipboard(Image.fromarray(flatten_alpha(frames[0], (255, 255, 255))))
        self.run_bg(work, lambda ok: self.status("Image copied" if ok else "Clipboard failed", 2500))

    # ------------------------------------------------------------------ composer
    def _update_rename_preview(self):
        p = self.current_path()
        if not p:
            self.rename_lbl.configure(text="")
            return
        art = sanitize_artist(self.entry.get())
        base = os.path.basename(p)
        self.rename_lbl.configure(text=f"→  [{art}] - {base}" if art else f"   {base}",
                                  text_color=T.ACCENT_HI if art else T.TEXT_3)

    def _on_tab(self, _e=None):
        if not self.entry.accept():
            if not self.entry.get().strip() and self.suggestions:
                self.entry.set(self.suggestions[0].artist)
        self._update_rename_preview()
        return "break"

    def _use_suggestion(self, k: int, commit: bool = False):
        if self.overlay or self.gallery or k >= len(self.suggestions) or not self.current_path():
            return
        self.entry.set(self.suggestions[k].artist)
        self._update_rename_preview()
        self.entry.focus()
        if commit:
            self.confirm()

    def _canonical_tag(self, text: str) -> str:
        return ", ".join(self.completer.canonical(a) for a in split_artists(sanitize_artist(text)))

    # ------------------------------------------------------------------ actions
    def confirm(self, focus: bool = False):
        if self.busy or self.overlay or not self.current_path():
            return
        tag = self._canonical_tag(self.entry.get())
        if not tag:
            if focus:
                self._open_focus_dialog()
            else:
                self.skip()
            return
        if not self.brain.ready:
            messagebox.showinfo("Please wait", "The AI is still loading.")
            return
        path = self.current_path()
        item = self.prepared.pop(path, None)
        top = self.suggestions[0] if self.suggestions else None
        rejected = top.artist if top and top.conf >= 0.3 and top.artist.lower() not in {a.lower() for a in split_artists(tag)} else None
        idx = self.idx

        def work():
            new = safe_rename(path, os.path.join(os.path.dirname(path), f"[{tag}] - {os.path.basename(path)}"))
            self.brain.note_move(path, new)
            key, vec = (item.key, item.vec) if item and item.vec is not None else self.brain.embed_file(new)
            prev = self.brain.db.labels_for(key)
            self.brain.learn(key, tag, vec, verified=True, rejected=rejected)
            return new, key, vec, prev

        # The UI moves on at once; renaming + learning run in order on a background thread.
        rec = {"path": path, "tag": tag, "idx": idx, "rejected": rejected, "focus": focus,
               "fut": self.tag_pool.submit(work)}
        rec["fut"].add_done_callback(lambda f: self.post(self._after_tag, rec))
        if 0 <= idx < len(self.files) and self.files[idx] == path:
            self.files.pop(idx)
        self.undo.append(("tag", rec))
        self.counts["TOTAL_NAMED"] += 1
        self.session_tagged += 1
        arts = split_artists(tag)
        self.completer.bump(arts)
        known = {h.lower() for h in self.history_names}
        self.history_names.extend(a for a in arts if a.lower() not in known)
        if self.focus_artist:
            self.focus_candidates = [c for c in self.focus_candidates if c[0] != path]
        self.status(f"Tagged  [{tag}]" + (f"   (taught: not {rejected})" if rejected else ""), 4000, T.OK_HI)
        if focus:
            self.entry.set("")
        else:
            self._show_current()

    def _unbusy(self):
        self.busy = False
        self._set_controls(bool(self.current_path()))

    def _after_tag(self, rec):
        err = rec["fut"].exception()
        if err is not None:
            logger.error(f"Tagging {rec['path']} failed: {err!r}")
            try:
                self.undo.remove(("tag", rec))
            except ValueError:
                pass
            self.counts["TOTAL_NAMED"] = max(0, self.counts["TOTAL_NAMED"] - 1)
            self.session_tagged = max(0, self.session_tagged - 1)
            if os.path.exists(rec["path"]) and rec["path"] not in self.files:
                cur = self.current_path()
                i = min(max(0, rec["idx"]), len(self.files))
                self.files.insert(i, rec["path"])
                if cur is not None and i <= self.idx:
                    self.idx += 1
            self._update_progress()
            messagebox.showerror("Could not tag file", f"{os.path.basename(rec['path'])}\n\n{err}\n\nIt was put back in the queue.")
            return
        self._save_counts()
        if self.focus_artist:
            self._save_session("review")
        self._set_controls(bool(self.current_path()))
        if rec["focus"]:
            self.start_focus_scan(split_artists(rec["tag"])[0])

    def skip(self):
        if self.busy or self.overlay or not self.current_path():
            return
        self.undo.append(("skip", self.idx))
        self.counts["TOTAL_SKIPPED"] += 1
        self._save_counts()
        self.idx += 1
        self._show_current()

    def delete_current(self):
        if self.busy or self.overlay or not self.current_path():
            return
        path = self.current_path()
        if not messagebox.askyesno("Delete file", f"Send this file to the Recycle Bin?\n\n{os.path.basename(path)}", icon="warning"):
            return
        self.busy = True
        self._set_controls(False)
        idx = self.idx

        def done(ok):
            self.busy = False
            if not ok:
                messagebox.showerror("Delete failed", "Could not move the file to the Recycle Bin (it may be open elsewhere).")
                self._set_controls(True)
                return
            if 0 <= idx < len(self.files) and self.files[idx] == path:
                self.files.pop(idx)
            self.prepared.pop(path, None)
            self.counts["TOTAL_DELETED"] += 1
            self._save_counts()
            self.status("Moved to Recycle Bin", 3000)
            self._show_current()
        self.run_bg(lambda: send_to_recycle_bin(path), done)

    def undo_last(self):
        if self.busy or self.overlay or not self.undo:
            return
        act = self.undo[-1]
        if act[0] == "skip":
            self.undo.pop()
            self.idx = max(0, act[1])
            self.counts["TOTAL_SKIPPED"] = max(0, self.counts["TOTAL_SKIPPED"] - 1)
            self._show_current()
            return
        rec = act[1]
        if not rec["fut"].done():
            self.after(40, self.undo_last)  # still being written; undo right after
            return
        self.undo.pop()
        if rec["fut"].exception() is not None:
            return
        new, key, vec, prev = rec["fut"].result()
        path, idx, rejected = rec["path"], rec["idx"], rec["rejected"]
        self.busy = True
        self._set_controls(False)

        def work():
            restored = safe_rename(new, path)
            self.brain.note_move(new, restored)
            prev_artists = [a for a, _ in prev]
            self.brain.db.set_labels_many([(key, prev_artists, any(v for _, v in prev))])
            if prev_artists:
                self.brain.index.add(key, prev_artists, vec, any(v for _, v in prev))
            else:
                self.brain.index.remove_key(key)
            if rejected:
                self.brain.db.write(lambda c: c.execute("DELETE FROM negatives WHERE fkey=? AND artist_l=?", (key, rejected.lower())))
                aid = self.brain.index.aid.get(rejected.lower())
                if aid:
                    self.brain.index.neg_by_key.get(key, set()).discard(aid)
            return restored

        def done(restored):
            self.busy = False
            idx_c = min(max(0, idx), len(self.files))
            self.files.insert(idx_c, restored)
            self.idx = idx_c
            self.counts["TOTAL_NAMED"] = max(0, self.counts["TOTAL_NAMED"] - 1)
            self.session_tagged = max(0, self.session_tagged - 1)
            self._save_counts()
            self.status(f"Undone: {os.path.basename(restored)}", 3000)
            self._show_current()

        def failed(e):
            logger.error(f"Undo failed: {e!r}")
            self._unbusy()
            messagebox.showerror("Undo failed", str(e))

        fut = self.tag_pool.submit(work)
        fut.add_done_callback(lambda f: self.post(done, f.result()) if f.exception() is None else self.post(failed, f.exception()))

    def _save_counts(self):
        self.counts["TOTAL_TAGGED"] = self.counts["TOTAL_NAMED"] + self.counts["TOTAL_SKIPPED"]
        atomic_write_json(Cfg.COUNTS_FILE, self.counts)


    # ------------------------------------------------------------------ long jobs
    def _begin_job(self, title: str, subtitle: str, accent: str = T.ACCENT) -> bool:
        if self.job_active or not self.brain.ready:
            if not self.brain.ready:
                messagebox.showinfo("Please wait", "The AI is still loading.")
            return False
        self.job_active = True
        self.job_stop.clear()
        self.overlay = TaskOverlay(self, title, subtitle, self._cancel_job, accent)
        return True

    def _cancel_job(self):
        if self.overlay:
            self.overlay.cancel_btn.configure(state="disabled", text="Stopping...")
        self.job_stop.set()

    def _end_job(self):
        self.job_active = False
        if self.overlay:
            self.overlay.destroy()
            self.overlay = None
        self.taskbar.clear(self.hwnd())
        self._refresh_brain_card()
        self.refresh_artists()
        self.entry.focus()

    # --- library sync (learn every tagged file) ---
    def start_sync(self):
        folder = self.folder
        if not folder or not messagebox.askyesno(
                "Sync library", f"Learn every tagged file under:\n{folder}\n\n(includes all subfolders)\n\nChoose 'No' to pick another folder."):
            folder = filedialog.askdirectory(title="Library folder to learn from", initialdir=folder or None)
            if not folder:
                return
        folder = os.path.normpath(folder)
        if not self._begin_job("Syncing library", f"Learning every \"[artist] - file\" under {folder}"):
            return

        def progress(done, total, detail, thumb, elapsed):
            self.post_latest("job", self.overlay_progress, done, total, detail, None, thumb)
            if thumb is not None:
                self.post(self.overlay_progress, done, total, detail, None, thumb)

        def work():
            return self.brain.sync_library(folder, self.job_stop.is_set, progress)

        def done(res):
            self._end_job()
            if res.get("cancelled"):
                self.status("Sync cancelled - progress so far is saved and will be reused.", 6000)
                return
            m = self.brain.index.metrics
            acc = f"\n\nSelf-test accuracy: top-1 {m['top1']:.0%}, top-5 {m['top5']:.0%}" if "top1" in m else ""
            messagebox.showinfo("Library synced", f"Tagged files found: {res['tagged']:,}\nLearned: {res['learned']:,}\n"
                                f"Label changes: {res.get('changed', 0):,}\nUnreadable: {res['errors']:,}{acc}")
            if self.current_path():
                self._show_current()
        self.run_bg(work, done, "Sync failed", on_error=self._end_job)

    def overlay_progress(self, done, total, detail="", eta=None, thumb=None, extra=None):
        if self.overlay:
            self.overlay.update_progress(done, total, detail, eta, thumb, extra)

    # --- projector training ---
    def start_training(self):
        if self.brain.ready and self.brain.index.size < 500:
            messagebox.showinfo("Train brain", "Tag (or sync) at least 500 files first.")
            return
        if not messagebox.askyesno("Train brain",
                                   "Train a style projection on your library?\n\n"
                                   "It learns what separates your artists from each other. A held-out set of files "
                                   "is used to check the result, and it is only kept if accuracy improves.\n\n"
                                   "Takes about a minute on the GPU."):
            return
        if not self._begin_job("Training the brain", "Metric learning on your tagged library (GPU)", T.CYAN):
            return

        def progress(done, total, detail, eta):
            self.post_latest("job", self.overlay_progress, done, total, detail, eta)

        def done(res):
            self._end_job()
            if res.get("reason") == "cancelled":
                self.status("Training cancelled.", 4000)
                return
            b, a = res["before_top1"], res["after_top1"]
            if res["accepted"]:
                messagebox.showinfo("Brain trained", f"Held-out accuracy improved and the new projection is active.\n\n"
                                    f"Top-1: {b:.1%}  ->  {a:.1%}\nTop-5: {res['before_top5']:.1%}  ->  {res['after_top5']:.1%}")
            else:
                messagebox.showinfo("No improvement", f"The trained projection did not beat the current one "
                                    f"({b:.1%} vs {a:.1%} top-1), so it was discarded.")
            if self.current_path():
                self._show_current()
        self.run_bg(lambda: self.brain.train(progress, self.job_stop.is_set), done, "Training failed", on_error=self._end_job)

    def _recalibrate(self):
        if not self.brain.ready:
            return
        self.status("Recalibrating...")
        self.run_bg(self.brain.index.calibrate, lambda m: (self._refresh_brain_card(), self.status("Calibrated.", 3000)))

    def reset_brain(self):
        if self.job_active or not self.brain.ready:
            return
        if not messagebox.askyesno("Reset brain", "Forget every artist the AI has learned?\n\nYour files are not touched, and "
                                   "cached image embeddings are kept so a later Sync Library is fast.", icon="warning"):
            return
        if not messagebox.askyesno("Are you sure?", "This cannot be undone.", icon="warning"):
            return
        self.brain.reset()
        self.brain.index.calibrate()
        self._refresh_brain_card()
        self.refresh_artists()
        self.status("Brain reset.", 4000, T.WARN)
        if self.current_path():
            self._show_current()

    # --- focus: find every file by one artist ---
    def _open_focus_dialog(self):
        if not self.files:
            messagebox.showinfo("Focus", "Open a folder with untagged files first.")
            return
        FocusDialog(self)

    def start_focus_scan(self, artist: str, resume_matches: Optional[List[Tuple[str, float]]] = None):
        artist = self.completer.canonical(artist)
        if not self.brain.index.has_artist(artist):
            messagebox.showwarning("Unknown artist", f"The brain has no files for [{artist}] yet.\nTag at least one file first.")
            return
        cands = [p for p in self.files if os.path.exists(p)]
        if not cands:
            messagebox.showinfo("Focus", "No untagged files to scan.")
            return
        if not self._begin_job(f"Finding {artist}", f"Scanning {len(cands):,} untagged files for this artist's style", T.ACCENT):
            return
        self.focus_artist = artist
        self._save_session("scan")
        floor = 0.05
        found: List[Tuple[str, float]] = []
        fm: Dict[str, Any] = {}
        vectors: Dict[str, Tuple[str, np.ndarray]] = {}
        buf: List[Tuple[str, str, np.ndarray]] = []
        t0 = time.time()
        total = len(cands)
        done_n = [0]

        def score(chunk):
            confs = self.brain.index.score_focus(np.stack([v for _, _, v in chunk]), [k for _, k, _ in chunk], fm["m"])
            for (p, _, _), c in zip(chunk, confs):
                if c >= floor:
                    found.append((p, float(c)))

        def on_item(r: EmbedResult):
            done_n[0] += 1
            if r.vec is not None:
                vectors[r.path] = (r.key, r.vec)
                buf.append((r.path, r.key, r.vec))
                if len(buf) >= 512:
                    score(buf)
                    buf.clear()
            if done_n[0] % 20 == 0 or r.thumb is not None or done_n[0] == total:
                self.post_latest("job", self.overlay_progress, done_n[0], total, os.path.basename(r.path), None, None,
                                 f"{len(found):,} likely matches so far")
                if r.thumb is not None:
                    self.post(self.overlay_progress, done_n[0], total, os.path.basename(r.path), None, r.thumb)

        def work():
            self.post(self.overlay_progress, 0, total, "Learning this artist's style...", None, None)
            fm["m"] = self.brain.index.focus_model(artist)
            if fm["m"] is None:
                raise RuntimeError(f"Not enough tagged files of {artist} to search for (need at least 2).")
            logger.info(f"Focus model for {artist}: held-out recall@1% {fm['m']['heldout_recall_at_1pct']:.0%}")
            keys = self.brain.resolve_keys(cands, self.job_stop.is_set)
            self.brain.embed_many(keys, self.job_stop.is_set, on_item)
            if buf:
                score(buf)
            found.sort(key=lambda x: x[1], reverse=True)
            return found

        def done(matches):
            cancelled = self.job_stop.is_set()
            self._end_job()
            self.focus_vectors = vectors
            self.focus_candidates = matches
            self._save_session("review")
            logger.info(f"Focus scan for {artist}: {len(matches)} candidates in {time.time() - t0:.1f}s")
            if cancelled:
                self.status(f"Scan stopped early - showing {len(matches)} candidates found so far.", 6000)
            self._show_focus_chip()
            self._open_gallery()
        self.run_bg(work, done, "Focus scan failed", on_error=self._end_job)

    def _index_loop(self):
        """Quietly embeds untagged files in the open folder (4 decode threads, pauses during jobs).
        Embeddings are stored, so a Focus scan only has to score them - seconds instead of minutes."""
        indexed: Set[str] = set()
        last_folder = None
        while not self._closing:
            time.sleep(1.5)
            try:
                if not self.brain.ready or self.job_active or not self.folder:
                    continue
                folder = self.folder
                if folder != last_folder:
                    indexed.clear()
                    last_folder = folder
                files = list(self.files) if not self.focus_artist else []
                pending = [p for p in files if p not in indexed]
                if not pending:
                    self.post_latest("indexer", self._set_index_state, len(files), len(files))
                    time.sleep(3)
                    continue

                def stop():
                    return self._closing or self.job_active or self.folder != folder or bool(self.focus_artist)
                keys = self.brain.resolve_keys(pending, stop)
                if stop():
                    continue
                have = self.brain.db.keys_with_embedding(self.brain.model_id)
                todo = {}
                for p, k in keys.items():
                    if k and k in have:
                        indexed.add(p)
                    else:
                        todo[p] = k
                for p in pending:
                    if p not in keys:
                        indexed.add(p)  # vanished
                total = len(files)
                done = [sum(1 for p in files if p in indexed)]
                self.post_latest("indexer", self._set_index_state, done[0], total)
                if not todo:
                    continue

                def on_item(r):
                    indexed.add(r.path)
                    done[0] += 1
                    if done[0] % 25 == 0:
                        self.post_latest("indexer", self._set_index_state, done[0], total)
                self.brain.embed_many(todo, stop, on_item, thumb_every=0, workers=4, vid_workers=1)
                self.post_latest("indexer", self._set_index_state, done[0], total)
            except Exception:
                logger.error("Background indexing: " + traceback.format_exc())
                time.sleep(10)

    def _set_index_state(self, done: int, total: int):
        if total and done < total:
            self.index_lbl.configure(text=f"Preparing Focus  {done:,}/{total:,}")
        else:
            self.index_lbl.configure(text="Focus ready" if total else "")

    def _show_focus_chip(self):
        if self.focus_artist:
            self.focus_chip.configure(text=f"  Focus: {self.focus_artist}  ({len(self.focus_candidates)})  ")
            self.focus_chip.grid(row=0, column=4, padx=6)
        else:
            self.focus_chip.grid_forget()

    def _open_gallery(self):
        if not self.focus_artist:
            return
        if self.gallery:
            self.gallery.destroy()
        self.gallery = FocusGallery(self, self.focus_artist)

    def close_gallery(self):
        if self.gallery:
            self.gallery.destroy()
            self.gallery = None
        self._show_focus_chip()
        self.entry.focus()

    def step_through_candidates(self, threshold: float):
        self.files = [p for p, c in self.focus_candidates if c >= threshold and os.path.exists(p)]
        self.idx = 0
        self.prepared.clear()
        self.close_gallery()
        self.status(f"Reviewing {len(self.files)} candidates for {self.focus_artist} one by one", 5000)
        self._show_current()

    def exit_focus(self, silent: bool = False):
        was = self.focus_artist
        self.focus_artist = None
        self.focus_candidates = []
        self.focus_vectors = {}
        self._clear_session()
        if self.gallery:
            self.gallery.destroy()
            self.gallery = None
        self._show_focus_chip()
        if was and not silent and self.folder:
            self.load_folder(self.folder)

    # --- focus session persistence ---
    def _save_session(self, phase: str):
        if not self.focus_artist or not self.folder:
            return
        atomic_write_json(Cfg.SESSION_FILE, {
            "artist": self.focus_artist, "folder": self.folder, "recursive": bool(self.recursive_var.get()),
            "phase": phase, "matches": self.focus_candidates if phase == "review" else [], "ts": time.time()})

    def _clear_session(self):
        try:
            os.remove(Cfg.SESSION_FILE)
        except OSError:
            pass

    def _check_session(self):
        data = read_json(Cfg.SESSION_FILE, None)
        if not isinstance(data, dict) or not data.get("artist") or not os.path.isdir(data.get("folder", "")):
            self._clear_session()
            self._autoload_last_folder()
            return
        artist, folder = data["artist"], data["folder"]
        if not messagebox.askyesno("Resume focus", f"Resume the focus session for [{artist}]?\n\n{folder}"):
            self._clear_session()
            self._autoload_last_folder()
            return
        self.recursive_var.set(bool(data.get("recursive")))
        self.settings["recursive"] = bool(data.get("recursive"))
        self.folder = folder
        self.folder_lbl.configure(text=folder)
        rec = bool(data.get("recursive"))
        matches = [(p, float(c)) for p, c in data.get("matches", []) if os.path.exists(p)]

        def scan():
            files = [p for p in list_media(folder, rec) if not parse_tagged_name(os.path.basename(p))]
            random.shuffle(files)
            return files

        self._scan_token = None

        def done(files):
            self._on_scanned(files)
            self.focus_artist = artist
            if data.get("phase") == "review" and matches:
                self.focus_candidates = matches
                self._show_focus_chip()
                self._open_gallery()
            else:
                self.start_focus_scan(artist)
        self.run_bg(scan, done)

    def _autoload_last_folder(self):
        last = self.settings.get("last_folder")
        if last and os.path.isdir(last) and not self.folder:
            self.load_folder(last)

    # ------------------------------------------------------------------ file ops shared by gallery/manager
    def tag_file(self, path: str, artists: List[str], vec_hint: Optional[Tuple[str, np.ndarray]] = None) -> str:
        """Worker-thread helper: renames to the tagged form and teaches the brain. Returns the new path."""
        parsed = parse_tagged_name(os.path.basename(path))
        base = parsed[1] if parsed else os.path.basename(path)
        tag = ", ".join(artists)
        new = safe_rename(path, os.path.join(os.path.dirname(path), f"[{tag}] - {base}"))
        self.brain.note_move(path, new)
        key, vec = vec_hint if vec_hint else self.brain.embed_file(new)
        self.brain.learn(key, tag, vec, verified=True)
        return new

    def retag_file(self, path: str, artists: List[str]) -> str:
        """Rewrites a file's tag (no brain changes). Empty list = untag."""
        parsed = parse_tagged_name(os.path.basename(path))
        base = parsed[1] if parsed else os.path.basename(path)
        name = f"[{', '.join(artists)}] - {base}" if artists else base
        new = safe_rename(path, os.path.join(os.path.dirname(path), name))
        self.brain.note_move(path, new)
        return new

    def tagged_files_for(self, artists: Iterable[str]) -> List[Tuple[str, List[str]]]:
        if not self.folder:
            return []
        want = {a.lower() for a in artists}
        out = []
        for p in list_media(self.folder, bool(self.settings.get("recursive"))):
            parsed = parse_tagged_name(os.path.basename(p))
            if parsed:
                arts = split_artists(parsed[0])
                if want & {a.lower() for a in arts}:
                    out.append((p, arts))
        return out

    def unify_artists(self, olds: List[str], target: str, rename_files: bool) -> Dict[str, int]:
        """Worker-thread: merges artist names in the brain and (optionally) in filenames."""
        target = sanitize_artist(target)
        olds = [o for o in olds if o and o.strip()]
        old_l = {o.lower() for o in olds}
        db_n = self.brain.merge_artists(olds, target)
        files_n = 0
        if rename_files:
            for p, arts in self.tagged_files_for(olds):
                new_arts, seen = [], set()
                for a in arts:
                    a2 = target if a.lower() in old_l else a
                    if a2.lower() not in seen:
                        seen.add(a2.lower())
                        new_arts.append(a2)
                if new_arts != arts:
                    try:
                        self.retag_file(p, new_arts)
                        files_n += 1
                    except OSError as e:
                        logger.warning(f"Rename failed for {p}: {e}")
        self.history_names = [h for h in self.history_names if h.lower() not in old_l] + [target]
        if self.focus_artist and self.focus_artist.lower() in old_l:
            self.focus_artist = target
        return {"db": db_n, "files": files_n}

    def untag_artists(self, names: List[str]) -> Dict[str, int]:
        """Worker-thread: removes artists from filenames in the open folder and from the brain."""
        low = {n.lower() for n in names}
        files_n = 0
        restored = []
        for p, arts in self.tagged_files_for(names):
            keep = [a for a in arts if a.lower() not in low]
            try:
                new = self.retag_file(p, keep)
                files_n += 1
                if not keep:
                    restored.append(new)
            except OSError as e:
                logger.warning(f"Untag failed for {p}: {e}")
        n = self.brain.delete_artists(names)
        self.history_names = [h for h in self.history_names if h.lower() not in low]
        self.post(self._add_untagged, restored)
        return {"db": n, "files": files_n}

    def _add_untagged(self, paths: List[str]):
        fresh = [p for p in paths if p not in set(self.files)]
        self.files.extend(fresh)
        self._update_progress()
        if fresh and not self.current_path():
            self._show_current()

    # ------------------------------------------------------------------ dialogs
    def open_artist_manager(self):
        if not self.brain.ready:
            messagebox.showinfo("Please wait", "The AI is still loading.")
            return
        try:
            if self.manager_win and self.manager_win.winfo_exists():
                self.manager_win.lift()
                self.manager_win.focus_force()
                return
        except tk.TclError:
            pass
        self.manager_win = ArtistManager(self)

    def open_settings(self):
        SettingsDialog(self)

    def _on_close(self):
        self._closing = True
        self.job_stop.set()
        try:
            self.tag_pool.shutdown(wait=True)  # never exit halfway through a rename
        except Exception:
            pass
        try:
            self._save_counts()
            atomic_write_json(Cfg.HISTORY_FILE, sorted(set(self.history_names), key=str.lower))
            if self.focus_artist:
                self._save_session("review" if self.focus_candidates else "scan")
        finally:
            with self.prep_cv:
                self.prep_cv.notify_all()
            self.destroy()
            os._exit(0)  # daemon threads may be mid-CUDA call; exit cleanly without waiting


def oled_titlebar(win):
    """Black Windows 11 title bar to match the OLED theme."""
    if sys.platform != "win32":
        return
    try:
        hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
        dwm = ctypes.windll.dwmapi
        on = ctypes.c_int(1)
        dwm.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(on), 4)          # immersive dark mode
        black = ctypes.c_int(0x00000000)
        dwm.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(black), 4)       # caption colour
        border = ctypes.c_int(0x001C1C1C)
        dwm.DwmSetWindowAttribute(hwnd, 34, ctypes.byref(border), 4)      # border colour
    except Exception:
        pass


# =================================================================================
#  FOCUS DIALOG & GALLERY
# =================================================================================
class FocusDialog(ctk.CTkToplevel):
    def __init__(self, app: VtaggerApp):
        super().__init__(app)
        self.app = app
        self.title("Find artist")
        self.configure(fg_color=T.SURFACE)
        self.resizable(False, False)
        self.transient(app)
        w, h = 520, 250
        self.geometry(f"{w}x{h}+{app.winfo_x() + (app.winfo_width() - w) // 2}+{app.winfo_y() + (app.winfo_height() - h) // 3}")
        self.after(30, lambda: oled_titlebar(self))
        label(self, "Find every file by an artist", 18, weight="bold").pack(anchor="w", padx=24, pady=(22, 2))
        label(self, f"Scans the {len(app.files):,} untagged files in this folder for the artist's style.", 12, T.TEXT_2).pack(anchor="w", padx=24)
        self.entry = GhostEntry(self, "Artist name", app.completer.complete_text, height=46, size=16,
                                options=app.completer.options)
        self.entry.pack(fill="x", padx=24, pady=(16, 4))
        self.msg = label(self, "Tab completes  -  Enter starts the scan", 11, T.TEXT_3)
        self.msg.pack(anchor="w", padx=24)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=24, pady=(14, 18))
        btn(row, "Cancel", self.destroy, "ghost", width=110).pack(side="left")
        btn(row, "Start scan", self.submit, "primary", width=150).pack(side="right")
        self.entry.bind_key("<Tab>", lambda e: (self.entry.accept(), "break")[1])
        self.entry.bind_key("<Return>", lambda e: self.submit())
        self.bind("<Escape>", lambda e: self.destroy())
        self.after(80, lambda: (self.grab_set(), self.entry.focus()))

    def submit(self):
        arts = split_artists(sanitize_artist(self.entry.get()))
        if not arts:
            return
        a = self.app.completer.canonical(arts[0])
        if not self.app.brain.index.has_artist(a):
            self.msg.configure(text=f"No files for [{a}] in the brain yet - tag one first.", text_color=T.DANGER_HI)
            return
        self.destroy()
        self.app.start_focus_scan(a)


class FocusGallery(ctk.CTkFrame):
    PAGE = 48

    def __init__(self, app: VtaggerApp, artist: str):
        super().__init__(app, fg_color=T.BG, corner_radius=0)
        self.app, self.artist = app, artist
        self.tab = "candidates"
        self.page = 0
        self.token = 0
        self.tagged: Optional[List[Tuple[str, List[str]]]] = None
        self.pool = ThreadPoolExecutor(6, thread_name_prefix="thumbs")
        self.thumb_refs: Dict[str, ctk.CTkImage] = {}
        self.cards: Dict[str, Dict[str, Any]] = {}
        self.threshold = float(app.settings.get("focus_min_conf", 0.35))
        self._render_job = None
        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.lift()

        head = ctk.CTkFrame(self, fg_color="transparent")
        head.pack(fill="x", padx=24, pady=(18, 10))
        left = ctk.CTkFrame(head, fg_color="transparent")
        left.pack(side="left")
        section_title(left, "Focus review").pack(anchor="w")
        label(left, artist, 26, weight="bold").pack(anchor="w")
        self.sub = label(left, "", 12, T.TEXT_2)
        self.sub.pack(anchor="w")
        right = ctk.CTkFrame(head, fg_color="transparent")
        right.pack(side="right")
        btn(right, "Exit focus", self._exit_focus, "danger", width=110).pack(side="right", padx=(6, 0))
        btn(right, "Close", app.close_gallery, "ghost", width=90).pack(side="right", padx=6)
        btn(right, "Open folder", lambda: open_external(app.folder), "ghost", width=110).pack(side="right", padx=6)
        self.step_btn = btn(right, "Step through", lambda: app.step_through_candidates(self.threshold), "accent", width=120)
        self.step_btn.pack(side="right", padx=6)
        self.batch_btn = btn(right, "", self._batch_confirm, "ok", width=190)
        self.batch_btn.pack(side="right", padx=6)

        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", padx=24, pady=(0, 8))
        self.seg = ctk.CTkSegmentedButton(bar, values=["Candidates", "Tagged"], command=self._on_tab, font=T.font(12),
                                          selected_color=T.ACCENT, selected_hover_color=T.ACCENT_HI,
                                          unselected_color=T.SURFACE_2, unselected_hover_color=T.HOVER,
                                          fg_color=T.SURFACE_2, text_color=T.TEXT, corner_radius=8, height=32)
        self.seg.set("Candidates")
        self.seg.pack(side="left")
        self.th_box = ctk.CTkFrame(bar, fg_color="transparent")
        self.th_box.pack(side="left", padx=24)
        self.th_lbl = label(self.th_box, "", 12, T.TEXT_2, mono=True, width=170, anchor="w")
        self.th_lbl.pack(side="left")
        self.slider = ctk.CTkSlider(self.th_box, from_=0.1, to=0.95, number_of_steps=85, width=240, command=self._on_threshold,
                                    progress_color=T.ACCENT, button_color=T.TEXT, button_hover_color="#ffffff", fg_color=T.SURFACE_3)
        self.slider.set(self.threshold)
        self.slider.pack(side="left")
        self.pager = ctk.CTkFrame(bar, fg_color="transparent")
        self.pager.pack(side="right")
        self.prev_btn = btn(self.pager, "‹", lambda: self._go(-1), "ghost", width=36)
        self.prev_btn.pack(side="left")
        self.page_lbl = label(self.pager, "", 12, T.TEXT_2, mono=True, width=120)
        self.page_lbl.pack(side="left")
        self.next_btn = btn(self.pager, "›", lambda: self._go(1), "ghost", width=36)
        self.next_btn.pack(side="left")

        self.grid_area = ctk.CTkScrollableFrame(self, fg_color=T.BG, scrollbar_button_color=T.SURFACE_3,
                                                scrollbar_button_hover_color=T.HOVER)
        self.grid_area.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        self.bind("<Escape>", lambda e: app.close_gallery())
        self.focus_set()
        self.render()

    def destroy(self):
        self.token += 1
        self.pool.shutdown(wait=False, cancel_futures=True)
        super().destroy()

    # ----- data -----
    def _visible_candidates(self) -> List[Tuple[str, float]]:
        return [(p, c) for p, c in self.app.focus_candidates if c >= self.threshold]

    def _items(self) -> List[Tuple[str, float, List[str]]]:
        if self.tab == "candidates":
            return [(p, c, []) for p, c in self._visible_candidates()]
        return [(p, 1.0, arts) for p, arts in (self.tagged or [])]

    def _on_tab(self, value):
        self.tab = "candidates" if value == "Candidates" else "tagged"
        self.page = 0
        if self.tab == "tagged" and self.tagged is None:
            self._load_tagged()
        self.render()

    def _load_tagged(self):
        self.app.run_bg(lambda: self.app.tagged_files_for([self.artist]), self._tagged_loaded)

    def _tagged_loaded(self, items):
        if self.winfo_exists():
            self.tagged = items
            self.render()

    def _on_threshold(self, v):
        self.threshold = round(float(v), 2)
        self.app.settings["focus_min_conf"] = self.threshold
        self._update_header()
        if self._render_job:
            self.after_cancel(self._render_job)
        self._render_job = self.after(250, self._rerender_first_page)

    def _rerender_first_page(self):
        self._render_job = None
        self.page = 0
        self.app.settings.save()
        self.render()

    def _go(self, d):
        self.page += d
        self.render()

    def _update_header(self):
        vis = len(self._visible_candidates())
        total = len(self.app.focus_candidates)
        tagged = f"{len(self.tagged):,}" if self.tagged is not None else "..."
        self.sub.configure(text=f"{total:,} candidates found   {vis:,} above threshold   {tagged} tagged in folder")
        self.th_lbl.configure(text=f"Min confidence {self.threshold:.0%}")
        self.batch_btn.configure(text=f"Tag all {vis:,} shown", state="normal" if vis else "disabled")
        self.step_btn.configure(state="normal" if vis else "disabled")
        is_c = self.tab == "candidates"
        for w in (self.batch_btn, self.step_btn):
            w.configure(state=("normal" if (is_c and vis) else "disabled"))

    # ----- rendering -----
    def render(self):
        self.token += 1
        token = self.token
        for w in self.grid_area.winfo_children():
            w.destroy()
        self.cards.clear()
        self.thumb_refs.clear()
        self._update_header()
        if self.tab == "tagged" and self.tagged is None:
            label(self.grid_area, "Listing tagged files...", 14, T.TEXT_3).pack(pady=60)
            return
        items = self._items()
        pages = max(1, math.ceil(len(items) / self.PAGE))
        self.page = max(0, min(self.page, pages - 1))
        self.page_lbl.configure(text=f"{self.page + 1} / {pages}")
        self.prev_btn.configure(state="normal" if self.page > 0 else "disabled")
        self.next_btn.configure(state="normal" if self.page < pages - 1 else "disabled")
        page_items = items[self.page * self.PAGE:(self.page + 1) * self.PAGE]
        if not page_items:
            msg = "Nothing above this confidence. Lower the threshold to see more." if self.tab == "candidates" else \
                  f"No files tagged [{self.artist}] in this folder yet."
            label(self.grid_area, msg, 14, T.TEXT_3).pack(pady=60)
            return
        width = max(600, int(self.winfo_width() / self._get_widget_scaling()) - 60)
        cols = max(3, width // (Cfg.THUMB + 34))
        for c in range(cols):
            self.grid_area.grid_columnconfigure(c, weight=1, uniform="g")
        for i, (p, conf, arts) in enumerate(page_items):
            self._make_card(i // cols, i % cols, p, conf, arts)
            self.pool.submit(self._load_thumb, p, token)

    def _make_card(self, r, c, path, conf, arts):
        cd = card(self.grid_area, fg_color=T.SURFACE, corner_radius=12)
        cd.grid(row=r, column=c, padx=6, pady=6, sticky="n")
        img = ctk.CTkLabel(cd, text="", width=Cfg.THUMB, height=Cfg.THUMB, fg_color=T.SURFACE_2, corner_radius=10)
        img.pack(padx=8, pady=(8, 6))
        img.bind("<Button-1>", lambda e: open_external(path))
        img.bind("<Button-3>", lambda e: reveal_in_explorer(path))
        img.configure(cursor="hand2")
        if self.tab == "candidates":
            pill = ctk.CTkLabel(cd, text=f" {conf:.0%} ", font=T.font(11, "bold"), fg_color=T.BG,
                                text_color=conf_color(conf), corner_radius=6, height=20)
            pill.place(x=14, y=14)
        base = os.path.basename(path)
        parsed = parse_tagged_name(base)
        shown = parsed[1] if parsed else base
        name = label(cd, shown if len(shown) <= 24 else shown[:22] + "...", 11, T.TEXT_2)
        name.pack(padx=8)
        row = ctk.CTkFrame(cd, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(4, 8))
        w = {"card": cd, "img": img, "name": name, "btns": []}
        if self.tab == "candidates":
            b1 = btn(row, "✓ Tag", lambda: self._confirm(path), "ok", width=72, height=28, size=12)
            b2 = btn(row, "✕", lambda: self._reject(path), "danger", width=40, height=28, size=12)
            b1.pack(side="left", expand=True, fill="x", padx=(0, 3))
            b2.pack(side="left", padx=3)
            w["btns"] = [b1, b2]
        else:
            b1 = btn(row, "Untag", lambda: self._untag(path, arts), "danger", width=72, height=28, size=12)
            b1.pack(side="left", expand=True, fill="x", padx=(0, 3))
            w["btns"] = [b1]
        btn(row, "↗", lambda: open_external(path), "ghost", width=36, height=28, size=12).pack(side="left", padx=(3, 0))
        self.cards[path] = w

    def _load_thumb(self, path: str, token: int):
        if token != self.token:
            return
        try:
            if media_kind(path) == "video":
                frames, _ = decode_video_frames(path, (0.5,), timeout=6)
                img = frames[0]
            else:
                img = decode_image(path)
            thumb = make_thumbnail(img, (Cfg.THUMB, Cfg.THUMB), bg=(15, 15, 15))
        except Exception:
            thumb = placeholder_tile(Cfg.THUMB, "no preview")
        self.app.post(self._set_thumb, path, thumb, token)

    def _set_thumb(self, path, thumb, token):
        w = self.cards.get(path)
        if token != self.token or not w or not w["img"].winfo_exists():
            return
        im = ctk_image(thumb)
        self.thumb_refs[path] = im
        w["img"].configure(image=im)

    def _mark(self, path, text, color, border):
        w = self.cards.get(path)
        if not w or not w["card"].winfo_exists():
            return
        w["name"].configure(text=text, text_color=color)
        w["card"].configure(border_color=border)
        for b in w["btns"]:
            b.configure(state="disabled")

    # ----- actions -----
    def _confirm(self, path):
        app = self.app
        hint = app.focus_vectors.get(path)
        self._mark(path, "tagging...", T.TEXT_3, T.BORDER_HI)

        def done(new):
            app.focus_candidates = [c for c in app.focus_candidates if c[0] != path]
            if path in app.files:
                app.files.remove(path)
            app.counts["TOTAL_NAMED"] += 1
            app.session_tagged += 1
            app._save_counts()
            app._save_session("review")
            if self.tagged is not None:
                self.tagged.append((new, [self.artist]))
            self._mark(path, "✓ tagged", T.OK_HI, "#164d2a")
            self._update_header()
        app.run_bg(lambda: app.tag_file(path, [self.artist], hint), done, "Tag failed")

    def _reject(self, path):
        app = self.app
        hint = app.focus_vectors.get(path)

        def work():
            key, vec = hint if hint else app.brain.embed_file(path)
            app.brain.reject(key, self.artist, vec)
        app.focus_candidates = [c for c in app.focus_candidates if c[0] != path]
        app._save_session("review")
        self._mark(path, "✕ not this artist", T.DANGER_HI, "#4a1616")
        self._update_header()
        app.run_bg(work)

    def _untag(self, path, arts):
        app = self.app
        keep = [a for a in arts if a.lower() != self.artist.lower()]

        def work():
            key, vec = app.brain.embed_file(path)
            new = app.retag_file(path, keep)
            if keep:
                app.brain.learn(key, ", ".join(keep), vec, verified=True)
                app.brain.reject(key, self.artist, vec)
            else:
                app.brain.forget_labels(key, rejected=self.artist, vec=vec)
            return new

        def done(new):
            if self.tagged is not None:
                self.tagged = [t for t in self.tagged if t[0] != path]
            if not keep:
                app._add_untagged([new])
            app.counts["TOTAL_NAMED"] = max(0, app.counts["TOTAL_NAMED"] - 1)
            app._save_counts()
            self._mark(path, "untagged", T.WARN, T.BORDER_HI)
            self._update_header()
        app.run_bg(work, done, "Untag failed")

    def _batch_confirm(self):
        app = self.app
        targets = self._visible_candidates()
        if not targets:
            return
        if not messagebox.askyesno("Tag all shown", f"Tag {len(targets):,} files as [{self.artist}]?\n\n"
                                   f"(everything at or above {self.threshold:.0%} confidence)", parent=self):
            return
        if not app._begin_job(f"Tagging {len(targets):,} files", f"as [{self.artist}]", T.OK):
            return
        artist = self.artist
        done_paths: List[Tuple[str, str]] = []

        def work():
            for i, (p, _) in enumerate(targets):
                if app.job_stop.is_set():
                    break
                try:
                    done_paths.append((p, app.tag_file(p, [artist], app.focus_vectors.get(p))))
                except Exception as e:
                    logger.warning(f"Batch tag failed for {p}: {e}")
                app.post_latest("job", app.overlay_progress, i + 1, len(targets), os.path.basename(p))
            return done_paths

        def done(pairs):
            app._end_job()
            moved = {p for p, _ in pairs}
            app.focus_candidates = [c for c in app.focus_candidates if c[0] not in moved]
            app.files = [f for f in app.files if f not in moved]
            app.counts["TOTAL_NAMED"] += len(pairs)
            app.session_tagged += len(pairs)
            app._save_counts()
            app._save_session("review")
            if self.tagged is not None:
                self.tagged.extend((n, [artist]) for _, n in pairs)
            app.status(f"Tagged {len(pairs):,} files as [{artist}]", 5000, T.OK_HI)
            if self.winfo_exists():
                self.render()
        app.run_bg(work, done, "Batch tag failed", on_error=app._end_job)

    def _exit_focus(self):
        self.app.exit_focus()


# =================================================================================
#  ARTIST MANAGER (duplicates, merge, browse)
# =================================================================================
def get_similar_artist_clusters(artists: List[str]) -> List[Dict[str, Any]]:
    """Groups spelling/format variants of the same artist name (union-find over prefix buckets)."""
    norm = {a: normalize_artist_name(a) for a in artists if a and len(normalize_artist_name(a)) >= 2}
    names = list(norm)
    if len(names) < 2:
        return []
    parent = {a: a for a in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    reasons: Dict[str, str] = {}

    def union(a, b, why):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
        reasons.setdefault(find(a), why)

    by_norm: Dict[str, List[str]] = defaultdict(list)
    for a in names:
        by_norm[norm[a]].append(a)
    for group in by_norm.values():
        for o in group[1:]:
            union(group[0], o, "Same name, different formatting")
    buckets: Dict[str, List[str]] = defaultdict(list)
    for n in by_norm:
        buckets[n[:2]].append(n)
    for b in buckets.values():
        for i, n1 in enumerate(b):
            for n2 in b[i + 1:]:
                if abs(len(n1) - len(n2)) > 2 or min(len(n1), len(n2)) < 4:
                    continue
                a1, a2 = by_norm[n1][0], by_norm[n2][0]
                if n1.startswith(n2) or n2.startswith(n1):
                    union(a1, a2, "Name extension / suffix")
                    continue
                sm = difflib.SequenceMatcher(None, n1, n2)
                th = 0.82 if min(len(n1), len(n2)) >= 6 else 0.86
                if sm.quick_ratio() >= th and sm.ratio() >= th:
                    union(a1, a2, f"Likely typo ({sm.ratio():.0%} similar)")
    groups: Dict[str, List[str]] = defaultdict(list)
    for a in names:
        groups[find(a)].append(a)
    out = []
    for root, mem in groups.items():
        if len(mem) > 1:
            why = next((reasons[find(m)] for m in mem if find(m) in reasons), "Variants")
            out.append({"members": sorted(mem, key=lambda x: (len(x), x.lower())), "reason": why})
    out.sort(key=lambda c: (-len(c["members"]), c["members"][0].lower()))
    return out


class ArtistManager(ctk.CTkToplevel):
    PAGE = 40

    def __init__(self, app: VtaggerApp):
        super().__init__(app)
        self.app = app
        self.title("Artists - Vtagger")
        self.geometry("1000x760")
        self.minsize(820, 600)
        self.configure(fg_color=T.BG)
        self.after(30, lambda: oled_titlebar(self))
        self.counts: Dict[str, int] = {}
        self.clusters: Optional[List[Dict[str, Any]]] = None
        self.group: List[str] = []
        self.filter = ""
        self.page = 0
        self.rename_files = ctk.BooleanVar(value=True)

        head = ctk.CTkFrame(self, fg_color="transparent")
        head.pack(fill="x", padx=22, pady=(18, 8))
        label(head, "Artists", 24, weight="bold").pack(side="left")
        self.summary = label(head, "", 12, T.TEXT_2)
        self.summary.pack(side="left", padx=14, pady=(8, 0))
        ctk.CTkCheckBox(head, text="Also rename files in the open folder", variable=self.rename_files, font=T.font(12),
                        text_color=T.TEXT_2, fg_color=T.ACCENT, hover_color=T.ACCENT_HI, border_color=T.BORDER_HI,
                        state="normal" if app.folder else "disabled").pack(side="right")
        self.seg = ctk.CTkSegmentedButton(self, values=["All artists", "Duplicates", "Merge"], command=self._on_tab,
                                          font=T.font(12), selected_color=T.ACCENT, selected_hover_color=T.ACCENT_HI,
                                          unselected_color=T.SURFACE_2, unselected_hover_color=T.HOVER,
                                          fg_color=T.SURFACE_2, corner_radius=8, height=32)
        self.seg.set("All artists")
        self.seg.pack(anchor="w", padx=22, pady=(0, 10))
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(fill="both", expand=True, padx=22, pady=(0, 18))
        self.tab = "All artists"
        self.reload()

    def reload(self):
        def work():
            return self.app.brain.artist_counts()

        def done(c):
            if not self.winfo_exists():
                return
            self.counts = c
            self.summary.configure(text=f"{len(c):,} artists   {sum(c.values()):,} tagged files")
            self.render()
        self.app.run_bg(work, done)

    def _on_tab(self, v):
        self.tab = v
        self.render()

    def _clear(self):
        for w in self.body.winfo_children():
            w.destroy()

    def render(self):
        self._clear()
        {"All artists": self._render_all, "Duplicates": self._render_dupes, "Merge": self._render_merge}[self.tab]()

    def _after_change(self, res=None, msg=""):
        self.clusters = None
        self.app.refresh_artists()
        self.app._refresh_brain_card()
        if msg:
            self.app.status(msg, 5000, T.OK_HI)
        self.reload()

    # ----- all artists -----
    def _render_all(self):
        top = ctk.CTkFrame(self.body, fg_color="transparent")
        top.pack(fill="x", pady=(0, 8))
        ent = ctk.CTkEntry(top, placeholder_text="Filter...", width=320, height=34, corner_radius=8, font=T.font(13),
                           fg_color=T.SURFACE_2, border_color=T.BORDER_HI, text_color=T.TEXT)
        ent.pack(side="left")
        if self.filter:
            ent.insert(0, self.filter)
        info = label(top, "", 12, T.TEXT_2, mono=True)
        info.pack(side="right")
        lst = ctk.CTkScrollableFrame(self.body, fg_color=T.BG, scrollbar_button_color=T.SURFACE_3)
        lst.pack(fill="both", expand=True)
        pager = ctk.CTkFrame(self.body, fg_color="transparent")
        pager.pack(fill="x", pady=(8, 0))

        def fill():
            for w in lst.winfo_children():
                w.destroy()
            for w in pager.winfo_children():
                w.destroy()
            f = self.filter.strip().lower()
            items = sorted(((a, n) for a, n in self.counts.items() if f in a.lower()), key=lambda x: (-x[1], x[0].lower()))
            pages = max(1, math.ceil(len(items) / self.PAGE))
            self.page = max(0, min(self.page, pages - 1))
            info.configure(text=f"{len(items):,} shown")
            for a, n in items[self.page * self.PAGE:(self.page + 1) * self.PAGE]:
                row = ctk.CTkFrame(lst, fg_color=T.SURFACE, corner_radius=10, border_width=1, border_color=T.BORDER)
                row.pack(fill="x", pady=3, padx=2)
                label(row, a, 14, weight="bold").pack(side="left", padx=14, pady=10)
                label(row, f"{n:,} files", 12, T.TEXT_3, mono=True).pack(side="left")
                btn(row, "Delete", lambda a=a: self._delete(a), "danger", width=72, height=28, size=12).pack(side="right", padx=(4, 10))
                btn(row, "Untag", lambda a=a: self._untag([a]), "danger", width=66, height=28, size=12).pack(side="right", padx=4)
                btn(row, "Rename", lambda a=a: self._rename(a), "ghost", width=72, height=28, size=12).pack(side="right", padx=4)
                btn(row, "+ Merge", lambda a=a: self._add_to_group(a), "accent", width=72, height=28, size=12).pack(side="right", padx=4)
                btn(row, "Find", lambda a=a: self._focus(a), "ghost", width=56, height=28, size=12).pack(side="right", padx=4)
            if pages > 1:
                btn(pager, "‹ Prev", lambda: (setattr(self, "page", self.page - 1), fill()), "ghost", width=80,
                    state="normal" if self.page > 0 else "disabled").pack(side="left")
                label(pager, f"Page {self.page + 1} / {pages}", 12, T.TEXT_2, mono=True).pack(side="left", expand=True)
                btn(pager, "Next ›", lambda: (setattr(self, "page", self.page + 1), fill()), "ghost", width=80,
                    state="normal" if self.page < pages - 1 else "disabled").pack(side="right")

        def on_key(_e=None):
            self.filter = ent.get()
            self.page = 0
            fill()
        ent.bind("<KeyRelease>", on_key)
        fill()

    def _focus(self, a):
        if not self.app.files:
            messagebox.showinfo("Find artist", "Open a folder with untagged files first.", parent=self)
            return
        self.app.start_focus_scan(a)

    def _rename(self, a):
        d = ctk.CTkInputDialog(text=f"New name for [{a}]:", title="Rename artist")
        new = sanitize_artist(d.get_input() or "")
        if new and new != a:
            self.app.run_bg(lambda: self.app.unify_artists([a], new, self.rename_files.get()),
                            lambda r: self._after_change(r, f"Renamed [{a}] -> [{new}]  ({r['files']} files)"))

    def _untag(self, names):
        if not messagebox.askyesno("Untag", f"Remove {', '.join(names[:5])}{'...' if len(names) > 5 else ''} from every "
                                   f"filename in the open folder and from the brain?", parent=self):
            return
        self.app.run_bg(lambda: self.app.untag_artists(names),
                        lambda r: self._after_change(r, f"Untagged {r['files']} files"))

    def _delete(self, a):
        if not messagebox.askyesno("Delete artist", f"Forget [{a}] in the brain?\n\nFilenames are not changed "
                                   f"(a later Sync Library would learn it again).", parent=self):
            return
        self.app.run_bg(lambda: self.app.brain.delete_artists([a]), lambda r: self._after_change(r, f"Forgot [{a}]"))

    def _add_to_group(self, a):
        if a not in self.group:
            self.group.append(a)
        self.seg.set("Merge")
        self._on_tab("Merge")

    # ----- duplicates -----
    def _render_dupes(self):
        if self.clusters is None:
            label(self.body, "Looking for duplicate names...", 14, T.TEXT_3).pack(pady=60)
            names = list(self.counts)

            def done(cl):
                self.clusters = cl
                if self.winfo_exists() and self.tab == "Duplicates":
                    self.render()
            self.app.run_bg(lambda: get_similar_artist_clusters(names), done)
            return
        if not self.clusters:
            label(self.body, "No duplicate artist names found.", 15, T.OK_HI).pack(pady=60)
            return
        label(self.body, f"{len(self.clusters):,} groups that look like the same artist", 12, T.TEXT_2).pack(anchor="w", pady=(0, 6))
        lst = ctk.CTkScrollableFrame(self.body, fg_color=T.BG, scrollbar_button_color=T.SURFACE_3)
        lst.pack(fill="both", expand=True)
        for cl in self.clusters[:60]:
            self._cluster_card(lst, cl)
        if len(self.clusters) > 60:
            label(lst, f"+ {len(self.clusters) - 60} more (resolve these first)", 12, T.TEXT_3).pack(pady=10)

    def _cluster_card(self, master, cl):
        mem = cl["members"]
        cd = card(master)
        cd.pack(fill="x", pady=5, padx=2)
        h = ctk.CTkFrame(cd, fg_color="transparent")
        h.pack(fill="x", padx=14, pady=(12, 4))
        label(h, f"{len(mem)} variants", 13, T.ACCENT_HI, weight="bold").pack(side="left")
        label(h, cl["reason"], 12, T.TEXT_3).pack(side="right")
        best = max(mem, key=lambda m: self.counts.get(m, 0))
        var = ctk.StringVar(value=best)
        for m in mem:
            ctk.CTkRadioButton(cd, text=f"{m}   ({self.counts.get(m, 0):,} files)", variable=var, value=m, font=T.font(13),
                               text_color=T.TEXT, fg_color=T.ACCENT, hover_color=T.ACCENT_HI,
                               border_color=T.BORDER_HI).pack(anchor="w", padx=18, pady=3)
        row = ctk.CTkFrame(cd, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(6, 12))
        custom = ctk.CTkEntry(row, placeholder_text="or type a new name", width=240, height=30, corner_radius=8,
                              fg_color=T.SURFACE_2, border_color=T.BORDER_HI, font=T.font(12))
        custom.pack(side="left")

        def unify():
            target = sanitize_artist(custom.get()) or var.get()
            self.app.run_bg(lambda: self.app.unify_artists(mem, target, self.rename_files.get()),
                            lambda r: (self._drop_cluster(cl, cd), self._after_change(r, f"Merged into [{target}] ({r['files']} files renamed)")))
        btn(row, "Untag all", lambda: self._untag(mem), "danger", width=90, height=30, size=12).pack(side="right", padx=(6, 0))
        btn(row, "Merge", unify, "ok", width=90, height=30, size=12).pack(side="right")

    def _drop_cluster(self, cl, cd):
        try:
            cd.destroy()
        except tk.TclError:
            pass

    # ----- manual merge -----
    def _render_merge(self):
        top = card(self.body)
        top.pack(fill="x", pady=(0, 10))
        label(top, "Add artists to merge into one name", 13, T.TEXT_2).pack(anchor="w", padx=16, pady=(14, 6))
        row = ctk.CTkFrame(top, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=(0, 14))
        ge = GhostEntry(row, "Artist name", lambda t: self.app.completer.complete_text(t), height=40, size=14,
                        options=self.app.completer.options)
        ge.pack(side="left", fill="x", expand=True, padx=(0, 10))

        def add(_e=None):
            v = sanitize_artist(ge.get())
            if v:
                v = self.app.completer.canonical(v)
                if v not in self.group:
                    self.group.append(v)
                self.render()
            return "break"
        ge.bind_key("<Tab>", lambda e: (ge.accept(), "break")[1])
        ge.bind_key("<Return>", add)
        btn(row, "Add", add, "accent", width=80, height=40).pack(side="right")
        self.after(50, ge.focus)

        box = card(self.body)
        box.pack(fill="both", expand=True)
        h = ctk.CTkFrame(box, fg_color="transparent")
        h.pack(fill="x", padx=16, pady=(14, 6))
        label(h, f"Selected ({len(self.group)})", 14, weight="bold").pack(side="left")
        if self.group:
            btn(h, "Clear", lambda: (self.group.clear(), self.render()), "subtle", width=60, height=26, size=12).pack(side="right")
        if not self.group:
            label(box, "Nothing selected yet. Add names above or use '+ Merge' in All artists.", 13, T.TEXT_3).pack(pady=40)
            return
        chips = ctk.CTkFrame(box, fg_color="transparent")
        chips.pack(fill="x", padx=12)
        for i, a in enumerate(self.group):
            ch = ctk.CTkFrame(chips, fg_color=T.SURFACE_3, corner_radius=8)
            ch.grid(row=i // 4, column=i % 4, padx=4, pady=4, sticky="w")
            label(ch, f"{a}  ({self.counts.get(a, 0):,})", 12).pack(side="left", padx=(10, 4), pady=4)
            btn(ch, "✕", lambda a=a: (self.group.remove(a), self.render()), "subtle", width=24, height=24, size=11).pack(side="left", padx=(0, 4))
        act = ctk.CTkFrame(box, fg_color="transparent")
        act.pack(fill="x", padx=16, pady=16, side="bottom")
        label(act, "Merge into", 13, T.TEXT_2).pack(side="left", padx=(0, 8))
        best = max(self.group, key=lambda m: self.counts.get(m, 0))
        var = ctk.StringVar(value=best)
        ctk.CTkOptionMenu(act, values=self.group, variable=var, width=220, font=T.font(13), fg_color=T.SURFACE_2,
                          button_color=T.SURFACE_3, button_hover_color=T.HOVER, dropdown_fg_color=T.SURFACE_2,
                          dropdown_hover_color=T.ACCENT_DIM, text_color=T.TEXT).pack(side="left")
        custom = ctk.CTkEntry(act, placeholder_text="or new name", width=200, height=32, corner_radius=8,
                              fg_color=T.SURFACE_2, border_color=T.BORDER_HI, font=T.font(12))
        custom.pack(side="left", padx=10)
        group = list(self.group)

        def merge():
            target = sanitize_artist(custom.get()) or var.get()
            self.app.run_bg(lambda: self.app.unify_artists(group, target, self.rename_files.get()),
                            lambda r: (self.group.clear(), self._after_change(r, f"Merged {len(group)} into [{target}] ({r['files']} files renamed)")))
        btn(act, "Merge", merge, "ok", width=100).pack(side="right")
        btn(act, "Untag all", lambda: self._untag(group), "danger", width=100).pack(side="right", padx=8)


# =================================================================================
#  SETTINGS
# =================================================================================
class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, app: VtaggerApp):
        super().__init__(app)
        self.app = app
        s = app.settings
        self.title("Settings")
        self.configure(fg_color=T.SURFACE)
        self.geometry("560x520")
        self.resizable(False, False)
        self.transient(app)
        self.after(30, lambda: oled_titlebar(self))
        label(self, "Settings", 22, weight="bold").pack(anchor="w", padx=24, pady=(20, 10))

        def row(title, hint):
            f = ctk.CTkFrame(self, fg_color="transparent")
            f.pack(fill="x", padx=24, pady=7)
            t = ctk.CTkFrame(f, fg_color="transparent")
            t.pack(side="left", fill="x", expand=True)
            label(t, title, 13, weight="bold", anchor="w").pack(anchor="w")
            label(t, hint, 11, T.TEXT_3, anchor="w", justify="left", wraplength=320).pack(anchor="w")
            return f

        labels = {b.label: k for k, b in BACKBONES.items()}
        self.bb = ctk.StringVar(value=BACKBONES.get(s["backbone"], SigLIP2Backbone).label)
        ctk.CTkOptionMenu(row("Vision model", "Changing it needs a restart and a Sync Library (embeddings are model specific)."),
                          values=list(labels), variable=self.bb, width=200, fg_color=T.SURFACE_2, button_color=T.SURFACE_3,
                          dropdown_fg_color=T.SURFACE_2, text_color=T.TEXT).pack(side="right")
        self.labels = labels
        self.tta = ctk.BooleanVar(value=bool(s["tta"]))
        ctk.CTkSwitch(row("Flip test-time augmentation", "Averages each image with its mirror. Slightly more accurate, half the speed."),
                      text="", variable=self.tta, progress_color=T.ACCENT).pack(side="right")
        self.bs = ctk.StringVar(value=str(s["batch_size"]))
        ctk.CTkOptionMenu(row("GPU batch size", "Lower it if you see out-of-memory errors."), values=["8", "16", "32", "48", "64"],
                          variable=self.bs, width=90, fg_color=T.SURFACE_2, button_color=T.SURFACE_3).pack(side="right")
        self.io = ctk.StringVar(value=str(s["io_workers"]))
        ctk.CTkOptionMenu(row("Decode threads", "Parallel file readers feeding the GPU."), values=["4", "6", "8", "12", "16", "24"],
                          variable=self.io, width=90, fg_color=T.SURFACE_2, button_color=T.SURFACE_3).pack(side="right")
        self.k = ctk.StringVar(value=str(s["knn_k"]))
        ctk.CTkOptionMenu(row("Neighbours (k)", "How many similar files vote on each suggestion."), values=["20", "30", "50", "80", "120"],
                          variable=self.k, width=90, fg_color=T.SURFACE_2, button_color=T.SURFACE_3).pack(side="right")
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(fill="x", padx=24, pady=(18, 20), side="bottom")
        btn(bar, "Cancel", self.destroy, "ghost", width=110).pack(side="left")
        btn(bar, "Save", self.save, "primary", width=130).pack(side="right")
        self.after(80, self.grab_set)

    def save(self):
        s = self.app.settings
        new_bb = self.labels[self.bb.get()]
        restart = new_bb != s["backbone"]
        s.update({"backbone": new_bb, "tta": bool(self.tta.get()), "batch_size": int(self.bs.get()),
                  "io_workers": int(self.io.get()), "knn_k": int(self.k.get())})
        s.save()
        idx = self.app.brain.index
        idx.k = s["knn_k"]
        if self.app.brain.embedder:
            e = self.app.brain.embedder
            e.tta, e.batch_size, e.workers = s["tta"], s["batch_size"], s["io_workers"]
        self.destroy()
        if restart:
            messagebox.showinfo("Restart needed", "Restart Vtagger to switch the vision model, then run Sync Library.")
        else:
            self.app._recalibrate()


# =================================================================================
#  ENTRY POINT
# =================================================================================
def main():
    setup_logging()
    logger.info(f"Vtagger {APP_VERSION} starting (Python {sys.version.split()[0]}, torch {torch.__version__}, "
                f"CUDA {'yes - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO'})")
    if not torch.cuda.is_available():
        r = tk.Tk()
        r.withdraw()
        if getattr(sys, "frozen", False):
            fix = "Fix: install or update your NVIDIA driver."
        else:
            fix = (f'Fix: run\n"{sys.executable}" -m pip install --upgrade torch torchvision '
                   f"--index-url {_TORCH_INDEX}")
        go = messagebox.askyesno("No CUDA GPU", "PyTorch cannot see your NVIDIA GPU, so Vtagger would run on the CPU "
                                 f"(very slow).\n\n{fix}\n\nContinue on CPU anyway?")
        r.destroy()
        if not go:
            return
    app = VtaggerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
