"""py2app build for a fully self-contained dist/OverlayCat.app.

Usage:
    cd packaging && ../.venv/bin/python -B setup_app.py py2app

Produces packaging/dist/OverlayCat.app (move to repo dist/ and ad-hoc sign
afterwards; see README). Requires the python.org framework
Python (py2app standalone mode needs a framework build).

Build-time choices, all deliberate:
- the icon is regenerated on every build from scripts/make_icon.py via
  iconutil (loud subprocess failures, no stale icns reuse);
- 'packages' forces overlay + every runtime dependency to ship as plain
  source trees under Resources/lib/python3.12/ instead of byte-compiled
  members of site-packages.zip — .pyc files embed absolute build paths
  (repo / venv) in code objects, source files do not, which keeps the
  bundle free of references to this checkout;
- numpy is force-included as a package (py2app's autopackages recipe does
  the same) so numpy/.dylibs BLAS dylibs are collected as real files;
- PIL ships as a full package so all plugin modules are importable;
- argv_emulation stays absent (False by default): it drags in Carbon and
  hangs on modern macOS;
- everything heavy that lives in this venv for the content pipeline is
  excluded explicitly; the runtime must never pull it.
"""

import plistlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from setuptools import setup

sys.dont_write_bytecode = True

PACKAGING_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGING_DIR.parent
APP_BUNDLE = PACKAGING_DIR / "dist" / "OverlayCat.app"

# modulegraph must resolve 'overlay' (lives at the repo root, not in
# site-packages); inserted at the front so the checkout always wins.
sys.path.insert(0, str(REPO_ROOT))

CLIPS_DIR = REPO_ROOT / "clips"
ENTRY_SCRIPT = PACKAGING_DIR / "overlaycat_entry.py"
ICONSET_DIR = PACKAGING_DIR / "build" / "cat.iconset"
ICNS_PATH = PACKAGING_DIR / "build" / "cat.icns"
EXPECTED_CLIP_PACKAGES = 18


def validate_clips() -> list[str]:
    """Return data_files sources for clips; raise loudly on a broken library."""
    index_path = CLIPS_DIR / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"{index_path} missing — run pipeline/build_index.py first")
    clip_dirs = sorted(
        p for p in CLIPS_DIR.iterdir()
        if p.is_dir() and (p / "manifest.json").is_file())
    if len(clip_dirs) != EXPECTED_CLIP_PACKAGES:
        raise RuntimeError(
            f"expected {EXPECTED_CLIP_PACKAGES} clip packages under "
            f"{CLIPS_DIR}, found {len(clip_dirs)}: "
            f"{[p.name for p in clip_dirs]}")
    return [str(index_path)] + [str(p) for p in clip_dirs]


def build_icns() -> None:
    """Regenerate packaging/build/cat.icns from scripts/make_icon.py."""
    ICNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-B", str(REPO_ROOT / "scripts" / "make_icon.py"),
         str(ICONSET_DIR)],
        check=True)
    subprocess.run(
        ["iconutil", "-c", "icns", str(ICONSET_DIR), "-o", str(ICNS_PATH)],
        check=True)


PLIST = {
    "CFBundleIdentifier": "ai.sobek.overlaycat",
    "CFBundleName": "OverlayCat",
    "CFBundleDisplayName": "Overlay Cat",
    "CFBundleShortVersionString": "0.5.0",
    "CFBundleVersion": "0.5.0",
    "LSUIElement": True,
    "NSHighResolutionCapable": True,
    "LSMinimumSystemVersion": "13.0",
}

OPTIONS = {
    "plist": PLIST,
    "iconfile": str(ICNS_PATH),
    "argv_emulation": False,
    # Ship as plain source packages: our package, pyobjc wrappers (their
    # lazy-import machinery loads submodules modulegraph cannot trace),
    # numpy (BLAS dylibs), PIL (plugins). PyObjCTools (imported by
    # objc._convenience) is a namespace package that py2app cannot
    # bootstrap via 'packages'; modulegraph zips it instead, which is safe:
    # py2app byte-compiles zip members with a RELATIVE purported filename
    # (module identifier), so no build path is baked into the zip.
    "packages": [
        "overlay",
        "objc",
        "AppKit",
        "Foundation",
        "CoreFoundation",
        "Quartz",
        "numpy",
        "PIL",
    ],
    # Heavy pipeline-only deps plus anything else known to lurk in this venv;
    # the runtime must not pull a single one of these.
    "excludes": [
        "torch", "torchvision", "torchaudio", "transformers", "cv2",
        "scipy", "matplotlib", "skimage", "kornia", "kornia_rs", "timm",
        "sympy", "networkx", "mpmath", "einops", "safetensors", "tokenizers",
        "huggingface_hub", "hf_xet", "regex", "yaml", "tifffile", "imageio",
        "lazy_loader", "pypdf", "rich", "pygments", "markdown_it", "mdurl",
        "click", "shellingham", "httpx", "httpcore", "h11", "anyio",
        "sniffio", "idna", "certifi", "charset_normalizer", "requests",
        "urllib3", "jinja2", "markupsafe", "fsspec", "filelock", "tqdm",
        "pytest", "pluggy", "iniconfig", "_pytest", "py2app", "modulegraph",
        "macholib", "altgraph", "setuptools", "pkg_resources", "pip",
        "wheel", "tkinter", "annotated_doc",
    ],
    "strip": True,
}


def strip_pycache(app: Path) -> int:
    """Remove __pycache__ dirs copied verbatim from site-packages: their
    .pyc embed absolute build-machine paths in code objects. Python falls
    back to the .py sources at runtime (and the entry point sets
    sys.dont_write_bytecode)."""
    dirs = sorted(p for p in app.rglob("__pycache__") if p.is_dir())
    for d in dirs:
        shutil.rmtree(d)
    return len(dirs)


def scrub_zip_pycache(app: Path) -> int:
    """Rewrite python312.zip without __pycache__ members: pip-compiled
    artifacts (e.g. numpy/random/_examples) ride along as package data and
    embed the venv path. Importable bytecode is unaffected — py2app names
    those members module.pyc, never __pycache__/*.cpython-312.pyc."""
    zip_path = app / "Contents" / "Resources" / "lib" / "python312.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(f"expected stdlib zip missing: {zip_path}")
    tmp_path = zip_path.with_suffix(".zip.tmp")
    dropped = 0
    with zipfile.ZipFile(zip_path) as src, \
            zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            if "__pycache__/" in item.filename:
                dropped += 1
                continue
            dst.writestr(item, src.read(item.filename))
    tmp_path.replace(zip_path)
    return dropped


def neutralize_plist(app: Path) -> None:
    """Drop py2app's informational PythonInfoDict/PythonExecutable key (the
    absolute build venv python). The standalone launcher locates the
    embedded framework via PyRuntimeLocations, not this key."""
    plist_path = app / "Contents" / "Info.plist"
    info = plistlib.loads(plist_path.read_bytes())
    py_info = info.get("PythonInfoDict")
    if not py_info or "PythonExecutable" not in py_info:
        raise RuntimeError(
            f"{plist_path}: expected py2app PythonInfoDict/PythonExecutable "
            f"(layout changed — re-audit path leaks)")
    del py_info["PythonExecutable"]
    plist_path.write_bytes(plistlib.dumps(info))


def scan_for_build_paths(app: Path) -> None:
    """Raise if any bundle file — or any python312.zip member — still
    references this checkout (covers the venv too: it lives inside it)."""
    needle = str(REPO_ROOT).encode()
    hits = []
    for p in sorted(app.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        if needle in p.read_bytes():
            hits.append(str(p))
    zip_path = app / "Contents" / "Resources" / "lib" / "python312.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(f"expected stdlib zip missing: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if needle in zf.read(name):
                hits.append(f"{zip_path}!{name}")
    if hits:
        raise RuntimeError(
            f"bundle still references {REPO_ROOT}:\n  " + "\n  ".join(hits))


def post_process(app: Path) -> None:
    if not app.is_dir():
        raise FileNotFoundError(f"py2app reported success but {app} is missing")
    removed = strip_pycache(app)
    dropped = scrub_zip_pycache(app)
    neutralize_plist(app)
    scan_for_build_paths(app)
    print(f"post-process ok: removed {removed} __pycache__ dirs, dropped "
          f"{dropped} __pycache__ zip members, PythonExecutable plist key "
          f"dropped, no build-path references")


def main() -> None:
    if not ENTRY_SCRIPT.is_file():
        raise FileNotFoundError(f"entry script missing: {ENTRY_SCRIPT}")
    data_sources = validate_clips()
    build_icns()
    setup(
        name="OverlayCat",
        app=[str(ENTRY_SCRIPT)],
        data_files=[("clips", data_sources)],
        options={"py2app": OPTIONS},
        setup_requires=["py2app"],
    )
    if "py2app" in sys.argv[1:]:
        post_process(APP_BUNDLE)


if __name__ == "__main__":
    main()
