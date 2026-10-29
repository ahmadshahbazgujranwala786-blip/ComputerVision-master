# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

PROJECT_ROOT = Path(SPECPATH)

hiddenimports = [
    "tkinter",
    "psycopg",
    "psycopg_binary",
    "cv2",
    "onnxruntime",
    "faiss",
    "sounddevice",
    "soundfile",
    "pywhatkit",
    "src.app.camera.camera",
    "src.app.embedding.scrfd",
    "src.app.embedding.arc_face",
    "src.app.alignment.alignment",
    "src.vector_store",
    "src.app.tts.notifier",
    "src.app.tts",
    "src.app.tts.service",
]
hiddenimports += collect_submodules("supertonic")

a = Analysis(
    [str(PROJECT_ROOT / "core_wrapper.py")],
    pathex=[str(PROJECT_ROOT), str(PROJECT_ROOT / "src")],
    binaries=[],
    datas=[
        (str(PROJECT_ROOT / "models"), "models"),
        (str(PROJECT_ROOT / "template"), "template"),
        (str(PROJECT_ROOT / "data"), "data"),
        (str(PROJECT_ROOT / ".env"), "."),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "tensorflow"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="SmartAttendance.Core.SIMPLE.TEST",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
)
