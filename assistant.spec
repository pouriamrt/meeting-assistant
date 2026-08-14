# PyInstaller build. Run through build.ps1, not directly.
#
# onedir, not onefile. A one-file build unpacks ~280 MB to %TEMP% on every
# launch, which is both a slow start and the single biggest trigger for
# antivirus false positives on PyInstaller output. The zipped folder is the
# same download size and starts immediately.

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

# Compiled extensions these three ship outside their Python packages. Without
# collecting them the exe builds cleanly and then dies at first use, which is
# the worst shape of failure: it looks packaged.
binaries = (
    collect_dynamic_libs("ctranslate2")
    + collect_dynamic_libs("onnxruntime")
    + collect_dynamic_libs("av")
)

# faster_whisper ships the Silero VAD network as faster_whisper/assets/
# silero_vad_v6.onnx. It is DATA, not a dynamic library, so collect_dynamic_libs
# above walks straight past it. Leaving it out builds and launches fine and
# then fails on every single audio chunk with NO_SUCHFILE, while the gate
# loop's error handling keeps the app upright - running, and transcribing
# nothing. Measured, not theoretical: the first build did exactly that.
datas = collect_data_files("faster_whisper") + collect_data_files("onnxruntime")

hiddenimports = (
    collect_submodules("faster_whisper")
    + collect_submodules("onnxruntime")
    + ["pyaudiowpatch", "soxr", "mss", "pynput.keyboard._win32", "pynput.mouse._win32"]
)

a = Analysis(
    ["run_assistant.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    excludes=[
        # Pulled in transitively and never used at runtime. Dropping them is
        # worth ~100 MB; anything actually needed shows up as an ImportError on
        # the smoke run, not as a subtle misbehaviour.
        "tkinter.test", "test", "unittest", "pydoc_data",
        "matplotlib", "PIL", "pandas", "scipy", "IPython", "pytest", "hypothesis",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="meeting-assistant",
    debug=False,
    strip=False,
    upx=False,          # UPX compression is itself an antivirus heuristic
    console=False,      # no console window; logs go to the file below
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="meeting-assistant",
)
