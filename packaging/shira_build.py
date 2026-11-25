"""Build backend for the SHIRA wheels (PEP 517, standard library only).

A SHIRA wheel carries the open Python layer and one prebuilt, closed libshira for one platform.
The Python layer finds the library two folders above its own package, so the wheel keeps the
layout the code was written and tested in:

    shira_dist/python/shira/        the drop-in modules
    shira_dist/python/shira_cpp/    the ctypes wrappers
    shira_dist/lib/libshira.*       the compiled library
    shira.pth                       one line, "shira_dist/python", which puts both packages on the path

Everything is read from pyproject.toml ([project] and [tool.shira]). The platform is the host's,
or SHIRA_WHEEL_PLATFORM (a key of [tool.shira.platforms]) to pack another platform's wheel.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import platform
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

LOG = logging.getLogger("shira_build")
ROOT = Path(__file__).resolve().parents[1]
MAX_FILES = 2000
FIXED_TIME = (2026, 11, 25, 0, 0, 0)
SKIP_PARTS = frozenset({"__pycache__", ".ruff_cache", "catboost_info", ".DS_Store"})


def _config() -> dict:
    """pyproject.toml as a dict."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    assert "project" in data and "shira" in data.get("tool", {}), "pyproject.toml lacks [project] or [tool.shira]"
    return data


def _platform_key(cfg: dict) -> str:
    """Which entry of [tool.shira.platforms] this build packs."""
    forced = os.environ.get("SHIRA_WHEEL_PLATFORM", "").strip()
    host = {("darwin", "arm64"): "macos-arm64", ("linux", "x86_64"): "linux-x86_64"}
    key = forced or host.get((sys.platform, platform.machine().lower()), "")
    assert key in cfg["tool"]["shira"]["platforms"], f"no SHIRA build for this platform: {key or sys.platform}"
    return key


def _metadata(cfg: dict) -> str:
    """The METADATA file."""
    project = cfg["project"]
    lines = ["Metadata-Version: 2.4", f"Name: {project['name']}", f"Version: {project['version']}",
             f"Summary: {project['description']}", f"Requires-Python: {project['requires-python']}"]
    lines += [f"License-File: {name}" for name in cfg["tool"]["shira"]["license-files"]]
    lines += [f"Classifier: {item}" for item in project.get("classifiers", [])]
    lines += [f"Requires-Dist: {item}" for item in project["dependencies"]]
    readme = ROOT / project["readme"]
    assert readme.is_file(), f"readme not found: {readme}"
    body = readme.read_text()
    return "\n".join(lines) + "\nDescription-Content-Type: text/markdown\n\n" + body


def _payload(cfg: dict, key: str) -> list[tuple[str, Path]]:
    """(path inside the wheel, source file) for the Python layer, the library and shira.pth's packages."""
    shira = cfg["tool"]["shira"]
    out: list[tuple[str, Path]] = []
    src = ROOT / shira["python-dir"]
    for package in shira["packages"]:
        files = sorted(p for p in (src / package).rglob("*") if p.is_file() and not SKIP_PARTS.intersection(p.parts))
        assert files and all(p.suffix == ".py" for p in files), f"{package}: expected only .py files"
        out += [(f"shira_dist/python/{p.relative_to(src).as_posix()}", p) for p in files]
    lib = ROOT / shira["platforms"][key]["library"]
    assert lib.is_file(), f"compiled library not found: {lib}"
    out.append((f"shira_dist/lib/{lib.name}", lib))
    assert len(out) <= MAX_FILES, f"unexpected file count {len(out)}"
    return out


def _record_line(name: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{name},sha256={digest},{len(data)}"


def _write(archive: zipfile.ZipFile, name: str, data: bytes, record: list[str], mode: int = 0o644) -> None:
    """Add one file with a fixed date (the same inputs give the same wheel)."""
    info = zipfile.ZipInfo(name, FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o100000 | mode) << 16
    archive.writestr(info, data)
    record.append(_record_line(name, data))
    assert len(record) <= MAX_FILES + 20, "too many wheel entries"


def build_wheel(wheel_directory: str, config_settings: dict | None = None, metadata_directory: str | None = None) -> str:
    """PEP 517: build the wheel for one platform; return its file name."""
    cfg = _config()
    key = _platform_key(cfg)
    project, shira = cfg["project"], cfg["tool"]["shira"]
    tag = f"{shira['python-tag']}-none-{shira['platforms'][key]['tag']}"
    stem = f"{project['name']}-{project['version']}"
    info_dir = f"{stem}.dist-info"
    name = f"{stem}-{tag}.whl"
    record: list[str] = []
    with zipfile.ZipFile(Path(wheel_directory) / name, "w") as archive:
        for inside, source in _payload(cfg, key):
            mode = 0o755 if inside.startswith("shira_dist/lib/") else 0o644
            _write(archive, inside, source.read_bytes(), record, mode)
        _write(archive, "shira.pth", b"shira_dist/python\n", record)
        _write(archive, f"{info_dir}/METADATA", _metadata(cfg).encode(), record)
        wheel = f"Wheel-Version: 1.0\nGenerator: shira_build\nRoot-Is-Purelib: false\nTag: {tag}\n"
        _write(archive, f"{info_dir}/WHEEL", wheel.encode(), record)
        for licence in shira["license-files"]:
            _write(archive, f"{info_dir}/licenses/{licence}", (ROOT / licence).read_bytes(), record)
        record.append(f"{info_dir}/RECORD,,")
        archive.writestr(zipfile.ZipInfo(f"{info_dir}/RECORD", FIXED_TIME), "\n".join(record) + "\n")
    assert (Path(wheel_directory) / name).stat().st_size > 0, "empty wheel"
    return name


def build_sdist(sdist_directory: str, config_settings: dict | None = None) -> str:
    """PEP 517: the package folder as it is (Python layer, prebuilt libraries, licences). The C++ core
    is not part of it: SHIRA cannot be built from this archive, only repacked."""
    cfg = _config()
    stem = f"{cfg['project']['name']}-{cfg['project']['version']}"
    name = f"{stem}.tar.gz"
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and not SKIP_PARTS.intersection(p.parts)
                   and "dist" not in p.relative_to(ROOT).parts[:1])
    assert 0 < len(files) <= MAX_FILES, f"unexpected file count {len(files)}"
    with tarfile.open(Path(sdist_directory) / name, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=f"{stem}/{path.relative_to(ROOT).as_posix()}")
        meta = _metadata(cfg).encode()
        info = tarfile.TarInfo(f"{stem}/PKG-INFO")
        info.size = len(meta)
        archive.addfile(info, io.BytesIO(meta))
    return name


def get_requires_for_build_wheel(config_settings: dict | None = None) -> list[str]:
    return []


def get_requires_for_build_sdist(config_settings: dict | None = None) -> list[str]:
    return []


def main(argv: list[str]) -> int:
    """Direct use, with no pip needed: shira_build.py <dist folder> [platform key]."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(argv) not in (2, 3):
        LOG.error("usage: shira_build.py <dist folder> [platform key]")
        return 2
    if len(argv) == 3:
        os.environ["SHIRA_WHEEL_PLATFORM"] = argv[2]
    Path(argv[1]).mkdir(parents=True, exist_ok=True)
    LOG.info("built %s", build_wheel(argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
