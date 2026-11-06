"""SynDrone archive access: member names, cached archive indexes, local-first reading.

Archives: `<base>/<Town>_Opt_120_color.zip` (rgb, semantic, camera, bboxes) and
`<Town>_Opt_120_depth.zip` (depth). Members are named
`<Town>_Opt_120/ClearNoon/{bboxes, height<H>m/{camera,rgb,semantic,depth}}/%05d.<ext>`
(read from the archive listings, 2026-11-04).

`MemberStore` reads a member from the local cache if present, otherwise from the remote
archive (CRC-checked), optionally keeping a local copy. Depth is normally streamed and not
kept.
"""

from __future__ import annotations

import json
import os
import shutil
import warnings
import zlib
from collections.abc import Iterable, Iterator
from pathlib import Path

from aerialrange.data.remote_zip import HttpRangeSource, Member, RemoteZip, write_atomic

WEATHER = "ClearNoon"
HEIGHTS = (20, 50, 80)
EXT = {"rgb": "jpg", "semantic": "png", "depth": "png", "camera": "json", "bboxes": "json"}
ARCHIVE_OF = {
    "rgb": "color",
    "semantic": "color",
    "camera": "color",
    "bboxes": "color",
    "depth": "depth",
}


def member_name(town: str, modality: str, frame: int, height: int | None = None) -> str:
    """Archive member name of one file. `height` is required except for `bboxes`."""
    base = f"{town}_Opt_120/{WEATHER}"
    if modality == "bboxes":
        return f"{base}/bboxes/{frame:05d}.json"
    if height not in HEIGHTS:
        raise ValueError(f"height must be one of {HEIGHTS}, got {height}")
    return f"{base}/height{height}m/{modality}/{frame:05d}.{EXT[modality]}"


def parse_member_name(name: str) -> tuple[str, str]:
    """(town, modality) of a member name."""
    return name.split("_Opt_120", 1)[0], name.split("/")[-2]


def archive_url(base_url: str, town: str, archive: str) -> str:
    return f"{base_url.rstrip('/')}/{town}_Opt_120_{archive}.zip"


class _IndexedZip(RemoteZip):
    """RemoteZip built from a cached member index instead of reading the central directory."""

    def __init__(
        self, src: HttpRangeSource, members: dict[str, Member], limit: dict[str, int]
    ) -> None:
        self.src = src
        self.members = members
        self._limit = limit


def open_archive(url: str, index_dir: Path) -> RemoteZip:
    """Open a remote archive, reusing a cached index (`index_dir/<name>.index.json`) if it
    matches the archive's current size; otherwise read the central directory and cache it."""
    src = HttpRangeSource(url)
    path = index_dir / (url.rsplit("/", 1)[-1] + ".index.json")
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        if d["size"] == src.size:
            members = {m[0]: Member(*m) for m in d["members"]}
            return _IndexedZip(src, members, d["limit"])
    rz = RemoteZip(src)
    payload = {
        "url": url,
        "size": src.size,
        "members": [
            [m.name, m.offset, m.csize, m.usize, m.crc, m.method] for m in rz.members.values()
        ],
        "limit": rz._limit,
    }
    write_atomic(path, json.dumps(payload).encode("utf-8"))
    return rz


def free_gb(path: Path) -> float:
    """Free space (GB) on the drive holding `path` (or its nearest existing parent)."""
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


class MemberStore:
    """Local-first access to SynDrone members.

    `root` is the local cache (`<data_root>/syndrone`); local files mirror member names.
    Remote archives are opened lazily and only if a member is missing locally.

    A local file is used only if its size and CRC-32 match the cached archive index
    (`root/_index`), when one exists. A file that fails (for example zero bytes left by a
    sleep or power loss mid-write) is moved to `root/_quarantine/` (never deleted) and
    the member is fetched again; without a `base_url` that is an error, never bad data.
    """

    def __init__(self, root: Path, base_url: str | None = None) -> None:
        self.root = Path(root)
        self.base_url = base_url
        self._zips: dict[tuple[str, str], RemoteZip] = {}
        self._listed: dict[tuple[str, str], dict[str, tuple[int, int]] | None] = {}
        self.n_quarantined = 0

    def local_path(self, name: str) -> Path:
        return self.root / name

    def _zip(self, town: str, archive: str) -> RemoteZip:
        key = (town, archive)
        if key not in self._zips:
            if self.base_url is None:
                raise FileNotFoundError("member not cached locally and no base_url given")
            url = archive_url(self.base_url, town, archive)
            self._zips[key] = open_archive(url, self.root / "_index")
        return self._zips[key]

    def _expected(self, town: str, archive: str) -> dict[str, tuple[int, int]] | None:
        """(size, CRC-32) per member from the cached index, or None if not cached."""
        key = (town, archive)
        if key not in self._listed:
            path = self.root / "_index" / f"{town}_Opt_120_{archive}.zip.index.json"
            self._listed[key] = (
                {
                    m[0]: (int(m[3]), int(m[4]))
                    for m in json.loads(path.read_text("utf-8"))["members"]
                }
                if path.exists()
                else None
            )
        return self._listed[key]

    def _local_ok(self, name: str, data: bytes, town: str, archive: str) -> bool:
        listed = self._expected(town, archive)
        if listed is None or name not in listed:
            return True  # nothing to check against (no index cached)
        usize, crc = listed[name]
        return len(data) == usize and zlib.crc32(data) == crc

    def _quarantine(self, name: str) -> None:
        dst = self.root / "_quarantine" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.local_path(name), dst)
        self.n_quarantined += 1
        warnings.warn(f"cached SynDrone member failed its CRC, quarantined: {name}", stacklevel=3)

    def read(self, name: str, keep: bool = False) -> bytes:
        return next(iter(self.iter_read([name], keep=keep)))[1]

    def iter_read(self, names: Iterable[str], keep: bool = False) -> Iterator[tuple[str, bytes]]:
        """Yield (name, data) for many members, fetching missing ones in merged spans.

        Local members come first, then remote ones in archive order (not input order).
        A quarantined member is fetched again and re-kept.
        """
        remote: dict[tuple[str, str], list[str]] = {}
        refetch: set[str] = set()
        for n in dict.fromkeys(names):
            p = self.local_path(n)
            town, modality = parse_member_name(n)
            archive = ARCHIVE_OF[modality]
            if p.exists():
                data = p.read_bytes()
                if self._local_ok(n, data, town, archive):
                    yield n, data
                    continue
                self._quarantine(n)
                refetch.add(n)
            remote.setdefault((town, archive), []).append(n)
        for (town, archive), ns in remote.items():
            for n, data in self._zip(town, archive).iter_members(ns):
                if keep or n in refetch:
                    write_atomic(self.local_path(n), data)
                yield n, data
