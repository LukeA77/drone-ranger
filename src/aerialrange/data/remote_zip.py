"""Read single members of a large remote ZIP archive with HTTP range requests.

SynDrone ships each town as one 5 to 8 GB archive. Reading only the members a step needs
(D-702) avoids holding whole archives on a nearly full disk:
the central directory is read once from the end of the file, then members are fetched in
merged byte spans, decompressed in memory and checked against the CRC-32 recorded in the
archive. Nothing is trusted that fails its CRC.

Only the standard library is used. The byte source is injectable so tests run offline.
"""

from __future__ import annotations

import http.client
import io
import os
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

# Local file header: signature, version, flags, method, time, date, crc, csize, usize,
# name length, extra length (APPNOTE.TXT 4.3.7).
_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
_LOCAL_SIG = 0x04034B50


class RangeSource:
    """A byte source that supports reading [start, end) and knows its size."""

    size: int

    def read_range(self, start: int, end: int) -> bytes:  # pragma: no cover - interface
        raise NotImplementedError


class BytesSource(RangeSource):
    """In-memory source (tests, or a downloaded archive)."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.size = len(data)
        self.calls = 0

    def read_range(self, start: int, end: int) -> bytes:
        self.calls += 1
        return self.data[start:end]


class HttpRangeSource(RangeSource):
    """HTTP(S) source using `Range: bytes=a-b`, with retries and length checks.

    The server must answer 206 with exactly the bytes asked for; anything else is retried
    and finally raised, never silently accepted. Each request also has an overall deadline,
    `deadline_base_s` plus the time the transfer takes at `min_rate_bps`, checked between
    1 MB reads: a connection that trickles or half-dies (seen after the laptop slept,
    2026-11-05) is abandoned and retried instead of stalling a run for hours.
    """

    def __init__(
        self,
        url: str,
        retries: int = 8,
        timeout_s: float = 30.0,
        backoff_s: float = 2.0,
        opener: Callable[..., object] | None = None,
        deadline_base_s: float = 60.0,
        min_rate_bps: float = 50e3,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.url = url
        self.retries = retries
        self.timeout_s = timeout_s
        self.backoff_s = backoff_s
        self._open = opener or urllib.request.urlopen
        self.deadline_base_s = deadline_base_s
        self.min_rate_bps = min_rate_bps
        self._clock = clock
        self.bytes_read = 0
        self.size = self._head_size()

    def _with_retries(self, fn: Callable[[], bytes | int]) -> bytes | int:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                return fn()
            except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as e:
                last = e
                time.sleep(self.backoff_s * (2**attempt) if self.backoff_s else 0)
        raise OSError(f"giving up on {self.url} after {self.retries} attempts: {last}")

    def _head_size(self) -> int:
        def head() -> int:
            req = urllib.request.Request(self.url, method="HEAD")
            with self._open(req, timeout=self.timeout_s) as r:  # type: ignore[attr-defined]
                n = r.headers.get("Content-Length")
                if n is None:
                    raise ValueError("no Content-Length")
                return int(n)

        return int(self._with_retries(head))

    def read_range(self, start: int, end: int) -> bytes:
        if not 0 <= start <= end <= self.size:
            raise ValueError(f"bad range [{start}, {end}) for size {self.size}")
        if start == end:
            return b""

        def get() -> bytes:
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
            with self._open(req, timeout=self.timeout_s) as r:  # type: ignore[attr-defined]
                if getattr(r, "status", 206) != 206:
                    raise ValueError(f"expected HTTP 206, got {r.status}")
                n = end - start
                deadline = self._clock() + self.deadline_base_s + n / self.min_rate_bps
                buf = bytearray()
                while len(buf) < n:
                    piece = r.read(min(1 << 20, n - len(buf)))
                    if not piece:
                        break
                    buf += piece
                    if self._clock() > deadline:
                        raise TimeoutError(f"range {start}-{end}: transfer below minimum rate")
                data = bytes(buf)
            if len(data) != end - start:
                raise ValueError(f"short read: {len(data)} of {end - start} bytes")
            return data

        data = bytes(self._with_retries(get))
        self.bytes_read += len(data)
        return data


class _CachedFile(io.RawIOBase):
    """Seekable read-only file over a RangeSource, with a small block cache.

    Only used to let `zipfile` parse the central directory; member data is fetched in bulk
    by `RemoteZip.iter_members`.
    """

    def __init__(self, src: RangeSource, block: int = 1 << 20, max_blocks: int = 64) -> None:
        self.src, self.block, self.max_blocks = src, block, max_blocks
        self.pos = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.src.size}[whence]
        self.pos = base + offset
        return self.pos

    def _block(self, i: int) -> bytes:
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        a = i * self.block
        data = self.src.read_range(a, min(self.src.size, a + self.block))
        self.cache[i] = data
        if len(self.cache) > self.max_blocks:
            self.cache.popitem(last=False)
        return data

    def readinto(self, b: bytearray) -> int:  # type: ignore[override]
        n = min(len(b), max(0, self.src.size - self.pos))
        out = bytearray()
        while len(out) < n:
            i, off = divmod(self.pos + len(out), self.block)
            out += self._block(i)[off : off + n - len(out)]
        b[:n] = out
        self.pos += n
        return n


@dataclass(frozen=True)
class Member:
    name: str
    offset: int  # local header offset
    csize: int
    usize: int
    crc: int
    method: int


class RemoteZip:
    """Index of a (possibly remote) ZIP archive; fetches members by merged byte spans."""

    def __init__(self, src: RangeSource) -> None:
        self.src = src
        with zipfile.ZipFile(io.BufferedReader(_CachedFile(src), buffer_size=1 << 16)) as z:
            infos = z.infolist()
            end_of_data = int(getattr(z, "start_dir", src.size))
        self.members: dict[str, Member] = {
            i.filename: Member(
                i.filename, i.header_offset, i.compress_size, i.file_size, i.CRC, i.compress_type
            )
            for i in infos
            if not i.is_dir()
        }
        offs = sorted(m.offset for m in self.members.values())
        # Upper bound of each member's record: the next local header (or the end of data).
        nxt = dict(zip(offs, offs[1:] + [end_of_data], strict=True))
        self._limit = {m.name: nxt[m.offset] for m in self.members.values()}

    def names(self) -> list[str]:
        return list(self.members)

    def _spans(self, names: list[str], max_gap: int, max_span: int) -> list[list[Member]]:
        ms = sorted((self.members[n] for n in names), key=lambda m: m.offset)
        spans: list[list[Member]] = []
        for m in ms:
            if spans:
                first, last = spans[-1][0], spans[-1][-1]
                gap = m.offset - self._limit[last.name]
                if gap <= max_gap and self._limit[m.name] - first.offset <= max_span:
                    spans[-1].append(m)
                    continue
            spans.append([m])
        return spans

    def iter_members(
        self, names: Iterable[str], max_gap: int = 1 << 20, max_span: int = 64 << 20
    ) -> Iterator[tuple[str, bytes]]:
        """Yield (name, data) for each requested member, CRC-checked, in archive order.

        Neighbouring members (gaps up to `max_gap` bytes) are fetched in one request of at
        most `max_span` bytes (a single larger member is fetched on its own).
        """
        names = list(dict.fromkeys(names))
        missing = [n for n in names if n not in self.members]
        if missing:
            raise KeyError(f"not in archive: {missing[:3]}{' ...' if len(missing) > 3 else ''}")
        for span in self._spans(names, max_gap, max_span):
            a, b = span[0].offset, self._limit[span[-1].name]
            buf = self.src.read_range(a, b)
            for m in span:
                yield m.name, self._decode(m, buf, m.offset - a)

    @staticmethod
    def _decode(m: Member, buf: bytes, at: int) -> bytes:
        sig, *_, nlen, xlen = _LOCAL_HEADER.unpack_from(buf, at)
        if sig != _LOCAL_SIG:
            raise ValueError(f"{m.name}: bad local header signature")
        start = at + _LOCAL_HEADER.size + nlen + xlen
        raw = buf[start : start + m.csize]
        if len(raw) != m.csize:
            raise ValueError(f"{m.name}: truncated member data")
        if m.method == zipfile.ZIP_STORED:
            data = raw
        elif m.method == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(raw, -15)
        else:
            raise ValueError(f"{m.name}: unsupported compression method {m.method}")
        if len(data) != m.usize or zlib.crc32(data) != m.crc:
            raise ValueError(f"{m.name}: CRC or size mismatch")
        return data


def write_atomic(path: Path, data: bytes) -> None:
    """Write via a temporary file and rename, so a crash never leaves a partial file.

    The data is flushed to disk (fsync) before the rename: without it, a sleep or power
    loss can leave the renamed file at full size but filled with zero bytes (seen on
    2026-11-04 with 41 SynDrone camera files on NTFS).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
