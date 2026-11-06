import http.client
import io
import zipfile

import pytest

from aerialrange.data.remote_zip import BytesSource, HttpRangeSource, RemoteZip, write_atomic


def _zip(members: dict[str, bytes], method=zipfile.ZIP_DEFLATED) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", compression=method) as z:
        for n, d in members.items():
            z.writestr(n, d)
    return b.getvalue()


MEMBERS = {f"T/h20/rgb/{i:05d}.jpg": bytes([i % 251]) * (1000 + 37 * i) for i in range(40)}
MEMBERS["T/bboxes/00000.json"] = b'[{"id": 1}]'


@pytest.mark.parametrize("method", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_reads_members_exactly(method):
    rz = RemoteZip(BytesSource(_zip(MEMBERS, method)))
    assert set(rz.names()) == set(MEMBERS)
    want = ["T/h20/rgb/00003.jpg", "T/bboxes/00000.json", "T/h20/rgb/00039.jpg"]
    got = dict(rz.iter_members(want))
    assert got == {n: MEMBERS[n] for n in want}


def test_neighbours_are_merged_into_few_requests():
    src = BytesSource(_zip(MEMBERS))
    rz = RemoteZip(src)
    before = src.calls
    names = [f"T/h20/rgb/{i:05d}.jpg" for i in range(10, 30)]
    assert len(dict(rz.iter_members(names))) == 20
    assert src.calls - before == 1
    before = src.calls
    list(rz.iter_members(names, max_gap=0, max_span=1))  # no merging at all
    assert src.calls - before == 20


def test_corruption_is_detected():
    data = bytearray(_zip(MEMBERS, zipfile.ZIP_STORED))
    rz = RemoteZip(BytesSource(bytes(data)))
    m = rz.members["T/h20/rgb/00005.jpg"]
    data[m.offset + 30 + len(m.name) + 10] ^= 0xFF  # flip a byte inside the member data
    rz.src = BytesSource(bytes(data))
    with pytest.raises(ValueError, match="CRC"):
        dict(rz.iter_members([m.name]))


def test_unknown_member_raises():
    rz = RemoteZip(BytesSource(_zip(MEMBERS)))
    with pytest.raises(KeyError):
        list(rz.iter_members(["nope"]))


class _Resp:
    def __init__(self, data, status=206, headers=None):
        self.data, self.status, self.headers = data, status, headers or {}
        self.pos = 0

    def read(self, n=-1):
        end = len(self.data) if n is None or n < 0 else self.pos + n
        piece = self.data[self.pos : end]
        self.pos += len(piece)
        return piece

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_http_source_retries_and_checks_length():
    blob = bytes(range(256)) * 4
    calls = {"n": 0}

    def opener(req, timeout):
        if req.get_method() == "HEAD":
            return _Resp(b"", 200, {"Content-Length": str(len(blob))})
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionResetError("reset")  # transient failure
        if calls["n"] == 2:
            return _Resp(b"short")  # wrong length: must be retried, not accepted
        if calls["n"] == 3:
            raise http.client.IncompleteRead(b"x", 10)  # connection dropped mid-body
        a, b = req.get_header("Range").split("=")[1].split("-")
        return _Resp(blob[int(a) : int(b) + 1])

    src = HttpRangeSource("https://example.org/x.zip", backoff_s=0, opener=opener)
    assert src.size == len(blob)
    assert src.read_range(10, 300) == blob[10:300]
    assert calls["n"] == 4


def test_http_source_gives_up():
    def opener(req, timeout):
        if req.get_method() == "HEAD":
            return _Resp(b"", 200, {"Content-Length": "10"})
        return _Resp(b"0123456789", status=200)  # server ignored Range

    src = HttpRangeSource("https://example.org/x.zip", retries=2, backoff_s=0, opener=opener)
    with pytest.raises(OSError):
        src.read_range(0, 5)


def test_write_atomic(tmp_path):
    p = tmp_path / "a" / "b.bin"
    write_atomic(p, b"xyz")
    assert p.read_bytes() == b"xyz"
    assert not (tmp_path / "a" / "b.bin.part").exists()


class _SlowResp(_Resp):
    """Delivers its body in 10-byte pieces, advancing a fake clock 100 s per piece."""

    def __init__(self, data, clock):
        super().__init__(data)
        self.clock = clock

    def read(self, n=-1):
        self.clock["t"] += 100.0
        piece = self.data[self.pos : self.pos + min(10, n if n > 0 else 10)]
        self.pos += len(piece)
        return piece


def test_http_source_abandons_a_stalled_transfer_and_retries():
    blob = bytes(range(200))
    clock = {"t": 0.0}
    calls = {"n": 0}

    def opener(req, timeout):
        if req.get_method() == "HEAD":
            return _Resp(b"", 200, {"Content-Length": str(len(blob))})
        calls["n"] += 1
        a, b = req.get_header("Range").split("=")[1].split("-")
        body = blob[int(a) : int(b) + 1]
        return _SlowResp(body, clock) if calls["n"] == 1 else _Resp(body)

    src = HttpRangeSource(
        "https://example.org/x.zip",
        backoff_s=0,
        opener=opener,
        deadline_base_s=150.0,
        clock=lambda: clock["t"],
    )
    assert src.read_range(0, 100) == blob[:100]
    assert calls["n"] == 2  # the trickling first attempt was abandoned
