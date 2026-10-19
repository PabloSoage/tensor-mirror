import pathlib
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "tools" / "kv260"))
import board_cases  # noqa: E402

VEC = "3c00 3c00 0 0 3c00 4000 0 0 3f800000 40400000 3f800001 4400\n" \
      "8001 0 0 0 1 0 0 0 1 1 33800000 1\n"


def test_pack_matches_the_s_axis_beat(tmp_path):
    (tmp_path / "v.txt").write_text(VEC)
    board_cases.pack(tmp_path / "v.txt", tmp_path / "c.bin")
    data = (tmp_path / "c.bin").read_bytes()
    assert len(data) == 64
    # tdata[63:0] = a, a[k] in bits 16k+15..16k; tdata[127:64] = b; tdata[159:128] = c
    w = struct.unpack_from("<8I", data, 0)
    assert w[:5] == (0x3C003C00, 0, 0x40003C00, 0, 0x3F800000) and w[5:] == (0, 0, 0)


def test_check_finds_one_wrong_unit(tmp_path):
    (tmp_path / "v.txt").write_text(VEC)
    rows = board_cases.read(tmp_path / "v.txt")
    good = b"".join(struct.pack("<IIHHI", r[9], r[10], r[11], 0, 0) for r in rows)
    (tmp_path / "r.bin").write_bytes(good)
    assert board_cases.check(tmp_path / "v.txt", tmp_path / "r.bin") == 0
    bad = bytearray(good)
    bad[16 + 8] ^= 1  # the f16 unit's result of case 1
    (tmp_path / "r.bin").write_bytes(bytes(bad))
    assert board_cases.check(tmp_path / "v.txt", tmp_path / "r.bin") == 1
