#!/usr/bin/env python3
"""tm_run.py - run a case file through tm_pl or tm_pl_x on the KV260 (after load.sh). As root:

    sudo python3 tm_run.py cases.bin results.bin [udmabuf0] [cases per transfer]

cases.bin holds 32 bytes per case for tm_pl (VERSION 1; its s_axis beat: a, b, c;
tools/kv260/board_cases.py pack), 32 * NB for tm_pl_x (VERSION 2, NB from its register;
tools/kv260/board_cases_x.py pack); results.bin gets 16 bytes per case (the m_axis beat).
The cases go through the AXI DMA in simple mode (AMD PG021), in transfers of up to
<cases per transfer>: S2MM is armed first with room for the results, then MM2S sends the
cases; the MM2S transfer ends with tlast, which the PL carries to its last result and ends
the S2MM transfer. CASES_IN counts beats. One key=value line per quantity goes to stdout;
exit 1 on a DMA error, a timeout or a count that does not add up.

The standard library only. Every register store is one 32-bit store through ctypes, never
struct.pack_into (CPython zero-fills the target first: a spurious write of 0; hetfx-docs
T-2.14). The u-dma-buf buffer is mapped non-cached (sync_mode 1 with O_SYNC), so the DMA's
writes are seen without cache maintenance; it is read and written by aligned 8-byte
accesses (device memory on arm64 faults on the unaligned ones of a memcpy)."""

import ctypes
import mmap
import os
import struct
import sys
import time

TM_BASE, DMA_BASE = 0xA000_0000, 0xA001_0000
ID, VERSION, BUILD, LATENCY = 0x000, 0x004, 0x008, 0x00C
CASES_IN, CASES_OUT, STALLS, CONTROL = 0x014, 0x018, 0x01C, 0x020
NB = 0x028  # tm_pl_x: s_axis beats per case
TM_ID = 0x544D_4952  # "TMIR"
MM2S_DMACR, MM2S_DMASR, MM2S_SA, MM2S_SA_MSB, MM2S_LENGTH = 0x00, 0x04, 0x18, 0x1C, 0x28
S2MM_DMACR, S2MM_DMASR, S2MM_DA, S2MM_DA_MSB, S2MM_LENGTH = 0x30, 0x34, 0x48, 0x4C, 0x58
DMACR_RS, DMACR_RESET = 1 << 0, 1 << 2
DMASR_HALTED, DMASR_IDLE = 1 << 0, 1 << 1
DMASR_ERRORS = 0x70  # DMAIntErr, DMASlvErr, DMADecErr
CRF_APB, RST_FPD_TOP, AFI_FM2_RESET = 0xFD1A_0000, 0x100, 1 << 9
BEAT_BYTES, OUT_BYTES = 32, 16


def rd(mem, off):
    return struct.unpack_from("<I", mem, off)[0]


def wr(mem, off, v):
    ctypes.c_uint32.from_buffer(mem, off).value = v


class Buffer:
    def __init__(self, name):
        cls = f"/sys/class/u-dma-buf/{name}"
        with open(f"{cls}/phys_addr") as f:
            self.phys = int(f.read(), 0)
        with open(f"{cls}/size") as f:
            self.size = int(f.read(), 0)
        with open(f"{cls}/sync_mode") as f:
            if int(f.read()) != 1:
                raise RuntimeError(f"{name}: sync_mode is not 1 (non-cached with O_SYNC)")
        self.fd = os.open(f"/dev/{name}", os.O_RDWR | os.O_SYNC)
        self.mem = mmap.mmap(self.fd, self.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

    def close(self):
        self.mem.close()
        os.close(self.fd)


def put(mem, off, data):
    arr = (ctypes.c_uint64 * (len(data) // 8)).from_buffer(mem, off)
    arr[:] = memoryview(data).cast("Q")
    del arr


def get(mem, off, n):
    w = n // 8
    return struct.pack(f"<{w}Q", *struct.unpack_from(f"<{w}Q", mem, off))


def dma_reset(dma):
    wr(dma, MM2S_DMACR, DMACR_RESET)
    wr(dma, S2MM_DMACR, DMACR_RESET)
    t = time.monotonic()
    while (rd(dma, MM2S_DMACR) | rd(dma, S2MM_DMACR)) & DMACR_RESET:
        if time.monotonic() - t > 1:
            raise RuntimeError("the DMA did not come out of reset")


def transfer(dma, buf, src_off, dst_off, n, in_bytes, timeout):
    """One simple-mode transfer of n cases; returns (seconds, problems)."""
    problems = []
    src, dst = buf.phys + src_off, buf.phys + dst_off
    wr(dma, S2MM_DMACR, DMACR_RS)
    wr(dma, S2MM_DA, dst & 0xFFFF_FFFF)
    wr(dma, S2MM_DA_MSB, dst >> 32)
    wr(dma, S2MM_LENGTH, n * OUT_BYTES)  # arms S2MM
    wr(dma, MM2S_DMACR, DMACR_RS)
    wr(dma, MM2S_SA, src & 0xFFFF_FFFF)
    wr(dma, MM2S_SA_MSB, src >> 32)
    t0 = time.monotonic()
    wr(dma, MM2S_LENGTH, n * in_bytes)  # starts MM2S
    while True:
        s2mm, mm2s = rd(dma, S2MM_DMASR), rd(dma, MM2S_DMASR)
        if (s2mm | mm2s) & DMASR_ERRORS:
            problems.append(f"DMASR MM2S 0x{mm2s:X} S2MM 0x{s2mm:X}")
            break
        if s2mm & DMASR_IDLE and mm2s & DMASR_IDLE:
            break
        if time.monotonic() - t0 > timeout:
            problems.append(f"timeout: DMASR MM2S 0x{mm2s:X} S2MM 0x{s2mm:X}")
            break
    dt = time.monotonic() - t0
    got = rd(dma, S2MM_LENGTH)  # simple mode: bytes actually written
    if not problems and got != n * OUT_BYTES:
        problems.append(f"S2MM wrote {got} bytes, expected {n * OUT_BYTES}")
    return dt, problems


def main(argv):
    if len(argv) < 3:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    with open(argv[1], "rb") as f:
        cases = f.read()
    name = argv[3] if len(argv) > 3 else "udmabuf0"
    buf = Buffer(name)
    fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    prot = mmap.PROT_READ | mmap.PROT_WRITE
    crf = mmap.mmap(fd, 0x1000, mmap.MAP_SHARED, mmap.PROT_READ, offset=CRF_APB)
    tm = mmap.mmap(fd, 0x1000, mmap.MAP_SHARED, prot, offset=TM_BASE)
    dma = mmap.mmap(fd, 0x1000, mmap.MAP_SHARED, prot, offset=DMA_BASE)
    problems, results, busy = [], bytearray(), 0.0
    try:
        if rd(crf, RST_FPD_TOP) & AFI_FM2_RESET:
            raise RuntimeError("AFI_FM2 (HP0) is in reset")
        if rd(tm, ID) != TM_ID:
            raise RuntimeError(f"tm_pl ID is 0x{rd(tm, ID):08X}")
        version = rd(tm, VERSION)
        nb = rd(tm, NB) if version == 2 else 1
        in_bytes = BEAT_BYTES * nb
        if len(cases) % in_bytes:
            raise ValueError(f"{argv[1]}: {len(cases)} bytes is not whole cases of {in_bytes}")
        n_all = len(cases) // in_bytes
        per = int(argv[4]) if len(argv) > 4 else buf.size // (in_bytes + OUT_BYTES)
        per = min(per, buf.size // (in_bytes + OUT_BYTES), ((1 << 26) - 1) // in_bytes)
        src_off, dst_off = 0, per * in_bytes
        print(f"version={version} build=0x{rd(tm, BUILD):08X} latency={rd(tm, LATENCY)} beats_per_case={nb} "
              f"buffer={name} phys=0x{buf.phys:X} size={buf.size} cases={n_all} per_transfer={per}")
        wr(tm, CONTROL, 1)
        dma_reset(dma)
        for first in range(0, n_all, per):
            n = min(per, n_all - first)
            put(buf.mem, src_off, cases[first * in_bytes : (first + n) * in_bytes])
            dt, pr = transfer(dma, buf, src_off, dst_off, n, in_bytes, timeout=10 + n * 1e-5)
            busy += dt
            if pr:
                problems += [f"transfer at case {first}: {p}" for p in pr]
                break
            results += get(buf.mem, dst_off, n * OUT_BYTES)
        cin, cout, stalls = rd(tm, CASES_IN), rd(tm, CASES_OUT), rd(tm, STALLS)
        dma_reset(dma)
    finally:
        for m in (crf, tm, dma):
            m.close()
        os.close(fd)
        buf.close()
    with open(argv[2], "wb") as f:
        f.write(results)
    print(f"cases_in={cin} cases_out={cout} stalls={stalls} results={len(results) // OUT_BYTES} "
          f"dma_seconds={busy:.3f} cases_per_second={len(results) / OUT_BYTES / busy if busy else 0:.0f}")
    if not problems and (cin != n_all * nb or cout != n_all):
        problems.append(f"tm counted {cin} beats in and {cout} out, expected {n_all * nb} and {n_all}")
    for p in problems:
        print(f"PROBLEM {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
