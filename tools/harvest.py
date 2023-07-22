#!/usr/bin/env python3

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sty import fg

script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
root_dir = script_dir / ".."
asm_dir = root_dir / "ver/current/asm/nonmatchings/"
build_dir = root_dir / "ver/current/build/"
elf_path = build_dir / "papermario.elf"
map_file_path = build_dir / "papermario.map"
rom_path = root_dir / "ver/current/baserom.z64"

OBJDUMP = "mips-linux-gnu-objdump"


@dataclass
class Bytes:
    offset: int
    normalized: str
    bytes: bytes


@dataclass
class Symbol:
    name: str
    rom_start: int
    ram: int
    current_file: Path
    prev_sym: str
    rom_end: Optional[int] = None

    def size(self):
        assert self.rom_end is not None
        return self.rom_end - self.rom_start


def read_rom() -> bytes:
    with open(rom_path, "rb") as f:
        return f.read()


def get_func_sizes() -> Dict[str, int]:
    try:
        result = subprocess.run(
            ["mips-linux-gnu-objdump", "-x", elf_path], stdout=subprocess.PIPE
        )
        nm_lines = result.stdout.decode().split("\n")
    except:
        print(
            f"Error: Could not run objdump on {elf_path} - make sure that the project is built"
        )
        sys.exit(1)

    sizes: Dict[str, int] = {}

    for line in nm_lines:
        if " F " in line:
            components = line.split()
            size = int(components[4], 16)
            name = components[5]
            sizes[name] = size

    return sizes


def get_symbol_bytes(func: str) -> Optional[Bytes]:
    if func not in syms or syms[func].rom_end is None:
        return None
    sym = syms[func]
    bs = list(rom_bytes[sym.rom_start : sym.rom_end])

    # trim nops
    while len(bs) > 0 and bs[-1] == 0:
        bs.pop()

    insns = bs[0::4]

    ret = []
    for ins in insns:
        ret.append(ins >> 2)

    return Bytes(0, bytes(ret).decode("utf-8"), rom_bytes[sym.rom_start : sym.rom_end])


def parse_map() -> OrderedDict[str, Symbol]:
    ram_offset = None
    cur_file = "<no file>"
    syms: OrderedDict[str, Symbol] = OrderedDict()
    prev_sym = ""
    prev_line = ""
    cur_sect = ""
    sect_re = re.compile(r"\(\..*\)")
    with open(map_file_path) as f:
        for line in f:
            sect = sect_re.search(line)
            if sect:
                sect_str = sect.group(0)
                if sect_str in ["(.text*)", "(.data*)", "(.rodata*)", "(.bss*)"]:
                    cur_sect = sect_str

            if "load address" in line:
                if "noload" in line or "noload" in prev_line:
                    ram_offset = None
                    continue
                ram = int(line[16 : 16 + 18], 0)
                rom = int(line[59 : 59 + 18], 0)
                ram_offset = ram - rom
                continue
            prev_line = line

            if (
                ram_offset is None
                or "=" in line
                or "*fill*" in line
                or " 0x" not in line
            ):
                continue
            ram = int(line[16 : 16 + 18], 0)
            rom = ram - ram_offset
            fn = line.split()[-1]
            if "0x" in fn:
                ram_offset = None
            elif "/" in fn:
                cur_file = fn
            else:
                if cur_sect != "(.text*)":
                    continue
                new_sym = Symbol(
                    name=fn,
                    rom_start=rom,
                    ram=ram,
                    current_file=Path(cur_file),
                    prev_sym=prev_sym,
                )
                if fn in func_sizes:
                    new_sym.rom_end = rom + func_sizes[fn]
                syms[fn] = new_sym
                prev_sym = fn

    # Calc end offsets
    for sym in syms:
        prev_sym = syms[sym].prev_sym
        if prev_sym and not syms[prev_sym].rom_end:
            syms[prev_sym].rom_end = syms[sym].rom_start

    return syms


LN_CACHE: Dict[Path, Dict[int, int]] = {}


def get_line_numbers(obj_file: Path) -> Dict[int, int]:
    if obj_file in LN_CACHE:
        return LN_CACHE[obj_file]

    objdump_out = (
        subprocess.run(
            [OBJDUMP, "-WL", obj_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        .stdout.decode("utf-8")
        .split("\n")
    )

    if not objdump_out:
        LN_CACHE[obj_file] = {}
    else:
        ret = {}
        for line in objdump_out[7:]:
            if not line:
                continue
            pieces = line.split()

            if len(pieces) < 3:
                continue

            fn = pieces[0]

            if fn == OBJDUMP or fn[0] == "<":
                continue

            starting_addr = int(pieces[2], 0)
            try:
                line_num = int(pieces[1])
                ret[starting_addr] = line_num
            except ValueError:
                continue
        LN_CACHE[obj_file] = ret

    return LN_CACHE[obj_file]


def get_tu_offset(obj_file: Path, symbol: str) -> Optional[int]:
    objdump = "mips-linux-gnu-objdump"

    objdump_out = (
        subprocess.run([objdump, "-t", obj_file], stdout=subprocess.PIPE)
        .stdout.decode("utf-8")
        .split("\n")
    )

    if not objdump_out:
        return None

    for line in objdump_out[4:]:
        if not line:
            continue
        pieces = line.split()

        if pieces[-1] == symbol:
            return int(pieces[0], 16)
    return None


@dataclass
class CRange:
    start: Optional[int] = None
    end: Optional[int] = None
    start_exact = False
    end_exact = False

    def has_info(self):
        return self.start is not None or self.end is not None

    def __str__(self):
        start_str = "?"
        end_str = "?"

        if self.start is not None:
            if self.start_exact:
                start_str = f"{self.start}"
            else:
                start_str = f"~{self.start}"

        if self.end is not None:
            if self.end_exact:
                end_str = f"{self.end}"
            else:
                end_str = f"~{self.end}"

        return f"{start_str} - {end_str}"


def get_c_range(insn_start: int, insn_end: int, line_numbers: Dict[int, int]) -> CRange:
    range = CRange()

    if insn_start in line_numbers:
        range.start = line_numbers[insn_start]
        range.start_exact = True
    else:
        keys = list(line_numbers.keys())
        for i, key in enumerate(keys[:-1]):
            if keys[i + 1] > insn_start:
                range.start = line_numbers[keys[i]]
                break

    if insn_end in line_numbers:
        range.end = line_numbers[insn_end]
        range.end_exact = True
    else:
        keys = list(line_numbers.keys())
        for i, key in enumerate(keys):
            if key > insn_end:
                range.end = line_numbers[key]
                break

    return range


parser = argparse.ArgumentParser(description="Harvester")
args = parser.parse_args()

if __name__ == "__main__":
    rom_bytes = read_rom()
    func_sizes = get_func_sizes()
    syms = parse_map()

    for symbol in syms:
        dog = 5
        if syms[symbol].current_file.name.endswith(".c.o"):
            c_file = str(syms[symbol].current_file)[13:-2]

            with open(c_file) as f:
                lines = f.readlines()

            start_line: Optional[int] = None
            end_line: Optional[int] = None

            for i, line in enumerate(lines):
                if symbol in line and line.endswith("{\n"):
                    start_line = i
                    break

            if start_line is None:
                continue

            for i, line in enumerate(lines[start_line + 1 :]):
                if line == "}\n":
                    end_line = start_line + i + 2
                    break

            if end_line is None:
                continue

            asm_glob = glob.glob(
                f"ver/us/asm/nonmatchings/**/{symbol}.s",
                recursive=True,
            )

            if len(asm_glob) < 1:
                print(f"Couldn't find asm for {symbol}")
                continue

            if len(asm_glob) > 1:
                print(f"Found multiple asm files for {symbol}")
                continue

            asm_file = asm_glob[0]

            out_dir = Path(f"harvest/{symbol}")
            out_dir.mkdir(parents=True, exist_ok=True)

            with open(out_dir / "raw.c", "w") as f:
                func = "".join(lines[start_line:end_line])
                f.write(func)

            shutil.copy(asm_file, out_dir / "raw.s")
