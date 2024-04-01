#!/usr/bin/env python3

import argparse
from functools import lru_cache
import glob
import os
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
from typing import Optional
from mapfile_parser import MapFile

from pycparser import c_ast as ca
from pycparser import c_generator, c_parser

from c_types import build_typemap, parse_struct, type_of_var_decl
from ast_types import decayed_expr_type, expr_type

script_dir = Path(os.path.dirname(os.path.realpath(__file__)))
root_dir = script_dir / "../.." / "papermario"
asm_dir = root_dir / "ver/current/asm/nonmatchings/"
build_dir = root_dir / "ver/current/build/"
elf_path = build_dir / "papermario.elf"
map_file_path = build_dir / "papermario.map"
rom_path = root_dir / "ver/current/baserom.z64"

OBJDUMP = "mips-linux-gnu-objdump"

CPP_FLAGS = [
    "-Iinclude",
    "-Isrc",
    "-Iassets",
    "-Iassets/us",
    "-Iver/current/build/include",
    "-D_LANGUAGE_C",
    "-DF3DEX_GBI_2",
    "-D_MIPS_SZLONG=32",
    "-DSCRIPT(...)={}",
    "-D__attribute__(...)=",
    "-D__asm__(...)=",
    "-ffreestanding",
    "-DM2CTX",
]


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


def parse_map() -> MapFile:
    mf = MapFile()
    mf.readMapFile(map_file_path)
    return mf


parser = argparse.ArgumentParser(description="Harvester")
args = parser.parse_args()
c_parser = c_parser.CParser()


def set_decl_name(decl: ca.Decl, name: str) -> None:
    if isinstance(decl, ca.ID):
        decl.name = name
        return

    type = type_of_var_decl(decl)
    while not isinstance(type, ca.TypeDecl):
        type = type.type
    type.declname = name
    decl.name = name


@lru_cache(maxsize=None)
def get_ast(file: Path):
    cpp_command = ["mips-linux-gnu-cpp", "-E", "-P", *CPP_FLAGS, file]

    file_text = subprocess.check_output(cpp_command, cwd=root_dir, encoding="utf-8")
    ast = c_parser.parse(file_text, filename=file.name)
    return ast


def get_struct_offset(typemap, struct_name, field_name):
    struct = None

    if struct_name in typemap.typedefs:
        c_type = typemap.typedefs[struct_name]
        if isinstance(c_type.type, ca.Struct):
            struct = parse_struct(c_type.type, typemap)
        else:
            raise Exception(f"???")
    else:
        raise Exception(f"???")

    if struct is None:
        raise ValueError(f"Couldn't find struct {struct_name}")

    for offset, field in struct.fields.items():
        for f in field:
            if f.name == field_name:
                return offset

    raise ValueError(f"Couldn't find field {field_name} in struct {struct_name}")


def synthesize_func(ast, symbol: str):
    func_ast = None
    for node in reversed(ast.ext):
        if isinstance(node, ca.FuncDef) and node.decl.name == symbol:
            func_ast = node
            break
    if func_ast is None:
        raise ValueError(f"Couldn't find function {symbol}")

    generator = c_generator.CGenerator()
    orig_c = generator.visit(func_ast)

    typemap = build_typemap(ast)

    rename_map = {}

    # Rename the function to "func"
    set_decl_name(func_ast.decl, "func")

    if symbol == "set_time_freeze_mode":
        dog = 5

    # Do the renaming
    class SymVisitor(ca.NodeVisitor):
        num_funcs = 0
        num_syms = 0
        num_types = 0

        def visit_FuncCall(self, node):
            func_name = node.name.name
            if func_name not in rename_map:
                rename_map[func_name] = "func" + str(self.num_funcs)
                self.num_funcs += 1
            self.generic_visit(node)
            node.name.name = rename_map[func_name]

        def visit_IdentifierType(self, node):
            for name in node.names:
                if name not in [
                    "void",
                    "s8",
                    "u8",
                    "s16",
                    "u16",
                    "s32",
                    "u32",
                    "f32",
                    "s64",
                    "u64",
                    "unsigned",
                    "signed",
                    "char",
                    "short",
                    "int",
                    "long",
                    "float",
                    "double",
                    "const",
                    "volatile",
                    "struct",
                    "union",
                    "enum",
                ]:
                    if name not in rename_map:
                        rename_map[name] = "type" + str(self.num_types)
                    node.names[node.names.index(name)] = rename_map[name]
            self.generic_visit(node)

        def visit_ID(self, node):
            if node.name not in rename_map:
                new_name = None
                # Try enum lookup first
                if node.name in typemap.enum_values:
                    new_name = f"0x{typemap.enum_values[node.name]:X}"
                if new_name is None:
                    new_name = "sym" + str(self.num_syms)
                    self.num_syms += 1
                rename_map[node.name] = new_name
            node.name = rename_map[node.name]
            self.generic_visit(node)

        # Field
        def visit_StructRef(self, node):
            try:
                typ = decayed_expr_type(node.name, typemap)
                if isinstance(typ, ca.PtrDecl):
                    struct_name = typ.type.type.names[0]
                else:
                    struct_name = typ.type.names[0]

                offset = get_struct_offset(typemap, struct_name, node.field.name)
                self.generic_visit(node)
                set_decl_name(node.field, f"field_{offset:X}")
            except Exception as e:
                self.generic_visit(node)
                set_decl_name(node.field, f"field_ERROR")

        def visit_ParamList(self, node):
            for i, param in enumerate(node.params):
                if isinstance(param, ca.EllipsisParam):
                    return

                if param.name not in rename_map:
                    rename_map[param.name] = "arg" + str(i)
                set_decl_name(param, rename_map[param.name])
                self.generic_visit(param)

        def visit_Decl(self, node):
            if node == func_ast.decl:
                self.generic_visit(node)
                return

            if node.name not in rename_map:
                rename_map[node.name] = "sym" + str(self.num_syms)
                self.num_syms += 1
            set_decl_name(node, rename_map[node.name])
            self.generic_visit(node)

    SymVisitor().visit(func_ast)

    # Write back to c
    generator = c_generator.CGenerator()
    new_c = generator.visit(func_ast)
    # print(out)

    return orig_c, new_c


def santize_asm(lines, symbol):
    sanitized = []
    num_labels = 0
    num_funcs = 0
    num_syms = 0
    rename_map = {}

    # Find all labels
    for line in lines:
        line = line.strip()

        label_match = re.match(r"\.(.+):", line)
        if label_match:
            label = label_match.group(1)
            if label not in rename_map:
                rename_map[label] = f"L{num_labels}"
                num_labels += 1

    for i, line in enumerate(lines):
        line = line.strip()

        # Remove comments
        while "/*" in line:
            start = line.index("/*")
            end = line.index("*/")
            line = (line[:start] + line[end + 2 :]).strip()

        # Remove extra spaces
        line = " ".join(line.split())

        # Rename function to func
        if line == "glabel " + symbol:
            line = "glabel func"
            sanitized.append(line)
            continue

        # Replace labels
        label_match = re.match(r".*\.([A-Z0-9]+)", line)
        if label_match:
            label = label_match.group(1)
            if label in rename_map:
                line = line.replace(label, rename_map[label])

        # Replace function calls
        jal_match = re.match(r"jal\s+(.+)", line)
        if jal_match:
            func = jal_match.group(1)
            if func not in rename_map:
                rename_map[func] = f"func{num_funcs}"
                num_funcs += 1
            line = line.replace(func, rename_map[func])

        # Replace symbols
        hi_match = re.match(r".*%hi\((.+)\)", line)
        if hi_match:
            sym = hi_match.group(1)
            if sym not in rename_map:
                rename_map[sym] = f"sym{num_syms}"
                num_syms += 1
            line = line.replace(sym, rename_map[sym])

        lo_match = re.match(r".*%lo\((.+)\)", line)
        if lo_match:
            sym = lo_match.group(1)
            if ")" in sym:
                sym = sym[: sym.index(")")]
            if sym not in rename_map:
                rename_map[sym] = f"sym{num_syms}"
                num_syms += 1
            line = line.replace(sym, rename_map[sym])

        sanitized.append(line)

    return "\n".join(sanitized)


if __name__ == "__main__":
    rom_bytes = read_rom()
    map = parse_map()

    for file in map.filesList:
        if file.segmentType == ".text" and file.filepath.name.endswith(".c.o"):
            c_file = str(file.getName())[6:]
            ast = get_ast(root_dir / c_file)

            for symbol in file.symbols:
                orig_c, santized_c = synthesize_func(ast, symbol.name)

                orig_asm_path = Path("data") / symbol.name / "raw.s"

                if not orig_asm_path.exists():
                    print(f"Skipping {symbol.name} - no original ASM found")
                    continue

                with open(orig_asm_path, "r") as f:
                    orig_asm = f.readlines()

                sanitized_asm = santize_asm(orig_asm, symbol.name)

                out_dir = Path(f"harvest/{symbol.name}")
                out_dir.mkdir(parents=True, exist_ok=True)

                with open(out_dir / "raw.c", "w") as f:
                    f.write(orig_c)

                with open(out_dir / "sanitized.c", "w") as f:
                    f.write(santized_c)

                with open(out_dir / "raw.s", "w") as f:
                    f.writelines(orig_asm)

                with open(out_dir / "sanitized.s", "w") as f:
                    f.write(sanitized_asm)

                # with open(out_dir / "raw.bin", "wb") as f:
                #     f.write(rom_bytes[symbol.vrom : symbol.vrom + symbol.size])
