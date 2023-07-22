#!/usr/bin/env python3

import os


for root, dirs, files in os.walk("data"):
    for file in files:
        if file == "raw.s":
            with open(os.path.join(root, file), "r") as f:
                lines = f.readlines()

            text = "".join(lines)

            out_lines = [lines[0]]

            for line in lines[1:]:
                if line.startswith("glabel"):
                    continue

                if line.endswith(":\n"):
                    label = line.split(":")[0]

                    num_occur = text.count(label)
                    if num_occur < 2:
                        continue

                if line.startswith("/*"):
                    line = line.split("*/")[1]

                line = line.lstrip()

                out_lines.append(line)

            with open(os.path.join(root, "clean.s"), "w") as f:
                f.writelines(out_lines)
