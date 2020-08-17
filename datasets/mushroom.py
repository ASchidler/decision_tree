# Converts the mushroom dataset into c4.5 format

import os

fn = f"mushroom.data"

with open(fn) as inp:
    with open(os.path.join("c-format", fn), "w") as outp:
        for c_ln in inp:
            c_fields = c_ln.strip().split(",")

            # put class at the end
            for c_f in range(1, len(c_fields)):
                outp.write(c_fields[c_f])
                outp.write(",")
            outp.write(c_fields[0])
            outp.write(os.linesep)

