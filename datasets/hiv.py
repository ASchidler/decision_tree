# Converts the HIV datasets into c4.5 format

import os

for suf in ["746", "1625", "impens", "schilling"]:
    fn = f"hiv_{suf}.data"

    with open(fn) as inp:
        with open(os.path.join("c-format", fn), "w") as outp:
            for c_ln in inp:
                for i in range(0, 8):
                    outp.write(c_ln[i:i+1])
                    outp.write(",")

                c_fields = c_ln.strip().split(",")
                outp.write(c_fields[-1])

                outp.write(os.linesep)

