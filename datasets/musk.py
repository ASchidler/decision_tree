# Converts the ida dataset into c4.5 format

import os

for i in range(1, 3):
    fn = f"musk{i}.data"

    with open(fn) as inp:
        with open(os.path.join("c-format", fn), "w") as outp:
            for c_ln in inp:
                c_fields = c_ln.strip().split(",")

                # Remove first two colums with identifier, at closing dot
                for c_f in range(2, len(c_fields)-1):
                    outp.write(c_fields[c_f])
                    outp.write(",")
                outp.write(c_fields[-1][:-1])
                outp.write(os.linesep)

