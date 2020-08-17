# Converts the monks dataset into c4.5 format

import os

for i in range(1, 4):
    for suff in ["test", "train"]:
        fn = f"monks-{i}.{suff}"
        with open(fn) as inp:
            with open(os.path.join("c-format", fn), "w") as outp:
                for c_ln in inp:
                    c_fields = c_ln.split()

                    # Remove ID at end and put class at the end
                    for c_f in range(1, len(c_fields)-1):
                        outp.write(c_fields[c_f])
                        outp.write(",")
                    outp.write(c_fields[0])
                    outp.write(os.linesep)
