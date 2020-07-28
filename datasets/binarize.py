import os

for r, d, f in os.walk("."):
    for fn in f:
        if fn.endswith(".data"):
            inp_path = os.path.join(r, fn)
            out_path = os.path.join(r, fn[-5] + "_bin.data")
            names_path = os.path.join(r, fn[-5] + "_bin.names")
            if not os.path.exists(out_path):
                with open(inp_path) as inf:
                    data = []
                    for _, ln in enumerate(inf):
                        fields = ln.split(",")
                        c_arr = []
                        for fd in fields:
                            try:
                                val = int(fd)
                                c_arr.append(val)
                            except ValueError:
                                try:
                                    val = float(fd)
                                    c_arr.append(int(val))
                                except ValueError:
                                    c_arr.append(fd)
                        data.append(c_arr)

                # Check consistency
                target = len(data[0])
                for c_arr in data:
                    if len(c_arr) != target:
                        print(f"Data array mismatch {target}/{len(c_arr)}")


                # Assign binary values
                for idx in range(0, target):
                    cnum = 0
                    assignment = {}

                    for c_arr in data:
                        if c_arr[idx] not in assignment:
                            assignment[c_arr[idx]] = bin(cnum)[2:]
                            cnum += 1
                        c_arr[idx] = assignment[c_arr[idx]]

                print("Done")