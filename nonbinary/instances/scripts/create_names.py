import os
from collections import defaultdict

base_path = ".."

header_names = None

for fl in sorted(os.listdir(base_path)):
    if fl.endswith(".data"):
        fl_path = os.path.join(base_path, fl)
        if not os.path.exists(fl_path[:-4] + "names"):
            with open(fl_path) as csv:
                domains = defaultdict(set)
                for i, cl in enumerate(csv):
                    fields = cl.strip().split(",")

                    for c_i in range(0, len(fields)):
                        fd = fields[c_i].strip()
                        try:
                            fd = int(fd)
                        except ValueError:
                            try:
                                fd = float(fd)
                            except ValueError:
                                pass
                        if fd != "?":
                            domains[c_i].add(fd)

            with open(os.path.join(base_path, fl[:-4] + "names"), "w") as data_file:
                for c_di, c_v in enumerate(domains[len(domains) - 1]):
                    data_file.write(f"{c_v}")
                    if c_di != len(domains[len(domains) - 1]) - 1:
                        data_file.write(", ")
                data_file.write("." + os.linesep)

                for c_i, c_d in domains.items():
                    if c_i == len(domains) - 1:
                        continue

                    is_cat = any(isinstance(x, str) for x in c_d)
                    if not is_cat:
                        data_file.write(f"att{c_i}: continuous." + os.linesep)
                    else:
                        data_file.write(f"att{c_i}: ")
                        for c_di, c_v in enumerate(c_d):
                            data_file.write(f"{c_v}")
                            if c_di != len(c_d) - 1:
                                data_file.write(", ")
                        data_file.write("." + os.linesep)
