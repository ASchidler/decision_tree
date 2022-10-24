import os
from collections import defaultdict

base_path = "nonbinary/instances"

header_names = None

for fl in sorted(os.listdir(base_path)):
    if fl.endswith(".csv"):
        with open(os.path.join(base_path, fl)) as csv:
            with open(os.path.join(base_path, fl[:-3] + "data"), "w") as data_file:
                domains = defaultdict(set)
                for i, cl in enumerate(csv):
                    if i == 0:
                        fields = cl.strip().replace(",", ";").split(";")
                        header_names = [x.strip() for x in fields]
                    else:
                        fields = cl.strip().replace(",", ";").split(";")

                        data_file.write(fields[0].strip())
                        for cf in fields[1:]:
                            data_file.write(","+ cf.strip())
                        data_file.write(os.linesep)

                        for c_i in range(0, len(fields)):
                            fd = fields[c_i].strip()
                            if fd.startswith("\"") and fd.endswith("\""):
                                fd = fd[1:-1]

                            try:
                                fd = int(fd)
                            except ValueError:
                                try:
                                    fd = float(fd)
                                except ValueError:
                                    pass
                            if fd != "?":
                                domains[c_i].add(fd)

            with open(os.path.join(base_path, fl[:-3] + "names"), "w") as data_file:
                for c_di, c_v in enumerate(domains[len(domains)-1]):
                    data_file.write(f"{c_v}")
                    if c_di != len(domains[len(domains)-1]) - 1:
                        data_file.write(", ")
                data_file.write("."+ os.linesep)

                for c_i, c_d in domains.items():
                    if c_i == len(domains) - 1:
                        continue

                    is_cat = any(isinstance(x, str) for x in c_d)
                    if not is_cat:
                        data_file.write(f"{header_names[c_i]}: continuous."+ os.linesep)
                    else:
                        data_file.write(f"{header_names[c_i]}: ")
                        for c_di, c_v in enumerate(c_d):
                            data_file.write(f"{c_v}")
                            if c_di != len(c_d) - 1:
                                data_file.write(", ")
                        data_file.write("."+os.linesep)
        os.remove(os.path.join(base_path, fl))
