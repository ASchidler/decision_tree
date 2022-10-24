import os
import nonbinary.nonbinary_instance as nbi

instance_pth = "instances"
ignore = set()
with open("results/ignore.txt") as ig:
    for cl in ig:
        cl.strip()
        ignore.add(cl)

fls = [x[:-7] for x in os.listdir(instance_pth) if x.endswith("1.data")]
fls = [x for x in fls if x not in ignore]

for c_file in sorted(fls):
    for c_slice in range(1, 6):
        try:
            instance, test, _ = nbi.parse("instances", c_file, c_slice, use_validation=True, use_test=True)
            if len(instance.classes) > 2:
                continue

            if os.path.exists(f"/media/asc/4ED79AA0509E44AA/binary_instances/{c_file}.{c_slice}.csv"):
                continue

            with open(f"/media/asc/4ED79AA0509E44AA/binary_instances/{c_file}.{c_slice}.csv", "w") as outp:
                with open(f"/media/asc/4ED79AA0509E44AA/binary_instances/{c_file}.{c_slice}.test.csv", "w") as outp2:
                    feature_map = {}
                    class_map = {x: i for i, x in enumerate(instance.classes)}
                    c_bf = 0
                    for c_f in range(1, instance.num_features+1):
                        feature_map[c_f] = {}
                        for c_i, c_v in enumerate(sorted(instance.domains[c_f])):
                            feature_map[c_f][c_v] = c_i
                            outp.write(f"a{c_bf};")
                            outp2.write(f"a{c_bf};")
                            c_bf += 1
                    outp.write("c"+os.linesep)

                    def find_smallest(t_value):
                        c_prev = None
                        for c_alt in sorted(instance.domains[c_f]):
                            if c_alt < t_value:
                                c_prev = c_alt
                                break

                        idx = feature_map[c_f][c_prev] if c_prev is not None else 0
                        return idx

                    for c_e in instance.examples:
                        for c_f in range(1, instance.num_features + 1):
                            is_cat = c_f in instance.is_categorical
                            if c_e.features[c_f] not in feature_map[c_f] and not is_cat:
                                c_v = find_smallest(c_e.features[c_f]) + 1
                            else:
                                c_v = feature_map[c_f][c_e.features[c_f]]

                            for _ in range(0, c_v):
                                outp.write("0;" if is_cat else "1;")
                            outp.write("0;" if not is_cat else "1;")
                            for _ in range(c_v+1, len(instance.domains[c_f])):
                                outp.write("0;")
                        outp.write(f"{class_map[c_e.cls]}"+os.linesep)

                    for c_e in test.examples:
                        for c_f in range(1, instance.num_features + 1):
                            is_cat = c_f in instance.is_categorical
                            if c_e.features[c_f] in feature_map[c_f]:
                                c_v = feature_map[c_f][c_e.features[c_f]]
                                for _ in range(0, c_v):
                                    outp2.write("0;" if is_cat else "1;")
                                outp2.write("0;" if not is_cat else "1;")
                                for _ in range(c_v+1, len(instance.domains[c_f])):
                                    outp2.write("0;")
                            elif is_cat:
                                for _ in range(0, len(instance.domains[c_f])):
                                    outp2.write("0;")
                            else:
                                c_prev = None
                                for c_alt in sorted(instance.domains[c_f]):
                                    if c_alt < c_e.features[c_f]:
                                        c_prev = c_alt
                                        break

                                c_v = feature_map[c_f][c_prev] if c_prev is not None else 0
                                for _ in range(0, c_v+1):
                                    outp2.write("1;")
                                for _ in range(c_v + 1, len(instance.domains[c_f])):
                                    outp2.write("0;")

                        outp2.write(f"{class_map[c_e.cls]}"+os.linesep)

                print(f"Finished {c_file}/{c_slice}")
        except FileNotFoundError:
            pass
