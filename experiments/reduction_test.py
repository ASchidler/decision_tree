import sys
import os
import parser
import time
import maxsat.maxsat_feature as maxsat_feature
import bdd_instance

path = sys.argv[1]
method = int(sys.argv[2])
runs = 10
outp = f"{sys.argv[3]}_{method}.csv"

# 0 = removal
# 1 = constructive random
# 2 = constructive greedy
# 3 = optimal

if method == 3 or method == 2:
    runs = 1

if not os.path.exists(outp):
    with open(outp, "w") as out_file:
        out_file.write(f"Instance;Best Key;Avg Key;Worst Key;Best Time;Avg Time;Worst time;Feat Old;Feat New;Samples Old;Samples New{os.linesep}")

with open(outp, "r+") as out_file:
    processed = set()
    out_file.seek(0)
    for _, cl in enumerate(out_file):
        fields = cl.split(";")
        processed.add(fields[0])

    with open(f"keys_{method}.txt", "a") as key_file:

        for r, d, f in os.walk(path):
            for fl in f:
                if fl.endswith(".csv") and fl not in processed:
                    print(fl)
                    inst = parser.parse(os.path.join(r, fl))
                    best_time = sys.maxsize
                    worst_time = 0
                    best_result = sys.maxsize
                    worst_result = 0
                    sum_result = 0
                    sum_time = 0
                    best_key = None

                    for _ in range(0, runs):
                        start_time = time.time()
                        key = None
                        if method == 0:
                            key = inst.min_key(randomize=True)
                        elif method == 1:
                            key = inst.min_key3()
                        elif method == 2:
                            key = inst.min_key2()
                        elif method == 3:
                            key = maxsat_feature.compute_features(inst, timeout=600)

                        best_result = min(best_result, len(key))
                        worst_result = max(worst_result, len(key))
                        best_time = min(best_time, time.time() - start_time)
                        worst_time = max(worst_time, time.time() - start_time)
                        sum_result += len(key)
                        sum_time += time.time() - start_time
                        if best_key is None or len(best_key) < len(key):
                            best_key = key

                        print(f"{fl}, finished run, key {len(key)} in {time.time() - start_time}")

                    old_feat = inst.num_features
                    old_exp = len(inst.examples)
                    bdd_instance.reduce(inst, min_key=best_key)
                    out_file.write("{};{};{};{};{};{};{};{};{};{};{};{}".format(fl, best_result, sum_result / runs, worst_result,
                                                     best_time, sum_time / runs, worst_time,
                                                     old_feat, inst.num_features, old_exp, len(inst.examples), os.linesep))
                    key_file.write(fl)
                    key_file.write(";")
                    for e in best_key:
                        key_file.write(f"{e},")
                    key_file.write(os.linesep)
