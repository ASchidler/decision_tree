import os
import sys
import parser

pth = sys.argv[1]

for fl in list(os.listdir(pth)):
    if os.path.isfile(os.path.join(pth, fl)) and fl.endswith(".csv"):
        if fl.endswith("test.csv"):
            test = os.path.join(pth, fl)
            training = os.path.join(pth, fl.replace("test.csv", "training.csv"))
            out_path = os.path.join(pth, fl[0:-1 * len("-un_test.csv")])
        else:
            test = os.path.join(pth, fl)
            training = test
            out_path = os.path.join(pth, fl[0:-1 * len("-un.csv")] + "-full")
        os.mkdir(os.path.join(pth, out_path))

        # Header and names file
        for nm, fn in [("test", test), ("training", training)]:
            inst = parser.parse(fn)

            # Names file
            if nm == "test":
                with open(os.path.join(out_path, "names"), "w") as names_file:
                    names_file.write(f"0,1.{os.linesep}")
                    for f in range(0, inst.num_features):
                        names_file.write(f"att{f}:0,1.{os.linesep}")

            with open(os.path.join(out_path, f"{nm}.data"), "w") as out_file:
                for ex in inst.examples:
                    for f in ex.features[1:]:
                        out_file.write(f"{f},")
                    out_file.write(f"{ex.cls}{os.linesep}")

        os.remove(test)
        if test != training:
            os.remove(training)

