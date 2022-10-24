import os
import sys
import tarfile

input_file = sys.argv[1]
output_file = ".".join(os.path.split(input_file)[-1].split(".")[:-2])
with open(output_file + "_0.csv", "w") as zf:
    with open(output_file + "_1.csv", "w") as of:
        zf.write("Instance;Samples;Features;Classes;Depth;Time;SAT;Size;DM" + os.linesep)
        of.write("Instance;Samples;Features;Classes;Depth;Time;SAT;Size;DM" + os.linesep)

        with tarfile.open(input_file) as tf:
            for ctf in tf:
                target_file = os.path.split(ctf.path)[-1]
                target_file = target_file.split(".")[0].split("-")[-1] == "0"
                target_file = zf if target_file else of
                instance_name = None

                cetf = tf.extractfile(ctf)
                for cln in cetf:
                    cln = cln.decode('ascii')
                    if cln.startswith("Instance: "):
                        instance_name = cln.split(":")[-1].strip()
                    elif cln.startswith("E:"):
                        fields = cln.split(" ")
                        fields = [x.split(":")[1].strip() for x in fields]

                        target_file.write(f"{instance_name};{fields[0]};{fields[3]};{fields[2]};{fields[6]};"
                                          f"{fields[1].replace('*', '')};{0 if fields[1].find('*') > -1 else 1};{fields[7]};{fields[5]}{os.linesep}")
