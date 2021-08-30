""" Print the time progression from a logfile"""

import sys
import tarfile
import matplotlib.pyplot as plt

logfiles = [
    ("dt-nb-k-1.tar.bz2", "SZ,EX"),
    ("dt-nb-m-1.tar.bz2", "SZ"),
    ("dt-nb-n-1.tar.bz2", "M"),
    ("dt-nb-q-1.tar.bz2", "None")
]
instance_name = sys.argv[1]
slice = sys.argv[2]

fields = [(1, "Depth"), (2, "Size")]
field_idx = 0

colors = ['black', '#eecc66', '#bb5566', '#004488']
symbols = ['s', 'o', 'x', 'v']

c_file = None

fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
min_x = sys.maxsize
max_x = 0
min_y = sys.maxsize
max_y = 0

legend = []

for logfile, logfile_name in logfiles:
    with tarfile.open(logfile) as tar_file:
        entries = []
        done = False
        for ctf in tar_file:
            if done:
                break
            file_parts = ctf.path.split(".")

            if file_parts[-2].startswith("e"):
                # Error file
                continue

            cetf = tar_file.extractfile(ctf)

            for ci, cl in enumerate(cetf):
                if type(cl) is not str:
                    cl = cl.decode('ascii')

                if ci == 0:
                    if not cl.startswith("Instance: "+instance_name) or cl.find("slice="+ slice) == -1:
                        break
                done = True
                if cl.startswith("START") or cl.startswith("END"):
                    cf = cl.split(",")
                    cv = [x.strip().split(":")[1].strip() for x in cf]
                    entries.append((cv[-1], cv[-5], cv[-4], cv[-3], cv[-2]))
                elif cl.startswith("Time:"):
                    # Time: 230.2385	Training 1.0000	Test 0.8406	Depth 019	Nodes 139	Method ma
                    cf = cl.split("\t")
                    cv = [x.strip().split(" ")[1].strip() for x in cf]
                    entries.append((cv[0], cv[-3], cv[-2], cv[1], cv[2]))

        x = []
        y = []
        for ce in entries:
            x.append(float(ce[0]) / 60)
            y.append(float(ce[fields[field_idx][0]]))

        max_x = max(max_x, max(x))
        min_x = min(min_x, min(x))
        max_y = max(max_y, max(y))
        min_y = min(min_y, min(y))

        ax.scatter(x, y, marker=symbols.pop(), s=10, color=colors.pop(), alpha=0.5)
        legend.append(logfile_name)

ax.set_axisbelow(True)
names = []

ax.set_xlabel('Minutes')
ax.set_ylabel(fields[field_idx][1])
# ax.set_title('scatter plot')
plt.rcParams["legend.loc"] = 'upper left'
plt.rcParams['savefig.pad_inches'] = 0
plt.autoscale(tight=True)
plt.legend(legend)
# if c_col == 1:
#     plt.xscale("log")
#     plt.yscale("log")
plt.xlim(min_x-0.5, max_x+0.5)
plt.ylim(min_y-0.5, max_y+0.5)
#ax.axline([0, 0], [1, 1], linestyle="--", color="grey")
#plt.plot(color='black', linewidth=0.5, linestyle='dashed', markersize=10)
#pname = "pruned" if use_pruning else "unpruned"
#colname = "d" if c_col == 0 else ("n" if c_col == 1 else "a")
plt.savefig(f"progression_{instance_name}_{field_idx}.pdf", bbox_inches='tight')
plt.show()

