import sys
import matplotlib.pyplot as plt
from collections import defaultdict

experiments = [("k", "SZ,EX"), ("m", "SZ"), ("n", "M"), ("q", "None")]

fields = [(5, "Depth"), (4, "Size"), (7, "Accuracy"), (8, "Avg. Decision Length")]
field_idx = 2

colors = ['black', '#eecc66', '#bb5566', '#004488']
symbols = ['s', 'o', 'x', 'v']

fig, ax = plt.subplots(figsize=(4, 2.5), dpi=300)
min_x = sys.maxsize * 1.0
max_x = 0.0
min_y = sys.maxsize * 1.0
max_y = 0.0

results = defaultdict(list)
legend = []

target_idx = fields[field_idx][0]

for c_experiment, c_ex_name in experiments:
    legend.append(c_ex_name)
    with open(f"results_{c_experiment}.csv") as inp:
        for i, cl in enumerate(inp):
            if i > 0:
                cf = cl.split(";")
                if len(results[cf[0]]) == 0:
                    results[cf[0]].append(cf[target_idx])
                # else:
                #     if results[cf[0]][0] != cf[target_idx]:
                #         raise RuntimeError("Base mismatch")
                results[cf[0]].append(cf[target_idx + 6])

y = []
X = [[] for _ in range(0, len(experiments))]

for cl in results.values():
    y.append(float(cl[0]))
    for i, cv in enumerate(cl[1:]):
        X[i].append(float(cv))

max_y = max(max_y, max(y))
min_y = min(min_y, min(y))

for x in X:
    max_x = max(max_x, max(x))
    min_x = min(min_x, min(x))

    ax.scatter(x, y, marker=symbols.pop(), s=10, color=colors.pop(), alpha=0.7)

ax.set_axisbelow(True)
names = []

ax.set_xlabel('DT-SLIM')
ax.set_ylabel('C4.5')
# ax.set_title('scatter plot')
plt.rcParams["legend.loc"] = 'lower right'
plt.rcParams['savefig.pad_inches'] = 0
plt.autoscale(tight=True)
plt.legend(legend)
if field_idx == 1:
    plt.xscale("log")
    plt.yscale("log")
if field_idx == 0 or field_idx == 3:
    plt.xlim(min_x - 1, max_x + 1)
    plt.ylim(min_y - 1, max_y + 1)
if field_idx == 1:
    plt.xlim(min_x-10, max_x+500)
    plt.ylim(min_y-10, max_y+500)
if field_idx == 2:
    plt.xlim(min_x-0.01, max_x+0.01)
    plt.ylim(min_y-0.01, max_y+0.01)
ax.axline([0, 0], [1, 1], linestyle="--", color="grey", zorder=0)

plt.savefig(f"scatter_{field_idx}.pdf", bbox_inches='tight')
plt.show()
