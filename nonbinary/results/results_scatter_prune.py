import sys
import matplotlib.pyplot as plt
from collections import defaultdict

cmp_heur = True
#experiments = [("k", "SZ,EX"), ("m", "SZ"), ("n", "M"), ("q", "None")]
experiments = [("k", "D-SZ-EX"), ("m", "D-SZ")]

fields = [(10, "Depth"), (9, "Size"), (12, "Accuracy"), (13, "Avg. Decision Length")]
field_idx = 3

colors = ['#228833', 'black', '#eecc66', '#bb5566', '#004488']
symbols = ['d', 'x', 's', 'v', 'o']

fig, ax = plt.subplots(figsize=(4, 2.5), dpi=300)
min_x = sys.maxsize * 1.0
max_x = 0.0
min_y = sys.maxsize * 1.0
max_y = 0.0

results = defaultdict(list)
legend = []

target_idx = fields[field_idx][0]

for c_experiment, c_ex_name in experiments:
    if cmp_heur or c_ex_name != experiments[0][1]:
        legend.append(c_ex_name)

    with open(f"results_{c_experiment}_comp.csv") as inp:
        for i, cl in enumerate(inp):
            if i > 0:
                cf = cl.split(";")
                if cmp_heur:
                    if len(results[cf[0]]) == 0:
                        results[cf[0]].append(cf[target_idx])
                    # else:
                    #     if results[cf[0]][0] != cf[target_idx]:
                    #         raise RuntimeError("Base mismatch")
                results[cf[0]].append(cf[target_idx + 12])

y = []
X = [[] for _ in range(0 if cmp_heur else 1, len(experiments))]
lts = [0 for _ in range(0 if cmp_heur else 1, len(experiments))]
lts2 = [0 for _ in range(0 if cmp_heur else 1, len(experiments))]
gts = [0 for _ in range(0 if cmp_heur else 1, len(experiments))]
gts2 = [0 for _ in range(0 if cmp_heur else 1, len(experiments))]

for cl in results.values():
    if not any(x == "-1" for x in cl):
        y.append(float(cl[0]))
        for i, cv in enumerate(cl[1:]):
            X[i].append(float(cv))
            if X[i][-1] < y[-1]:
                lts[i] += 1
                if X[i][-1] < y[-1] * 0.9:
                    lts2[i] += 1
            elif X[i][-1] > y[-1]:
                gts[i] += 1
                if X[i][-1] > y[-1] * 1.1:
                    gts2[i] += 1

for i in range(0, len(lts)):
    print(f"{i}: {lts[i]} ({lts2[i]}) {gts[i]} ({gts2[i]})")

max_y = max(max_y, max(y))
min_y = min(min_y, min(y))

for x in X:
    max_x = max(max_x, max(x))
    min_x = min(min_x, min(x))

    ax.scatter(x, y, marker=symbols.pop(), s=10, alpha=0.7 if colors[-1] != '#eecc66' else 1,
               zorder=2 if colors[-1] != '#eecc66' else 1, color=colors.pop())

ax.set_axisbelow(True)
names = []

ax.set_xlabel('DT-SLIM')
ax.set_ylabel('C4.5' if cmp_heur else experiments[0][1])
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
    plt.xlim(min_x, max_x+200)
    plt.ylim(min_y, max_y+200)
if field_idx == 2:
    plt.xlim(min_x-0.01, max_x+0.01)
    plt.ylim(min_y-0.01, max_y+0.01)
ax.axline([0, 0], [1, 1], linestyle="--", color="grey", zorder=0)

plt.savefig(f"scatter_prune_{field_idx}.pdf", bbox_inches='tight')
plt.show()
