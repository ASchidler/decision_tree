import math
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

cmp_heur = False
#experiments = [("m", "DP-SZ"), ("k", "DP-SL-SZ"), ("n", "MT-DP"), ("q", "DP"), ("f", "SZ-DP")]
#experiments = [("m", "DP-SZ"), ("o", "Old"), ("r", "Cat")]
#experiments = [("m", "DP-SZ"), ("c", "Old")]
experiments = [("m", "DP-SZ"), ("g", "Dynamic Runtime Prediction"), ("x", "Static Runtime Prediction")]
#experiments = [("m", "DP-SZ"), ("g", "Encoding")]
#experiments = [("o", "DP-SZ"), ("c", "DT Budget")]
experiments = [("m", "DP-SZ"), ("a", "Dynamic Runtime Prediction"), ("b", "Static Runtime Prediction")]

fields = [(5, "Depth"), (4, "Size"), (7, "Accuracy"), (8, "Avg. Decision Length")]
field_idx = 2
bar_idx = [0, 1]

colors = ['#228833', 'black', '#eecc66', '#bb5566', '#004488']
symbols = ['d', 'x', 's', 'v', 'o']

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

    with open(f"results_{c_experiment}.csv") as inp:
        for i, cl in enumerate(inp):
            if i > 0:
                cf = cl.split(";")
                if cmp_heur:
                    if len(results[cf[0]]) == 0:
                        results[cf[0]].append([cf[x[0]] for x in fields])
                    # else:
                    #     if results[cf[0]][0] != cf[target_idx]:
                    #         raise RuntimeError("Base mismatch")
                results[cf[0]].append([cf[x[0] + 6].strip() for x in fields])


y = []
X = [[] for _ in range(0 if cmp_heur else 1, len(experiments))]
lts = [[0 for _ in range(0, len(fields))] for _ in range(0 if cmp_heur else 1, len(experiments))]
lts2 = [[0 for _ in range(0, len(fields))] for _ in range(0 if cmp_heur else 1, len(experiments))]
gts = [[0 for _ in range(0, len(fields))] for _ in range(0 if cmp_heur else 1, len(experiments))]
gts2 = [[0 for _ in range(0, len(fields))] for _ in range(0 if cmp_heur else 1, len(experiments))]
min_counts = {i: 0 for i in range(0, len(legend))}
max_counts = {i: 0 for i in range(0, len(legend))}

for cl in results.values():
    if len(cl) < len(experiments) + (1 if cmp_heur else 0):
        continue

    min_val = min(float(cv[field_idx]) for cv in cl[1:])
    max_val = max(float(cv[field_idx]) for cv in cl[1:])

    if not any(x == "-1" for x in cl):
        y.append(float(cl[0][field_idx]))
        for i, cv in enumerate(cl[1:]):
            if math.fabs(float(cv[field_idx]) - min_val) < 0.0001:
                min_counts[i] += 1
            if math.fabs(float(cv[field_idx]) - max_val) < 0.0001:
                max_counts[i] += 1

            X[i].append(float(cv[field_idx]))
            for idx in range(0, len(fields)):
                if float(cv[idx]) < float(cl[0][idx]):
                    lts[i][idx] += 1
                    if (idx != 2 and float(cv[idx]) < float(cl[0][idx]) * 0.9) or \
                        (idx == 2 and float(cl[0][idx]) - float(cv[idx]) >= 0.01):
                        lts2[i][idx] += 1
                elif float(cv[idx]) > float(cl[0][idx]):
                    gts[i][idx] += 1
                    if (idx != 2 and float(cv[idx]) * 0.9 > float(cl[0][idx])) or \
                            (idx == 2 and float(cv[idx]) - float(cl[0][idx]) >= 0.01):
                        gts2[i][idx] += 1

for i in range(0, len(lts)):
    print(f"{i}: {lts[i]} ({lts2[i]}) {gts[i]} ({gts2[i]})")
    print(f"{min_counts[i]}")
    print(f"{max_counts[i]}")

# Bar plot
fig, ax = plt.subplots(figsize=(4, 0.6 * len(lts)))
yticks = []
ylabels = []

bar_height = 0.06
y_scale = len(lts) * 2 * bar_height + 0.1
# # Example data
# for i in range(0, len(lts)):
#     plt.text(0, -i * y_scale, legend[i], fontsize=10, ha='center')
#     yticks.append(-i)
#     ylabels.append(legend[i])
#     plt.plot([0, 0], [-i*y_scale - 0.045, -i*y_scale - 0.045 - len(bar_idx) * (bar_height + 0.02) + 0.027], color='black', lw=0.5)
#     for idx_idx, idx in enumerate(bar_idx):
#         y_pos = -1 * (i * y_scale + (bar_height + 0.02) * idx_idx + 0.1)
#
#         ax.barh(y_pos, gts[i][idx], align='center', color='grey', height=bar_height)
#         plt.text(max(gts[i][idx] + 0.1, 3), y_pos - 0.01, f"{gts[i][idx]} ({gts2[i][idx]})", ha='left', va='center', size=8)
#         ax.barh(y_pos, -lts[i][idx], align='center', color='grey', height=bar_height)
#         plt.text(min(-lts[i][idx] - 0.1, -3), y_pos - 0.01, f"{lts[i][idx]} ({lts2[i][idx]})", ha='right', va='center', size=8)
#         ax.barh(y_pos, gts2[i][idx], align='center', color=colors[3], height=bar_height)
#         plt.text(0.2, y_pos - 0.01, fields[idx][1], ha='center', va='center', size=8)
#         ax.barh(y_pos, -lts2[i][idx], align='center', color=colors[3], height=bar_height)

# bar_height = 0.06
# y_scale = len(lts) * 2 * bar_height + 0.1
# for idx_idx, idx in enumerate(bar_idx):
#     plt.text(0, -idx_idx * y_scale, fields[idx][1], fontsize=10, ha='left')
#
#     for i in range(0, len(lts)):
#         #plt.plot([0, 0], [-i*y_scale - 0.045, -i*y_scale - 0.045 - len(bar_idx) * (bar_height + 0.02) + 0.027], color='black', lw=0.5)
#         y_pos = -1 * (idx_idx * y_scale + 2 * i * (bar_height + 0.01) + 0.04)
#
#         plt.text(0.2, y_pos, f"{legend[i]} >", ha='left', va='center', size=8) #  {'C4.5' if cmp_heur else experiments[0][1]}
#         plt.text(0.2, y_pos - bar_height, f"{legend[i]} <", ha='left', va='center', size=8)
#
#         ax.barh(y_pos, gts[i][idx], align='center', color='grey', height=bar_height)
#         ax.barh(y_pos, gts2[i][idx], align='center', color=colors[3 if idx != 2 else 0], height=bar_height)
#         plt.text(max(gts[i][idx] + 0.1, 3), y_pos - 0.01, f"{gts[i][idx]} ({gts2[i][idx]})", ha='left', va='center', size=8)
#
#         ax.barh(y_pos - bar_height, lts[i][idx], align='center', color='grey', height=bar_height)
#         ax.barh(y_pos - bar_height, lts2[i][idx], align='center', color=colors[0 if idx != 2 else 3], height=bar_height)
#         plt.text(max(lts[i][idx] + 0.1, 3), y_pos - bar_height - 0.01, f"{lts[i][idx]} ({lts2[i][idx]})", ha='left', va='center', size=8)

bar_height = 0.06
y_scale = len(lts) * bar_height + 0.1
for idx_idx, idx in enumerate(bar_idx):
    plt.text(len(results)/2, -idx_idx * y_scale, fields[idx][1], fontsize=10, ha='center')

    for i in range(0, len(lts)):
        #plt.plot([0, 0], [-i*y_scale - 0.045, -i*y_scale - 0.045 - len(bar_idx) * (bar_height + 0.02) + 0.027], color='black', lw=0.5)
        y_pos = -1 * (idx_idx * y_scale + i * (bar_height + 0.01) + 0.04)

        plt.text(0.2, y_pos, f"{legend[i]}", ha='left', va='center', size=8, fontweight=600) #  {'C4.5' if cmp_heur else experiments[0][1]}
        # plt.text(0.2, y_pos - bar_height, f"{legend[i]} <", ha='left', va='center', size=8)
        ax.barh(y_pos, lts2[i][idx], align='center', color='#4eb265' if idx != 2 else '#d6604d', height=bar_height, label=lts2[i][idx])
        ax.barh(y_pos, lts[i][idx] - lts2[i][idx], align='center', color='#cae0ab' if idx != 2 else '#F5a582', height=bar_height, left=lts2[i][idx])

        ax.barh(y_pos, len(results) - gts[i][idx] - lts[i][idx], left=lts[i][idx], align='center', color='#FFEE99',
                height=bar_height)

        ax.barh(y_pos, gts[i][idx] - gts2[i][idx], align='center', color='#F5a582' if idx != 2 else '#cae0ab', height=bar_height, left=len(results) - gts[i][idx])
        ax.barh(y_pos, gts2[i][idx], align='center', color='#d6604d' if idx != 2 else '#4eb265', height=bar_height, left=len(results) - gts2[i][idx])


        #plt.text(max(gts[i][idx] + 0.1, 3), y_pos - 0.01, f"{gts[i][idx]} ({gts2[i][idx]})", ha='left', va='center',
        #         size=8)
        #plt.text(max(lts[i][idx] + 0.1, 3), y_pos - bar_height - 0.01, f"{lts[i][idx]} ({lts2[i][idx]})", ha='left', va='center', size=8)



# people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
# y_pos = np.arange(len(people))
# performance = 3 + 10 * np.random.rand(len(people))
# error = np.random.rand(len(people))
#fig.axes[0].get_xaxis().set_visible(False)
ax.set_yticks([])
ax.set_yticklabels([])
ax.spines['left'].set_color('none')
#set_position('zero')
#ax.spines['bottom'].set_color('none')

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

#ax.invert_yaxis()  # labels read top-to-bottom
#ax.set_xlabel('Performance')
plt.rcParams['savefig.pad_inches'] = 0
plt.autoscale(tight=True)

plt.savefig(f"bar.pdf", bbox_inches='tight')
plt.show()

max_y = max(max_y, max(y))
min_y = min(min_y, min(y))
fig, ax = plt.subplots(figsize=(4, 2.5), dpi=300)
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
    plt.xlim(min_x-10, max_x+500)
    plt.ylim(min_y-10, max_y+500)
if field_idx == 2:
    plt.xlim(min_x-0.01, max_x+0.01)
    plt.ylim(min_y-0.01, max_y+0.01)
ax.axline([0, 0], [1, 1], linestyle="--", color="grey", zorder=0)

plt.savefig(f"scatter_{field_idx}.pdf", bbox_inches='tight')
plt.show()
