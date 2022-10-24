import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import math

cmp_heur = True
use_cart = False
use_binoct = True
use_pruning = True
binary_only = False

experiment = "m"

experiments = [("m", "DT-SLIM")]
use_encodings = ["0", "4", "5", "7"]
#use_encodings = ["7"]

fields = [(10, "Depth"), (9, "Size"), (12, "Accuracy"), (13, "Avg. Decision Length"),
          (11, "Training Accuracy")
          ]
field_idx = 2
bar_idx = [0, 1, 2]
pruning_offset_encoding = 1 * 6
pruning_offset = 1 * 6
#offset = 66
#offset = 48

colors = ['black', '#eecc66', '#228833', '#777777', '#a50266', '#004488']
symbols = ['d', 'x', 's', 'v', 'o']
ignore = set()
with open("ignore.txt") as ip:
    for _, cl in enumerate(ip):
        if len(cl.strip()) > 0:
            ignore.add(cl.strip())


min_x = sys.maxsize * 1.0
max_x = 0.0
min_y = sys.maxsize * 1.0
max_y = 0.0

encoding_results = list()
encoding_compare = list()
heuristic_results = list()
heuristic_compare = list()
legend = []

target_idx = fields[field_idx][0]

binoct_results = None
if use_binoct:
    binoct_results = dict()
    with open("results_binoct.csv") as inp:
        for i, cl in enumerate(inp):
            if i > 0:
                fds = cl.strip().split(";")
                binoct_results[fds[0]] = [float(fds[-4]), float(fds[-5]), float(fds[-2]), float(fds[-1]), float(fds[-3])]


# Extract encoding information
with open(f"results_e_comp{'' if not use_cart else '_c'}.csv") as inp:
    starts = dict()
    last_encoding = False
    encoding_field_idx = target_idx - 8

    for i, cl in enumerate(inp):
        c_fields = cl.strip().split(";")
        if i == 0:
            for cfi, cf in enumerate(c_fields):
                try:
                    int(cf[0])
                    if cf[0] != last_encoding:
                        starts[cf[0]] = cfi
                        last_encoding = cf[0]
                except ValueError:
                    pass
        else:
            if c_fields[0] not in ignore or (binary_only and c_fields[3] != "2"):
                continue
            c_min = None
            c_cmp = None

            if cmp_heur:
                if use_binoct:
                    c_cmp = binoct_results[c_fields[0]]
                elif use_pruning:
                    c_cmp = [float(c_fields[x[0]]) for x in fields]
                else:
                    c_cmp = [float(c_fields[x[0] - 5]) for x in fields]

            for c_enc in use_encodings:
                c_t = starts[c_enc]
                if c_fields[c_t + pruning_offset_encoding + encoding_field_idx] != "" and c_fields[c_t + pruning_offset_encoding + encoding_field_idx] != "-1":
                    if c_min is None or float(c_fields[c_t + 1 + pruning_offset_encoding]) < c_min[1]:
                        c_min = [float(c_fields[c_t + x[0] + pruning_offset_encoding - 8].strip()) for x in fields]
                    if not cmp_heur:
                        if c_cmp is None or float(c_fields[c_t + 1 + 12]) < c_cmp[1]:
                            c_cmp = [float(c_fields[c_t + x[0] - 8 + 12].strip()) for x in fields]
                if c_t + 18 + pruning_offset_encoding + encoding_field_idx < len(fields) and c_fields[c_t + encoding_field_idx + 18 + pruning_offset_encoding] != "" and c_fields[c_t + encoding_field_idx + 18 + pruning_offset_encoding] != "-1":
                    if c_min is None or float(c_fields[c_t + 1 + pruning_offset_encoding + 18]) < c_min[0]:
                        c_min = [float(c_fields[c_t + x[0] + pruning_offset_encoding - 8 + 18].strip()) for x in fields]
                    if not cmp_heur:
                        if c_cmp is None or float(c_fields[c_t + 1 + 18 + 12]) < c_cmp[1]:
                            c_cmp = [float(c_fields[c_t + x[0] - 8 + 18 + 12].strip()) for x in fields]
            if c_min is not None:
                encoding_results.append(c_min)
                encoding_compare.append(c_cmp)
            else:
                print("Error")

with open(f"results_{experiment}_comp{''if not use_cart else '_c'}.csv") as inp:
    for i, cl in enumerate(inp):
        if i > 0:
            cf = cl.split(";")
            if binary_only and cf[3] != "2":
                continue
            if cmp_heur:
                if use_binoct:
                    heuristic_compare.append(binoct_results[cf[0]])
                elif use_pruning:
                    heuristic_compare.append([float(cf[x[0]]) for x in fields])
                else:
                    heuristic_compare.append([float(cf[x[0] - 5]) for x in fields])
                # else:
                #     if results[cf[0]][0] != cf[target_idx]:
                #         raise RuntimeError("Base mismatch")
            else:
                heuristic_compare.append([float(cf[x[0] + 6].strip()) for x in fields])

            heuristic_results.append([float(cf[x[0] + 6 + pruning_offset].strip()) for x in fields])

X = []
y = []

fig, ax = plt.subplots(figsize=(3, 1.5), dpi=300)

# for x in X:
#     max_x = max(max_x, max(x))
#     min_x = min(min_x, min(x))
#
#     ax.scatter(x, y, marker=symbols.pop(), s=10, alpha=0.7 if colors[-1] != '#eecc66' else 1,
#                zorder=2 if colors[-1] != '#eecc66' else 1, color=colors.pop())


X2 = []
y2 = []
all_values = []
for idx, values in enumerate(heuristic_results):
    # X.append((round(values[field_idx], 2), 0))
    # y.append(round(heuristic_compare[idx][field_idx], 2))
    X.append((values[field_idx], 0))
    y.append(heuristic_compare[idx][field_idx])
for idx, values in enumerate(encoding_results):
    # X.append((round(values[field_idx], 2), 1))
    # y.append(round(encoding_compare[idx][field_idx], 2))
    X.append((values[field_idx], 1))
    y.append(encoding_compare[idx][field_idx])

# X[i].append(float(cv[field_idx]))
xp = [(X[i][0] - y[i], X[i][1]) for i in range(0, len(X))]
xp.sort()

legend = ["DT-SLIM", "Encoding"]
ax.scatter([idx + 1 for idx in range(0, len(xp)) if xp[idx][1] == 0], [v[0] for v in xp if v[1] == 0], marker=symbols.pop(), s=3, alpha=0.7 if colors[-1] != '#eecc66' else 1,
           zorder=2 if colors[-1] != '#eecc66' else 1, color=colors.pop())
ax.scatter([idx + 1 for idx in range(0, len(xp)) if xp[idx][1] == 1], [v[0] for v in xp if v[1] == 1], marker=symbols.pop(), s=3, alpha=0.7 if colors[-1] != '#eecc66' else 1,
           zorder=2 if colors[-1] != '#eecc66' else 1, color=colors.pop())

ax.set_xlabel('Instances')
ax.set_ylabel('Difference')
# ax.set_title('scatter plot')
plt.rcParams["legend.loc"] = 'upper left'
plt.rcParams['savefig.pad_inches'] = 0
plt.autoscale(tight=True)
#plt.legend(legend)

found_zero = False
start_neutral = None
end_neutral = None
for c_idx, c_v in enumerate(xp):
    if c_v[0] == 0 and not found_zero:
        found_zero = True
        if c_idx > 0:
            plt.axhspan(-100000, 0, xmin=0,xmax=(c_idx + 0.5) / 70, facecolor='#f67e4b' if field_idx == 2 else '#4eb265', alpha=0.5)
            #plt.axhspan(-10000, 0, facecolor='#d6604d', alpha=0.5)
            #plt.axvspan(0, c_idx + 1 - 0.5, facecolor='#d6604d', alpha=0.5)
            ax.axvline(c_idx + 1 - 0.5, linestyle="--", color="grey", zorder=0, linewidth=0.5)
            start_neutral = c_idx
    elif c_v[0] > 0:
        if c_idx > 0:

            plt.axhspan(0, 100000, xmin=(c_idx + 0.5)/70.0, facecolor='#f67e4b' if field_idx != 2 else '#4eb265', alpha=0.5)
            #plt.axhspan(0, 10000, facecolor='#4eb265', alpha=0.5)
            #plt.axvspan(c_idx + 1 - 0.5, 70, facecolor='#4eb265', alpha=0.5)
            ax.axvline(c_idx + 1 - 0.5, linestyle="--", color="grey", zorder=0, linewidth=0.5)
            end_neutral = c_idx
        break

if start_neutral is not None and end_neutral is not None:
    plt.axvspan(start_neutral + 0.5, end_neutral + 0.5, facecolor='#FFEE99', alpha=0.5)

ax.set_axisbelow(True)
names = []

plt.xlim(0.5, len(xp) + 0.5)
plt.ylim(min(x[0] for x in xp) - abs(min(x[0] for x in xp)) * 0.1, max(x[0] for x in xp) + abs(max(x[0] for x in xp)) * 0.4)
#ax.axline([0, 0], [1, 1], linestyle="--", color="grey", zorder=0)

# for idx in range(0, len(encoding_results)):
#     encoding_results[idx] = [round(x, 2) for x in encoding_results[idx]]
#     encoding_compare[idx] = [round(x, 2) for x in encoding_compare[idx]]
# for idx in range(0, len(heuristic_results)):
#     heuristic_results[idx] = [round(x, 2) for x in heuristic_results[idx]]
#     heuristic_compare[idx] = [round(x, 2) for x in heuristic_compare[idx]]

if field_idx == 1:
    #xticks = ax.yaxis.get_major_ticks()
    # xticks[0].label1.set_visible(False)
    #plt.xscale("log")
    plt.yscale("symlog")
# if field_idx == 0 or field_idx == 3:

#     plt.ylim(min_y - 1, max_y + 1)
# if field_idx == 1:
#     plt.xlim(min_x, max_x+200)
#     plt.ylim(min_y, max_y+200)
# if field_idx == 2:
#     plt.xlim(min_x-0.01, max_x+0.01)
#     plt.ylim(min_y-0.01, max_y+0.01)
# ax.axline([0, 0], [1, 1], linestyle="--", color="grey", zorder=0)

plt.axhline(linestyle="-", color="grey", zorder=0, linewidth=0.5, y=0)

#ax.axes.get_yaxis().set_ticks([10000, 1000, 100, 10, 1, -1, -10])
#ax.axes.get_yaxis().set_ticks([10, 1, -1, -10, -100, -1000, -10000])
#ax.axes.get_yaxis().set_ticks([1000, 100, 10, 1, -1, -10, -100])
plt.savefig(f"scatter2_prune_{field_idx}.pdf", bbox_inches='tight')
plt.show()

better = [0, 0, 0]
worse = [0, 0, 0]
equal = [0, 0, 0]
better_const_acc = [0, 0]
better_const_size = 0

better_acc_cum = [0 for i in range(0, 50)]
better_size_cum = [0 for i in range(0, 50)]

for c_origin_r, c_origin_c in [(encoding_results, encoding_compare), (heuristic_results, heuristic_compare)]:
    for idx in range(0, len(c_origin_r)):
        for c_field in range(0, 3):
            if (c_field < 2 and c_origin_r[idx][c_field] < c_origin_c[idx][c_field]) or (c_field == 2 and c_origin_r[idx][c_field] > c_origin_c[idx][c_field]):
                better[c_field] += 1
            elif (c_field < 2 and c_origin_r[idx][c_field] > c_origin_c[idx][c_field]) or (c_field == 2 and c_origin_r[idx][c_field] < c_origin_c[idx][c_field]):
                worse[c_field] += 1
            elif (c_origin_r[idx][c_field] == c_origin_c[idx][c_field]):
                equal[c_field] += 1

        if c_origin_r[idx][2] >= c_origin_c[idx][2] and c_origin_r[idx][0] < c_origin_c[idx][0]:
            better_const_acc[0] += 1
        if c_origin_r[idx][2] >= c_origin_c[idx][2] and c_origin_r[idx][1] < c_origin_c[idx][1]:
            better_const_acc[1] += 1
        if c_origin_r[idx][1] <= c_origin_c[idx][1] and c_origin_r[idx][2] > c_origin_c[idx][2]:
            better_const_size += 1

    for c_diff in range(0, len(better_acc_cum)):
        for idx in range(0, len(c_origin_r)):
            if c_origin_r[idx][2] >= c_origin_c[idx][2] - c_diff * 0.01 and c_origin_r[idx][1] < c_origin_c[idx][1]:
                better_size_cum[c_diff] += 1
            if c_origin_r[idx][1] <= c_origin_c[idx][1] * (1 + 0.1 * float(c_diff)) and c_origin_r[idx][2] > c_origin_c[idx][2]:
                better_acc_cum[c_diff] += 1

sizes = [0, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 1000000]
s_better = [[0, 0, 0] for _ in range(0, len(sizes))]
s_counts = [0 for _ in range(0, len(sizes))]

for c_origin_r, c_origin_c in [(encoding_results, encoding_compare), (heuristic_results, heuristic_compare)]:
    for idx in range(0, len(c_origin_r)):
        for s_idx, c_s in enumerate(sizes):
            if c_origin_r[idx][1] >= c_s:
                s_counts[s_idx] += 1
                if c_origin_r[idx][2] > c_origin_c[idx][2]:
                    s_better[s_idx][2] += 1
                if c_origin_r[idx][1] < c_origin_c[idx][1]:
                    s_better[s_idx][1] += 1
                if c_origin_r[idx][0] < c_origin_c[idx][0]:
                    s_better[s_idx][0] += 1

print(s_counts)
print(s_better)

print(better)
print(equal)
print(worse)
print(better_const_acc)
print(better_const_size)
print(list(enumerate(better_acc_cum)))
print(list(enumerate(better_size_cum)))
print()
print(f"{len(encoding_results)}")
print(f"{len(heuristic_results)}")
