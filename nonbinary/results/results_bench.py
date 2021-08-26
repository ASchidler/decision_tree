import tarfile
import sys
from os import linesep
from sklearn.linear_model import LinearRegression, Lasso


class BenchmarkResult:
    def __init__(self, instance, instance_slice, duration, timeout, depth, examples, classes, features, max_domain, sum_domain, encoding_size, reduced):
        self.instance = instance
        self.slice = instance_slice
        self.duration = duration
        self.timeout = timeout
        self.depth = depth
        self.examples = examples
        self.classes = classes
        self.features = features
        self.max_domain = max_domain
        self.sum_domain = sum_domain
        self.encoding_size = encoding_size
        self.reduced = reduced

def parse_file(c_file):
    instance_name = None
    instance_slice = None
    reduced = None
    c_bound = 0
    results = []
    for i, cl in enumerate(c_file):
        if type(cl) is not str:
            cl = cl.decode('ascii')
        if i == 0:
            instance_name = cl.split(",")[0].split(":")[1].strip()
            slice_idx = cl.index("slice=")
            end_idx = cl[slice_idx:].index(",")
            instance_slice = cl[slice_idx+6: slice_idx+end_idx]
            reduced = 0 if cl.find("reduce=False") > -1 else 1
        if cl.startswith("Running"):
            c_bound = int(cl.split(",")[0].split(" ")[1])
        elif cl.startswith("E:") and not cl.find("Time: -1") > -1:
            fields = cl.split(" ")
            fields = [int(float(x.split(":")[1])) for x in fields]
            results.append(BenchmarkResult(instance_name, instance_slice, fields[1], False, fields[6], fields[0],
                                           fields[2], fields[3], fields[5], fields[4], fields[7], reduced))
    return results


with tarfile.open(sys.argv[1]) as tar_file:
    results = []
    for ctf in tar_file:
        file_parts = ctf.path.split(".")

        if file_parts[-2].startswith("e"):
            # Error file
            continue
        cetf = tar_file.extractfile(ctf)

        results.extend(parse_file(cetf))
#
# with open("benchmark_results.csv", "w") as outp:
#     outp.write("Instance;Slice;Reduced;Duration;Encoding Size;Depth;Domain Sizes;Examples;Classes;Features;Nodes;Decisions"+linesep)
#     for cr in results:
#         outp.write(f"{cr.instance};{cr.slice};{cr.reduced};{cr.duration};{cr.encoding_size};{cr.depth};{cr.sum_domain};{cr.examples};{cr.classes};{cr.features};{2**cr.depth};{2**cr.depth * cr.sum_domain}"+linesep)


X = []
y = []

for cr in results:
    if cr.reduced == 1:
        X.append([cr.depth, 2**cr.depth, cr.sum_domain, cr.classes, cr.encoding_size, cr.reduced, cr.examples, 2**cr.depth * cr.sum_domain])
        # X.append([2 ** cr.depth, cr.sum_domain, cr.reduced, cr.examples,
        #           2 ** cr.depth * cr.sum_domain, cr.encoding_size])
        y.append(cr.duration)

reg = LinearRegression().fit(X, y)
#reg = Lasso(alpha=0.1).fit(X, y)
print(f"{reg.score(X,y)}")
print(f"{reg.coef_}")
