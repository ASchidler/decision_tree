import subprocess
import sys
import os
import parser
from sat.size_narodytska import SizeNarodytska
from sat.depth_avellaneda import DepthAvellaneda
from sat.depth_partition import DepthPartition
from sat.switching_encoding import SwitchingEncoding
from pysat.solvers import Glucose3

instance = parser.parse(sys.argv[1])
test_instance = instance
if os.path.exists(sys.argv[1][:-4]+"test"):
    test_instance = parser.parse(sys.argv[1][:-4] + "test")

#encoding = DepthAvellaneda()
#encoding = DepthPartition()
#encoding = SizeNarodytska()
encoding = SwitchingEncoding()
tree = encoding.run(instance, Glucose3)
tree.check_consistency()
# Verify tree
total = 0
correct = 0
for e in instance.examples:
    decision = tree.decide(e.features)
    total += 1
    if decision != e.cls:
        print(f"ERROR: Decision mismatch, should be {e.cls} but is {decision}.")
    else:
        correct += 1
print(f"Accuracy: {correct/total}")

print(f"Final result: {tree.get_depth()}")
if tree is not None:
    print(f"Accuracy: {tree.get_accuracy(test_instance.examples)}")
