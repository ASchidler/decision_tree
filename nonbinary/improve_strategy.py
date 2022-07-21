import math
import sys
import time

import nonbinary.depth_avellaneda_base as bs
import nonbinary.depth_avellaneda_base_benchmark as bbs
import nonbinary.depth_avellaneda_sat as s1
import nonbinary.depth_avellaneda_sat2 as s2
import nonbinary.depth_avellaneda_sat3 as s3
import nonbinary.improver as improver
from nonbinary.nonbinary_instance import Example

sample_limit_short = [175,
                      175, 175, 175, 175, 175,
                      175, 175, 175, 150, 120,
                      70, 70, 70, 70, 70,
                      0]

sample_limit_mid = [300,
                    300, 300, 300, 270, 270,
                    270, 270, 270, 270, 250,
                    200, 100, 0]

sample_limit_long = [50000,
                     50000, 50000, 25000, 7000, 7000,
                     1500, 800, 600, 500, 250,
                     250, 250, 250, 250, 250,
                     250, 250, 250, 250, 250,
                     250, 250, 250, 250, 250,
                     250, 250, 250, 250, 250,
                     250, 250, 250, 250, 250,
                     250, 250, 250, 250, 250]

literal_limit = 200 * 1000 * 1000 * 1000

time_limits = [60, 300, 300]
depth_limits = [12, 12, 14]

reduce_runs = 1

encodings = [None, s1, s2, s3]


class SlimParameters:
    def __init__(self, tree, instance, encoding, slv, opt_size, opt_slim, maintain, reduce_numeric, reduce_categoric, timelimit, use_dt, benchmark,
                 size_first, use_enc_dt, reduce_first, reduce_alternate):
        self.tree = tree
        self.instance = instance
        self.encoding = encoding
        self.slv = slv
        self.opt_size = opt_size
        self.opt_slim = opt_slim
        self.maintain = maintain
        self.reduce_numeric_full = reduce_numeric
        self.reduce_categoric_full = reduce_categoric
        self.timelimit = timelimit
        self.use_smt = not self.encoding.is_sat()
        self.use_dt = use_dt
        self.example_decision_tree = None
        self.maximum_examples = 25000
        self.maximum_depth = 14
        self.sample_limits = [self.maximum_examples for _ in range(0, self.maximum_depth + 1)]
        self.solver_time_limit = 300
        self.benchmark = benchmark
        self.size_first = size_first
        self.use_enc_dt = use_enc_dt
        self.enc_decision_tree = None
        self.reduce_first = reduce_first
        self.reduce_alternate = reduce_alternate

    def call_solver(self, new_instance, new_ub, cd, leaves):
        if not self.benchmark:
            use_encoding = self.encoding
            if self.enc_decision_tree is not None:
                sample = self._create_tree_sample(new_instance, min(new_ub, cd - 1) if not self.size_first else min(new_ub, cd))
                target_encoding = int(self.enc_decision_tree.root.decide(sample)[0])
                if target_encoding <= 0:
                    return None
                # Encoding 3 can only run on non threshold-reduced instances, use encoding 1 as default replacement
                # This is not valid anymore, but since thresholds can only used as thresholds and not for =,
                # using encoding 3 for threshold reduced instances makes little sense
                if target_encoding == 3 and new_instance.reduced_key is not None and \
                    any(x[1] is not None for x in new_instance.reduced_key):
                    target_encoding = 1
                use_encoding = encodings[target_encoding]

            if self.encoding.is_sat():
                return bs.run(use_encoding, new_instance, self.slv, start_bound=min(new_ub, cd - 1),
                                  timeout=self.solver_time_limit,
                                  ub=min(new_ub, cd - 1), opt_size=self.opt_size, slim=self.opt_slim,
                                  maintain=self.maintain,
                                  limit_size=leaves, c_depth=cd, size_first=self.size_first)
            else:
                return self.encoding.run(new_instance, start_bound=min(new_ub, cd - 1), timeout=self.solver_time_limit,
                                                   ub=min(new_ub, cd - 1), opt_size=self.opt_size,
                                                   slim=self.opt_slim, maintain=self.maintain)
        else:
            return bbs.run(new_instance, self.slv, start_bound=min(new_ub, cd - 1),
                          timeout=self.solver_time_limit,
                          ub=min(new_ub, cd - 1), opt_size=self.opt_size, slim=self.opt_slim,
                          maintain=self.maintain,
                          limit_size=leaves)

    def _create_tree_sample(self, new_instance, c_bound):
        return Example(None, [new_instance.reduced_key is not None, self.encoding.estimate_size(new_instance, c_bound),
                      c_bound, sum(len(new_instance.domains[x]) for x in range(1, new_instance.num_features + 1)),
                      max(len(new_instance.domains[x]) for x in range(1, new_instance.num_features + 1)),
                      len(new_instance.examples), len(new_instance.classes), new_instance.num_features,
                      -1 * sum(x / len(new_instance.examples) * math.log2(x / len(new_instance.examples)) for x in
                               new_instance.class_distribution.values())
            ], None)

    def decide_instance(self, new_instance, c_bound):
        if len(new_instance.examples) > self.maximum_examples or c_bound > self.maximum_depth \
                or self.encoding.estimate_size(new_instance, c_bound) > literal_limit or \
                len(new_instance.examples) == 0:
            return False

        if self.example_decision_tree is not None:
            sample = self._create_tree_sample(new_instance, c_bound if not self.size_first else c_bound+1)
            return self.example_decision_tree.root.decide(sample)[0] == "1"
        elif self.enc_decision_tree is not None:
            sample = self._create_tree_sample(new_instance, c_bound if not self.size_first else c_bound + 1)
            return self.enc_decision_tree.root.decide(sample)[0] != "-1"
        else:
            return self.sample_limits[c_bound if not self.size_first else c_bound+1] >= len(new_instance.examples)

    def get_max_bound(self, new_instance):
        new_ub = -1
        if len(new_instance.examples) == 0 or len(new_instance.examples) > self.maximum_examples:
            return -1

        sample = None if self.example_decision_tree is None and self.enc_decision_tree is None\
            else self._create_tree_sample(new_instance, new_ub)
        for cb, sl in enumerate(self.sample_limits):
            if 1 <= cb <= self.maximum_depth and self.encoding.estimate_size(new_instance, cb) < literal_limit:
                if self.example_decision_tree is not None:
                    sample.features[3] = cb
                    if self.example_decision_tree.root.decide(sample)[0] == "1":
                        new_ub = cb
                    else:
                        break
                elif self.enc_decision_tree is not None:
                    sample.features[3] = cb
                    if self.enc_decision_tree.root.decide(sample)[0] != "-1":
                        new_ub = cb
                    else:
                        break
                else:
                    if sl >= len(new_instance.examples):
                        new_ub = cb
                    else:
                        break

        return new_ub


def find_deepest_leaf(tree, ignore=None):
    if not ignore:
        ignore = set()

    q = [(0, tree.root)]
    c_max = (-1, None)
    parent = {tree.root.id: None}

    while q:
        c_d, c_n = q.pop()

        if c_d > c_max[0] and c_n.id not in ignore:
            c_max = (c_d, c_n)

        if not c_n.is_leaf:
            if c_n.left.id:
                parent[c_n.left.id] = c_n
                q.append((c_d+1, c_n.left))
            if c_n.right.id:
                parent[c_n.right.id] = c_n
                q.append((c_d+1, c_n.right))

    if c_max[1] is None:
        return None

    c_node = c_max[1]
    path = []

    while c_node is not None:
        path.append(c_node)
        c_node = parent[c_node.id]

    return path


def clear_ignore(ignore, root):
    q = [root]

    while q:
        c_n = q.pop()
        ignore.discard(c_n.id)
        if not c_n.is_leaf:
            q.append(c_n.left)
            q.append(c_n.right)


def run(parameters, test, limit_idx=1):
    parameters.sample_limits = [sample_limit_short, sample_limit_mid, sample_limit_long][limit_idx]
    parameters.maximum_depth = depth_limits[limit_idx]
    parameters.maximum_examples = 500 if parameters.example_decision_tree is not None else parameters.sample_limits[2]
    parameters.solver_time_limit = time_limits[limit_idx]
    # Select nodes based on the depth
    c_ignore = set()
    c_ignore_reduce = set()

    start_time = time.time()

    def process_change(mth):
        print(f"Time {time.time() - start_time:.4f}\t"
              f"Training {parameters.tree.get_accuracy(parameters.instance.examples):.4f}\t"
              f"Test {parameters.tree.get_accuracy(test.examples):.4f}\t"
              f"Depth {parameters.tree.get_depth():03}\t"              
              f"Nodes {parameters.tree.get_nodes()}\t"
              f"Avg. Length {parameters.tree.get_avg_length(parameters.instance.examples)}\t"
              f"Method {mth}\t")
        sys.stdout.flush()

    assigned = parameters.tree.assign(parameters.instance)
    tree_size = parameters.tree.get_nodes()
    if parameters.reduce_first:
        c_ignore.update(x.id for x in parameters.tree.nodes if x is not None)

    while tree_size > len(c_ignore_reduce):
        allow_reduction = False
        pth = find_deepest_leaf(parameters.tree, c_ignore)

        if pth is None:
            allow_reduction = True
            pth = find_deepest_leaf(parameters.tree, c_ignore_reduce)
            if pth is None:
                return

        while pth:
            root = pth.pop()

            if (not allow_reduction and root.id in c_ignore) or (allow_reduction and root.id in c_ignore_reduce):
                continue

            op = None
            result = False
            is_done = False
            if not allow_reduction:
                ran = False
                print("ls")
                result, orig = improver.leaf_select(parameters, root, assigned, parameters.instance)
                if result:
                    op = "ls"
                elif result is None or not orig:
                    print("la")
                    result, _, ran = improver.leaf_reduced(parameters, root, assigned, parameters.instance, False)
                    if result:
                        op = "la"
                elif result == False:
                    is_done = True

                if not result and ran:
                    print("lr")
                    result, orig, _ = improver.leaf_reduced(parameters, root, assigned, parameters.instance, True)
                    if result:
                        op = "lr"

                if result is None:
                    print("ma")
                    result, _ = improver.mid_reduced(parameters, root, assigned, parameters.instance, False, rerun=True)
                    if result:
                        op = "ma"
            else:
                if len(assigned[root.id]):
                    result, orig, _ = improver.leaf_reduced(parameters, root, assigned, parameters.instance, True)
                    print("lr")
                    if result:
                        op = "lr"
                    if result is None or not orig:
                        print("mr")
                        result, _ = improver.mid_reduced(parameters, root, assigned, parameters.instance, True)
                        if result:
                            op = "mr"
                    elif result == False:
                        is_done = True


            if result:
                process_change(op)

            if 0 < parameters.timelimit < (time.time() - start_time):
                return

            def mark_as_done(c_root, target):
                c_q = [c_root]
                while c_q:
                    c_n = c_q.pop()
                    target.add(c_n.id)
                    if not c_n.is_leaf:
                        c_q.extend([c_n.left, c_n.right])

            if not allow_reduction:
                c_ignore.add(root.id)
                if is_done:
                    mark_as_done(root, c_ignore)
                    break
            else:
                c_ignore_reduce.add(root.id)
                if is_done:
                    mark_as_done(root, c_ignore_reduce)
                    break

            if result:
                # TODO: This could be more efficient... We only have to re-compute assigned from the current root!
                assigned = parameters.tree.assign(parameters.instance)
                tree_size = parameters.tree.get_nodes()
                # May have been replaced if we reduce the tree to a leaf
                if parameters.tree.nodes[root.id] == root:
                    clear_ignore(c_ignore, root)
                    clear_ignore(c_ignore_reduce, root)
                # Break as the path may have become invalid
                break
            # else:
            #     print("None")
