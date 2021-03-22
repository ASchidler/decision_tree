from sat.depth_avellaneda import DepthAvellaneda
from sat.depth_partition import DepthPartition
from sys import maxsize
from threading import Timer


class SwitchingEncoding:
    def __init__(self,):
        self.last_limit = 0
        self.switch_threshold = 10
        self.enc1 = DepthAvellaneda()
        self.enc2 = DepthPartition()

    def decode(self, model, instance, depth):
        if self.last_limit < self.switch_threshold:
            return self.enc1.decode(model, instance, depth)
        else:
            return self.enc2.decode(model, instance, depth)

    @staticmethod
    def lb():
        return 1

    @staticmethod
    def max_instances(num_features, limit):
        if num_features < 20:
            return 50
        if num_features < 35:
            return 40
        return 25

    @staticmethod
    def new_bound(tree, instance):
        if tree is None:
            return 1

        def dfs_find(node, level):
            if node.is_leaf:
                return level
            else:
                return max(dfs_find(node.left, level + 1), dfs_find(node.right, level + 1))

        return dfs_find(tree.root, 0)

    def encode(self, instance, depth):
        self.last_limit = depth
        if depth >= self.switch_threshold:
            return self.enc2.encode(instance, depth)
        return self.enc1.encode(instance, depth)

    def run(self, instance, solver, start_bound=1, timeout=0, ub=maxsize):
        c_bound = start_bound
        lb = 0
        best_model = None

        while lb < ub:
            print(f"Running {c_bound}")
            with solver() as slv:
                enc = self.enc2 if c_bound >= self.switch_threshold else self.enc1

                enc.encode(instance, c_bound)
                slv.append_formula(enc.formula)
                if timeout == 0:
                    solved = slv.solve()
                else:
                    def interrupt(s):
                        s.interrupt()

                    timer = Timer(timeout, interrupt, [slv])
                    timer.start()
                    solved = slv.solve_limited(expect_interrupt=True)

                if solved:
                    model = {abs(x): x > 0 for x in slv.get_model()}
                    best_model = enc.decode(model, instance, c_bound)
                    ub = c_bound
                    c_bound -= 1
                else:
                    c_bound += 1
                    lb = c_bound + 1

        return best_model