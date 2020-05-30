from base_encoding import BaseEncoding
from tree_depth_encoding import TreeDepthEncoding
from aaai_encoding import AAAIEncoding


class SwitchingEncoding(BaseEncoding):
    def __init__(self, stream):
        BaseEncoding.__init__(self, stream)
        self.last_limit = 0
        self.switch_threshold = 10
        self.enc1 = AAAIEncoding(stream)
        self.enc2 = TreeDepthEncoding(stream)

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
        if num_features < 50:
            return 100
        if num_features < 100:
            return 70
        return 50

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

