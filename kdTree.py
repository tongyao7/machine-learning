from operator import itemgetter


class Node:
    def __init__(self, data):
        self.data = data
        self.l_child = None
        self.r_child = None


class kdtree(Node):
    def __init__(self):
        pass

    def create_tree(self, tree, point_list, depth=0):
        try:
            k = len(point_list[0])  # 维度
        except IndexError as e:
            return None
        # 选择分割轴
        axis = depth % k

        # 对列表中的点根据axis维度排序
        point_list.sort(key=itemgetter(axis))
        median = len(point_list) // 2

        # 创建子节点和子树
        if point_list is not None:
            tree = Node(point_list[median])
            tree.l_child = self.create_tree(tree.l_child, point_list[:median], depth + 1)
            tree.r_child = self.create_tree(tree.r_child, point_list[median + 1:], depth + 1)

        return tree

    def pre_order(self, tree):
        if tree is not None:
            print(str(tree.data) + '\t')
            self.pre_order(tree.l_child)
            self.pre_order(tree.r_child)

    def in_order(self, tree):
        if tree is not None:
            self.in_order(tree.l_child)
            print(str(tree.data) + '\t')
            self.in_order(tree.r_child)

    def post_order(self, tree):
        if tree is not None:
            self.post_order(tree.l_child)
            self.post_order(tree.r_child)
            print(str(tree.data) + '\t')

    # def find_nearest(self,tree,point):


point_list = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]
tree = kdtree()
root = tree.create_tree(None, point_list)
tree.post_order(root)
