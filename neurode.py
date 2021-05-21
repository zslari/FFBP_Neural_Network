from multi_link_node import MultiLinkNode
import random


class Neurode(MultiLinkNode):
    def __init__(self, node_type, learning_rate=.05):
        super().__init__()
        self._value = 0
        self._node_type = node_type
        self._learning_rate = learning_rate
        self._weights = dict()

    def _process_new_neighbor(self, node, side: MultiLinkNode.Side):
        """ Processes upstream neighboring nodes """
        if side is MultiLinkNode.Side.UPSTREAM:
            self._weights[node] = random.random()

    def _check_in(self, node, side: MultiLinkNode.Side):
        """ Tells nodes they have been processed """
        node_number = self._neighbors[side].index(node)
        self._reporting_nodes[side] = self._reporting_nodes[side] | 1 << node_number
        if self._reporting_nodes[side] == self._reference_value[side]:
            self._reporting_nodes[side] = 0
            return True
        else:
            return False

    @property
    def value(self):
        return self._value

    @property
    def node_type(self):
        return self._node_type

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        self._learning_rate = learning_rate

    def weight(self, node):
        return self._weights[node]