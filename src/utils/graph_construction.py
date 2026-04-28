import numpy as np


class Graph():
    """ 
    The Graph to model the skeletons
    """

    def __init__(self, layout='skeleton', strategy='spatial'):
        """
        Parameters
        ----------
        layout: the layout of the graph, e.g., 'skeleton'
        strategy: the strategy to construct the adjacency matrix, e.g., 'uniform', 'spatial'
        For more information, please refer to the section 'Partition Strategies' in paper (https://arxiv.org/abs/1801.07455).
        """
        if layout == 'skeleton':
            self.num_node = 21  # number of joints in the skeleton
            self.self_link = [(i, i) for i in range(self.num_node)]
            self.neighbor_link_centripetal = [(1, 0), (2, 1), (3, 2), (4, 3),
                                              (5, 0), (6, 5), (7, 6), (8, 7),
                                              (9, 0), (10, 9), (11, 10), (12, 11),
                                              (13, 0), (14, 13), (15, 14), (16, 15),
                                              (17, 0), (18, 17), (19, 18), (20, 19)]
            self.neighbor_link_centrifugal = [(j, i) for (i, j) in self.neighbor_link_centripetal]
            self.edge = self.self_link + self.neighbor_link_centripetal
            self.neighbor = self.neighbor_link_centrifugal + self.neighbor_link_centripetal
            self.center = 0 
        else:
            raise ValueError("Do Not Exist This Layout.")
        self.A = self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_adjacency(self, strategy):
        adjacency_matrix = edge2mat(self.self_link + self.neighbor, self.num_node)
        normalized_matrix = normalize_digraph(adjacency_matrix)
        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalized_matrix
            return A
        elif strategy == 'spatial':
            I = edge2mat(self.self_link, self.num_node, normalized_matrix)
            In = edge2mat(self.neighbor_link_centripetal, self.num_node, normalized_matrix)
            Out = edge2mat(self.neighbor_link_centrifugal, self.num_node, normalized_matrix)
            A = np.stack((I, normalize_digraph(In), normalize_digraph(Out)))
            return A
        else:
            raise ValueError("Do Not Exist This Strategy")

def edge2mat(link, num_node, matrix=None):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        if matrix is None:
            A[j, i] = 1
        else:
            A[j, i] = matrix[j, i]
    return A

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD
