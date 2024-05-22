import random
import igraph as ig
import numpy as np

def dag2edges(A):
    edges = []
    for i in range(len(A)):
        for j in range(len(A[0])):
            if (A[i][j] == 1):
                edges.append((i,j))
    return edges


def edges2dag(n, edges):
    A = np.zeros((n,n))
    for edge in edges:
        A[edge[0]][edge[1]] = 1
    return A


def simulate_dag(d, s0, graph_type, triu=True):
    """Simulate random DAG with some expected number of edges.
    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        if triu:
            return np.triu(B_und, k=1)
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=False)
        B_und = _graph_to_adjmat(G)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
        
    if not triu:
        B = _random_permutation(B)
    assert ig.Graph.Adjacency(B.tolist()).is_dag()
    return B

def simulate_SummaryDAG(d, s0, s1, s2, graph_type, triu=True):
    SummaryDAG = simulate_dag(d, s0, graph_type)

    edges = dag2edges(SummaryDAG)
    random.shuffle(edges)

    AutoREG_1st = edges2dag(d, edges[:s1])
    AutoREG_2nd = edges2dag(d, edges[:s2])

    return SummaryDAG, AutoREG_1st, AutoREG_2nd

