import time
import torch
import numpy as np

def independence_test_V2(SCORE_Ordering, CInd, Ind, IVs, P, M, alpha = 0.01, detail = False):
    print('With Interventions. ')
    n, d = M.shape
    Matrix = np.zeros((d,d))
    start_time = time.time()
    for i in range(d):
        parents_of_i = []
        for p4i in np.nonzero(SCORE_Ordering[:,i])[0]:
            if p4i in IVs:
                pass
            else:
                parents_of_i.append(p4i)
        ConditionalSet = P[:,parents_of_i]
        # ConditionalSet = torch.cat([P[:,:i], P[:,(i+1):]], 1)

        for j in range(d):
            if SCORE_Ordering[i,j] == 0:
                Matrix[i,j] = 0.5
            else:
                if i in IVs:
                    p = Ind(P[:,i:i+1],M[:,j:j+1],ConditionalSet)
                else:
                    p = CInd(P[:,i:i+1],M[:,j:j+1],ConditionalSet)
                Matrix[i,j] = p
                oneRUN = time.time() - start_time
                if detail:
                    print("{} -> {} ({}): {}.".format(i, j, oneRUN, p<alpha))
                    
    return Matrix.round(4)

def independence_test(SCORE_Ordering, CInd, P, M, alpha = 0.01, detail = False):
    print('Without Interventions. ')
    n, d = M.shape
    Matrix = np.zeros((d,d))
    start_time = time.time()

    for i in range(d):
        parents_of_i = []
        for p4i in np.nonzero(SCORE_Ordering[:,i])[0]:
            parents_of_i.append(p4i)
        ConditionalSet = P[:,parents_of_i]
        # ConditionalSet = torch.cat([P[:,:i], P[:,(i+1):]], 1)

        for j in range(d):
            if SCORE_Ordering[i,j] == 0:
                Matrix[i,j] = 0.5
            else:
                p = CInd(P[:,i:i+1],M[:,j:j+1],ConditionalSet)
                Matrix[i,j] = p
                oneRUN = time.time() - start_time
                if detail:
                    print("{} -> {} ({}): {}.".format(i, j, oneRUN, p<alpha))
    
    return Matrix.round(4)

def Topo_layers(Top_KCI, alpha):
    '''Based on the p-value matrix and the threshold, generate Topo-Graph Order
    Args:
        Top_KCI: the p-value matrix
        alpha: If p<alpha, Node(i) -> Node(j)
    Returns:
        layers (list): Topo-Graph Layers Order
    '''
    current_res = list(range(len(Top_KCI)))
    np.fill_diagonal(Top_KCI, np.ones(len(Top_KCI)))

    def push_layer(Top_KCI, res_index, alpha):
        Top_KCI_new = Top_KCI[res_index,:][:,res_index]
        now_d = len(Top_KCI_new)
        now_index = np.argwhere(np.sum(Top_KCI_new>=alpha, 1)==now_d)[:,0]

        while len(now_index)==0:
            index_pair = np.where(Top_KCI_new<alpha)
            max_index = list(Top_KCI_new[index_pair]).index(max(Top_KCI_new[index_pair]))
            Top_KCI_new[index_pair[0][max_index],index_pair[1][max_index]] += alpha
            now_index = np.argwhere(np.sum(Top_KCI_new>=alpha, 1)==now_d)[:,0]

        push_index = np.array(res_index)[now_index]
        res_index = list(set(res_index)-set(push_index))
        return push_index, res_index

    layers = []
    while True:
        current_layer, current_res = push_layer(Top_KCI, current_res, alpha)
        layers.append(current_layer)
        print(current_layer)
        if len(current_res) == 0:
            break

    return layers

def Topo2DAG(top_layers):
    '''Topo-Graph Layers Order to Topo-Graph DAG
    Args:
        top_layers: Topo-Graph Layers
    Returns:
        A (np.ndarray): [d, d] Topo-Graph DAG
    '''
    L = len(top_layers)
    
    d = 0
    for i in range(L):
        d += len(top_layers[i])

    A = np.zeros((d,d))

    for i in range(L):
        for j in range(i+1, L):
            for item in top_layers[j]:
                A[item, list(top_layers[i])] = 1

    return A