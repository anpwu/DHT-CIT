import os
import numpy as np
from .simulate_dag import simulate_SummaryDAG
from .simulate_dist import Dist

def generateData(exps = 10, graph_type = 'ER', d = 10, s0=20, s1=0, s2=0, n = 1000, gunc='sigmoid', func='sigmoid', noiseType = 'Gauss', norm=True, IVlayer=0, IVs=[]):
    
    DAGname='{}_{}_{}_{}_{}'.format(graph_type, d, s0, s1, s2)
    
    if IVlayer==12 or IVlayer==23:
        savePath = '/DAG_{}/{}({})_{}_{}_{}_{}/'.format(DAGname, noiseType, n, gunc, func, IVlayer, len(IVs))
    else:
        savePath = '/DAG_{}/{}({})_{}_{}/'.format(DAGname, noiseType, n, gunc, func)
    
    for exp in range(exps):
        expPath = f'./Data/{savePath}/{exp}/'
        os.makedirs(os.path.dirname(expPath), exist_ok=True)

        SummaryDAG, AutoREG_1st, AutoREG_2nd = simulate_SummaryDAG(d, s0, s1, s2, graph_type)

        np.save(expPath+"adjacency.npy", SummaryDAG)
        np.savetxt(expPath+'SummaryDAG.csv', SummaryDAG, delimiter=',', fmt='%.0f')
        np.savetxt(expPath+'AutoREG_1st.csv', AutoREG_1st, delimiter=',', fmt='%.0f')
        np.savetxt(expPath+'AutoREG_2nd.csv', AutoREG_2nd, delimiter=',', fmt='%.0f')
        np.savetxt(expPath+'IVs.csv', np.array(IVs), delimiter=',', fmt='%d')
        
        teacher = Dist(d, SummaryDAG, AutoREG_1st, AutoREG_2nd, noise_type = [noiseType]*12, gunc=gunc, func=func, norm=norm, IVlayer=IVlayer)
        Datas, Interventions = teacher.sample(expPath, n, IVlayer, IVs)

    return savePath
