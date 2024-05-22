import numpy as np
import os
import torch

from .cit4dtg import independence_test, independence_test_V2, Topo_layers, Topo2DAG
from .cam_prune import cam_pruning
from .score import compute_top_order, full_DAG
from utils import backRE

def SCORE_CAM(X, eta_G=0.001, eta_H=0.001, cutoff=0.001, normalize_var=False, dispersion="var"):
    top_order = compute_top_order(X, eta_G, eta_H, normalize_var, dispersion)
    G_pred = full_DAG(top_order)

    return G_pred

def FullOne(C):
    n, d = C.shape
    G_pred = np.ones((d, d))
    np.fill_diagonal(G_pred, 0)
    return G_pred


class DHTCIT(object):
    def __init__(self, exps=10) -> None:
        self.config = {
                    'name': 'DHT-CIT',
                    'exps': exps, 
                    'alpha': 0.05,
                    'cutoff': 0.001, 
                    'eta_G': 0.001,
                    'eta_H': 0.001,
                    'cam_cutoff': 0.001, 
                    'detail': True,
                    'device': 'cpu',
                    'seed': 2024,
                    }

    def set_Configuration(self, config):
        self.config = config

    def run(self, CInd, dataPath, Pind=1, Mind=4, config=None, IVlayer=0, IVs=[], Ind=None, useScore=0):
        if config is None:
            config = self.config

        self.IVlayer = IVlayer
        self.IVs = IVs
        self.name = config['name']
        self.exps = config['exps']
        self.alpha = config['alpha']
        self.cutoff = config['cutoff']
        self.eta_G = config['eta_G']
        self.eta_H = config['eta_H']
        self.cam_cutoff = config['cam_cutoff']
        self.detail = config['detail']
        self.device = torch.device(config['device'])
        self.seed = config['seed']

        self.name = self.name + '_{}_{}_{}_{}{}'.format(useScore, IVlayer, len(IVs), Pind, Mind)

        if self.IVlayer == 12:
            Pind, Mind, Cind = 1, 2, -1
        elif self.IVlayer == 23:
            Pind, Mind, Cind = 2, 3, -1
        else:
            Cind = max(Pind, 1)

        savePath = './Result/{}/{}/'.format(dataPath, self.name)
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        for exp in range(self.exps):
            path = './Data/{}/{}/'.format(dataPath, exp)
            REpath = './Result/{}/{}/{}/'.format(dataPath, self.name, exp)
            logfile = REpath + 'log.txt'
            os.makedirs(os.path.dirname(REpath), exist_ok=True)

            tar_DAG = np.load(path+'adjacency.npy')
            X = torch.load(path+'X.pt')

            P = X[Pind]
            M = X[Mind]
            C = X[Cind]

            if useScore:
                print("With Score. ")
                SCORE_Ordering = SCORE_CAM(C, self.eta_G, self.eta_H, self.cam_cutoff)
            else:
                print("Without Score. ")
                SCORE_Ordering = FullOne(C)

            LayerOrd, DTG, alterDTG, pred_DAG = self.single(CInd, P, M, C, SCORE_Ordering, self.alpha, self.cutoff, self.detail, Ind=Ind)
            
            torch.save(P,     REpath+'P.pt')
            torch.save(M,     REpath+'M.pt')
            np.savetxt(REpath+"tar_DAG.csv", tar_DAG, fmt="%d", delimiter=',')
            np.savetxt(REpath+"pred_DAG.csv", pred_DAG, fmt="%d", delimiter=',')
            np.savetxt(REpath+"alterDTG.csv", alterDTG, fmt="%d", delimiter=',')

            prune_sum  = np.sum((pred_DAG - tar_DAG)==1)
            SHD, SID = backRE(tar_DAG, pred_DAG)
            print(f"{self.name} - SHD: {SHD}, SID: {SID}, Prune: {prune_sum}. ")


    def single(self, CInd, P, M, C, SCORE_Ordering, alpha=0.01, cutoff=0.001, detail=False, Ind=None):
        n, d = M.shape

        if self.IVlayer == 12 or self.IVlayer == 23:
            Matrix = independence_test_V2(SCORE_Ordering, CInd, Ind, self.IVs, P, M, alpha, detail)
        else:
            Matrix = independence_test(SCORE_Ordering, CInd, P, M, alpha, detail)
        layers = Topo_layers(Matrix, alpha) 
        LayerOrd = Topo2DAG(layers)

        DTG = (Matrix < alpha).astype(int)

        alterDTG = ((LayerOrd == 1) & (DTG == 1)).astype(int)

        pred_DAG = cam_pruning(alterDTG, C, cutoff)

        return LayerOrd, DTG, alterDTG, pred_DAG