from causallearn.utils.KCI.KCI import *
from .algorithm import DHTCIT

CInd = KCI_CInd()
UInd = KCI_UInd()

def KCI_test(X, Y, Z):
    try:
        X=X.numpy()
        Y=Y.numpy()
        Z=Z.numpy()
    except:
        pass

    if Z.shape[1]<1:
        return UInd.compute_pvalue(X,Y)[0]
    else:
        return CInd.compute_pvalue(X,Y,Z)[0]

def KI_test(X, Y, Z):
    try:
        X=X.numpy()
        Y=Y.numpy()
        Z=Z.numpy()
    except:
        pass
    
    return UInd.compute_pvalue(X,Y)[0]