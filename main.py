from src import DHTCIT, KCI_test, KI_test
from generator import generateData

if __name__ == "__main__":
    dataPath = generateData()
    method = DHTCIT(1)
    method.run(CInd=KCI_test, dataPath=dataPath, Pind=1, Mind=4, Ind=KI_test)
