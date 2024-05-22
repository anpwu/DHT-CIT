import uuid
import os
import numpy as np 
import pandas as pd
from cdt.utils.R import RPackages, launch_R_script
from cdt.metrics import retrieve_adjacency_matrix

def np_to_csv(array, save_path):
    """
    Convert np array to .csv
    array: numpy array
        the numpy array to convert to csv
    save_path: str
        where to temporarily save the csv
    Return the path to the csv file
    """
    id = str(uuid.uuid4())
    output = os.path.join(os.path.dirname(save_path), 'tmp_' + id + '.csv')

    df = pd.DataFrame(array)
    df.to_csv(output, header=False, index=False)

    return output

def cam_pruning(A, X, cutoff=0.001):
    save_path = "./Result/tmp/"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data_np = np.array(X.detach().cpu().numpy())
    data_csv_path = np_to_csv(data_np, save_path)
    dag_csv_path = np_to_csv(A, save_path)

    arguments = dict()
    arguments['{PATH_DATA}'] = data_csv_path
    arguments['{PATH_DAG}'] = dag_csv_path
    arguments['{PATH_RESULTS}'] = os.path.join(save_path, "results.csv")
    arguments['{ADJFULL_RESULTS}'] = os.path.join(save_path, "adjfull.csv")
    arguments['{CUTOFF}'] = str(cutoff)
    arguments['{VERBOSE}'] = "TRUE"

    def retrieve_result():
        A = pd.read_csv(arguments['{PATH_RESULTS}']).values
        os.remove(arguments['{PATH_RESULTS}'])
        os.remove(arguments['{PATH_DATA}'])
        os.remove(arguments['{PATH_DAG}'])
        return A

    dag = launch_R_script("utils/Rscript/cam_pruning.R", arguments, output_function=retrieve_result)
    
    return dag