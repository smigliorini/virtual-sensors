import pandas as pd
import os
import numpy as np
import csv
from constants import *
from sklearn.metrics import r2_score, mean_squared_error

def my_rmse(target,predicted):
    mse = mean_squared_error(target,predicted)
    # Raise the mean squared error to the power of 0.5
    rmse = (mse) ** (1 / 2)
    return rmse

def writeToCsvFile(to_csv,outFile):
    keys = to_csv[0].keys()
    with open(outFile, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(to_csv)

if __name__ == '__main__':
    tipiSplit = ("Shuffle","FromTop")
    for tipoSplit in tipiSplit:
        inFilename =DIR_ESITI + os.path.sep + "mieidaticonpredizione_epochs_400_timesteps_5_"+tipoSplit+".csv"
        outFilename = DIR_ESITI + os.path.sep + "metriche_"+tipoSplit+".csv"

        dfDati = pd.read_csv(inFilename)
        result=[]
        for sensore in range(1, 8):
            nomeColTarget = 'target_' + str(sensore)
            nomeColYTEST = 'y_test_' + str(sensore)
            nomeColPredicted = 'predicted_' + str(sensore)

            ColTarget = dfDati[nomeColTarget]
            ColYTEST = dfDati[nomeColYTEST]
            ColPredicted = dfDati[nomeColPredicted]

            R2_RNN = r2_score(ColTarget,ColPredicted)
            RMSE_RNN = my_rmse(ColTarget,ColPredicted)

            fileNameDatiOrig = DIR_DATI + os.path.sep + "Dataset_original_sens_" + str(sensore) + ".csv"
            dfOrig = pd.read_csv(fileNameDatiOrig)
            dfOrigNoNan = dfOrig.replace([np.nan, -np.inf], 0)
            dfOrigPocheColonne = dfOrigNoNan.loc[:, ["target", "evja_temp"]]

            ColOrig = dfOrigPocheColonne["target"]
            ColEvja = dfOrigPocheColonne["evja_temp"]

            R2_EVJA = r2_score(ColOrig, ColEvja)
            RMSE_EVJA = my_rmse(ColOrig, ColEvja)
            rigaEsiti ={"Sensore":str(sensore),"R2_RNN":R2_RNN,"RMSE_RNN":RMSE_RNN,"R2_EVJA":R2_EVJA,"RMSE_EVJA":RMSE_EVJA}
            result.append(rigaEsiti)
        print(result)
        writeToCsvFile(result,outFilename)
    print('finito')