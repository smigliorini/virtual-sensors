from modelEvaluation import ModelEvaluator
from hyperParameters import HyperParameters
from constants import *

SENSORE = 4
default_time_lag = 7
default_epochs = 350
nunits = 256
thisdropout = 0.2
numlayers = 2
batch_size = 1
validation_split = 0.1
# DOPPI PER VELOCITA'
#default_epochs = 50
#default_time_lag = 4
#batch_size = 128
#thisdropout = 0.2
thisdropout = 0.0
batch_size = 32

def defineHyperParams(datadescr,time_lag=-1,epochs=-1):
    hyperparameterValues = {
        BATCH_SIZE_LABEL: batch_size,
        VALIDATION_SPLIT_LABEL: validation_split,
        DROPOUT_LABEL: thisdropout,
        NUNITS_LABEL: nunits,
        TIME_LAG_LABEL: default_time_lag,
        EPOCHS_LABEL: default_epochs,
        NUMLAYERS_LABEL: numlayers,
        DATADESCR_LABEL:datadescr
    }
    if time_lag >0:
        hyperparameterValues[TIME_LAG_LABEL] = time_lag
    if epochs > 0:
        hyperparameterValues[EPOCHS_LABEL] = epochs

    result = HyperParameters(hyperparameterValues)
    return result

def execEval(datadescr='-',time_lag=-1,epochs=-1):
    hyperparameterValues = defineHyperParams(datadescr,time_lag=time_lag,epochs=-epochs)
    compare = ModelEvaluator(hyperparameterValues, SENSORE,data_headers.get(datadescr))
    sklearn_metrics_mape,predicted = compare.evaluate()
    print('sklearn.metrics.mape ', sklearn_metrics_mape) 
    return sklearn_metrics_mape

def myMain(time_lag=-1,epochs=-1):
    result = {}
    result['base'] = execEval(datadescr='base',time_lag=time_lag,epochs=-epochs)
    #
    # out['esteso'] = execEval(datadescr='esteso')
    #
    result['meno_tempo'] = execEval(datadescr='meno_tempo',time_lag=time_lag,epochs=-epochs)
    result['senza_distanza'] = execEval(datadescr='senza_distanza',time_lag=time_lag,epochs=-epochs)
    result['senza_evja'] = execEval(datadescr='senza_evja',time_lag=time_lag,epochs=-epochs)
    return result


if __name__ == '__main__':
    print('iniziato')
    out={}
    '''
    for epoch in range(35,55,5):
        for timestep in range(4,8,1):
            out = myMain(time_lag=timestep,epochs=epoch)
            print(str(out))    
    '''
    out = myMain(epochs = 400,time_lag = 7)
    print(str(out))
    print('finito')
