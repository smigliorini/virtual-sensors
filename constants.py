filenamebase = 'Tutti_i_valori_CON_MISSING_VALUES_per_Sens0'
dir_modelli = 'modelli'
dir_esiti = 'esiti'
dir_dati = 'dati'
data_header_index = 'timestamp_normalizzato'
data_header = [
    'distanza_da_centralina_cm',
 #   'angolo_gradi_verso_CN0',
    'anno_acquisizione',
    'mese_acquisizione',
    'settimana_acquisizione_su_anno',
    'giorno_acquisizione_su_anno',
    'giorno_acquisizione_su_mese',
    'giorno_acquisizione_su_settimana',
    'ora_acquisizione',
    'timestamp_normalizzato',
    'evja_temp',
    'Barometer_HPa',
    'Temp__C',
    'HighTemp__C',
    'LowTemp__C',
    'Hum__',
    'DewPoint__C',
    'WetBulb__C',
    'WindSpeed_Km_h',
    'WindDirection_',
    'WindRun_Km',
    'HighWindSpeed_Km_h',
    'HighWindDirection_',
    'WindChill__C',
    'HeatIndex__C',
    'THWIndex__C',
    'THSWIndex__C',
    'Rain_Mm',
    'RainRate_Mm_h',
    'SolarRad_W_m_2',
    'SolarEnergy_Ly',
    'HighSolarRad_W_m_2',
    'ET_Mm',
    'UVIndex_',
    'UVDose_MEDs_',
    'HighUVIndex_',
    'HeatingDegreeDays',
    'CoolingDegreeDays',
    'Humidity__RH_',
    'Solar_klux_',
    'target'
    ]
###########################################################################
DROPOUT_LABEL ='dropout'
NUNITS_LABEL = 'nunits'
NUMFEATURES_LABEL = 'numfeatures'
EPOCHS_LABEL = 'epochs'
SENSOR_LABEL = 'sensor'
NUMLAYERS_LABEL = 'num_layers'
XTRAIN_LABEL = 'x_train'
YTRAIN_LABEL = 'y_train'
XTEST_LABEL = 'x_test'
YTEST_LABEL = 'y_test'
DROPOUTS_VALUES_LABEL = 'dropout_values'
NUNITS_VALUES_LABEL = 'nunits_values'
EPOCHS_VALUES_LABEL = 'epochs_values'
MAXNUMLAYERS_LABEL = 'max_num_layers'
BATCH_SIZE_LABEL = 'batch_size'
VALIDATION_SPLIT_LABEL = 'validation_split'
MAX_TIME_LAG_LABEL = 'max_time_lag'
TIME_LAG_LABEL = 'time_lag'
PERFORMANCES_LABEL = 'performances'
THIS_MODEL_LABEL = 'thismodel'
BEST_ONE_LABEL = 'bestOne'
EVALUATION_METRIC_LABEL = 'evalMetric'
LSTM_MODEL_LABEL = 'LSTM_model'
UNDERSCORE ='_'
CSV_EXTENSION = '.csv'
#
CHOSEN_LOSS = 'mean_absolute_error'
MAPE_LABEL = 'mape'
MAE_LABEL = 'mae'
MSE_LABEL = 'mse'
R2_LABEL = 'r2_score'
#
LOG_SHAPE_OF_MSG = 'Shape of '
LOG_DATA_WORD_MSG = 'data '
LOG_TRAIN_WORD_MSG = 'train '
LOG_TEST_WORD_MSG = 'test '
LOG_RESULT_WORD_MSG = 'result '
LOG_VALUES_WORD_MSG = 'values '
LOG_BEST_ONE_MSG = 'il migliore ora risulta quello con: '
