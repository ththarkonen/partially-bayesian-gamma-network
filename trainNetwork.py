import sys
import h5py
import numpy as np

from datetime import datetime
from scipy.io import savemat
from sklearn.model_selection import train_test_split

from gammanet import createArchitecture
from dataLoaders import loadData

import pandas as pd
import tensorflow as tf

tf.keras.backend.clear_session()
tf.compat.v1.enable_eager_execution()

fileName = sys.argv[1]
mode = sys.argv[2]
epochs = int( sys.argv[3] )
batch_size = int( sys.argv[4] )
validationSplit = float( sys.argv[5] )

timeStamp = datetime.now()
timeStamp = datetime.timestamp( timeStamp )
timeStamp = int( timeStamp )
timeStamp = str( timeStamp )

def scheduler( epoch, lr):
    if epoch < 10: return lr
    elif lr > 0.000001: return lr * tf.math.exp(-0.1)
    else: return 0.000001
        

callbackSchedule = tf.keras.callbacks.LearningRateScheduler(scheduler)
callbackStopping = tf.keras.callbacks.EarlyStopping( monitor = 'val_loss',
                                             patience = 50, 
                                             min_delta = 0, 
                                             mode = 'auto', 
                                             baseline = None, 
                                             restore_best_weights = True)

X, y = loadData( fileName, mode)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = validationSplit)

nInputs = X_train.shape[1]
nOutputs = y_train.shape[1]

model_ii = createArchitecture( nInputs, nOutputs)
model_ii.summary()

h = model_ii.fit( X_train, y_train, epochs = epochs,
                    verbose = 1,
                    validation_data = ( X_test, y_test),
                    batch_size = batch_size,
                    shuffle = True,
                    callbacks = [callbackSchedule, callbackStopping])

modelSavePath = './models/' + mode + '/gamma-specnet-model-' + timeStamp
historySavePath = './models/' + mode + '/gamma-specnet-history-' + timeStamp + ".json"

hist_df = pd.DataFrame( h.history ) 

# save to json:  
with open( historySavePath, mode = 'w') as f:
    hist_df.to_json( f )

file = h5py.File( modelSavePath, 'w')

weights = model_ii.get_weights()
nWeights = len( weights )

for ii in range( nWeights ):
   file.create_dataset( 'weight' + str(ii), data = weights[ii])
file.close()
