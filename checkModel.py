import os
import sys

import h5py
import numpy as np

from scipy.io import savemat
from dataLoaders import loadInputs
from gammanet import createArchitecture

n = int( sys.argv[1] )
filePath = sys.argv[2]
mode = sys.argv[3]

if mode == "cars":
    modelDir = "./models/cars/"
else:
    modelDir = "./models/raman/"

dirs = os.listdir( modelDir )
dirs = sorted( dirs )
nModels = len( dirs )

nInputs = 640
nOutputs = 640

modelPath = modelDir + dirs[nModels - 1]
print( modelPath )

model = createArchitecture( nInputs, nOutputs)

file = h5py.File( modelPath, 'r')
weights = []
nWeights = len( file.keys() )

for ii in range( nWeights ):
   weights.append( file['weight' + str(ii)][:] )

model.set_weights( weights )
model.summary()

X = loadInputs( filePath, mode)
inputDims = (1, nInputs, 1)

nDataSets = X.shape[0]
result = {}

nDataSets = min( n, nDataSets)

for ii in range(nDataSets):

    input = np.empty( inputDims )
    input[ 0, :, 0] = X[ ii, :, 0]

    yHat = model( input )

    median = yHat.quantile( 0.50 )
    lowerBound = yHat.quantile( 0.05 )
    upperBound = yHat.quantile( 0.95 )

    tempIndex = "spectrum_" + str(ii)

    result[ tempIndex ] = {}
    result[ tempIndex ]["input"] = input
    result[ tempIndex ]["median"] = median.numpy()
    result[ tempIndex ]["lowerBound"] = lowerBound.numpy()
    result[ tempIndex ]["upperBound"] = upperBound.numpy()

    savemat("./modelCheck.mat", result)