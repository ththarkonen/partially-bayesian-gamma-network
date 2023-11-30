import h5py
import numpy as np

def loadData( fileName, mode, maximumDataSets = 1000000):

    modes = {"raman", "cars"}

    if mode not in modes:
        raise ValueError('Mode must be either "raman" or "cars". Current: %s.', mode)

    data = h5py.File( fileName, 'r')

    measurementData = data.get( mode + "Data" )
    measurementData = np.array( measurementData )

    imChi3Data = data.get('beta')
    imChi3Data = np.array( imChi3Data )

    nPoints = measurementData.shape[0]
    nDataSets = measurementData.shape[1]

    if( nDataSets > maximumDataSets ):
        nDataSets = maximumDataSets

    xDims = ( nDataSets, nPoints, 1)
    yDims = ( nDataSets, nPoints)

    X = np.empty( xDims )
    y = np.empty( yDims )

    for ii in range( nDataSets ):

        X[ ii, :, 0] = measurementData[ :, ii]
        y[ ii, :] = imChi3Data[ :, ii]

    return X, y

def loadInputs( fileName, mode, maximumDataSets = 1000000):

    modes = {"raman", "cars"}

    if mode not in modes:
        raise ValueError('Mode must be either "raman" or "cars". Current: %s.', mode)

    data = h5py.File( fileName, 'r')

    measurementData = data.get( mode + "Data" )
    measurementData = np.array( measurementData )

    nPoints = measurementData.shape[0]
    nDataSets = measurementData.shape[1]

    if( nDataSets > maximumDataSets ):
        nDataSets = maximumDataSets

    xDims = ( nDataSets, nPoints, 1)

    X = np.empty( xDims )

    for ii in range( nDataSets ):
        X[ ii, :, 0] = measurementData[ :, ii]

    return X

def loadDataField( fileName, field, maximumDataSets = 1000000):

    data = h5py.File( fileName, 'r')

    fieldData = data.get( field )
    fieldData = np.array( fieldData )

    nOutputPoints = fieldData.shape[0]
    nDataSets = fieldData.shape[1]

    if( nDataSets > maximumDataSets ):
        nDataSets = maximumDataSets

    yDims = ( nDataSets, nOutputPoints)
    y = np.empty( yDims )

    for ii in range( nDataSets ):
            y[ ii, :] = fieldData[ :, ii]
    
    return y
