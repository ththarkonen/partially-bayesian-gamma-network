import keras
from keras.layers import Activation, BatchNormalization, Conv1D, Dense, Dropout, Flatten
from keras.models import Sequential

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers

optimizer = keras.optimizers.Adam( learning_rate = 0.001 )

def createArchitecture( inputSize, outputSize):

    inputShape = ( inputSize, 1)

    negativeLogLikelihood = lambda y, rv_y: -rv_y.log_prob( y )
    gammaLayer = lambda input: tfd.Gamma( concentration = tf.nn.relu( input[..., 0:outputSize] + 1e-12 ),
                                          log_rate      = tf.nn.relu( input[..., outputSize:]) )

    outputLayer = tfp.layers.DistributionLambda( gammaLayer )

    model = Sequential()
    model.add( BatchNormalization(  axis = -1,
                                    momentum = 0.99,
                                    epsilon = 0.001,
                                    center = True,
                                    scale = True,
                                    beta_initializer = 'zeros',
                                    gamma_initializer = 'ones',
                                    moving_mean_initializer = 'zeros',
                                    moving_variance_initializer = 'ones',
                                    beta_regularizer = None,
                                    gamma_regularizer = None,
                                    beta_constraint = None,
                                    gamma_constraint = None,
                                    input_shape = inputShape
                                 ))

    model.add( Activation('relu'))

    model.add( tfp.layers.Convolution1DFlipout( 128, activation = 'relu', kernel_size = (32)) )
    model.add( Conv1D( 64, activation = 'relu', kernel_size = (16)) )
    model.add( Conv1D( 16, activation = 'relu', kernel_size = (8)) )
    model.add( Conv1D( 16, activation = 'relu', kernel_size = (8)) )
    model.add( Conv1D( 16, activation = 'relu', kernel_size = (8)) )

    model.add( Dense(  32, activation = 'relu') )
    model.add( Dense(  16, activation = 'relu') )
    model.add( Flatten() )
    model.add( Dropout( 0.25 ) )
    
    model.add( Dense( outputSize + outputSize, activation = 'relu' ) )
    model.add( outputLayer )

    model.compile( loss = negativeLogLikelihood, optimizer = optimizer)

    return model