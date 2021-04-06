'''
Custom TensorFlow Optimizer that implements ARC
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Optimizer

import hessian as H
import sub_problem as cubic

'''Stochastic Adaptive Regularization with Cubics'''
class SARC(Optimizer):
    '''
    Initialize the optimizer.
    '''
    def __init__(self, name='SARC_Optimizer', sigma=1, eta_1=0.1, eta_2=0.9,
                    gamma_1=2, gamma_2=2 , **kwargs):

        super().__init__(name, **kwargs)
        self._set_hyper('sigma', sigma)
        self._set_hyper('eta_1', eta_1)
        self._set_hyper('eta_2', eta_2)
        self._set_hyper('gamma_1', gamma_1)
        self._set_hyper('gamma_2', gamma_2)

    '''Get optimizer configuration details.'''
    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            'sigma': self._serialize_hyperparameter('sigma'),
            'eta_1': self._serialize_hyperparameter('eta_1'),
            'eta_2': self._serialize_hyperparameter('eta_2'),
            'gamma_1': self._serialize_hyperparameter('gamma_1'),
            'gamma_2': self._serialize_hyperparameter('gamma_2')
        }

    '''Not Implemented.'''
    def _create_slots(self, vars):
        pass

    '''Note Implemented'''
    def __prepare_local(self):
        pass

    '''Not Implemented.'''
    @tf.function
    def _resource_apply_sparse(self):
        pass

    '''
    Main functionality to update parameters of model.

    grad -> Model gradient
    hess -> Function to compute model hessian vector product
    vars -> Model parameters

    Thoughts: Do we need closure? We potentially need to compute a forward pass
    with potential update, but if we do this it seems like a waste
    '''
    @tf.function
    def _resource_apply_dense(self, grad, hess, vars):
        pass

        #Do arc stuff here with calls to sub_problem
