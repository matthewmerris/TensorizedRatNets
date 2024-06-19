import numpy as np
import math
import matlab.engine

def loewnerize_acts(inputs_raw, outputs_raw, targets, tensorlab_path):
    """Construct a loewner tensor out of layer activations
    Parameters
    __________
    inputs_raw : ndarray
    outputs_raw : ndarray

    Returns
    __________
    loewner_x : ndarray
    """

    # reshape inputs and outputs to matrices
    inputs = inputs_raw.reshape(-1, inputs_raw.shape[-1])
    outputs = outputs_raw.reshape(-1, outputs_raw.shape[-1])

    # instantiate empty containers for splitting activations according to target
    i_targets = list()
    o_targets = list()
    for i in range(0,10):
        i_targets.append(np.empty((0,inputs.shape[1])))
        o_targets.append(np.empty((0,outputs.shape[1])))

    # split activations according to targets
    for idx, x in enumerate(targets):
        o_targets[x] = np.vstack([o_targets[x], outputs[idx, :]])
        i_targets[x] = np.vstack([i_targets[x], inputs[idx, :]])

    # Loewnerize activations
    eng = matlab.engine.start_matlab()
    s = eng.genpath(tensorlab_path)
    eng.addpath(s, nargout=0)

    I = math.ceil(inputs.shape[1]/2)
    J = inputs.shape[1] - I

    loewner_tens = np.empty(10)
    # loewner_tens = list()
    for j in range(0,10):
        num_rows = o_targets[j].shape[0]
        loewner_tens[j] = np.empty((I, J, 0), dtype=Common_DataType)
        # loewner_tens.append(np.empty((I, J, 0)))
        for i in range(0,num_rows):
            loewner_tens[j] = np.dstack((loewner_tens[0], eng.loewnerize(o_targets[j][i, :], 'T', i_targets[j][i, :], nargout=1)))

    return loewner_tens




