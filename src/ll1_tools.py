import numpy as np
import matlab.engine
import matlab

def split_data(inputs_raw, outputs_raw, targets, tensorlab_path, num_sets):
    """Randomly split activations into a specified number of subsets and lownerize

    Parameters
    ___________

    Returns
    ___________
    lwn_tns :
    observations:
    """

    # reshape inputs and outputs to matrices
    inputs = inputs_raw.reshape(-1, inputs_raw.shape[-1])
    outputs = outputs_raw.reshape(-1, outputs_raw.shape[-1])

    num_obs = inputs.shape[0]
    num_set_rows = num_obs // num_sets
    # split into specified number of sets
    # @@ probably should do some error handling on here for num sets vs num observations
    observations = []
    obs_targets = []
    for i in range(num_sets):
        tmp_obs = outputs[(i*num_set_rows):(i*num_set_rows + num_set_rows)]
        observations.append(tmp_obs)
        tmp_trgts = targets[(i*num_set_rows):(i*num_set_rows + num_set_rows)]
        obs_targets.append(tmp_trgts)

    # Loewnerize activations
    eng = matlab.engine.start_matlab()
    s = eng.genpath(tensorlab_path)
    eng.addpath(s, nargout=0)

    lwn_tns = list()
    for idx, obs in enumerate(observations):
        




def pack_ll1(U):
    """Helper function to unpack Tensorlab's (matlab) LL1
    tensor (3 factor matrices and a core tensor)
    Parameters
    __________
    U : list of lists of matlab.double arrays (i.e. a list of arrays
        for each term of LL1 decomposition)

    Returns
    __________
    U_mod : list of numpy arrays (i.e. factor matrices) consolidating
            the column vectors for each term and their respective modes.
    """

    # gather approriate dimensional info
    num_terms = len(U)
    L = np.asarray(U[0][0]).shape[1]

    dim_mode0 = np.asarray(U[0][0]).shape[0]
    dim_mode1 = np.asarray(U[0][1]).shape[0]
    dim_mode2 = np.asarray(U[0][2]).shape[0]

    # make factor matrix containers
    factor_0 = np.empty((dim_mode0, L*num_terms))
    factor_1 = np.empty((dim_mode1, L*num_terms))
    factor_2 = np.empty((dim_mode2, num_terms))

    # cycle through each term and collect relavent columns
    for i in range(num_terms):
        if L > 1:
            lo = i * L
            hi = lo + L
            factor_0[:, lo:hi] = np.asarray(U[i][0])
            factor_1[:, lo:hi] = np.asarray(U[i][1])
            factor_2[:, lo:hi] = np.asarray(U[i][2])
        else:
            factor_0[:, i] = np.asarray(U[i][0])
            factor_1[:, i] = np.asarray(U[i][1])
            factor_2[:, i] = np.asarray(U[i][2])

    # construct the block structured core tensor
    core = np.zeros((L*num_terms, L*num_terms))
    for i in range(L*num_terms):
        core[i,i] = 1.0

    U_mod = [factor_0, factor_1, factor_2, core]

    return U_mod

def unpack_ll1(U):
    """ Helper function, unpack LL1 factor matrices into individual
    factors (ala tensorlab's LL1 format)
    Parameters
    __________
    U : list of factor matrices (numpy arrays) for LL1 decomposition

    Returns
    __________
    U_mod : list of numpy arrays (i.e. factor matrices) consolidating
            the column vectors for each term and their respective modes.
    """

    # gather dimensional info
    num_terms = U[2].shape[1]
    L = U[0].shape[1] // num_terms
    # print('Num terms:')
    # print(num_terms)
    # print('L: ')
    # print(L)

    U_mod = list()
    for i in range(num_terms):
        if L > 1:
            lo = i*L
            hi = lo + L
            tmp_0 = U[0][:, lo:hi]
            tmp_1 = U[1][:, lo:hi]
            tmp_2 = U[2][:, i]
            tmp_3 = np.zeros((L, L))
            for j in range(L):
                tmp_3[j, j] = 1
        else:
            tmp_0 = U[0][:, i]
            tmp_1 = U[1][:, i]
            tmp_2 = U[2][:, i]
            tmp_3 = np.ones(1)

        tmp_list = [tmp_0, tmp_1, tmp_2, tmp_3]
        # print('Unpacked into:')
        # for factor in tmp_list:
        #     print(factor)
        # print('Yay!')
        U_mod.append(tmp_list)

    return U_mod

def convert_ll1(U, convert_to):
    """Helper function to convert between numpy array and matlab.double
    Parameters
    __________
    U : list of lists of matlab.double arrays or numpy arrays (i.e. a list of arrays
        for each term of LL1 decomposition)
    type: string (matlab or numpy), designates conversion type

    Returns
    __________
    U_mod : list of numpy arrays (i.e. factor matrices) consolidating
            the column vectors for each term and their respective modes.
    """
    convert_to = convert_to.lower()

    if convert_to == 'numpy':
        for term_list in U:
            for idx, factor in enumerate(term_list):
                temp = np.array(factor)
                term_list[idx] = temp

    elif convert_to == 'matlab':
        # convert from numpy to matlab.double
        eng = matlab.engine.start_matlab()
        for term_list in U:
            for idx, factor in enumerate(term_list):
                term_list[idx] = matlab.double(factor)
        eng.quit()
    else:
        print("Conversion type not supported, data returned as received")

    return U

def recover_sources(X,M):
    """Reconver source matrix in BSS problem (X = MS)
    Parameters
    __________
    X : numpy array (matrix), observed data
    M : numpy array (matrix), mixing matrix recovered from (L,L,1) decomp
    """
    S = np.matmul(np.linalg.pinv(M), X)
    return S