import numpy as np

def unpack_ll1(U):
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
        factor_0 = np.append(factor_0, np.asarray(U[i][0]))
        factor_1 = np.append(factor_1, np.asarray(U[i][1]))
        factor_2 = np.append(factor_2, np.asarray(U[i][2]))

    # construct the block structured core tensor
    core = np.zeros((L*num_terms, L*num_terms))
    for i in range(L*num_terms):
        core[i,i] = 1.0

    U_mod = [factor_0, factor_1, factor_2, core]

    return U_mod

