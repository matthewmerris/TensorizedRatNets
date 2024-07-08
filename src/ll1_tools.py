import numpy as np
import matlab.engine
import matlab

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
        lo = i * L
        hi = lo + L
        factor_0[:, lo:hi] = np.asarray(U[i][0])
        factor_1[:, lo:hi] = np.asarray(U[i][1])
        factor_2[:, lo:hi] = np.asarray(U[i][2])

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
    L = U[0].shape[1] / num_terms

    U_mod = list()
    for i in range(num_terms):
        lo = i*L
        hi = lo + L
        tmp_0 = U[0][:, lo:hi]
        tmp_1 = U[1][:, lo:hi]
        tmp_2 = U[2][:, i]
        tmp_3 = np.zeros((L, L))
        for j in range(L):
            tmp_3[j, j] = 1

        tmp_list = [tmp_0, tmp_1, tmp_2, tmp_3]
        U_mod.append(tmp_list)

    return U_mod

def convert_ll1(U, convert_to):
    """Helper function to convert between numpy array and matlab.double
    __________
    U : list of lists of matlab.double arrays or numpy arrays (i.e. a list of arrays
        for each term of LL1 decomposition)
    type: string (matlab or numpy), designates conversion type

    Returns
    __________
    U_mod : list of numpy arrays (i.e. factor matrices) consolidating
            the column vectors for each term and their respective modes.
    """
    convert_to = type.lower()

    if convert_to == 'numpy':
        for term_list in U:
            for idx, factor in enumerate(term_list):
                temp = np.array(factor)
                term_list[idx] = temp

    elif convert_to == 'matlab':
        # convert from numpy to matlab.double
        for term_list in U:
            for idx, factor in enumerate(term_list):
                term_list[idx] = matlab.double(factor)
    else:
        print("Conversion type not supported, data returned as received")

    return U

