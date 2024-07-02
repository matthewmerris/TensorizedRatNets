import numpy as np
import matlab.engine
import matlab

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
                temp = np.array(factor.tolist())
                term_list[idx] = temp.reshape(factor.size).transpose()

    elif convert_to == 'matlab':
        # convert from numpy to matlab.double
        for term_list in U:
            for idx, factor in enumerate(term_list):
                term_list[idx] = matlab.double(factor.tolist())
    else:
        print("Conversion type not supported, data returned as received")

    return U
