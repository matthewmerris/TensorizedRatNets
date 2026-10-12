import numpy as np
import tensorly as tl
from scipy.linalg import subspace_angles

def unfold_two_modes(tensor):
    """
    Unfolds a 3rd-order tensor T of shape (I, J, K) along its first two modes.
    
    In TensorLy, mode-2 unfolding produces a matrix of shape (K, I * J),
    where rows represent mode 2 (K) and columns represent the flattened 
    modes 0 and 1 (I, J). Transposing this yields shape (I * J, K).
    
    Parameters
    ----------
    tensor : ndarray or TensorLy tensor
        A third-order tensor of shape (I, J, K).
        
    Returns
    -------
    unfolded_matrix : ndarray
        The matricized tensor T_{IJ x K} of shape (I * J, K).
    """
    # Mode-2 unfolding yields shape (K, I * J)
    unfolded_k = tl.unfold(tensor, mode=2)
    
    # Transpose to obtain shape (I * J, K)
    return tl.transpose(unfolded_k)
# Set TensorLy backend (numpy)


def compute_two_mode_similarity(tensor1, tensor2, n_components):
    """
    Computes two-mode tensor similarity by comparing dominant subspaces 
    via principal angles without calculating tensor decompositions.
    
    Parameters
    ----------
    tensor1 : ndarray
        First tensor of shape (I, J, K).
    tensor2 : ndarray
        Second tensor of shape (I, J, K).
    n_components : int
        Number of dominant components N (e.g., rank R for CPD).
        
    Returns
    -------
    principal_angles : ndarray
        Array of N principal angles (in radians) in [0, pi/2].
    """
    # Step 1: Unfold both tensors along their first two modes -> (I*J, K)
    T1_unfolded = unfold_two_modes(tensor1)
    T2_unfolded = unfold_two_modes(tensor2)
    
    # Step 2: Extract orthonormal bases U and V for the top N singular vectors
    # U, V have dimensions (I*J, N)
    U, _, _ = tl.truncated_svd(T1_unfolded, n_eigenvecs=n_components)
    V, _, _ = tl.truncated_svd(T2_unfolded, n_eigenvecs=n_components)
    
    # Step 3: Compute N principal angles between span{U} and span{V}
    angles = subspace_angles(U, V)
    
    return angles


