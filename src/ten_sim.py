import numpy as np
import tensorly as tn
from scipy.linalg import subspace_angles, pinv

def mode_n_matricization(tensor,n):
    """
    Unfolds a 3rd-order tensor T of shape (I, J, K) along its first two modes.
    
    Forms matrix columns as column-wise vectorization of n-th mode frontal slices
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
    unfolded_k = []
    for ndx in range(tensor.shape[n]):
        if n == 0:
            vect_mat = tn.reshape(tensor[ndx,:,:], -1, 'F')
        elif n == 1:
            vect_mat = tn.reshape(tensor[:,ndx,:], -1, 'F')
        elif n == 2:
            vect_mat = tn.reshape(tensor[:,:,ndx], -1, 'F')
        unfolded_k.append(vect_mat)
    unfolded_k = np.column_stack(unfolded_k)
    return tn.tensor(unfolded_k)
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
    T1_unfolded = mode_n_matricization(tensor1, 2)
    T2_unfolded = mode_n_matricization(tensor2, 2)
    
    # Step 2: Extract orthonormal bases U and V for the top N singular vectors
    # U, V have dimensions (I*J, N)
    U, _, _ = tn.truncated_svd(T1_unfolded, n_eigenvecs=n_components)
    V, _, _ = tn.truncated_svd(T2_unfolded, n_eigenvecs=n_components)
    
    # Step 3: Compute N principal angles between span{U} and span{V}
    angles = subspace_angles(U, V)
    
    return angles

def compress_tensors(A, B, n_modes=None, tol=1e-10):
    """
    Simultaneous MLSVD compression to ensure mode-n unfolding of A
    has full column rank before solving for linear mapping.
    """
    if n_modes is None:
        n_modes = tn.ndim(A)

    compressed_A = tn.copy(A)
    compressed_B = tn.copy(B)

    for mode in range(n_modes):
        A_unfolded = mode_n_matricization(compressed_A, mode)
        _,S,Vh = tn.svd(A_unfolded, full_matrices=False)
        rank = int(np.sum(S > tol * S) if len(S) > 0 else 0)
        if rank == 0:
            continue

        U_n = Vh[:rank,:]

        # compress
        compressed_A = tn.tenalg.mode_dot(compressed_A, U_n.conj(), mode=mode)
        compressed_B = tn.tenalg.mode_dot(compressed_B, U_n.conj(), mode=mode)

    return compressed_A, compressed_B

def compute_linear_mapping(A, B, mode):
    """
    Solves B_n = A_n M_n for linear mapping M_n
    """
    A_n = mode_n_matricization(A,mode)
    B_n = mode_n_matricization(B,mode)
    
    # solve for M_n
    M_n = pinv(A_n) @ B_n

    return M_n

def analyze_linear_mapping(M_n, tol=1e-4):
    """
    Computes EVD on M_N to extract structural parameters:
        - R: number of terms (distinct eigenvalues)
        - scaling_factors: distinct eigenvalues (lambda_r)
        - multilinear_ranks: algebraic multiplicities in mode n (L_nr)
    """
    eigvals, eigvects = np.linalg.eig(M_n)
    distinct_lambdas = []
    multiplicities = []

    # Group eigenvalues according to tol
    for val in eigvals:
        matched = False
        for idx, d_val in enumerate(distinct_lambdas):
            if np.isclose(val, d_val, atol=tol, rtol=tol):
                multiplicities[idx] += 1
                matched = True
                break
            if not matched:
                distinct_lambdas.append(val)
                multiplicities.append(1)

    return {
        'R': len(distinct_lambdas),
        'scaling_factors': np.array(distinct_lambdas),
        'multilinear_ranks': np.array(multiplicities),
        'eigenvalues': eigvals,
        'eigenvectors': eigvects
    }
def tensor_similarity_score(A,B):
    """
    Computes a normalized similarity score between tensors A and B.
    Based on Domanov & De Lathauwer (2019)
    """
    N_modes = tn.ndim(A)
    A_compressed, B_compressed = compress_tensors(A,B)

    fit_scores = []
    mode_lambdas = []

    for mode in range(N_modes):
        # solve for linear mapping M_n
        A_n = mode_n_matricization(A_compressed, mode)
        B_n = mode_n_matricization(B_compressed, mode)

        M_n = pinv(A_n) @ B_n
        
        # evaluate residual error in n-th mode
        B_pred = A_n @ M_n
        norm_B = tn.norm(B_n)
        rel_err = tn.norm(B_n - B_pred) / (norm_B + 1e-12)
        fit_scores.append(max(0.0, 1.0 - float(rel_err)))

        # extract eigenvalues of M_n
        eigvals = np.linalg.eigvals(M_n)
        eigvals_sorted = eigvals[np.argsort(-np.abs(eigvals))]
        mode_lambdas.append(eigvals_sorted)

    avg_fit_score = float(np.mean(fit_scores))

    # measure cross-mode spectral consistency
    min_len = min(len(l) for l in mode_lambdas)
    if min_len > 0:
        truncated_lambdas = np.array([l[:min_len] for l in mode_lambdas])
        mean_spec = np.mean(truncated_lambdas, axis=0)
        std_spec = np.std(truncated_lambdas, axis=0)
        rel_disp = np.mean(std_spec / (np.abs(mean_spec) + 1e-12))
        spec_consistency = max(0.0, float(1 - rel_disp))
    else:
        spec_consistency = 0.0
        mean_spec = np.array([])

    # aggregate composite similarity score
    overall_sim = avg_fit_score * (0.7 + 0.3 * spec_consistency)

    return {
        'similarity_score': round(overall_sim, 6),
        'subspace_fit_score': round(avg_fit_score, 6),
        'spectral_consistency': round(spec_consistency, 6),
        'mode_residuals': [round(1.0 - f, 6) for f in fit_scores],
        'shared_rank_R': min_len,
        'estimated_scaling_factors': np.round(mean_spec, 4).tolist()
    }

