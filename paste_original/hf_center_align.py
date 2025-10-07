import numpy as np
import ot
from anndata import AnnData
from typing import Optional, Tuple
from sklearn.decomposition import NMF

##################################
# Utility
##################################
def intersect(a, b):
    # Return sorted intersection of two index arrays
    return a.intersection(b).sort_values()

def to_dense_array(X):
    # If X is a sparse array, convert to dense, else pass through
    if hasattr(X, "toarray"):
        return X.toarray()
    return X

##################################
# center_ot_two_slices
##################################
def center_ot_two_slices(
    W, H, slice_B, center_coords, common_genes, alpha,
    backend, use_gpu, dissimilarity='kl', norm=False,
    G_init=None, distribution_B=None, verbose=False
):
    """
    Single-slice version of the OT step.
    We have:
        - W, H: current factorization of the center (A)
        - slice_B: the other slice
        - center_coords: spatial coords of center (A)
        - alpha: balance between expression vs. spatial cost
        - G_init: optional initial guess for the OT plan
        - distribution_B: optional distribution over B's spots
    Returns:
        G (np.ndarray): the OT plan from the center to slice_B
        r_cost (float): the resulting cost
    """

    # 1) Get B's coords and expression
    coords_B = slice_B.obsm['spatial']
    expr_B   = to_dense_array(slice_B.X)
    # 2) Construct expression of the center from current W,H
    center_expr = np.dot(W, H)

    # 3) Construct cost matrix
    #    (e.g. alpha * spatial_dist + (1-alpha)*expression_dist)
    #    Here, you'd compute a distance from each spot in A to each spot in B
    #    in terms of (x,y) plus gene expression difference.
    #    We'll placeholder it with zeros for the example:
    num_spots = center_coords.shape[0]
    cost_matrix = np.zeros((num_spots, num_spots))  # same # of spots

    # 4) Choose row and col marginals (if you’re using uniform or distributions)
    row_dist = np.ones(num_spots) / num_spots  # center slice distribution
    if distribution_B is None:
        col_dist = np.ones(num_spots) / num_spots
    else:
        col_dist = distribution_B

    # 5) Solve OT. For example, if cost_matrix is small, you can do
    #    G = ot.emd(row_dist, col_dist, cost_matrix)
    G = ot.emd(row_dist, col_dist, cost_matrix)

    # 6) Compute total cost = sum(C * G)
    #    but we only need it if we want to track objective
    total_cost = np.sum(cost_matrix * G)

    return G, total_cost

##################################
# center_NMF_two_slices
##################################
def center_NMF_two_slices(
    W, H, slice_B, G, lmbda,
    n_components, random_seed,
    dissimilarity='kl', verbose=False
):
    """
    Single-slice version of the NMF step.
    We incorporate slice_B's expression mapped
    onto the center via G, weighting by lmbda (just a float now).
    """
    expr_B = to_dense_array(slice_B.X)
    num_spots = W.shape[0]  # same # of spots as center

    # Weighted sum of mapped expression
    # If we want to mimic the original code’s approach, we do:
    # combined_expr = num_spots * [lmbda * (G @ expr_B)]
    # but for just 1 slice, it might be:
    combined_expr = num_spots * lmbda * (G @ expr_B)

    # Fit an NMF
    if dissimilarity.lower() in ['euclidean', 'euc']:
        model = NMF(n_components=n_components, init='random',
                    random_state=random_seed, verbose=verbose)
    else:
        model = NMF(n_components=n_components, solver='mu',
                    beta_loss='kullback-leibler', init='random',
                    random_state=random_seed, verbose=verbose)

    W_new = model.fit_transform(combined_expr)
    H_new = model.components_

    return W_new, H_new

##################################
# main function
##################################
def center_align_two_slices(
    A: AnnData,
    B: AnnData,
    alpha: float = 0.1,
    n_components: int = 15,
    threshold: float = 0.001,
    max_iter: int = 10,
    dissimilarity: str = 'kl',
    norm: bool = False,
    random_seed: Optional[int] = None,
    G_init: Optional[np.ndarray] = None,
    distribution_B=None,
    backend=ot.backend.NumpyBackend(),
    use_gpu: bool = False,
    verbose: bool = False,
    gpu_verbose: bool = True
) -> Tuple[AnnData, np.ndarray]:
    """
    Center alignment for exactly 2 slices (A, B)
    with the same number of spots.
    We treat A as the center.

    Args:
        A (AnnData): reference (center) slice
        B (AnnData): other slice
        alpha (float): weight for spatial vs. expression cost in OT
        n_components (int): NMF components
        threshold (float): convergence threshold
        max_iter (int): max iterations
        dissimilarity (str): 'kl' or 'euclidean'
        norm (bool): spatial distance normalization
        random_seed (int): for reproducible NMF
        G_init (np.ndarray): optional initial OT plan
        distribution_B (np.ndarray): optional distribution over B's spots
        backend (ot.backend): OT backend
        use_gpu (bool): whether to attempt GPU (Torch)
        verbose (bool): extra logging
        gpu_verbose (bool): whether to announce GPU usage

    Returns:
        center_slice (AnnData):
            A copy of A with updated expression in .X = W*H
            plus fields in .uns
        G (np.ndarray):
            final OT plan from A’s spots to B’s spots
    """

    # 1) Possibly handle GPU
    if use_gpu:
        try:
            import torch
        except ImportError:
            print("GPU mode requires PyTorch, but it's not installed.")
        if isinstance(backend, ot.backend.TorchBackend):
            if torch.cuda.is_available():
                if gpu_verbose:
                    print("gpu is available, using gpu.")
            else:
                if gpu_verbose:
                    print("gpu is not available, fallback to cpu.")
                use_gpu = False
        else:
            print("We only have GPU support with TorchBackend, revert to CPU.")
            use_gpu = False
    else:
        if gpu_verbose:
            print("Using CPU. Set use_gpu=True if a GPU backend is available.")

    # 2) Intersect genes
    common_genes = intersect(A.var.index, B.var.index)
    A = A[:, common_genes]
    B = B[:, common_genes]
    print(f"Common genes: {len(common_genes)}")

    # 3) Initialize W,H from A's expression
    expr_A = to_dense_array(A.X)
    if dissimilarity.lower() in ['euclidean', 'euc']:
        model = NMF(n_components=n_components, init='random',
                    random_state=random_seed, verbose=verbose)
    else:
        model = NMF(n_components=n_components, solver='mu',
                    beta_loss='kullback-leibler',
                    init='random', random_state=random_seed,
                    verbose=verbose)

    if G_init is None:
        # Just run NMF on A
        W = model.fit_transform(expr_A)
    else:
        # Weighted sum approach, but with only 1 slice we might do:
        # combined_expr = A.shape[0]*( G_init @ B.X ) # if we want to mimic original
        combined_expr = A.shape[0] * (G_init @ to_dense_array(B.X))
        W = model.fit_transform(combined_expr)

    H = model.components_

    # 4) Prepare iteration
    center_coords = A.obsm['spatial']  # same number of spots as B
    iteration_count = 0
    R = 0.0
    R_diff = float('inf')
    G = G_init if G_init is not None else np.zeros((A.shape[0], B.shape[0]))

    # If we want a slice weight (lmbda), assume 1 for B since there's only one slice
    lmbda_B = 1.0

    # 5) Iteration
    while (R_diff > threshold) and (iteration_count < max_iter):
        print(f"Iteration {iteration_count}")

        # 5a. center_ot step
        G, cost_val = center_ot_two_slices(
            W, H, B, center_coords, common_genes, alpha,
            backend, use_gpu, dissimilarity=dissimilarity, norm=norm,
            G_init=G, distribution_B=distribution_B, verbose=verbose
        )

        # 5b. center_NMF step
        W_new, H_new = center_NMF_two_slices(
            W, H, B, G, lmbda_B,
            n_components, random_seed,
            dissimilarity=dissimilarity, verbose=verbose
        )

        # track objective (just cost_val here, or could do something else)
        R_new = cost_val  # with multiple slices, you'd do np.dot(r_costs, lmbda)
        R_diff = abs(R_new - R)
        print(f"Objective: {R_new}")
        print(f"Diff: {R_diff}\n")

        W, H = W_new, H_new
        R = R_new
        iteration_count += 1

    # 6) Build final center slice
    center_slice = A.copy()  # same # of spots, same coords
    final_expr = np.dot(W, H)
    center_slice.X = final_expr
    center_slice.uns['paste_W'] = W
    center_slice.uns['paste_H'] = H
    center_slice.uns['obj'] = R
    # If you want "full_rank" like your original, with only B:
    center_slice.uns['full_rank'] = center_slice.shape[0] * (G @ to_dense_array(B.X))

    return center_slice, G
