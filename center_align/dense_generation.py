import torch
import numpy as np
from sklearn.decomposition import PCA
from anndata import AnnData
from scipy.sparse import issparse
from scipy.spatial import cKDTree

def _to_dense(X):
    return X.toarray() if issparse(X) else np.asarray(X)

def _make_grid(coords, h):
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    n_cols = int(np.ceil((x_max - x_min) / h)) + 1
    n_rows = int(np.ceil((y_max - y_min) / h)) + 1
    gx = np.arange(x_min, x_min + n_cols * h, h)
    gy = np.arange(y_min, y_min + n_rows * h, h)
    return np.array([(x, y) for y in gy for x in gx])

@torch.no_grad()
def _sinkhorn_log_masked(a, b, C, mask, eps_schedule=(10., 5.), iters_per_eps=200, tol=1e-6, device="cuda"):
    """
    a: (n,) source marginal (sum=1)
    b: (m,) target marginal (sum=1)
    C: (n,m) cost tensor (float32), large where mask==False is ok (ignored via mask)
    mask: (n,m) bool tensor; False -> forbidden pairs
    eps_schedule: tuple/list high->low epsilon values (annealing)
    """
    n, m = C.shape
    log_a = torch.log(a)
    log_b = torch.log(b)

    # Initialize duals
    f = torch.zeros(n, device=device)
    g = torch.zeros(m, device=device)

    # Pre-mask: set forbidden entries to +inf in cost to keep them out of logsumexp
    C_masked = C.clone()
    C_masked[~mask] = torch.inf

    for eps in eps_schedule:
        inv_eps = 1.0 / max(eps, 1e-12)
        for _ in range(iters_per_eps):
            f_prev = f.clone()

            # Update f: ensure sum_j exp((f_i + g_j - C_ij)/eps) = a_i
            # logsumexp over masked columns
            # lse_i = logsumexp_j [ (g_j - C_ij)/eps ] on allowed j
            lse_i = torch.logsumexp((g[None, :] - C_masked) * inv_eps, dim=1)
            f = eps * (log_a - lse_i)

            # Update g similarly
            lse_j = torch.logsumexp((f[:, None] - C_masked) * inv_eps, dim=0)
            g = eps * (log_b - lse_j)

            if torch.max(torch.abs(f - f_prev)).item() < tol:
                break

    # Recover transport only on mask: T_ij = exp((f_i + g_j - C_ij)/eps)
    eps_final = eps_schedule[-1]
    inv_eps = 1.0 / max(eps_final, 1e-12)
    # For numerical safety, subtract a max before exp
    Z = (f[:, None] + g[None, :] - C_masked) * inv_eps
    Z[~mask] = -torch.inf
    T = torch.exp(Z)
    # normalize tiny drift
    T = T * (a.sum() / (T.sum() + 1e-12))
    return T, f, g

def spatial_regrid_fuse_optimized(
    adata,
    grid_size=2,
    fuse_radius=3.0,      # strongly recommended (e.g., 2*grid_size)
    k_mask=32,             # additionally keep up to K nearest grids per spot
    alpha=0.7,
    pca_dim=30,
    eps_schedule=(10.0, 5.0, 2.5),  # anneal to target epsilon
    iters_per_eps=200,
    tol=1e-6,
    use_cosine=True,
    device="cuda",
):
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    X = _to_dense(adata.X).astype(np.float32)
    n, g = X.shape

    # 1) grid + keep only near cells by fuse_radius
    grid_full = _make_grid(coords, grid_size)
    kd_spots = cKDTree(coords)
    if fuse_radius is not None:
        dmin, _ = kd_spots.query(grid_full, k=1)
        keep = dmin <= fuse_radius
        grid = grid_full[keep]
    else:
        grid = grid_full
    m = grid.shape[0]
    if m == 0:
        raise ValueError("No grid cells within fuse_radius.")

    # 2) Warm start mass with radius-limited KNN (CPU, cheap), then move to GPU
    kd_grid = cKDTree(grid)
    col_mass0 = np.zeros(m, dtype=np.float64)
    for i, spot in enumerate(coords):
        idxs = kd_grid.query_ball_point(spot, r=fuse_radius) if fuse_radius is not None else list(range(m))
        if len(idxs) == 0:
            continue
        sub = grid[idxs]
        d = np.linalg.norm(sub - spot[None, :], axis=1)
        order = np.argsort(d)[:min(k_mask, len(d))]
        sel = np.asarray(idxs, dtype=int)[order]
        w = 1.0 / (d[order] + 1e-6); w /= w.sum()
        col_mass0[sel] += w
    keep_mass = col_mass0 > 1e-12
    grid = grid[keep_mass]
    col_mass0 = col_mass0[keep_mass]
    m = grid.shape[0]
    if m == 0:
        raise ValueError("No grid cells received warm-start mass; increase fuse_radius or k_mask.")

    # 3) PCA for gene distance (CPU fit once)
    if pca_dim and 0 < pca_dim < min(n, g):
        pca = PCA(n_components=pca_dim, random_state=0)
        Xp = pca.fit_transform(X)
    else:
        pca = None
        Xp = X

    # 4) Build masked cost on GPU
    x_spot = torch.tensor(coords, device=device, dtype=torch.float32)            # (n,2)
    x_grid = torch.tensor(grid,   device=device, dtype=torch.float32)            # (m,2)

    # spatial cost (squared euclidean, auto-scale)
    Cs = torch.cdist(x_spot, x_grid, p=2.0)**2                                   # (n,m)
    s_med = torch.median(Cs[Cs>0]) if (Cs>0).any() else torch.tensor(1.0, device=device)
    Cs = Cs / torch.clamp(s_med, min=1e-6)

    # neighborhood mask: within radius and (optional) K nearest per spot
    if fuse_radius is not None:
        mask_radius = (torch.cdist(x_spot, x_grid, p=2.0) <= fuse_radius)
    else:
        mask_radius = torch.ones((n, m), dtype=torch.bool, device=device)
    if k_mask is not None and k_mask < m:
        # topk on -distance to get k nearest; ensure radius respected
        dists = torch.cdist(x_spot, x_grid, p=2.0)
        topk_idx = torch.topk(-dists, k=min(k_mask, m), dim=1).indices           # (n,k)
        mask_knn = torch.zeros_like(mask_radius)
        mask_knn.scatter_(1, topk_idx, True)
        mask = mask_radius & mask_knn
    else:
        mask = mask_radius

    # gene cost to grid needs grid expressions; we’ll start from spot PCA means of neighbors (quick init)
    Xp_spot = torch.tensor(Xp, device=device, dtype=torch.float32)               # (n,dg)
    # quick init: grid embedding = avg of nearby spots (masked)
    with torch.no_grad():
        # weights ~ exp(-spatial_dist / median_dist) within mask
        d = torch.cdist(x_grid, x_spot, p=2.0)                                   # (m,n)
        med = torch.median(d[d>0]) if (d>0).any() else torch.tensor(1.0, device=device)
        W = torch.exp(-d / torch.clamp(med, min=1e-6))
        # zero out spots not connected to grid by the transpose of mask
        W *= mask.T.float()
        denom = W.sum(dim=1, keepdim=True).clamp_min(1e-8)
        Xp_grid = (W @ Xp_spot) / denom                                          # (m,dg)

    # gene dissimilarity
    if use_cosine:
        Xn = Xp_spot / (Xp_spot.norm(dim=1, keepdim=True) + 1e-8)
        Yn = Xp_grid / (Xp_grid.norm(dim=1, keepdim=True) + 1e-8)
        Cg = 1.0 - (Xn @ Yn.T)                                                   # (n,m)
    else:
        # ||x-y||^2 = |x|^2 + |y|^2 - 2 x y^T
        xsq = (Xp_spot*Xp_spot).sum(dim=1, keepdim=True)
        ysq = (Xp_grid*Xp_grid).sum(dim=1, keepdim=True).T
        Cg = xsq + ysq - 2.0 * (Xp_spot @ Xp_grid.T)
    g_med = torch.median(Cg[Cg>0]) if (Cg>0).any() else torch.tensor(1.0, device=device)
    Cg = Cg / torch.clamp(g_med, min=1e-6)

    # combined cost
    C = alpha * Cs + (1.0 - alpha) * Cg
    # enforce mask
    C[~mask] = torch.inf

    # 5) Sinkhorn (log-domain, annealed)
    a = torch.full((n,), 1.0/n, device=device, dtype=torch.float32)
    b = torch.tensor(col_mass0 / col_mass0.sum(), device=device, dtype=torch.float32)

    T, f, g_dual = _sinkhorn_log_masked(a, b, C, mask, eps_schedule=eps_schedule,
                                        iters_per_eps=iters_per_eps, tol=tol, device=device)

    # 6) Barycentric fusion on GPU
    X_spot = torch.tensor(X, device=device, dtype=torch.float32)                 # (n,G)
    col_mass = T.sum(dim=0).clamp_min(1e-12)                                     # (m,)

    X_grid = (T.T @ X_spot) / col_mass[:, None]                                  # (m,G)


    # 7) Back to AnnData (CPU)
    new_adata = AnnData(X=X_grid.cpu().numpy(), var=adata.var.copy())
    new_adata.obsm["spatial"] = grid.astype(np.float32)
    new_adata.obs_names = [f"grid_{i}" for i in range(new_adata.n_obs)]
    new_adata.uns.update(dict(
        grid_size=grid_size,
        fuse_radius=fuse_radius,
        alpha=alpha,
        pca_dim=pca_dim,
        eps_schedule=list(eps_schedule),
        iters_per_eps=iters_per_eps
    ))
    # Optionally store marginals if needed
    new_adata.obsm["transport_col_mass"] = col_mass.cpu().numpy()
    return new_adata
