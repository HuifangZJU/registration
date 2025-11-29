import numpy as np
import torch
from anndata import AnnData
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

# ---------- small helpers ----------
def _to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

def _make_grid(coords: np.ndarray, grid: float) -> np.ndarray:
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    n_cols = int(np.ceil((x_max - x_min) / grid)) + 1
    n_rows = int(np.ceil((y_max - y_min) / grid)) + 1
    gx = np.arange(x_min, x_min + n_cols * grid, grid)
    gy = np.arange(y_min, y_min + n_rows * grid, grid)
    return np.array([(x, y) for y in gy for x in gx], dtype=np.float32)

def _build_lattice_edges(grid_xy: np.ndarray, grid_size: float):
    x0, y0 = grid_xy.min(axis=0)
    gx = np.rint((grid_xy[:, 0] - x0) / grid_size).astype(np.int64)
    gy = np.rint((grid_xy[:, 1] - y0) / grid_size).astype(np.int64)
    key = (gx << 32) + gy
    idx_map = {int(k): i for i, k in enumerate(key)}
    offsets = [(1,0),(-1,0),(0,1),(0,-1)]
    ei, ej = [], []
    for i, (ix, iy) in enumerate(zip(gx, gy)):
        for dx, dy in offsets:
            j = idx_map.get(int(((ix+dx) << 32) + (iy+dy)), -1)
            if j >= 0 and j > i:
                ei.append(i); ej.append(j)
    if not ei:
        # fallback: connect to nearest neighbor so diffusion has support
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2).fit(grid_xy)
        nbrs = nn.kneighbors(return_distance=False)
        for i in range(grid_xy.shape[0]):
            j = int(nbrs[i, 1])
            if j > i: ei.append(i); ej.append(j)
    return np.asarray(ei, np.int64), np.asarray(ej, np.int64)

@torch.no_grad()
def _sinkhorn_log_masked(a, b, C, mask, eps_schedule=(12.,6.,3.), iters_per_eps=120, tol=1e-6):
    device = C.device
    n, m = C.shape
    C = C.clone(); C[~mask] = torch.inf
    f = torch.zeros(n, device=device); g = torch.zeros(m, device=device)
    log_a = torch.log(a); log_b = torch.log(b)
    for eps in eps_schedule:
        inv = 1.0/eps
        for _ in range(iters_per_eps):
            f_prev = f
            f = eps * (log_a - torch.logsumexp((g[None,:]-C)*inv, dim=1))
            g = eps * (log_b - torch.logsumexp((f[:,None]-C)*inv, dim=0))
            if torch.max(torch.abs(f-f_prev)).item() < tol: break
    Z = (f[:,None] + g[None,:] - C) / eps_schedule[-1]
    Z[~mask] = -torch.inf
    T = torch.exp(Z)
    T = T * (a.sum() / T.sum())
    return T, f, g

# ---------- adaptive SVT (no explicit rank) ----------
def _tau_for_nuclear_fraction(s: torch.Tensor, keep_frac: float, iters: int = 40) -> float:
    """Find tau so sum(max(s - tau, 0))/sum(s) ~= keep_frac."""
    keep_frac = float(np.clip(keep_frac, 0.0, 1.0))
    if keep_frac >= 0.9999: return 0.0
    if keep_frac <= 1e-6:   return float(s.max().item())  # kill almost all
    lo, hi = 0.0, float(s.max().item())
    total = s.sum().item()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        kept = torch.clamp(s - mid, min=0).sum().item()
        if kept / total >= keep_frac:
            lo = mid
        else:
            hi = mid
    return hi

def _svt_adaptive(X: torch.Tensor, keep_nuc_frac: float = 0.9, knee: bool = False):
    """
    Adaptive soft-threshold on singular values.
    If knee=True, detect elbow on log(s) and set tau between s_k and s_{k+1}.
    Else choose tau to keep a fraction of nuclear norm (keep_nuc_frac).
    Returns shrunk X and (A, Vt) where X ~= A @ Vt for downstream latent diffusion.
    """
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)  # S sorted desc
    if knee:
        s = S
        log_s = torch.log(s + 1e-12)
        d = log_s[:-1] - log_s[1:]
        k = int(torch.argmax(d).item())  # elbow index
        tau = float(s[k+1].item()) if k + 1 < s.numel() else 0.0
    else:
        tau = _tau_for_nuclear_fraction(S, keep_nuc_frac)
    S_shr = torch.clamp(S - tau, min=0.0)
    q = int((S_shr > 0).sum().item())
    if q == 0:
        return torch.zeros_like(X), torch.zeros((X.shape[0],1), device=X.device), torch.zeros((1,X.shape[1]), device=X.device)
    Uq = U[:, :q]; Sq = S_shr[:q]; Vtq = Vh[:q, :]
    A = Uq * Sq  # (m,q)
    X_shr = A @ Vtq
    return X_shr, A, Vtq

def _latent_anisotropic_diffusion(A, ei, ej, iters=10, tau=0.15, kappa=1.0):
    """Edge-preserving diffusion on latent factors over lattice edges."""
    device = A.device
    ei_t = torch.tensor(ei, device=device, dtype=torch.long)
    ej_t = torch.tensor(ej, device=device, dtype=torch.long)
    for _ in range(iters):
        diff = A[ej_t] - A[ei_t]                    # (E, q)
        w = torch.exp(-(diff.pow(2).sum(dim=1) / (kappa**2 + 1e-12))).unsqueeze(1)
        flux = w * diff
        dA = torch.zeros_like(A)
        dA.index_add_(0, ei_t, flux)
        dA.index_add_(0, ej_t, -flux)
        A = A + tau * dA
    return A

# ---------- main: adaptive low-rank, no KNN ----------
def spatial_regrid_fuse_gpu_robust(
    adata: AnnData,
    grid_size: float = 50.0,
    fuse_radius: float = None,
    k_mask: int = 64,
    alpha: float = 0.5,
    pca_dim: int = 64,
    outer_loops: int = 2,
    # adaptive rank controls (pick ONE behavior):
    keep_nuc_frac: float = 0.90,   # keep 90% of nuclear norm (soft shrink)
    knee_mode: bool = False,       # if True, ignore keep_nuc_frac and use elbow
    # latent diffusion (edge-preserving)
    aniso_iters: int = 10,
    aniso_tau: float = 0.15,
    aniso_kappa: float = 1.0,
    # OT params
    eps_schedule=(12., 6., 3.),
    iters_per_eps: int = 120,
    device: str = "cuda",
    use_raw: bool = False
) -> AnnData:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # --- inputs
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    X_spot_np = _to_dense(adata.raw.X if (use_raw and adata.raw is not None) else adata.X).astype(np.float32)
    var_ref = (adata.raw.var if (use_raw and adata.raw is not None) else adata.var).copy()

    n, G = X_spot_np.shape
    x_spot = torch.tensor(coords, device=device)
    X_spot = torch.tensor(X_spot_np, device=device)

    # --- grid + radius pruning
    grid_full = _make_grid(coords, grid_size)
    if fuse_radius is not None:
        dmin, _ = cKDTree(coords).query(grid_full, k=1)
        keep = dmin <= fuse_radius
        grid = grid_full[keep]
    else:
        grid = grid_full
    if grid.shape[0] == 0:
        raise ValueError("No grid cells within fuse_radius.")
    x_grid = torch.tensor(grid, device=device)

    # --- gene embedding for cost: lib-norm -> log1p -> PCA -> L2
    Xn = X_spot_np / np.clip(X_spot_np.sum(axis=1, keepdims=True), 1.0, None) * 1e4
    Xn = np.log1p(Xn)
    pca = PCA(n_components=min(pca_dim, min(n, Xn.shape[1])), random_state=0)
    Xp_spot_np = pca.fit_transform(Xn).astype(np.float32)
    Xp_spot_np /= (np.linalg.norm(Xp_spot_np, axis=1, keepdims=True) + 1e-8)
    Xp_spot = torch.tensor(Xp_spot_np, device=device)

    # --- spatial cost + mask
    d = torch.cdist(x_spot, x_grid, p=2.0)
    Cs = d**2
    s_med = torch.median(Cs[Cs > 0]) if (Cs > 0).any() else torch.tensor(1.0, device=device)
    Cs = Cs / s_med.clamp(min=1e-6)
    mask_r = (d <= fuse_radius) if fuse_radius is not None else torch.ones_like(Cs, dtype=torch.bool)
    if k_mask is not None and k_mask < x_grid.shape[0]:
        topk_idx = torch.topk(-d, k=min(k_mask, x_grid.shape[0]), dim=1).indices
        mask_k = torch.zeros_like(mask_r); mask_k.scatter_(1, topk_idx, True)
        allowed = mask_r & mask_k
    else:
        allowed = mask_r
    row_ok = allowed.any(dim=1)
    if (~row_ok).any():
        nearest = torch.argmin(d, dim=1)
        allowed[torch.arange(n, device=device), nearest] = True
    col_ok = allowed.any(dim=0)
    if (~col_ok).any():
        x_grid = x_grid[col_ok]; allowed = allowed[:, col_ok]; Cs = Cs[:, col_ok]; d = d[:, col_ok]
        grid = grid[col_ok.cpu().numpy()]
    m = x_grid.shape[0]

    # --- warm-start target marginal b (CPU)
    kd = cKDTree(grid)
    b0 = np.zeros(m, dtype=np.float64)
    for i, s in enumerate(coords):
        idxs = kd.query_ball_point(s, r=fuse_radius) if fuse_radius is not None else list(range(m))
        if len(idxs) == 0:
            j = int(np.argmin(np.linalg.norm(grid - s[None,:], axis=1))); idxs = [j]
        sub = grid[idxs]; dst = np.linalg.norm(sub - s[None,:], axis=1)
        sel = np.argsort(dst)[:min(k_mask, len(dst))]
        sel_idx = np.asarray(idxs, int)[sel]
        w = 1.0 / (dst[sel] + 1e-6); w /= w.sum(); b0[sel_idx] += w
    keepm = b0 > 1e-12
    if not keepm.all():
        grid = grid[keepm]; x_grid = x_grid[keepm]
        allowed = allowed[:, keepm]; Cs = Cs[:, keepm]; d = d[:, keepm]
        b0 = b0[keepm]; m = x_grid.shape[0]
    a = torch.full((n,), 1.0/n, device=device)
    b = torch.tensor(b0 / b0.sum(), device=device, dtype=torch.float32)

    # --- init grid embedding from nearby spot embeddings
    with torch.no_grad():
        dgs = torch.cdist(x_grid, x_spot, p=2.0)
        med = torch.median(dgs[dgs > 0]) if (dgs > 0).any() else torch.tensor(1.0, device=device)
        W = torch.exp(-dgs / med.clamp(min=1e-6)) * allowed.T.float()
        denom = W.sum(dim=1, keepdim=True).clamp(min=1e-8)
        Xp_grid = (W @ Xp_spot) / denom
        Xp_grid = Xp_grid / (Xp_grid.norm(dim=1, keepdim=True) + 1e-8)

    # lattice edges for latent diffusion
    ei_np, ej_np = _build_lattice_edges(grid, grid_size)

    # ---------- outer loops ----------
    for _ in range(outer_loops):
        # gene cost (cosine) + combine
        Cg = 1.0 - (Xp_spot @ Xp_grid.T)
        g_med = torch.median(Cg[Cg > 0]) if (Cg > 0).any() else torch.tensor(1.0, device=device)
        Cg = Cg / g_med.clamp(min=1e-6)
        C = alpha * Cs + (1.0 - alpha) * Cg
        C[~allowed] = torch.inf

        # Sinkhorn
        T, _, _ = _sinkhorn_log_masked(a, b, C, allowed, eps_schedule=eps_schedule, iters_per_eps=iters_per_eps, tol=1e-6)

        # fuse
        col_mass = T.sum(dim=0).clamp(min=1e-12)
        X_grid = (T.T @ X_spot) / col_mass[:, None]   # (m,G)

        # ---- adaptive SVT (no explicit rank) ----
        X_grid, A_latent, Vt = _svt_adaptive(
            X_grid, keep_nuc_frac=keep_nuc_frac, knee=knee_mode
        )

        # ---- latent anisotropic diffusion (edge-preserving; optional) ----
        if aniso_iters > 0 and A_latent.shape[1] > 0 and len(ei_np) > 0:
            A_latent = _latent_anisotropic_diffusion(
                A_latent, ei_np, ej_np, iters=aniso_iters, tau=aniso_tau, kappa=aniso_kappa
            )
            X_grid = A_latent @ Vt  # reconstruct after diffusion

        # update grid embedding for next cost (reusing same PCA)
        Xg_np = X_grid.detach().cpu().numpy()
        # --- simple sanitize to avoid log1p warnings / NaNs ---
        Xg_np = np.nan_to_num(Xg_np, nan=0.0, posinf=0.0, neginf=0.0)
        Xg_np[Xg_np < 0.0] = 0.0
        row_sum = Xg_np.sum(axis=1, keepdims=True); row_sum[row_sum == 0.0] = 1.0
        Xg_norm = (Xg_np / row_sum) * 1e4
        Xg_norm = np.log1p(Xg_norm)
        Xg_norm = np.nan_to_num(Xg_norm, nan=0.0, posinf=0.0, neginf=0.0)

        Xg_emb = pca.transform(Xg_norm)
        Xg_emb /= (np.linalg.norm(Xg_emb, axis=1, keepdims=True) + 1e-8)
        Xp_grid = torch.tensor(Xg_emb, device=device, dtype=torch.float32)

        b = col_mass / col_mass.sum()

    # pack
    new_adata = AnnData(X=X_grid.detach().cpu().numpy(), var=var_ref.copy())
    new_adata.obsm["spatial"] = grid.astype(np.float32)
    new_adata.obs_names = [f"grid_{i}" for i in range(new_adata.n_obs)]
    new_adata.uns.update(dict(
        grid_size=float(grid_size),
        fuse_radius=None if fuse_radius is None else float(fuse_radius),
        alpha=float(alpha),
        pca_dim=int(pca_dim),
        outer_loops=int(outer_loops),
        keep_nuc_frac=float(keep_nuc_frac),
        knee_mode=bool(knee_mode),
        aniso_iters=int(aniso_iters),
        aniso_tau=float(aniso_tau),
        aniso_kappa=float(aniso_kappa),
        eps_schedule=[float(e) for e in eps_schedule],
        iters_per_eps=int(iters_per_eps),
    ))
    return new_adata
