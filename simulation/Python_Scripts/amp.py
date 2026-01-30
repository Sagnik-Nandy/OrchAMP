import warnings
import importlib
# force reload of key modules so edits are picked up
import emp_bayes, pca_pack, preprocessing
importlib.reload(emp_bayes)
importlib.reload(pca_pack)
importlib.reload(preprocessing)

import numpy as np
from scipy.linalg import block_diag
from emp_bayes import ParametricEB, MultiModalParametricEB
from pca_pack import MultiModalityPCA
from preprocessing import LowDimModalityLoadings

def ebamp_multimodal(pca_model, pca_model_low_dim=None, eb_models_v=None, eb_model_u=None, amp_iters=10, muteu=False, mutev=False):
    """
    Multimodal Gaussian Bayes AMP with per-modality denoising for V and per-cluster denoising for U. We also incorporate the low-dim modalities

    Parameters
    ----------
    pca_model : MultiModalityPCA
        PCA model containing PCA results for each high dimensional modality.
    pca_model_low_dim : LowDimModalityLoadings
        Pre-processed model for the low dimensional modalities with the estimated loadings L_k's.
    eb_models_v : list of ParametricEB
        Mapping from modality index k to a ParametricEB instance for V-denoising.
    eb_model_u : ParametricEB
        ParametricEB instance for U-denoising across all modalities.
    amp_iters : int
        Number of AMP iterations.
    muteu, mutev : bool
        If True, use the identity map as the denoiser in that direction.
    
    Returns
    -------
    U_dict, V_dict : dict
        Denoised U and V matrices for each high-dimensional modality.
    U_dict_low : dict
        Denoised U for each low-dimensional modality.
    """

    X_list = [pca_model.pca_results[k].X for k in pca_model.pca_results.keys()]
    n_samples, _ = X_list[0].shape  
    m_hd = len(X_list)  # Number of high-dim modalities
    gamma_list = [X.shape[1] / n_samples for X in X_list]  # Aspect ratios
    # Only extract low-dim if pca_model_low_dim is not None
    if pca_model_low_dim is not None:
        X_low_dim_list = [pca_model_low_dim.pca_results[k].X for k in pca_model_low_dim.pca_results.keys()]
        m_ld = len(X_low_dim_list)
    else:
        X_low_dim_list = []
        m_ld = 0

    # Initialize storage dictionaries
    U_dict, V_dict, U_dict_denois, V_dict_denois = {}, {}, {}, {}
    mu_dict_u, sigma_sq_dict_u, mu_dict_v, sigma_sq_dict_v = {}, {}, {}, {}
    b_bar_dict_u = {}
    U_dict_ld_denois = {}
    mu_dict_ld, sigma_sq_dict_ld = {}, {}

    # Store diagonal signal matrices for each modality
    signal_diag_dict = {k: np.diag(pca_model.pca_results[k].signals) for k in range(m_hd)}

    # Initialize storage per modality
    for k in range(m_hd):
        pca_k = pca_model.pca_results[k]
        U_init, V_init = pca_k.U, pca_k.V

        # Normalize U and V
        f_k = U_init / np.linalg.norm(U_init, axis=0) * np.sqrt(n_samples)
        g_k = V_init / np.linalg.norm(V_init, axis=0) * np.sqrt(X_list[k].shape[1])

        # Initialize mu and sigma_sq for V denoising (first step uses cluster model)
        mu_dict_v[k] = np.diag(pca_k.feature_aligns)
        sigma_sq_dict_v[k] = np.diag(1 - pca_k.feature_aligns**2)
        mu_dict_u[k] = np.diag(pca_k.sample_aligns)
        sigma_sq_dict_u[k] = np.diag(1 - pca_k.sample_aligns**2)

        # Store initial values
        U_dict[k] = f_k[:, :, np.newaxis]
        V_dict[k] = g_k[:, :, np.newaxis]
        U_dict_denois[k] = (f_k @ np.sqrt(sigma_sq_dict_v[k]))[:, :, np.newaxis]
        V_dict_denois[k] = g_k[:, :, np.newaxis]
    
    # Initialize low-dimensional modality dictionaries only if pca_model_low_dim is provided
    if pca_model_low_dim is not None:
        for j in range(m_ld):
            ld_pca_pack = pca_model_low_dim.pca_results[j]
            mu_dict_ld[j] = ld_pca_pack.sample_aligns
            sigma_sq_dict_ld[j] = np.eye(ld_pca_pack.sample_aligns.shape[0])
            U_dict_ld_denois[j] = ld_pca_pack.U[:, :, np.newaxis]
        # Precompute M_ld_list and S_ld_list for low-dim modalities (sorted by key for consistent ordering)
        M_ld_list = [mu_dict_ld[j] for j in sorted(mu_dict_ld.keys())]
        S_ld_list = [sigma_sq_dict_ld[j] for j in sorted(sigma_sq_dict_ld.keys())]

    for t in range(amp_iters):

        #print(f"\n--- AMP Iteration {t + 1} ---")

        # ---- Step 1: Denoising V (PER-MODALITY) ----
        for k in range(m_hd):

            gamma_k = gamma_list[k]

            vdenoiser = eb_models_v[k]  # dict with 'denoise' and 'ddenoise'

            g_k = V_dict[k][:, :, -1]  # Latest estimate of V
            mu_k, sigma_sq_k = mu_dict_v[k], sigma_sq_dict_v[k]
            u_k = U_dict_denois[k][:, :, -1]

            if not mutev:
                v_k = vdenoiser["denoise"](g_k, mu_k, sigma_sq_k)
                V_dict_denois[k] = np.dstack((V_dict_denois[k], v_k[:, :, np.newaxis]))
                # Compute correction term
                b_k = gamma_k * np.mean(vdenoiser["ddenoise"](g_k, mu_k, sigma_sq_k), axis=0)
                sigma_bar_sq = v_k.T @ v_k / n_samples
                mu_bar = sigma_bar_sq * pca_model.pca_results[k].signals
            else:
                # Identity denoiser
                mu_inv = np.linalg.pinv(mu_k)
                v_k = g_k @ mu_inv.T
                V_dict_denois[k] = np.dstack((V_dict_denois[k], v_k[:, :, np.newaxis]))
                b_k = mu_inv * gamma_k
                mu_bar = np.diag(pca_model.pca_results[k].signals) * gamma_k
                sigma_bar_sq = (np.eye(v_k.shape[1]) + mu_inv @ sigma_sq_k @ mu_inv.T) * gamma_k
            

            # Update f_k using v_k
            f_k = X_list[k] @ v_k - u_k @ b_k.T
            U_dict[k] = np.dstack((U_dict[k], f_k[:, :, np.newaxis]))

            # Store updated mu and sigma_sq for next U denoising
            mu_dict_u[k] = mu_bar
            sigma_sq_dict_u[k] = sigma_bar_sq

            #print(f"[V] iter={t}, modality={k} | sum(mu_bar)={mu_bar.sum():.4g}, sum(sigma_bar_sq)={sigma_bar_sq.sum():.4g}")

        # ---- Step 2: Denoising U (ALL MODALITIES) ----
        # Concatenate high-dimensional and low-dimensional U estimates if present
        f_hd = [U_dict[k][:, :, -1] for k in range(m_hd)]
        if pca_model_low_dim is not None:
            f_ld = X_low_dim_list
            f_concat = np.hstack([*f_hd, *f_ld])
        else:
            f_concat = np.hstack(f_hd)

        # Build block-diagonal M including low-dim loadings L_k if present
        M_hd = [mu_dict_u[k] for k in range(m_hd)]
        if pca_model_low_dim is not None:
            M_concat = block_diag(*[*M_hd, *M_ld_list])
        else:
            M_concat = block_diag(*M_hd)

        # Build block-diagonal S using prior variances and identity for low-dim if present
        S_hd = [sigma_sq_dict_u[k] for k in range(m_hd)]
        if pca_model_low_dim is not None:
            S_concat = block_diag(*[*S_hd, *S_ld_list])
        else:
            S_concat = block_diag(*S_hd)
        # Check singularity of S_concat
        if np.linalg.matrix_rank(S_concat) < S_concat.shape[0]:
            warnings.warn("S_concat is singular or ill-conditioned", UserWarning)
        udenoiser = eb_model_u  # dict with 'denoise' and 'ddenoise'

        # Build signal strengths: high-dim PCA signals + identity signals for low-dim if present
        signals_hd = np.concatenate([pca_model.pca_results[k].signals for k in range(m_hd)])
        if pca_model_low_dim is not None:
            signals_ld = np.concatenate([
                np.ones(pca_model_low_dim.pca_results[j].sample_aligns.shape[0])
                for j in range(m_ld)
            ])
            full_signals = np.concatenate([signals_hd, signals_ld])
        else:
            full_signals = signals_hd

        if not muteu:
            u_concat = udenoiser["denoise"](f_concat, M_concat, S_concat)
            b_bar = np.mean(udenoiser["ddenoise"](f_concat, M_concat, S_concat), axis=0)
            sigma_sq = u_concat.T @ u_concat / n_samples
            mu_bar_all = sigma_sq * full_signals
        else:
            mu_inv = np.linalg.pinv(M_concat)
            u_concat = f_concat @ mu_inv.T
            b_bar = mu_inv
            mu_bar_all = np.diag(full_signals)
            sigma_sq = np.eye(u_concat.shape[1]) + mu_inv @ S_concat @ mu_inv.T

        #print(f"[U] iter={t} | sum(mu_bar_all)={mu_bar_all.sum():.4g}, sum(sigma_sq)={sigma_sq.sum():.4g}")

        # Split back per modality
        start = 0
        for k in range(m_hd):
            width = U_dict[k].shape[1]
            U_dict_denois[k] = np.dstack((U_dict_denois[k], u_concat[:, start:start+width][:, :, np.newaxis]))
            b_bar_dict_u[k] = b_bar[start:start+width, start:start+width]
            mu_dict_v[k] = mu_bar_all[start:start+width, start:start+width]
            sigma_sq_dict_v[k] = sigma_sq[start:start+width, start:start+width]
            # Update V estimate
            g_k = X_list[k].T @ U_dict_denois[k][:, :, -1] - V_dict_denois[k][:, :, -1] @ b_bar_dict_u[k].T
            V_dict[k] = np.dstack((V_dict[k], g_k[:, :, np.newaxis]))
            start += width

        # Split back for low-dimensional modalities if present
        if pca_model_low_dim is not None:
            start_ld = sum(U_dict[k].shape[1] for k in range(m_hd))
            for j in range(m_ld):
                width_ld = U_dict_ld_denois[j].shape[1]
                U_dict_ld_denois[j] = np.dstack((U_dict_ld_denois[j], u_concat[:, start_ld:start_ld+width_ld][:, :, np.newaxis]))
                start_ld += width_ld


    # Return dict, only include U_ld_denoised if present
    result = {
        "U_non_denoised": U_dict,
        "U_denoised": U_dict_denois,
        "V_non_denoised": V_dict,
        "V_denoised": V_dict_denois,
        "signal_diag_dict": signal_diag_dict,
        "eb_models_v": eb_models_v,
        "eb_model_u": eb_model_u
    }
    if pca_model_low_dim is not None:
        result["U_ld_denoised"] = U_dict_ld_denois
    return result


def ebamp_separate(
    pca_model,
    pca_model_low_dim=None,
    eb_models_v=None,
    eb_models_u=None,
    amp_iters=10,
    muteu=False,
    mutev=False,
    instability_threshold=30.0,  # <-- threshold for sums of mu_bar or sigma_bar
):
    """
    Run AMP separately for each modality, using per-modality EB models for V and U.
    Stops early (and rolls back to last clean iteration) if instability is detected.
    """

    # ---------- helpers ----------
    def _snapshot_state():
        """Deep-ish copy of current high-dim AMP iterates to return as 'last clean'."""
        return {
            "U_non_denoised": {k: U_dict[k].copy() for k in U_dict},
            "U_denoised": {k: U_dict_denois[k].copy() for k in U_dict_denois},
            "V_non_denoised": {k: V_dict[k].copy() for k in V_dict},
            "V_denoised": {k: V_dict_denois[k].copy() for k in V_dict_denois},
        }

    # ---------- set up ----------
    X_list = [pca_model.pca_results[k].X for k in pca_model.pca_results.keys()]
    n_samples, _ = X_list[0].shape
    m_hd = len(X_list)  # Number of high-dim modalities
    gamma_list = [X.shape[1] / n_samples for X in X_list]  # Aspect ratios

    # Only extract low-dim if pca_model_low_dim is not None
    if pca_model_low_dim is not None:
        X_low_dim_list = [pca_model_low_dim.pca_results[k].X for k in pca_model_low_dim.pca_results.keys()]
        m_ld = len(X_low_dim_list)
    else:
        X_low_dim_list = []
        m_ld = 0

    # Initialize storage dictionaries
    U_dict, V_dict, U_dict_denois, V_dict_denois = {}, {}, {}, {}
    mu_dict_u, sigma_sq_dict_u, mu_dict_v, sigma_sq_dict_v = {}, {}, {}, {}
    U_dict_ld_denois = {}
    U_dict_ld_non_denoised = {}
    mu_dict_ld, sigma_sq_dict_ld = {}, {}

    # Store diagonal signal matrices for each modality
    signal_diag_dict = {k: np.diag(pca_model.pca_results[k].signals) for k in range(m_hd)}

    # Initialize storage per modality
    for k in range(m_hd):
        pca_k = pca_model.pca_results[k]
        U_init, V_init = pca_k.U, pca_k.V

        # Normalize U and V
        f_k = U_init / np.linalg.norm(U_init, axis=0) * np.sqrt(n_samples)
        g_k = V_init / np.linalg.norm(V_init, axis=0) * np.sqrt(X_list[k].shape[1])

        # Initialize mu and sigma_sq for V denoising (first step uses cluster model)
        mu_dict_v[k] = np.diag(pca_k.feature_aligns)
        sigma_sq_dict_v[k] = np.diag(1 - pca_k.feature_aligns**2)
        mu_dict_u[k] = np.diag(pca_k.sample_aligns)
        sigma_sq_dict_u[k] = np.diag(1 - pca_k.sample_aligns**2)

        # Store initial values
        U_dict[k] = f_k[:, :, np.newaxis]
        V_dict[k] = g_k[:, :, np.newaxis]
        U_dict_denois[k] = (f_k @ np.sqrt(sigma_sq_dict_v[k]))[:, :, np.newaxis]
        V_dict_denois[k] = g_k[:, :, np.newaxis]

    # Initialize low-dimensional modality dictionaries only if pca_model_low_dim is provided
    if pca_model_low_dim is not None:
        for j in range(m_ld):
            ld_pca_pack = pca_model_low_dim.pca_results[j]
            mu_dict_ld[j] = ld_pca_pack.sample_aligns
            sigma_sq_dict_ld[j] = np.eye(ld_pca_pack.sample_aligns.shape[0])
            U_dict_ld_non_denoised[j] = ld_pca_pack.U[:, :, np.newaxis]
            U_dict_ld_denois[j] = ld_pca_pack.U[:, :, np.newaxis]

    # Snapshot the *initial* clean state (iteration = -1)
    last_clean_state = _snapshot_state()

    early_stop = False  # flag to halt outer loop after instability
    # ---------- AMP iterations ----------
    for t in range(amp_iters):

        # ---- Step 1: Denoising V (PER-MODALITY) ----
        for k in range(m_hd):
            gamma_k = gamma_list[k]
            vdenoiser = eb_models_v[k]  # dict with 'denoise' and 'ddenoise'
            g_k = V_dict[k][:, :, -1]  # Latest estimate of V
            mu_k, sigma_sq_k = mu_dict_v[k], sigma_sq_dict_v[k]
            u_k = U_dict_denois[k][:, :, -1]

            if not mutev:
                v_k = vdenoiser["denoise"](g_k, mu_k, sigma_sq_k)
                V_dict_denois[k] = np.dstack((V_dict_denois[k], v_k[:, :, np.newaxis]))
                # Compute correction term
                b_k = gamma_k * np.mean(vdenoiser["ddenoise"](g_k, mu_k, sigma_sq_k), axis=0)
                sigma_bar_sq = v_k.T @ v_k / n_samples
                mu_bar = sigma_bar_sq * pca_model.pca_results[k].signals
            else:
                # Identity denoiser
                mu_inv = np.linalg.pinv(mu_k)
                v_k = g_k @ mu_inv.T
                V_dict_denois[k] = np.dstack((V_dict_denois[k], v_k[:, :, np.newaxis]))
                b_k = mu_inv * gamma_k
                mu_bar = np.diag(pca_model.pca_results[k].signals) * gamma_k
                sigma_bar_sq = (np.eye(v_k.shape[1]) + mu_inv @ sigma_sq_k @ mu_inv.T) * gamma_k

            # Update f_k using v_k
            f_k = X_list[k] @ v_k - u_k @ b_k.T
            U_dict[k] = np.dstack((U_dict[k], f_k[:, :, np.newaxis]))

            # Store updated mu and sigma_sq for next U denoising
            mu_dict_u[k] = mu_bar
            sigma_sq_dict_u[k] = sigma_bar_sq
            # (Optional debug)
            # print(f"[Separate V] iter={t}, modality={k} | sum(mu_bar)={mu_bar.sum():.4g}, sum(sigma_bar)={sigma_bar_sq.sum():.4g}")

        # ---- Step 2: Denoising U (Separately) ----
        for k in range(m_hd):
            # get latest f_k and EB model for U
            f_k = U_dict[k][:, :, -1]
            mu_u_k = mu_dict_u[k]
            sigma_u_k = sigma_sq_dict_u[k]
            udenoiser = eb_models_u[k]  # U EB model

            if not muteu:
                # denoise
                u_k = udenoiser["denoise"](f_k, mu_u_k, sigma_u_k)
                # store denoised U
                U_dict_denois[k] = np.dstack((U_dict_denois[k], u_k[:, :, np.newaxis]))
                # Onsager term
                b_k = np.mean(udenoiser["ddenoise"](f_k, mu_u_k, sigma_u_k), axis=0)
                # update prior for next V step
                sigma_bar = u_k.T @ u_k / n_samples
                mu_bar = sigma_bar * pca_model.pca_results[k].signals
            else:
                # identity denoiser
                mu_inv = np.linalg.pinv(mu_u_k)
                u_k = f_k @ mu_inv.T
                U_dict_denois[k] = np.dstack((U_dict_denois[k], u_k[:, :, np.newaxis]))
                b_k = mu_inv
                mu_bar = np.diag(pca_model.pca_results[k].signals)
                sigma_bar = (np.eye(u_k.shape[1]) + mu_inv @ sigma_u_k @ mu_inv.T)

            mu_dict_v[k] = mu_bar
            sigma_sq_dict_v[k] = sigma_bar
            print(f"[Separate U] iter={t}, modality={k} | sum(mu_bar)={mu_bar.sum():.4g}, sum(sigma_bar)={sigma_bar.sum():.4g}")

            # ---- instability guard: stop ALL further AMP iterations; keep last clean snapshot ----
            if (np.sum(mu_bar) > instability_threshold) or (np.sum(sigma_bar) > instability_threshold):
                warnings.warn(
                    f"AMP stopped early at iter={t}, modality={k} due to instability: "
                    f"sum(mu_bar)={np.sum(mu_bar):.4g}, sum(sigma_bar)={np.sum(sigma_bar):.4g}",
                    UserWarning
                )
                early_stop = True
                break  # breaks the modality loop

            # Update V estimate for modality k
            v_k = V_dict_denois[k][:, :, -1]
            g_k = X_list[k].T @ u_k - v_k @ b_k.T
            V_dict[k] = np.dstack((V_dict[k], g_k[:, :, np.newaxis]))

        # If instability was detected, break the outer t-loop too (we will return last_clean_state)
        if early_stop:
            break

        # Reaching here means the whole iteration t completed cleanly -> refresh snapshot
        last_clean_state = _snapshot_state()

    # ---- Low-dim modality denoising (single-shot) ----
    if pca_model_low_dim is not None:
        for j in range(m_ld):
            # raw low-dim observations
            f_ld = X_low_dim_list[j]
            # corresponding EB denoiser for U: positioned after high-dim models
            udenoiser_ld = eb_models_u[m_hd + j]
            # identity prior for low-dim
            mu_ld = mu_dict_ld[j]
            sigma_ld = sigma_sq_dict_ld[j]
            # denoise (or identity if muteu)
            if not muteu:
                u_ld = udenoiser_ld["denoise"](f_ld, mu_ld, sigma_ld)
            else:
                u_ld = f_ld.copy()
            # store the one-shot denoised low-dim U
            U_dict_ld_denois[j] = np.dstack((U_dict_ld_denois[j], u_ld[:, :, np.newaxis]))

    # ---- Assemble return: use the *last clean snapshot* for high-dim iterates ----
    result = {
        "U_non_denoised": last_clean_state["U_non_denoised"],
        "U_denoised": last_clean_state["U_denoised"],
        "V_non_denoised": last_clean_state["V_non_denoised"],
        "V_denoised": last_clean_state["V_denoised"],
        "signal_diag_dict": signal_diag_dict,
        "eb_models_v": eb_models_v,
        "eb_models_u": eb_models_u,
    }
    if pca_model_low_dim is not None:
        result["U_ld_denoised"] = U_dict_ld_denois
        result["U_ld_non_denoised"] = U_dict_ld_non_denoised

    return result


# # Old version without instability check


# def ebamp_separate(
#     pca_model,
#     pca_model_low_dim=None,
#     eb_models_v=None,
#     eb_models_u=None,
#     amp_iters=10,
#     muteu=False,
#     mutev=False,
# ):
#     """
#     Run AMP separately for each modality, using per-modality EB models for V and U.
#     Does not integrate across modalities.

#     Parameters
#     ----------
#     pca_model : MultiModalityPCA
#         PCA results for high-dimensional modalities.
#     pca_model_low_dim : LowDimModalityLoadings
#         PCA-like results for low-dimensional modalities.
#     eb_models_v : list of dict
#         Per-modality EB denoisers for V (each dict has 'denoise','ddenoise').
#     eb_models_u : list of dict
#         Per-modality EB denoisers for U (same structure).
#     amp_iters : int
#         Number of AMP iterations.
#     muteu, mutev : bool
#         If True, skip denoising for U or V, respectively.

#     Returns
#     -------
#     dict
#         Dictionary with the following keys:
#             "U_non_denoised": dict of U estimates (before denoising) for each high-dim modality.
#             "U_denoised": dict of U estimates (after denoising) for each high-dim modality.
#             "V_non_denoised": dict of V estimates (before denoising) for each high-dim modality.
#             "V_denoised": dict of V estimates (after denoising) for each high-dim modality.
#             "U_ld_denoised": dict of denoised U for each low-dim modality.
#             "U_ld_non_denoised": dict of non-denoised low-dim U (pre-denoised).
#             "eb_models_v": list of V denoisers per modality.
#             "eb_models_u": list of U denoisers per modality.
#     """
#     X_list = [pca_model.pca_results[k].X for k in pca_model.pca_results.keys()]
#     n_samples, _ = X_list[0].shape
#     m_hd = len(X_list)  # Number of high-dim modalities
#     gamma_list = [X.shape[1] / n_samples for X in X_list]  # Aspect ratios
#     # Only extract low-dim if pca_model_low_dim is not None
#     if pca_model_low_dim is not None:
#         X_low_dim_list = [pca_model_low_dim.pca_results[k].X for k in pca_model_low_dim.pca_results.keys()]
#         m_ld = len(X_low_dim_list)
#     else:
#         X_low_dim_list = []
#         m_ld = 0

#     # Initialize storage dictionaries
#     U_dict, V_dict, U_dict_denois, V_dict_denois = {}, {}, {}, {}
#     mu_dict_u, sigma_sq_dict_u, mu_dict_v, sigma_sq_dict_v = {}, {}, {}, {}
#     b_bar_dict_u = {}
#     U_dict_ld_denois = {}
#     U_dict_ld_non_denoised = {}
#     mu_dict_ld, sigma_sq_dict_ld = {}, {}

#     # Store diagonal signal matrices for each modality
#     signal_diag_dict = {k: np.diag(pca_model.pca_results[k].signals) for k in range(m_hd)}

#     # Initialize storage per modality
#     for k in range(m_hd):
#         pca_k = pca_model.pca_results[k]
#         U_init, V_init = pca_k.U, pca_k.V

#         # Normalize U and V
#         f_k = U_init / np.linalg.norm(U_init, axis=0) * np.sqrt(n_samples)
#         g_k = V_init / np.linalg.norm(V_init, axis=0) * np.sqrt(X_list[k].shape[1])

#         # Initialize mu and sigma_sq for V denoising (first step uses cluster model)
#         mu_dict_v[k] = np.diag(pca_k.feature_aligns)
#         sigma_sq_dict_v[k] = np.diag(1 - pca_k.feature_aligns**2)
#         mu_dict_u[k] = np.diag(pca_k.sample_aligns)
#         sigma_sq_dict_u[k] = np.diag(1 - pca_k.sample_aligns**2)

#         # Store initial values
#         U_dict[k] = f_k[:, :, np.newaxis]
#         V_dict[k] = g_k[:, :, np.newaxis]
#         U_dict_denois[k] = (f_k @ np.sqrt(sigma_sq_dict_v[k]))[:, :, np.newaxis]
#         V_dict_denois[k] = g_k[:, :, np.newaxis]

#     # Initialize low-dimensional modality dictionaries only if pca_model_low_dim is provided
#     if pca_model_low_dim is not None:
#         for j in range(m_ld):
#             ld_pca_pack = pca_model_low_dim.pca_results[j]
#             mu_dict_ld[j] = ld_pca_pack.sample_aligns
#             sigma_sq_dict_ld[j] = np.eye(ld_pca_pack.sample_aligns.shape[0])
#             U_dict_ld_non_denoised[j] = ld_pca_pack.U[:, :, np.newaxis]
#             U_dict_ld_denois[j] = ld_pca_pack.U[:, :, np.newaxis]

#     for t in range(amp_iters):

#         # ---- Step 1: Denoising V (PER-MODALITY) ----
#         for k in range(m_hd):
#             gamma_k = gamma_list[k]
#             vdenoiser = eb_models_v[k]  # dict with 'denoise' and 'ddenoise'
#             g_k = V_dict[k][:, :, -1]  # Latest estimate of V
#             mu_k, sigma_sq_k = mu_dict_v[k], sigma_sq_dict_v[k]
#             u_k = U_dict_denois[k][:, :, -1]

#             if not mutev:
#                 v_k = vdenoiser["denoise"](g_k, mu_k, sigma_sq_k)
#                 V_dict_denois[k] = np.dstack((V_dict_denois[k], v_k[:, :, np.newaxis]))
#                 # Compute correction term
#                 b_k = gamma_k * np.mean(vdenoiser["ddenoise"](g_k, mu_k, sigma_sq_k), axis=0)
#                 sigma_bar_sq = v_k.T @ v_k / n_samples
#                 mu_bar = sigma_bar_sq * pca_model.pca_results[k].signals
#             else:
#                 # Identity denoiser
#                 mu_inv = np.linalg.pinv(mu_k)
#                 v_k = g_k @ mu_inv.T
#                 V_dict_denois[k] = np.dstack((V_dict_denois[k], v_k[:, :, np.newaxis]))
#                 b_k = mu_inv * gamma_k
#                 mu_bar = np.diag(pca_model.pca_results[k].signals) * gamma_k
#                 sigma_bar_sq = (np.eye(v_k.shape[1]) + mu_inv @ sigma_sq_k @ mu_inv.T) * gamma_k

#             # Update f_k using v_k
#             f_k = X_list[k] @ v_k - u_k @ b_k.T
#             U_dict[k] = np.dstack((U_dict[k], f_k[:, :, np.newaxis]))

#             # Store updated mu and sigma_sq for next U denoising
#             mu_dict_u[k] = mu_bar
#             sigma_sq_dict_u[k] = sigma_bar_sq
#             #print(f"[Separate V] iter={t}, modality={k} | sum(mu_bar)={mu_bar.sum():.4g}, sum(sigma_bar)={sigma_bar_sq.sum():.4g}")

#         # ---- Step 2: Denoising U (Separately) ----
#         for k in range(m_hd):
#             # get latest f_k and EB model for U
#             f_k = U_dict[k][:, :, -1]
#             mu_u_k = mu_dict_u[k]
#             sigma_u_k = sigma_sq_dict_u[k]
#             udenoiser = eb_models_u[k]  # U EB model

#             if not muteu:
#                 # denoise
#                 u_k = udenoiser["denoise"](f_k, mu_u_k, sigma_u_k)
#                 # store denoised U
#                 U_dict_denois[k] = np.dstack((U_dict_denois[k], u_k[:, :, np.newaxis]))
#                 # Onsager term
#                 b_k = np.mean(udenoiser["ddenoise"](f_k, mu_u_k, sigma_u_k), axis=0)
#                 # update prior for next V step
#                 sigma_bar = u_k.T @ u_k / n_samples
#                 mu_bar = sigma_bar * pca_model.pca_results[k].signals
#             else:
#                 # identity denoiser
#                 mu_inv = np.linalg.pinv(mu_u_k)
#                 u_k = f_k @ mu_inv.T
#                 U_dict_denois[k] = np.dstack((U_dict_denois[k], u_k[:, :, np.newaxis]))
#                 b_k = mu_inv
#                 mu_bar = np.diag(pca_model.pca_results[k].signals)
#                 sigma_bar = (np.eye(u_k.shape[1]) + mu_inv @ sigma_u_k @ mu_inv.T)

#             mu_dict_v[k] = mu_bar
#             sigma_sq_dict_v[k] = sigma_bar
#             print(f"[Separate U] iter={t}, modality={k} | sum(mu_bar)={mu_bar.sum():.4g}, sum(sigma_bar)={sigma_bar.sum():.4g}")

#             # Check for instability: stop if sum(mu_bar) or sum(sigma_bar) exceeds 30
#             if np.sum(mu_bar) > 30 or np.sum(sigma_bar) > 30:
#                 warnings.warn(f"AMP stopped early at iter={t}, modality={k} due to instability: sum(mu_bar)={np.sum(mu_bar):.4g}, sum(sigma_bar)={np.sum(sigma_bar):.4g}", UserWarning)
#                 break

#             # Update V estimate for modality k
#             v_k = V_dict_denois[k][:, :, -1]
#             g_k = X_list[k].T @ u_k - v_k @ b_k.T
#             V_dict[k] = np.dstack((V_dict[k], g_k[:, :, np.newaxis]))

#     # ---- Low-dim modality denoising (single-shot) ----
#     if pca_model_low_dim is not None:
#         for j in range(m_ld):
#             # raw low-dim observations
#             f_ld = X_low_dim_list[j]
#             # corresponding EB denoiser for U: positioned after high-dim models
#             udenoiser_ld = eb_models_u[m_hd + j]
#             # identity prior for low-dim
#             mu_ld = mu_dict_ld[j]
#             sigma_ld = sigma_sq_dict_ld[j]
#             # denoise (or identity if muteu)
#             if not muteu:
#                 u_ld = udenoiser_ld["denoise"](f_ld, mu_ld, sigma_ld)
#             else:
#                 u_ld = f_ld.copy()
#             # store the one-shot denoised low-dim U
#             U_dict_ld_denois[j] = np.dstack((U_dict_ld_denois[j], u_ld[:, :, np.newaxis]))

#     # Return dict, only include low-dim keys if present
#     result = {
#         "U_non_denoised": U_dict,
#         "U_denoised": U_dict_denois,
#         "V_non_denoised": V_dict,
#         "V_denoised": V_dict_denois,
#         "signal_diag_dict": signal_diag_dict,
#         "eb_models_v": eb_models_v,
#         "eb_models_u": eb_models_u
#     }
#     if pca_model_low_dim is not None:
#         result["U_ld_denoised"] = U_dict_ld_denois
#         result["U_ld_non_denoised"] = U_dict_ld_non_denoised
#     return result