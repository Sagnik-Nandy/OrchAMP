#!/usr/bin/env python3
import os
import sys
import importlib
import numpy as np
import pandas as pd
import complete_pipeline
importlib.reload(complete_pipeline)
MultimodalPCAPipeline = complete_pipeline.MultimodalPCAPipeline
SeparatePCAPipeline = complete_pipeline.SeparatePCAPipeline

np.random.seed(10)

def generate_tri_modal(
    N,
    gamma1=0.25,
    gamma2=0.25,
    r1=2,
    r2=3,                 # screenshot uses 3 columns in U2
    r3=2,
    rho=0.85,             # dependence level used in U2 and U3
    tildrho=0.85,         # if you want U3 to use a different rho; else set = rho
    lam=1,
    seed=None
):
    """
    Generate X1 (high-d), X2 (high-d), X3 (low-d) with latent factors U1,U2,U3
    according to the spec in the provided screenshot.

    Key constructs:
      U1_i = sqrt(2/3) * (T1_i + T2_i),  T1_i ~ Unif(S^{r1-1}),  T2_i ~ Unif({±1}^{r1})
      U2_·1 = ρ * stdz(truncate-linear(U1_·1)) + sqrt(1-ρ^2) * Z,  Z ~ N(0,1)
      U2_·2 = ρ * stdz(max(0, U1_·2)) + sqrt(1-ρ^2) * B,         B ~ Rademacher(±1)
      U2_·3 ~ Laplace(0, 1/√2)  (extra cols, if any, are independent Laplace)
      U3_i = ρ̃ * U1_i + sqrt(1-ρ̃^2) * T3_i,  T3_i ~ Unif({±1}^{r1})
      Rows of V_h are i.i.d. √(r_h) * Unif(S^{r_h-1})
      D1 = diag(lam + (3 - i)), i=1..r1
      D2 = diag(2*(4 - i)), i=1..r2
      L3 = 5 I_{r3} + 2.5 11ᵀ
    """
    if seed is not None:
        np.random.seed(seed)

    # Feature dimensions
    p1 = int(round(gamma1 * N))
    p2 = int(round(gamma2 * N))

    # -----------------------
    # U1 : sqrt(2/3) * (Unif(S^{r1-1}) + Rademacher^{r1})
    # -----------------------
    T1 = np.random.randn(N, r1)
    T1 = T1 / np.linalg.norm(T1, axis=1, keepdims=True)        # Unif on unit sphere
    T2 = np.random.choice([-1, 1], size=(N, r1))               # Unif on {±1}^{r1}
    U1 = np.sqrt(2.0 / 3.0) * (T1 + T2)

    # -----------------------
    # U2 : 3 columns per spec (expand/shrink to r2 if needed)
    # -----------------------

    # The order of the columns are reversed compared to the manuscript. Please match by signal strength

    # col 1: truncate then linear map then standardize, mixed with Gaussian
    expU = np.clip(U1[:, 0], -0.25, 0.25) * 4.0 + 1.0  
    expU = (expU - expU.mean()) / expU.std()
    U2_col0 = rho * expU + np.sqrt(1 - rho**2) * np.random.randn(N)

    # col 2: ReLU transform with Rademacher noise
    reluU = np.maximum(0.0, U1[:, 1])
    reluU = (reluU - reluU.mean()) / (reluU.std() if reluU.std() > 0 else 1.0)
    U2_col1 = rho * reluU + np.sqrt(1 - rho**2) * np.random.choice([-1, 1], size=N)

    # col 3: independent Laplace(0, 1/sqrt(2))
    laplace_scale = 1.0 / np.sqrt(2.0)
    U2_col2 = np.random.laplace(loc=0.0, scale=laplace_scale, size=N)

    U2_base = np.column_stack([U2_col0, U2_col1, U2_col2])

    # If r2 != 3, adapt by trimming or adding extra independent Laplace columns
    if r2 < 3:
        U2 = U2_base[:, :r2]
    elif r2 == 3:
        U2 = U2_base
    else:
        extra = np.random.laplace(loc=0.0, scale=laplace_scale, size=(N, r2 - 3))
        U2 = np.column_stack([U2_base, extra])

    # -----------------------
    # U3 : noisy linear function of U1 with Rademacher noise
    # -----------------------
    T3 = np.random.choice([-1, 1], size=(N, r1))
    U3 = tildrho * U1 + np.sqrt(1 - tildrho**2) * T3

    # -----------------------
    # Loadings V1, V2 : rows ~ √(r_h) * Unif(S^{r_h-1})
    # -----------------------
    V1 = np.random.randn(p1, r1)
    V1 = V1 / np.linalg.norm(V1, axis=1, keepdims=True)
    V1 = np.sqrt(r1) * V1

    V2 = np.random.randn(p2, r2)
    V2 = V2 / np.linalg.norm(V2, axis=1, keepdims=True)
    V2 = np.sqrt(r2) * V2

    # -----------------------
    # Strength matrices and low-dim loading
    # -----------------------
    D1 = np.diag([lam + (3 - i) for i in range(1, r1 + 1)])
    D2 = np.diag([2 * i for i in range(1, r2 + 1)])
    L3 = 5.0 * np.eye(r3) + 2.5 * np.ones((r3, r3))

    # -----------------------
    # Observed data
    # -----------------------
    X1 = (U1 @ D1 @ V1.T) / N + np.random.randn(N, p1) / np.sqrt(N)
    X2 = (U2 @ D2 @ V2.T) / N + np.random.randn(N, p2) / np.sqrt(N)
    X3 = (U3 @ L3.T) + np.random.randn(N, r3)

    return (X1, X2, X3), (U1, U2, U3)


def rec_error(Uhat, Utrue):
    """
    Reconstruction error:
      E_rec = ||Uhat Uhat^T - Utrue Utrue^T||_F^2 / N^2
    """
    N = Utrue.shape[0]
    diff = Uhat @ Uhat.T - Utrue @ Utrue.T
    return np.linalg.norm(diff, 'fro')**2 / (N**2)




def run_data_integration(N, lam, rho, gamma1, gamma2, reps=50):
    """
    Run the data integration experiment for given parameters.

    Returns:
      errors: (reps, 12) array with per-rep errors in this order
              [joint_U1, joint_U2, joint_U3,
               sep_U1,   sep_U2,   sep_U3,
               svd_U1,   svd_U2,   svd_U3,
               U2_first2_joint, U2_first2_sep, U2_first2_svd]
      mean:   (12,) mean over reps (same order)
      std:    (12,) std  over reps (same order)
    """
    errors = np.zeros((reps, 12), dtype=float)  # 3 methods × 3 modalities + 3 for U2 first two cols

    for i in range(reps):
        # --- generate one replication ---
        (X1, X2, X3), (U1, U2, U3) = generate_tri_modal(
            N,
            gamma1=gamma1,
            gamma2=gamma2,
            r1=2,
            r2=3,
            r3=2,
            rho=rho,
            tildrho=rho,
            lam=lam,
            seed=None
        )

        # === Joint AMP ===
        print(f"[{i+1}/{reps}] Joint AMP", flush=True)
        pipeline = MultimodalPCAPipeline(
            n_components_u=4,
            n_components_v_list=[4, 4]
        )
        amp_out = pipeline.denoise_amp(
            X_list=[X1, X2],
            X_low_dim_list=[X3],
            K_list=[2, 3],          # r1=2, r2=3
            amp_iters=25,
            muteu=False,
            mutev=False
        )
        Uhat1 = amp_out["U_denoised"][0][:, :, -1]
        Uhat2 = amp_out["U_denoised"][1][:, :, -1]
        Uhat3 = amp_out["U_ld_denoised"][0][:, :, -1]
        errors[i, 0] = rec_error(Uhat1, U1)
        errors[i, 1] = rec_error(Uhat2, U2)
        errors[i, 2] = rec_error(Uhat3, U3)

        # === Separate AMP ===
        print(f"[{i+1}/{reps}] Separate AMP", flush=True)
        sep_pipeline = SeparatePCAPipeline(
            n_components_list_v=[4, 4],
            n_components_list_u=[4, 4, 4]
        )
        sep_out = sep_pipeline.denoise_amp(
            X_list=[X1, X2],
            X_low_dim_list=[X3],
            K_list=[2, 3],
            amp_iters=25,
            muteu=False,
            mutev=False,
            preprocess=False
        )
        Uhat1_sep = sep_out["U_denoised"][0][:, :, -1]
        Uhat2_sep = sep_out["U_denoised"][1][:, :, -1]
        Uhat3_sep = sep_out["U_ld_denoised"][0][:, :, -1]
        errors[i, 3] = rec_error(Uhat1_sep, U1)
        errors[i, 4] = rec_error(Uhat2_sep, U2)
        errors[i, 5] = rec_error(Uhat3_sep, U3)

        # === SVD baseline ===
        print(f"[{i+1}/{reps}] SVD baseline", flush=True)
        from complete_pipeline import compute_svd_only, LowDimModalityLoadings
        pca_model_ld = LowDimModalityLoadings()
        pca_model_ld.fit([X3])
        U_svd, _, U_ld = compute_svd_only([X1, X2], [X3], pca_model_ld, K_list=[2, 3])
        errors[i, 6] = rec_error(U_svd[0], U1)
        errors[i, 7] = rec_error(U_svd[1], U2)
        errors[i, 8] = rec_error(U_ld[0], U3)

        # === NEW: U2 first two columns only (for all three methods) ===
        errors[i, 9]  = rec_error(Uhat2[:, :2],     U2[:, :2])   # joint
        errors[i, 10] = rec_error(Uhat2_sep[:, :2], U2[:, :2])   # separate
        errors[i, 11] = rec_error(U_svd[1][:, :2],  U2[:, :2])   # svd

    return errors, errors.mean(axis=0), errors.std(axis=0)


def main():
    if len(sys.argv) != 4:
        print("Usage: python effect_of_data_integration.py <setting_label> <lambda> <rho>", flush=True)
        sys.exit(1)

    label = sys.argv[1]
    lam = float(sys.argv[2])
    rho = float(sys.argv[3])

    gamma_map = {
        "std": (0.25, 0.25),
        "sparse": (0.25, 0.05),
    }
    if label not in gamma_map:
        print(f"Unknown setting label '{label}'. Valid options are: {list(gamma_map.keys())}", flush=True)
        sys.exit(1)

    gamma1, gamma2 = gamma_map[label]
    print(f"Running: setting={label}, lambda={lam}, rho={rho}", flush=True)

    # ==== run reps ====
    reps = 50
    errors, E_mean, E_std = run_data_integration(
        N=10000,
        lam=lam,
        rho=rho,
        gamma1=gamma1,
        gamma2=gamma2,
        reps=reps
    )

    # ===== save results =====
    out_dir = "/home/nandy.15/Research/Experiments_revision/Results/Effect_of_data_integration"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Per-iteration results (all metrics together)
    cols = [
        "E1_joint", "E2_joint", "E3_joint",
        "E1_sep",   "E2_sep",   "E3_sep",
        "E1_svd",   "E2_svd",   "E3_svd",
        # NEW: U2 first two columns only
        "E2ab_joint", "E2ab_sep", "E2ab_svd"
    ]
    df_iters = pd.DataFrame(errors, columns=cols)
    df_iters.insert(0, "iter", range(1, reps + 1))
    df_iters.insert(0, "rho", rho)
    df_iters.insert(0, "lambda", lam)
    df_iters.insert(0, "setting", label)

    iters_path = os.path.join(
        out_dir, f"reconstruction_{label}_lam{lam}_rho{rho}_iters.csv"
    )
    df_iters.to_csv(iters_path, index=False)

    # 2) Summary (mean & std)
    result_dict = {
        "setting": label,
        "lambda": lam,
        "rho": rho,
        "E1_joint": E_mean[0], "E2_joint": E_mean[1], "E3_joint": E_mean[2],
        "E1_sep":   E_mean[3], "E2_sep":   E_mean[4], "E3_sep":   E_mean[5],
        "E1_svd":   E_mean[6], "E2_svd":   E_mean[7], "E3_svd":   E_mean[8],
        # NEW means:
        "E2ab_joint": E_mean[9],  "E2ab_sep": E_mean[10], "E2ab_svd": E_mean[11],

        "E1_joint_std": E_std[0], "E2_joint_std": E_std[1], "E3_joint_std": E_std[2],
        "E1_sep_std":   E_std[3], "E2_sep_std":   E_std[4], "E3_sep_std":   E_std[5],
        "E1_svd_std":   E_std[6], "E2_svd_std":   E_std[7], "E3_svd_std":   E_std[8],
        # NEW stds:
        "E2ab_joint_std": E_std[9], "E2ab_sep_std": E_std[10], "E2ab_svd_std": E_std[11],
        "reps": reps,
    }
    df_summary = pd.DataFrame([result_dict])

    summary_path = os.path.join(
        out_dir, f"reconstruction_{label}_lam{lam}_rho{rho}.csv"
    )
    df_summary.to_csv(summary_path, index=False)

    # 3) Three per-iteration CSVs (one per modality)
    #    X1 (modality 1): columns 0,3,6 → joint/sep/svd for U1
    df_X1 = pd.DataFrame({
        "setting": [label]*reps,
        "lambda":  [lam]*reps,
        "rho":     [rho]*reps,
        "iter":    range(1, reps+1),
        "E_joint": errors[:, 0],
        "E_sep":   errors[:, 3],
        "E_svd":   errors[:, 6],
    })
    path_X1 = os.path.join(out_dir, f"reconstruction_{label}_lam{lam}_rho{rho}_iters_X1.csv")
    df_X1.to_csv(path_X1, index=False)

    #    X2 (modality 2): columns 1,4,7 → joint/sep/svd for U2
    df_X2 = pd.DataFrame({
        "setting": [label]*reps,
        "lambda":  [lam]*reps,
        "rho":     [rho]*reps,
        "iter":    range(1, reps+1),
        "E_joint": errors[:, 1],
        "E_sep":   errors[:, 4],
        "E_svd":   errors[:, 7],
    })
    path_X2 = os.path.join(out_dir, f"reconstruction_{label}_lam{lam}_rho{rho}_iters_X2.csv")
    df_X2.to_csv(path_X2, index=False)

    #    X3 (modality 3): columns 2,5,8 → joint/sep/svd for U3
    df_X3 = pd.DataFrame({
        "setting": [label]*reps,
        "lambda":  [lam]*reps,
        "rho":     [rho]*reps,
        "iter":    range(1, reps+1),
        "E_joint": errors[:, 2],
        "E_sep":   errors[:, 5],
        "E_svd":   errors[:, 8],
    })
    path_X3 = os.path.join(out_dir, f"reconstruction_{label}_lam{lam}_rho{rho}_iters_X3.csv")
    df_X3.to_csv(path_X3, index=False)

    # 4) NEW: per-iteration CSV for U2 first two columns only
    df_X2ab = pd.DataFrame({
        "setting": [label]*reps,
        "lambda":  [lam]*reps,
        "rho":     [rho]*reps,
        "iter":    range(1, reps+1),
        "E_joint": errors[:, 9],
        "E_sep":   errors[:, 10],
        "E_svd":   errors[:, 11],
    })
    path_X2ab = os.path.join(out_dir, f"reconstruction_{label}_lam{lam}_rho{rho}_iters_X2_first2.csv")
    df_X2ab.to_csv(path_X2ab, index=False)

    print(f"Saved per-iteration (all): {iters_path}", flush=True)
    print(f"Saved per-iteration X1   : {path_X1}", flush=True)
    print(f"Saved per-iteration X2   : {path_X2}", flush=True)
    print(f"Saved per-iteration X2 first-two-cols: {path_X2ab}", flush=True)
    print(f"Saved per-iteration X3   : {path_X3}", flush=True)
    print(f"Saved summary            : {summary_path}", flush=True)

if __name__ == "__main__":
    main()
