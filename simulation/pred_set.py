#!/usr/bin/env python3
import sys
import importlib
import numpy as np
import pandas as pd
import complete_pipeline_pred
importlib.reload(complete_pipeline_pred)
MultimodalPCAPipeline = complete_pipeline_pred.MultimodalPCAPipeline


np.random.seed(50)

def generate_bi_modal(
    N,
    gamma1=2,
    gamma2=2,
    r1=2,
    r2=3,
    rho=0.85,
    lam=1,
    seed=None
):
    """
    Generate two high-dimensional modalities X1, X2 and one low-dimensional modality X3,
    along with latent factors U1, U2, U3. U1 is Rademacher (±1), U3 is a correlated version of U1,
    and U2 contains non-Gaussian components: one exponential, one ReLU-transformed, and two Laplace-distributed columns.
    Feature dimensions p1 and p2 are determined by gamma1 * N and gamma2 * N, respectively.
    """
    if seed is not None:
        np.random.seed(seed)

    # Determine feature dimensions from gamma ratios
    p1 = int(round(gamma1 * N))
    p2 = int(round(gamma2 * N))

    # U1: i.i.d. Rademacher(±1) variables
    # Old scheme for U1:
    T1 = np.random.randn(N, r1)
    T1 = T1 / np.linalg.norm(T1, axis=1, keepdims=True)        # Unif on unit sphere
    T2 = np.random.choice([-1, 1], size=(N, r1))               # Unif on {±1}^{r1}
    U1 = np.sqrt(2.0 / 3.0) * (T1 + T2)

    # -----------------------
    # U2 : 3 columns per spec (expand/shrink to r2 if needed)
    # -----------------------
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


    # Loadings drawn from standard normal, projected to unit sphere, then standardized
    V1 = np.random.randn(p1, r1)
    V1 = V1 / np.linalg.norm(V1, axis=1, keepdims=True)
    V1 = (V1 - np.mean(V1, axis=0)) / np.std(V1, axis=0)

    V2 = np.random.randn(p2, r2)
    V2 = V2 / np.linalg.norm(V2, axis=1, keepdims=True)
    V2 = (V2 - np.mean(V2, axis=0)) / np.std(V2, axis=0)

    D1 = np.diag([lam + (3 - i) for i in range(1, r1 + 1)])
    D2 = np.diag([2 * i for i in range(1, r2 + 1)])

    # Observations with data integration scaling and noise variance 1/N
    # X1 = (1/N) * U1 D1 V1^T + Z1, Z1 ~ N(0, I/N)
    X1 = (U1 @ D1 @ V1.T) / N + np.random.randn(N, p1) / np.sqrt(N)
    # X2 = (1/N) * U2 D2 V2^T + Z2, Z2 ~ N(0, I/N)
    X2 = (U2 @ D2 @ V2.T) / N + np.random.randn(N, p2) / np.sqrt(N)

    return (X1, X2), (U1, U2), (V1, V2)


def run_data_integration_with_coverage(N, lam, rho, gamma1, gamma2, reps=100, alpha=95, num_samples=100000):
    """
    Run prediction-set coverage experiment.
    Train joint AMP on two modalities, then for each replication:
      - simulate test X1, X2, true U1, U2 via generate_bi_modal
      - fit pipeline, extract signal diagonals and denoised V
      - generate posterior GMM on U given only modality 1 via predict_with_missing
      - draw num_samples, compute radius at alpha, test if true concatenated U falls inside
    Returns average coverage probability.
    """
    coverage = []
    for i in range(reps):
        print(f"[Iteration {i+1}/{reps}]")
        # train
        (X1_train, X2_train), (U1_train, U2_train), (V1_train, V2_train) = generate_bi_modal(N, gamma1, gamma2, rho=rho, lam=lam)
        pipeline = MultimodalPCAPipeline(
            n_components_u=4, 
            n_components_v_list=[4, 4]
        )
        amp_out = pipeline.denoise_amp(
            X_list=[X1_train, X2_train],
            K_list=[2, 3],
            amp_iters=25,
            muteu=False,
            mutev=False
        )
        V_dict = {k: amp_out["V_denoised"][k][:, :, -1] for k in range(2)}
        D_dict = {k: np.diag(pipeline.pca_model.pca_results[k].signals) for k in range(2)}
        
        # generate multiple test points and randomly select one using fixed V1_train, V2_train
        n_test_points = N

        # Generate fresh latent factors U1 and U2 for test data
        U1_all = np.random.randn(n_test_points, 2)
        U1_all = U1_all / np.linalg.norm(U1_all, axis=1, keepdims=True)
        shift = np.random.choice([-1, 1], size=(n_test_points, 2))
        U1_all = U1_all + shift
        U1_all = np.sqrt(2.0 / 3.0) * U1_all

        W2 = np.random.randn(n_test_points, 1)
        exp_vals = np.clip(U1_all[:, 0], -0.25, 0.25) * 4.0 + 1.0
        exp_vals = (exp_vals - np.mean(exp_vals)) / np.std(exp_vals)
        U2_col0 = rho * exp_vals + np.sqrt(1 - rho**2) * W2[:, 0]


        relu_centered = np.maximum(0, U1_all[:, 1])
        relu_scaled = (relu_centered - np.mean(relu_centered)) / np.std(relu_centered)
        U2_col1 = rho * relu_scaled + np.sqrt(1 - rho**2) * np.random.choice([-1, 1], size=n_test_points)

        laplace_scale = 1 / np.sqrt(2)
        U2_col2 = np.random.laplace(loc=0, scale=laplace_scale, size=n_test_points)
        U2_all = np.column_stack([U2_col0, U2_col1, U2_col2])

        D1 = np.diag([lam + (3 - i) for i in range(1, 3)])
        D2 = np.diag([2 * i for i in range(1, 4)])

        # Generate test observations using training V1/V2
        X1_all = (U1_all @ D1 @ V1_train.T) / N + np.random.randn(n_test_points, X1_train.shape[1]) / np.sqrt(N)
        X2_all = (U2_all @ D2 @ V2_train.T) / N + np.random.randn(n_test_points, X2_train.shape[1]) / np.sqrt(N)

        # Select one test sample
        idx = np.random.choice(n_test_points)
        X1_test = X1_all[idx:idx+1]
        X2_test = X2_all[idx:idx+1]
        U1_test = U1_all[idx:idx+1]
        U2_test = U2_all[idx:idx+1]


        # only modality 1 is observed at index 0
        observed_indices = [0]
        U_joint_hat, gmms = pipeline.predict_with_missing(
            [X1_test], observed_indices=observed_indices
        )
        # true joint latent
        U_joint_true = np.hstack([U1_test, U2_test])
        # posterior GMM for joint embedding
        gmm_post = gmms[0]
        # sample from posterior
        samples = gmm_post.sample(num_samples)[0]
        # compute deviations
        deltas = samples - U_joint_hat
        radii = np.linalg.norm(deltas, axis=1)
        r_alpha = np.percentile(radii, alpha)
        # align true signals to predicted signs
        U_joint_true_aligned = np.abs(U_joint_true) * np.sign(U_joint_hat)
        # Check whether the true joint latent falls within the prediction radius
        dist = np.linalg.norm(U_joint_true_aligned - U_joint_hat)
        inside = (dist <= r_alpha)
        coverage.append(inside)
        if inside:
           print(f"Coverage hit: distance {dist:.4f} ≤ radius {r_alpha:.4f}")
        if not inside:
           print(f"Coverage not hit: distance {dist:.4f} greater than radius {r_alpha:.4f}")
        

    return np.mean(coverage)



def main():
    if len(sys.argv) != 4:
        print("Usage: python pred_set_calibration.py <N> <lambda> <rho>", flush=True)
        sys.exit(1)
    N = int(sys.argv[1])
    lam = float(sys.argv[2])
    rho = float(sys.argv[3])
    gamma1, gamma2 = 0.25, 0.25
    reps = 50
    alpha = 95
    num_samples = 100000

    cov = run_data_integration_with_coverage(
        N, lam, rho, gamma1, gamma2,
        reps=reps,
        alpha=alpha,
        num_samples=num_samples
    )
    print(f"Coverage: {cov:.3f}", flush=True)
    # Save results to CSV
    result_dict = {
        "N": N,
        "lambda": lam,
        "rho": rho,
        "coverage": cov
    }
    results_path = f"/home/nandy.15/Research/Experiments_revision/Results/Pred_set_calibrate_mod_1/coverage_lam{lam}_N{N}_rho{rho}.csv"
    pd.DataFrame([result_dict]).to_csv(results_path, index=False)
    print(f"Results saved to {results_path}", flush=True)

if __name__ == "__main__":
    main()
