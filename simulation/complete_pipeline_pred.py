import numpy as np
from sklearn.mixture import GaussianMixture
import importlib
from scipy.linalg import block_diag, sqrtm
from scipy.sparse.linalg import svds

# Dynamically import required modules
pca_pack = importlib.import_module("pca_pack")
emp_bayes = importlib.import_module("emp_bayes")
amp = importlib.import_module("amp")
preprocessing = importlib.import_module("preprocessing")
importlib.reload(pca_pack)
importlib.reload(emp_bayes)
importlib.reload(amp)
importlib.reload(preprocessing)

from emp_bayes import ParametricEB, MultiModalParametricEB
from preprocessing import LowDimModalityLoadings

def extract_normalized_U(pca_model, X_list):
    """
    Extracts and normalizes the left singular vectors (U) for each modality
    from a fitted MultiModalityPCA object.

    Parameters
    ----------
    pca_model : MultiModalityPCA
        Fitted PCA model containing PCA results.
    X_list : list of ndarray
        List of original modality matrices.

    Returns
    -------
    normalized_U_list : list of ndarray
        List of normalized U_k matrices for each modality.
    """
    normalized_U_list = []
    for k in range(len(X_list)):
        U_k = pca_model.pca_results[k].U
        n = U_k.shape[0]
        U_k_normalized = U_k / np.sqrt((U_k**2).sum(axis=0)) * np.sqrt(n)
        normalized_U_list.append(U_k_normalized)
    return normalized_U_list

def extract_normalized_V(pca_model, X_list):
    """
    Extracts and normalizes the right singular vectors (V) for each modality
    from a fitted MultiModalityPCA object.

    Parameters
    ----------
    pca_model : MultiModalityPCA
        Fitted PCA model containing PCA results.
    X_list : list of ndarray
        List of original modality matrices.

    Returns
    -------
    normalized_U_list : list of ndarray
        List of normalized U_k matrices for each modality.
    """
    normalized_V_list = []
    for k in range(len(X_list)):
        V_k = pca_model.pca_results[k].V
        p = V_k.shape[0]
        V_k_normalized = V_k / np.sqrt((V_k**2).sum(axis=0)) * np.sqrt(p)
        normalized_V_list.append(V_k_normalized)
    return normalized_V_list

class MultimodalPCAPipeline:
    """
    Implements a full pipeline for multimodal PCA denoising using Gaussian Bayes AMP.

    Steps:
    1. Preprocesses raw modality matrices (normalize observations and PCs).
    2. Runs PCA to extract principal components and estimates noise structure.
    3. Constructs empirical Bayes models for U and per-modality denoisers for V.
    4. Runs AMP to obtain denoised U and V matrices.
    Attributes
    ----------
    pca_model : pca_pack.MultiModalityPCA
        PCA results after fitting.
    eb_models_v : list of ParametricEB
        Per-modality empirical Bayes models for V.
    eb_model_u : ParametricEB
        Empirical Bayes model for U across all modalities.
    amp_results : dict
        Stores U, V, denoised and raw versions.

    Methods
    -------
    denoise_amp(X_list, X_low_dim_list, K_list, amp_iters=10, muteu=False, mutev=False, preprocess=False):
        Executes the full denoising pipeline on input data.
    """

    def __init__(self, n_components_u=None, n_components_v_list=None):
        self.pca_model = None
        self.pca_model_low_dim = None
        self.eb_models_v = None
        self.eb_model_u = None
        self.eb_param_u = None  # Store ParametricEB instance for U
        self.amp_results = None
        self.n_components_u = n_components_u
        self.n_components_v_list = n_components_v_list

    def denoise_amp(self, X_list, X_low_dim_list=None, K_list=None, amp_iters=10, muteu=False, mutev=False, preprocess=False, n_components_list_v=None, n_components_u=5):
        """
        Runs the full denoising pipeline: Preprocessing, PCA, empirical Bayes modeling, and AMP.

        Parameters
        ----------
        X_list : list of ndarray
            List of m data matrices X_k of shape (n, r_k).
        X_low_dim_list : list of ndarray, optional
            List of low-dimensional representations of each modality.
        K_list : list of int
            List of m values specifying the number of principal components per modality.
        amp_iters : int, optional
            Number of AMP iterations (default is 5).
        muteu, mutev : bool, optional
            If True, disables denoising in U or V direction (default is False).
        
        Returns
        -------
        amp_results : dict
            Contains the following structured results:
            - "U_non_denoised": dict of non-denoised U matrices
            - "U_denoised": dict of denoised U matrices
            - "V_non_denoised": dict of non-denoised V matrices
            - "V_denoised": dict of denoised V matrices
        """
        
        X_low_dim_list = X_low_dim_list or []

        if preprocess:
            #print("\n=== Step 1: Preprocessing ===")
            diagnostic_tool = preprocessing.MultiModalityPCADiagnostics()
            X_preprocessed = diagnostic_tool.normalize_obs(X_list, K_list)

        #print("\n=== Step 2a: PCA ===")
        self.pca_model = pca_pack.MultiModalityPCA()

        if preprocess:
           self.pca_model.fit(X_preprocessed, K_list, plot_residual=False)
        else:
            self.pca_model.fit(X_list, K_list, plot_residual=False)

        if X_low_dim_list:
            # Fit low-dimensional PCA loadings
            #print("\n=== Step 2b: Low-Dimensional PCA ===")
            self.pca_model_low_dim = LowDimModalityLoadings()
            self.pca_model_low_dim.fit(X_low_dim_list)

        #print("\n=== Step 3: Constructing Empirical Bayes Models ===")
        n = X_list[0].shape[0]
        r_list = [X.shape[1] for X in X_list]
        U_normalized_list = extract_normalized_U(self.pca_model, X_list)
        V_normalized_list = extract_normalized_V(self.pca_model, X_list)

        m_ld = len(X_low_dim_list)
        pca_model_low_dim = self.pca_model_low_dim if X_low_dim_list else None

        # Construct M and S matrices using PCA feature aligns and sample aligns
        M_matrices_v = [np.diag(self.pca_model.pca_results[k].feature_aligns) for k in range(len(X_list))]
        S_matrices_v = [np.diag(1 - self.pca_model.pca_results[k].feature_aligns**2) for k in range(len(X_list))]
        # High-dim M and S
        m_hd = len(X_list)
        M_hd = [np.diag(self.pca_model.pca_results[k].sample_aligns) for k in range(m_hd)]
        S_hd = [np.diag(1 - self.pca_model.pca_results[k].sample_aligns**2) for k in range(m_hd)]
        if X_low_dim_list:
            # Low-dim M and S
            M_ld = [pca_model_low_dim.pca_results[j].sample_aligns for j in range(m_ld)]
            S_ld = [np.eye(pca_model_low_dim.pca_results[j].sample_aligns.shape[0]) for j in range(m_ld)]
            M_matrices_u = M_hd + M_ld
            S_matrices_u = S_hd + S_ld
        else:
            M_matrices_u = M_hd
            S_matrices_u = S_hd

        # Build per-modality ParametricEB models for V
        factory_v = MultiModalParametricEB(n_components_list=self.n_components_v_list)
        self.eb_models_v = factory_v.fit(V_normalized_list, [[M] for M in M_matrices_v], [[S] for S in S_matrices_v])

        # Single-modality EB for U
        factory_u = ParametricEB(n_components=self.n_components_u)
        if X_low_dim_list:
            data_u = np.hstack(U_normalized_list + X_low_dim_list)
        else:
            data_u = np.hstack(U_normalized_list)
        factory_u.estimate_prior(data_u, M_matrices_u, S_matrices_u)
        # Store U denoiser (includes prior and posterior info)
        self.eb_model_u = factory_u.get_denoisers()
        # Retain ParametricEB instance for weight calculations
        self.eb_param_u = factory_u


        #print("\n=== Step 4: Running AMP ===")
        self.amp_results = amp.ebamp_multimodal(
            self.pca_model, self.pca_model_low_dim if X_low_dim_list else None, self.eb_models_v, self.eb_model_u,
            amp_iters=amp_iters, muteu=muteu, mutev=mutev
        )
        #print("\n=== Denoising Complete! ===")
        return self.amp_results


    def predict_embedding_and_density(self, X_list, observed_indices, M_pred=None, S_pred=None):
        """
        Compute test-time embeddings and GMM posterior weights using estimated priors (only high dim modalities supported).

        Parameters
        ----------
        X_list : list of ndarray
            Test data matrices, one per modality.
        observed_indices : list of int
            Indices of modalities in X_list that are actually observed.

        Returns
        -------
        U_hat : dict
            Estimated latent embeddings for each modality.
        W : ndarray
            Posterior component responsibilities for concatenated embedding.
        """
        if not observed_indices:
            raise ValueError("predict_embedding_and_density requires a non-empty list of observed_indices")

        # pull out last AMP-denoised V and signal diagonals
        V_dict = self.amp_results["V_denoised"]
        D_dict = self.amp_results["signal_diag_dict"]
        n = self.pca_model.pca_results[0].X.shape[0]
        print(n, flush=True)

        # 1) compute per-modality embeddings for observed modalities
        U_hat = {}
        for idx, k in enumerate(observed_indices):
            Xk = X_list[idx]
            V_k = V_dict[k][:, :, -1]
            D_k = D_dict[k]
            A_k = (1.0 / n) * D_k @ V_k.T
            AtA_inv = np.linalg.pinv(A_k @ A_k.T)
            U_hat[k] = Xk @ A_k.T @ AtA_inv

        # 2) concatenate to form observed joint embedding
        U_joint = np.concatenate([U_hat[k] for k in sorted(observed_indices)], axis=1)

        # 3) build mapping from observed dims to full joint dims
        dims = [self.pca_model.pca_results[k].U.shape[1] for k in range(len(self.pca_model.pca_results))]
        starts = np.cumsum([0] + dims[:-1])
        d_obs = sum(dims[k] for k in observed_indices)
        d_joint = sum(dims)
        M_block = np.zeros((d_obs, d_joint))
        row = 0
        for k in observed_indices:
            d_k = dims[k]
            col = starts[k]
            M_block[row:row + d_k, col:col + d_k] = np.eye(d_k)
            row += d_k

        # 4) default or user-supplied prediction transforms
        M_pred_final = M_block if M_pred is None else M_pred
        if S_pred is None:
            S_blocks = []
            for k in sorted(observed_indices):
                V_k = V_dict[k][:, :, -1]
                D_k = D_dict[k]
                D_inv = np.linalg.inv(D_k)
                Sigma_k = (1.0 / n) * (V_k.T @ V_k)
                S_blocks.append(D_inv @ np.linalg.inv(Sigma_k) @ D_inv)
            S_pred_final = block_diag(*S_blocks)
        else:
            S_pred_final = S_pred

        # 5) compute posterior responsibilities
        W = self.eb_param_u._get_W(U_joint, M_pred_final, S_pred_final)
        return U_hat, W

        
    def get_conditional_posterior_gmm(self, observed_embedding, m_known, m_unknown,
                                      cov_known, cov_unknown, cross_cov, weights):
        """
        Compute the conditional GMM posterior of the unobserved latent part given the observed part.

        Parameters
        ----------
        observed_embedding : ndarray of shape (1, d_obs)
            Observed test-time embedding (e.g., U_x for query).
        m_known : ndarray of shape (K, d_obs)
            Component means for the observed part.
        m_unknown : ndarray of shape (K, d_unobs)
            Component means for the unobserved part.
        cov_known : ndarray of shape (K, d_obs, d_obs)
            Covariance blocks for the observed part.
        cov_unknown : ndarray of shape (K, d_unobs, d_unobs)
            Covariance blocks for the unobserved part.
        cross_cov : ndarray of shape (K, d_unobs, d_obs)
            Cross-covariance blocks between unobserved and observed.
        weights : ndarray of shape (K,)
            GMM mixture weights.

        Returns
        -------
        means_cond : ndarray of shape (K, d_unobs)
            Component means of the conditional posterior.
        covs_cond : ndarray of shape (K, d_unobs, d_unobs)
            Component covariances of the conditional posterior.
        weights_cond : ndarray of shape (K,)
            Posterior component weights evaluated at the observed_embedding.
        """
        K = len(weights)
        d_obs = observed_embedding.shape[1]
        means_cond = np.zeros((K, m_unknown.shape[1]))
        covs_cond = np.zeros((K, m_unknown.shape[1], m_unknown.shape[1]))

        for k in range(K):
            A = cross_cov[k] @ np.linalg.pinv(cov_known[k])
            delta = observed_embedding - m_known[k]
            means_cond[k] = m_unknown[k] + delta @ A.T
            covs_cond[k] = cov_unknown[k] - A @ cross_cov[k].T

        return means_cond, covs_cond


    def predict_with_missing(self, X_list, observed_indices, M_pred=None, S_pred=None):
        """
        Predict full latent embeddings when some modalities are missing.

        Parameters
        ----------
        X_list : list of ndarray
            Test data matrices for observed modalities, in increasing modality index order.
        observed_indices : list of int
            Indices of modalities in X_list that are actually observed.
        M_pred, S_pred : optional
            Custom mean/covariance blocks for posterior, defaults to model prior.

        Returns
        -------
        U_hat_obs : dict
            Predicted U embeddings for observed modalities.
        U_hat_missing : ndarray, shape (n_samples, total_missing_dim)
            Point predictions for missing latent embeddings.
        gmms : list of GaussianMixture
            Posterior GMM objects for each sample over missing dims.
        """
        # AMP outputs
        V_dict = self.amp_results["V_denoised"]
        D_dict = self.amp_results["signal_diag_dict"]

        # Determine total and missing modalities
        total_modalities = len(self.pca_model.pca_results)
        missing_indices = [k for k in range(total_modalities) if k not in observed_indices]
        n = self.pca_model.pca_results[0].X.shape[0]

        # Build per-modality latent-dimension slices
        dims = [self.pca_model.pca_results[k].U.shape[1] for k in range(total_modalities)]
        starts = np.cumsum([0] + dims[:-1])
        slices = {k: slice(starts[k], starts[k] + dims[k]) for k in range(total_modalities)}
        obs_slices = [slices[k] for k in observed_indices]
        miss_slices = [slices[k] for k in missing_indices]

        # Construct mapping from observed dims to full joint dims
        d_obs = sum(dims[k] for k in observed_indices)
        d_joint = sum(dims)
        M_block = np.zeros((d_obs, d_joint))
        row = 0
        for k in observed_indices:
            d_k = dims[k]
            col = starts[k]
            M_block[row:row + d_k, col:col + d_k] = np.eye(d_k)
            row += d_k

        # Default prediction transforms if not provided
        if M_pred is None:
            M_pred = M_block
        if S_pred is None:
            S_blocks = []
            for k in observed_indices:
                V_k = V_dict[k][:, :, -1]
                D_k = D_dict[k]
                D_inv = np.linalg.inv(D_k)
                Sigma = (1.0 / n) * (V_k.T @ V_k)
                S_blocks.append(D_inv @ np.linalg.inv(Sigma) @ D_inv)
            S_pred = block_diag(*S_blocks)

        # 1) Compute observed embeddings and responsibilities
        U_hat_obs, W = self.predict_embedding_and_density(X_list, observed_indices, M_pred=M_pred, S_pred=S_pred)
        U_joint_obs = np.concatenate([U_hat_obs[k] for k in sorted(observed_indices)], axis=1)

        # 2) Extract GMM prior for full joint latent
        m_prior, cov_prior, weights_prior = self.eb_model_u['prior']

        # 3) Compute conditional GMM parameters
        m_known = m_prior @ M_block.T
        cov_known = np.array([
            M_block @ C @ M_block.T + S_pred for C in cov_prior
        ])
        cov_unknown = np.array(cov_prior)
        cross_cov = np.array([
            C @ M_block.T for C in cov_prior
        ])

        # 4) Predict missing latents per sample
        n_test = X_list[0].shape[0]
        U_hat = np.zeros((n_test, d_joint))
        gmms = []
        for i in range(n_test):
            x_obs = U_joint_obs[i:i + 1, :]
            means_c, covs_c = self.get_conditional_posterior_gmm(
                x_obs, m_known, m_prior,
                cov_known, cov_unknown, cross_cov, weights_prior
            )
            w = W[i]
            U_hat[i] = np.sum(w[:, None] * means_c, axis=0)

            gmm = GaussianMixture(n_components=len(w), covariance_type='full')
            gmm.means_ = means_c
            gmm.covariances_ = covs_c
            gmm.weights_ = w
            gmms.append(gmm)

        return U_hat, gmms