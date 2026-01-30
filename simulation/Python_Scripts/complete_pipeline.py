import numpy as np
import importlib
from scipy.linalg import block_diag
from scipy.sparse.linalg import svds
from typing import Any, List

# Dynamically import required modules
pca_pack = importlib.import_module("pca_pack")
emp_bayes = importlib.import_module("emp_bayes")
amp = importlib.import_module("amp")
preprocessing = importlib.import_module("preprocessing")

# Tracey Zhong's modules
preprocessing_tz = importlib.import_module("preprocessing_tz")
amp_tz = importlib.import_module("amp_tz")
emp_bayes_tz = importlib.import_module("empbayes_tz")
pca_tz = importlib.import_module("pca_tz")

importlib.reload(pca_pack)
importlib.reload(emp_bayes)
importlib.reload(amp)
importlib.reload(preprocessing)

importlib.reload(preprocessing_tz)
importlib.reload(amp_tz)
importlib.reload(emp_bayes_tz)
importlib.reload(pca_tz)

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
            print("\n=== Step 1: Preprocessing ===")
            diagnostic_tool = preprocessing.MultiModalityPCADiagnostics()
            X_preprocessed = diagnostic_tool.normalize_obs(X_list, K_list)

        print("\n=== Step 2a: PCA ===")
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

        print("\n=== Step 3: Constructing Empirical Bayes Models ===")
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


        #print("\n=== Step 4: Running AMP ===")
        self.amp_results = amp.ebamp_multimodal(
            self.pca_model, self.pca_model_low_dim if X_low_dim_list else None, self.eb_models_v, self.eb_model_u,
            amp_iters=amp_iters, muteu=muteu, mutev=mutev
        )

        #print("\n=== Denoising Complete! ===")
        return self.amp_results

class SeparatePCAPipeline(MultimodalPCAPipeline):
    """
    Pipeline for PCA denoising using separate EB AMP per modality (no data integration).

    Inherits preprocessing and PCA fitting from MultimodalPCAPipeline, but builds
    per-modality EB models for both V and U and uses amp.ebamp_separate.
    """
    def __init__(self, n_components_list_v=None, n_components_list_u=None):
        super().__init__()
        self.n_components_list_v = n_components_list_v
        self.n_components_list_u = n_components_list_u

    def denoise_amp(self, X_list, X_low_dim_list=None, K_list=None,
                    amp_iters=10, muteu=False, mutev=False,
                    n_components_list_v=None, n_components_list_u=None, preprocess=False):
        n_components_list_v = n_components_list_v if n_components_list_v is not None else self.n_components_list_v
        n_components_list_u = n_components_list_u if n_components_list_u is not None else self.n_components_list_u
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
        n_components_list_v : list of int, optional
            Number of GMM components per modality for V direction.
        n_components_list_u : list of int, optional
            Number of GMM components per modality for U direction.
        
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

        # Reuse preprocessing and PCA fitting as in base class
        if preprocess:
            diagnostic_tool = preprocessing.MultiModalityPCADiagnostics()
            X_list = diagnostic_tool.normalize_obs(X_list, K_list)

        # PCA high-dim
        self.pca_model = pca_pack.MultiModalityPCA()
        self.pca_model.fit(X_list, K_list, plot_residual=False)
        # PCA low-dim
        if X_low_dim_list:
            self.pca_model_low_dim = LowDimModalityLoadings()
            self.pca_model_low_dim.fit(X_low_dim_list)

        # Build EB models for V (same as base)
        U_norm = extract_normalized_U(self.pca_model, X_list)
        V_norm = extract_normalized_V(self.pca_model, X_list)
        M_v = [np.diag(self.pca_model.pca_results[k].feature_aligns) for k in range(len(X_list))]
        S_v = [np.diag(1 - self.pca_model.pca_results[k].feature_aligns**2) for k in range(len(X_list))]
        factory_v = MultiModalParametricEB(n_components_list=n_components_list_v)
        self.eb_models_v = factory_v.fit(V_norm, [[M] for M in M_v], [[S] for S in S_v])

        if X_low_dim_list:
            U_u_list = U_norm + [ld.X for ld in self.pca_model_low_dim.pca_results.values()]
            M_u_blocks = (
                [np.diag(self.pca_model.pca_results[k].sample_aligns) for k in range(len(X_list))] +
                [ld.sample_aligns for ld in self.pca_model_low_dim.pca_results.values()]
            )
            S_u_blocks = (
                [np.diag(1 - self.pca_model.pca_results[k].sample_aligns**2) for k in range(len(X_list))] +
                [np.eye(ld.sample_aligns.shape[0]) for ld in self.pca_model_low_dim.pca_results.values()]
            )
        else:
            U_u_list = U_norm
            M_u_blocks = [np.diag(self.pca_model.pca_results[k].sample_aligns) for k in range(len(X_list))]
            S_u_blocks = [np.diag(1 - self.pca_model.pca_results[k].sample_aligns**2) for k in range(len(X_list))]

        factory_u = MultiModalParametricEB(n_components_list=n_components_list_u)
        self.eb_models_u = factory_u.fit(
            U_u_list,
            [[M] for M in M_u_blocks],
            [[S] for S in S_u_blocks]
        )

        # Run separate AMP
        self.amp_results = amp.ebamp_separate(
            self.pca_model, self.pca_model_low_dim if X_low_dim_list else None,
            self.eb_models_v, self.eb_models_u,
            amp_iters=amp_iters, muteu=muteu, mutev=mutev
        )
        return self.amp_results
    
    
def compute_svd_only(X_list, X_low_dim_list=None, pca_model_low_dim=None, K_list=None):
    """
    Compute raw SVD-based factors for high-dimensional modalities and simple low-dimensional factors.

    For each high-dim X in X_list:
      - Compute full SVD X = U S V^T.
      - Normalize U so that U^T U = n I and V so that V^T V = p I by scaling U *= sqrt(n), V *= sqrt(p).

    For each low-dim modality, returns L^{-1} X using the loading matrix L = sample_aligns from the low-dim PCA packs.

    Parameters
    ----------
    X_list : list of ndarray, shape (n, p_k)
        High-dimensional data matrices.
    X_low_dim_list : list of ndarray, shape (n, q_k), optional
        Low-dimensional data matrices (centered).
    pca_model_low_dim : LowDimModalityLoadings
        Fitted low-dim PCA object with pca_results containing sample_aligns.
    r_list : list of int
        Number of singular values/vectors to compute for each modality.

    Returns
    -------
    U_svd : list of ndarray
        List of normalized left factors U (n x r) for each high-dim modality.
    V_svd : list of ndarray
        List of normalized right factors V (p x r) for each high-dim modality.
    U_ld : list of ndarray
        List of low-dimensional factors computed as X_low_dim @ pinv(L) for each modality.
    """

    U_svd = []
    V_svd = []
    for X, r in zip(X_list, K_list):
        n, p = X.shape
        # truncated SVD
        U, S, Vt = svds(X, k=r)
        idx = np.argsort(S)[::-1]
        U, S, Vt = U[:, idx], S[idx], Vt[idx, :]
        # scale U and V
        U_svd.append(U * np.sqrt(n))
        V_svd.append(Vt.T * np.sqrt(p))

    # low-dim factors
    if X_low_dim_list is None:
        U_ld = []
    else:
        U_ld = []
        for X_ld, ld_pack in zip(X_low_dim_list, pca_model_low_dim.pca_results.values()):
            L = ld_pack.sample_aligns
            # invert L
            L_inv = np.linalg.pinv(L)
            U_low_dim_emb = (X_ld @ L_inv) 
            U_ld.append(U_low_dim_emb)

    return U_svd, V_svd, U_ld


Array = np.ndarray

class run_EBAMP_ZF:
    """
    Single-modality runner for the original EB-AMP (Zhong et al.).

    Enforces a single matrix Y (n x p) and a scalar rank K.
    Passing multiple modalities or a list rank raises ValueError.
    """

    def __init__(self):
        pass

    def run_amp_tz(
        self,
        Y: Array, # type: ignore
        K: int,
        amp_iters: int = 5,
        preprocess: bool = True,
    ) -> dict[str, Any]:
        """
        Parameters
        ----------
        Y : np.ndarray
            Data matrix of shape (n, p). Must NOT be a list/sequence.
        K : int
            Target rank (positive integer). Must NOT be a list/sequence.
        amp_iters : int
            Number of AMP iterations (default 5).
        preprocess : bool
            If True, apply tz normalize_obs before PCA (recommended).

        Returns
        -------
        dict with keys:
            - "U_denoised": (n, K)
            - "V_denoised": (p, K)
            - "U_pca": (n, K) or None
            - "V_pca": (p, K) or None
            - "rank", "n", "p"
        """
        # --- Strict single-modality checks ---
        if isinstance(Y, (list, tuple)):
            raise ValueError(
                f"run_EBAMP_ZF expects a single matrix ndarray; got a sequence of length {len(Y)}."
            )
        if not isinstance(Y, np.ndarray):
            raise TypeError(f"Y must be a NumPy ndarray; got {type(Y)}.")

        if isinstance(K, (list, tuple, np.ndarray)):
            raise ValueError("K must be a scalar integer for single-modality.")
        try:
            K = int(K)
        except Exception as e:
            raise ValueError(f"K must be castable to int; got {K!r}.") from e
        if K <= 0:
            raise ValueError("K must be positive.")

        if Y.ndim != 2:
            raise ValueError(f"Y must be 2D (n x p); got shape {Y.shape} with ndim={Y.ndim}.")

        # --- 1) Preprocess (tz) ---
        X = preprocessing_tz.normalize_obs(Y, K) if preprocess else Y

        # --- 2) PCA (tz) ---
        pcapack = pca_tz.get_pca(X, K)
        U_pca = getattr(pcapack, "U", None)     # (n, K) or None
        V_pca = getattr(pcapack, "V", None)     # (p, K) or None

        # --- 3) EB-AMP (tz) ---
        U_seq, V_seq = amp_tz.ebamp_gaussian(pcapack, amp_iters=amp_iters, warm_start=False)

        # Last iterate, robust to 2D/3D
        U_den = U_seq if U_seq.ndim == 2 else U_seq[:, :, -1]
        V_den = V_seq if V_seq.ndim == 2 else V_seq[:, :, -1]

        return {
            "U_denoised": U_den,
            "V_denoised": V_den,
            "U_pca": U_pca,
            "V_pca": V_pca,
            "rank": K,
            "n": Y.shape[0],
            "p": Y.shape[1],
        }