from abc import ABC, abstractmethod
import warnings

import numpy as np
import scipy as sp
from sklearn.mixture import GaussianMixture
import seaborn as sns
from rpy2.robjects.packages import importr

# === Utility Functions ===

def near_psd(A, epsilon=1e-4):
    """
    Project A onto the PSD cone by clamping its eigenvalues.

    Parameters
    ----------
    A : (n, n) array, symmetric (or nearly so)
    epsilon : float
        Smallest allowed eigenvalue (default 1e-8).

    Returns
    -------
    A_psd : (n, n) array
        The nearest PSD matrix (in Frobenius norm) with eigenvalues >= epsilon.
    """
    # 1) Make symmetric (in case of numerical asymmetry)
    A = (A + A.T) / 2

    # 2) Eigen-decompose
    eigvals, eigvecs = np.linalg.eigh(A)

    # 3) Clamp eigenvalues
    eigvals[eigvals < epsilon] = epsilon

    # 4) Reconstruct
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def inv_reg(matrix, epsilon=1e-3):
    """
    Invert a matrix with regularization to avoid singularity.

    Parameters:
    - matrix (np.ndarray): Matrix to invert.
    - epsilon (float): Regularization strength. Default is 1e-3.

    Returns:
    - np.ndarray: Regularized inverse of the matrix.
    """
    regularized = matrix + epsilon * np.eye(matrix.shape[0])
    return np.linalg.inv(regularized)


def replace_zeros(array, small_value=1e-30):
    """
    Replace zero entries in a NumPy array with a small nonzero value.

    Parameters:
    - array (np.ndarray): Input array.
    - small_value (float): Value to replace zeros with. Default is 1e-30.

    Returns:
    - np.ndarray: Modified array with zeros replaced.
    """
    array = np.copy(array)  # avoid modifying the original array
    array[array == 0] = small_value
    return array

# === Abstract Base Class for Empirical Bayes ===
class _BaseEmpiricalBayes(ABC):
    """
    Abstract base class for empirical Bayes estimators.
    """
    pass

class ParametricEB(_BaseEmpiricalBayes):
    """
    Parametric empirical Bayes via a Gaussian Mixture Model (GMM).

    Methods
    -------
    estimate_prior(data, M_list, D_list):
        Fit a GMM on data, adjusting means and covariances via block–diagonal M/D.
    denoise(data):
        Compute posterior means in observed space.
    ddenoise(data):
        Compute derivative of the denoising map ∂μ̂(x)/∂x, shape (N, P, P).
    """

    def __init__(self, n_components=5, max_iter=500, choose_comp=False):
        super().__init__()
        self.n_components = n_components
        self.max_iter     = max_iter
        self.choose_comp  = choose_comp

        # These get filled in estimate_prior:
        self.m_prior   = None   # (K, P)
        self.cov_prior = None   # (K, P, P)
        self.weights   = None   # (K,)

        # Store input blocks and their block-diagonals:
        self.M_list = None      # list of (p_i × p_i)
        self.D_list = None      # list of (p_i × p_i)
        self.M_bd   = None      # (P × P), P = sum p_i
        self.D_bd   = None      # (P × P)

    def estimate_prior(self, data, M_list, D_list):
        """
        Fit the GMM and define the empirical‐Bayes prior in observed space.

        Parameters
        ----------
        data : ndarray, shape (N, P)
            Observed concatenated modalities.
        M_list : list of ndarrays
            Each block loading matrix (p_i × p_i).
        D_list : list of ndarrays
            Each noise covariance (p_i × p_i).

        Returns
        -------
        m_prior   : (K, P)
        cov_prior : (K, P, P)
        weights   : (K,)
        """
        # 1) Build and store block-diagonals
        self.M_list = M_list
        self.D_list = D_list
        self.M_bd   = sp.linalg.block_diag(*M_list)   # → (P, P)
        self.D_bd   = sp.linalg.block_diag(*D_list)   # → (P, P)

        # 2) Fit GMM in observed space (optionally select by BIC)
        if self.choose_comp:
            print(f"Choosing number of mixing components by BIC", flush=True)
            best_bic = np.inf
            gmm = None
            for k in range(1, self.n_components + 1):
                cand = GaussianMixture(n_components=k, max_iter=self.max_iter)
                cand.fit(data)
                bic = cand.bic(data)
                if bic < best_bic:
                    best_bic = bic
                    gmm = cand
            print(f"Selected {gmm.n_components} components", flush=True)
        else:
            gmm = GaussianMixture(n_components=self.n_components, max_iter=self.max_iter)
            gmm.fit(data)

        # 3) Adjust means into the prior
        inv_M = np.linalg.inv(self.M_bd)              # (P, P)
        means_obs = gmm.means_                        # (K, P)
        self.m_prior = means_obs @ inv_M              # (K, P)

        # 4) Adjust covariances into isotropic prior
        cov_obs = gmm.covariances_                    # (K, P, P)
        cov_adj = cov_obs - self.D_bd                 # remove noise
        traces  = np.maximum([np.trace(c) for c in cov_adj], 0)
        factor  = np.trace(self.M_bd @ self.M_bd.T)
        self.cov_prior = np.array([
            (t / factor) * np.eye(self.M_bd.shape[0])
            for t in traces
        ])                                            # (K, P, P)

        # 5) Store weights
        self.weights = gmm.weights_                   # (K,)

        return self.m_prior, self.cov_prior, self.weights

    def _get_W(self, data, mu, cov):
        """
        Compute responsibilities W[i,k] = P(Z=k | x_i) in observed space.

        Returns
        -------
        W : ndarray, shape (N, K)
        """
        M_bd_local = sp.linalg.block_diag(*mu) if isinstance(mu, list) else mu
        D_bd_local = sp.linalg.block_diag(*cov) if isinstance(cov, list) else cov

        K, N = len(self.cov_prior), data.shape[0]

        # accumulate log-weights
        logits = np.zeros((N, K))
        for k in range(K):
            # build component covariance in observed space
            Sigma_k = D_bd_local + M_bd_local @ self.cov_prior[k] @ M_bd_local.T
            inv_S   = np.linalg.inv(Sigma_k)

            # --- NEW: log determinant term (constant per component) ---
            _, logdet = np.linalg.slogdet(Sigma_k)

            # quadratic forms
            fsq = np.einsum("ij,ij->i", data @ inv_S, data) / 2
            mz  = self.m_prior[k] @ M_bd_local.T               # (P,)
            zsq = (mz @ inv_S @ mz) / 2
            fz  = data @ inv_S @ mz                      # (N,)

            logits[:, k] = np.log(self.weights[k]) - fsq + fz - zsq - 0.5 * logdet 

        # stable softmax
        mx  = logits.max(axis=1, keepdims=True)
        mx[~np.isfinite(mx)] = -700
        diff = logits - mx
        diff = np.clip(diff, -700, 700)
        W   = np.exp(diff)
        W   = W / W.sum(axis=1, keepdims=True)
        return W                                       # (N, K)

    def denoise(self, data, mu, cov):
        """
        Posterior mean μ̂(x_i) in observed space.

        Returns
        -------
        X_hat : ndarray, shape (N, R)
        """
        M_bd_local = sp.linalg.block_diag(*mu) if isinstance(mu, list) else mu
        D_bd_local = sp.linalg.block_diag(*cov) if isinstance(cov, list) else cov

        W    = self._get_W(data, mu, cov)       # (N, K)
        K, N = len(self.cov_prior), data.shape[0]
        P    = M_bd_local.shape[0]

        inv_D = np.linalg.inv(D_bd_local)           # (P, P)
        inv_C = [np.linalg.pinv(c) for c in self.cov_prior]

        result = np.zeros((N, P))
        for k in range(K):
            # posterior precision in latent space:
            A = M_bd_local.T @ inv_D @ M_bd_local + inv_C[k]
            prec = np.linalg.inv(A)                # (P, P)

            # two contributions
            part_data  = data @ inv_D.T @ M_bd_local @ prec.T
            part_prior = (self.m_prior[k] @ inv_C[k].T) @ prec.T
            part_prior = np.broadcast_to(part_prior, part_data.shape)

            result += W[:, k:k+1] * (part_data + part_prior)

        return result                               # (N, R)

    def ddenoise(self, data, mu, cov):
        """
        Derivative ∂μ̂(x_i)/∂x_i, shape (N, R, R).

        Returns
        -------
        J : ndarray, shape (N, R, R)
        """
        M_bd_local = sp.linalg.block_diag(*mu) if isinstance(mu, list) else mu
        D_bd_local = sp.linalg.block_diag(*cov) if isinstance(cov, list) else cov

        W    = self._get_W(data, mu, cov)       # (N, K)
        K, N = len(self.cov_prior), data.shape[0]
        P    = M_bd_local.shape[0]

        inv_D = np.linalg.inv(D_bd_local)
        inv_C = [np.linalg.pinv(c) for c in self.cov_prior]

        # temporary storage
        mat1 = np.zeros((K, N, P))
        mat2 = np.zeros((K, N, P))
        mat3 = np.zeros((K, P, P))

        # build component pieces
        for k in range(K):
            A     = M_bd_local.T @ inv_D @ M_bd_local + inv_C[k]
            prec  = np.linalg.inv(A)                   # (R, R)
            Sigma_k = D_bd_local + M_bd_local @ self.cov_prior[k] @ M_bd_local.T
            adj   = np.linalg.inv(Sigma_k)             # (R, R)

            mat1[k] = (data @ inv_D @ M_bd_local + self.m_prior[k] @ inv_C[k]) @ prec.T
            mat2[k] = (data - self.m_prior[k] @ M_bd_local.T) @ adj.T
            mat3[k] = prec @ M_bd_local.T @ inv_D

        # combine
        J = np.zeros((N, P, P))
        # diagonal terms
        for k in range(K):
            outer = np.einsum("ij,ik->ijk", mat1[k], mat2[k]) - mat3[k]
            w = W[:, k:k+1].flatten()  # shape (N,1,1)
            J = J - w[:, np.newaxis, np.newaxis] * outer

        # cross terms
        for k1 in range(K):
            for k2 in range(K):
                outer = np.einsum("ij,ik->ijk", mat1[k1], mat2[k2])
                w12 = (W[:, k1] * W[:, k2]).flatten()
                J = J + w12[:, np.newaxis, np.newaxis] * outer
        return J                                     # (N, R, R)

    def get_denoisers(self):
        """
        Return fitted prior parameters and denoising functions for use in other scripts.

        Returns
        -------
        eb_info : dict
            Dictionary with the following keys:
              - 'prior': tuple (m_prior, cov_prior, weights)
              - 'denoise': callable function (data, mu, cov) -> denoised output
              - 'ddenoise': callable function (data, mu, cov) -> derivative tensor
        """
        return {
            "prior": (self.m_prior, self.cov_prior, self.weights),
            "denoise": lambda data, mu, cov: self.denoise(data, mu, cov),
            "ddenoise": lambda data, mu, cov: self.ddenoise(data, mu, cov),
        }


# === Multi-modal Parametric Empirical Bayes ===
class MultiModalParametricEB:
    """
    Fits a separate ParametricEB model to each modality, allowing
    different GMM component counts per modality.
    """

    def __init__(self, n_components=5, choose_comp=False, max_iter=500, n_components_list=None):
        """
        Parameters
        ----------
        n_components : int
            Default number of GMM components (used if n_components_list is None).
        max_iter : int
            Max EM iterations for GMM.
        n_components_list : list of int, length m (optional)
            If provided, modality k uses n_components_list[k] components.
        """
        self.n_components = n_components
        self.max_iter     = max_iter
        self.n_components_list = n_components_list
        self.eb_models_v  = {}
        self.choose_comp  = choose_comp

    def fit(self, data_list, M_list_list, D_list_list):
        """
        Fit ParametricEB for each modality, using per-modality component counts.

        Returns
        -------
        eb_models_v : dict
            Mapping modality index k → dict with keys:
              - 'prior': (m_prior, cov_prior, weights)
              - 'denoise': callable(data, mu, cov)
              - 'ddenoise': callable(data, mu, cov)
        """
        self.eb_models_v = {}
        for k, (data, M_list, D_list) in enumerate(zip(data_list, M_list_list, D_list_list)):
            # choose components for this modality
            comps = (self.n_components_list[k]
                     if self.n_components_list is not None
                     else self.n_components)

            eb = ParametricEB(n_components=comps, max_iter=self.max_iter, choose_comp=self.choose_comp)
            m_prior, cov_prior, weights = eb.estimate_prior(data, M_list, D_list)
            self.eb_models_v[k] = {
                'prior':   (m_prior, cov_prior, weights),
                "denoise":   (lambda eb: (lambda data, mu, cov, _eb=eb: _eb.denoise(data, mu, cov)))(eb),
                "ddenoise": (lambda eb: (lambda data, mu, cov, _eb=eb: _eb.ddenoise(data, mu, cov)))(eb)
            }

        return self.eb_models_v