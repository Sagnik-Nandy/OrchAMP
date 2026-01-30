import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import importlib
from numpy.linalg import svd

# Define named tuple for PCA results
PcaPack = namedtuple("PCAPack", ["X", "U", "V", "mu", "K", 
    "n_samples", "n_features", "signals", "sample_aligns", "feature_aligns"])


class MultiModalityPCA:
    """Performs PCA on multiple modality matrices and analyzes residual spectra."""

    def __init__(self):
        self.pca_results = {}

    def fit(self, X_list, K_list, plot_residual=False):
        """
        Perform PCA for multiple modality matrices.

        Parameters
        ----------
        X_list : list of ndarray
            List of m data matrices X_k of shape (n_samples, p_k).
        K_list : list of int
            List of m values specifying the number of principal components per modality.
        plot_residual : bool, optional
            Whether to plot the residual spectrum (default is False).
        """
        if len(X_list) != len(K_list):
            raise ValueError("Mismatch: Number of modalities and K values must be the same.")

        for k, (X_k, K_k) in enumerate(zip(X_list, K_list)):
            #print(f"Running PCA for Modality {k}: Shape {X_k.shape}, K={K_k}")
            self.pca_results[k] = self._get_pca(X_k, K_k)

            if plot_residual:
                self._plot_residual_spectrum(k)

    def _get_pca(self, X, K, update_signal=True):
        """
        Perform PCA on a given modality.

        Parameters
        ----------
        X : ndarray (n_samples, n_features)
            Data matrix.
        K : int
            Number of principal components.
        update_signal : bool, optional
            Whether to estimate the signal strength using the Marčenko-Pastur correction.

        Returns
        -------
        PcaPack : namedtuple
            Contains PCA components, singular values, and alignments.
        """
        if K <= 0:
            raise ValueError("K must be greater than zero.")

        n_samples, n_features = X.shape
        U_full, Lambdas_full, Vh_full = svd(X, full_matrices=False)
        U = U_full[:, :K]
        Lambdas = Lambdas_full[:K]
        Vh = Vh_full[:K, :]

        # Extract top-K components
        U_K, Lambda_K, V_K = U, np.diag(Lambdas), Vh

        # Compute residual singular values
        X_residual = X - U_K @ Lambda_K @ V_K
        _, residual_singular_values, _ = np.linalg.svd(X_residual, full_matrices=False)

        aspect_ratio = n_features / n_samples

        # Compute estimated signal strength
        greek_lambda = Lambdas / np.sqrt(aspect_ratio)

        if update_signal:
            singval_threshold = 1 + np.sqrt(aspect_ratio)
            if np.min(Lambdas) < singval_threshold:
                print("Warning: Signal does not separate from the bulk. Proceeding without updating greek_lambda.", flush=True)
            else:
                greek_lambda = np.sqrt(
                    (greek_lambda**2 * aspect_ratio - 1 - aspect_ratio + 
                     np.sqrt((greek_lambda**2 * aspect_ratio - 1 - aspect_ratio)**2 - 4 * aspect_ratio))
                    / (2 * aspect_ratio)
                )
                print([greek_lambda, 1 + np.sqrt(aspect_ratio)], flush=True)

        # Compute alignments
        sample_align = np.sqrt(np.maximum(1 - (1 + greek_lambda**2) / 
                                          (greek_lambda**2 * (aspect_ratio * greek_lambda**2 + 1)), 0))
        feature_align = np.sqrt(np.maximum(1 - (1 + aspect_ratio * greek_lambda**2) / 
                                           (aspect_ratio * greek_lambda**2 * (greek_lambda**2 + 1)), 0))

        return PcaPack(X=X, U=U_K, V=V_K.T, mu=residual_singular_values, 
                       n_samples=n_samples, n_features=n_features, K=K,
                       signals=greek_lambda, sample_aligns=sample_align, feature_aligns=feature_align)

    def check_residual_spectrum(self, show=False, save=False, save_path="./residual_spectrum.pdf"):
        """
        Plot residual singular values and compare with the Marčenko-Pastur (MP) law.

        Parameters
        ----------
        show : bool
            Display the plot.
        save : bool
            Save the plot.
        save_path : str
            File path to save.
        """
        for k in self.pca_results.keys():
            self._plot_residual_spectrum(k, show, save, save_path)

    def _plot_residual_spectrum(self, modality_idx, show=False, save=False, save_path="./residual_spectrum.pdf"):
        """
        Helper function to plot residual spectrum for a single modality.
        """
        pca_pack = self.pca_results[modality_idx]
        mu, n_samples, n_features = pca_pack.mu, pca_pack.n_samples, pca_pack.n_features

        fig, ax = plt.subplots()
        ax.hist(mu[:min(n_samples, n_features)], density=True, bins=50, label="Sample singular values")
        
        aspect_ratio = n_features / n_samples
        x_vals = np.linspace(mu.min() - 0.1, mu.max(), num=50)
        ax.plot(x_vals, aspect_ratio * np.array(self._sqrtMPlaw(x_vals, n_samples, n_features)), label="MP law prediction")

        ax.legend()
        ax.set_title(f"Residual Singular Values (Modality {modality_idx})")

        if save:
            fig.savefig(save_path.replace(".pdf", f"_modality{modality_idx}.pdf"))
        if show:
            plt.show()
        else:
            plt.close(fig)

    def generate_synthetic_data(self, n_samples, p_list, K_list):
        """
        Generate synthetic data for multiple modalities.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        p_list : list of int
            List of feature dimensions for each modality.
        K_list : list of int
            List of principal component counts per modality.

        Returns
        -------
        X_list : list of ndarray
            Generated data matrices.
        """
        X_list = []
        for k, p_k in enumerate(p_list):
            aspect_ratio = p_k / n_samples
            lambda_threshold = aspect_ratio ** (-1/4)

            U_k, V_k = np.random.randn(n_samples, K_list[k]), np.random.randn(p_k, K_list[k])
            D_k = np.diag(lambda_threshold + 10.0 + np.random.rand(K_list[k]) * 4)

            X_k = (1 / n_samples) * U_k @ D_k @ V_k.T + np.random.randn(n_samples, p_k) / np.sqrt(n_samples)
            X_list.append(X_k)

        return X_list

    def _sqrtmplaw(self, x, n, p):
        aspect_ratio = p / n
        lambda_plus = (1 + np.sqrt(aspect_ratio))**2
        lambda_minus = (1 - np.sqrt(aspect_ratio))**2
        if x**2 < lambda_minus or x**2 > lambda_plus:
            return 0
        return 1 / (np.pi * aspect_ratio * x) * np.sqrt((lambda_plus - x**2) * (x**2 - lambda_minus))

    def _sqrtMPlaw(self, arr, n, p):
        return [self._sqrtmplaw(x, n, p) for x in arr]