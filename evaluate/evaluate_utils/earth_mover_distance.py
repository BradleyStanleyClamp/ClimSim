"""
Script containing code for calculating the Earth Mover Distance between two probability distributions.
"""

import logging
import numpy as np
from sklearn.decomposition import PCA
import time
import torch
from typing import Optional, Tuple
import math
from typing import Union, Optional
from scipy.stats import wasserstein_distance_nd
from .kl_divergence import pca_gpu


class EMDMetric:
    def __init__(
        self,
        n_components: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Earth Mover Distance (EMD) estimate between two distributions on (optionally PCA-reduced) data.

        Args:
            n_components: if not None and < data dim, perform PCA to this many components.
            device: torch.device or None (use CUDA if available).
        """
        self.n_components = n_components
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def __call__(self, X_tensor: torch.Tensor, Y_tensor: torch.Tensor) -> float:
        """
        Returns the EMD between two datasets X and Y.
        If multivariate and n_components is smaller than data dimension, PCA is applied first.
        """
        device = self.device
        X = X_tensor.to(device=device, dtype=torch.float32)
        Y = Y_tensor.to(device=device, dtype=torch.float32)

        # Optionally PCA-reduce
        if (
            self.n_components is not None
            and self.n_components < X.shape[1]
            and X.shape[1] > 1
        ):
            start_time = time.perf_counter()
            Xp, Yp = pca_gpu(X, Y, n_components=self.n_components, device=device)
            X = Xp
            Y = Yp
            end_time = time.perf_counter()
            # logging.info(f"PCA reduction took {end_time - start_time:.4f} seconds")

        start_time = time.perf_counter()
        emd = wasserstein_distance_nd(X.cpu().numpy(), Y.cpu().numpy())
        end_time = time.perf_counter()
        # logging.info(
        #     f"KL divergence calculation took {end_time - start_time:.4f} seconds"
        # )

        return emd
