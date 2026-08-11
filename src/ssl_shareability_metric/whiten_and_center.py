import numpy as np

class WhitenAndCenter:
        def __init__(self):
                self.mean = None
                self.inv_sqrt_cov = None

        def fit(self, x):
                self.mean = np.mean(x, axis=0)
                covariance = np.cov(x, rowvar=False)
                covariance_eigenvalues, covariance_eigenvectors = np.linalg.eigh(covariance)
                covariance_eigenvalues = np.clip(covariance_eigenvalues, 1e-8, None)
                self.inv_sqrt_cov = covariance_eigenvectors @ np.diag(1 / np.sqrt(covariance_eigenvalues)) @ covariance_eigenvectors.T
                return self

        def transform(self, x):
                return (x - self.mean) @ self.inv_sqrt_cov
        
        def fit_transform(self, x):
                return self.fit(x).transform(x)