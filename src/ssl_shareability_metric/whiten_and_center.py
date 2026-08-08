import numpy as np

def whiten_and_center(vector):
        vector_mean = np.mean(vector, axis=0)
        
        vector_covariance = np.cov(vector, rowvar=False)
        
        vector_covariance_eigenvalues, vector_covariance_eigenvectors = np.linalg.eigh(vector_covariance)
        
        vector_inv_sqrt_cov = vector_covariance_eigenvectors @ np.diag(1 / np.sqrt(vector_covariance_eigenvalues)) @ vector_covariance_eigenvectors.T

        whitened_and_centered_vector = (vector - vector_mean) @ vector_inv_sqrt_cov
        
        return whitened_and_centered_vector