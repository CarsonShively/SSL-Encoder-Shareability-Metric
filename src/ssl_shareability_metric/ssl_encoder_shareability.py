import numpy as np

def ssl_shareability(current, future):
        current = np.array(current)
        future = np.array(future)
        
        if current.shape != future.shape:
            raise ValueError("Shape mismatch")

        current_mean = np.mean(current, axis=0)
        future_mean = np.mean(future, axis=0)
        
        current_covariance = np.cov(current, rowvar=False)
        future_covariance = np.cov(future, rowvar=False)
        
        current_covariance_eigenvalues, current_covariance_eigenvectors = np.linalg.eigh(current_covariance)
        future_covariance_eigenvalues, future_covariance_eigenvectors = np.linalg.eigh(future_covariance)
        
        current_inv_sqrt_cov = current_covariance_eigenvectors @ np.diag(1 / np.sqrt(current_covariance_eigenvalues)) @ current_covariance_eigenvectors.T
        future_inv_sqrt_cov = future_covariance_eigenvectors @ np.diag(1 / np.sqrt(future_covariance_eigenvalues)) @ future_covariance_eigenvectors.T
        
        whitened_and_centered_current = (current - current_mean) @ current_inv_sqrt_cov
        whitened_and_centered_future = (future - future_mean) @ future_inv_sqrt_cov
        
        M_cross_covariance = (whitened_and_centered_future.T @ whitened_and_centered_current) / whitened_and_centered_current.shape[0]
        
        S_symmetric_cross_covariance = (M_cross_covariance + M_cross_covariance.T) / 2
        
        symmetric_eigenvalues = np.linalg.eigvalsh(S_symmetric_cross_covariance)
        
        J_shared = np.max(np.abs(symmetric_eigenvalues))
        
        J_sep = np.linalg.svd(M_cross_covariance, compute_uv=False)[0]
        
        P = J_shared / J_sep
        
        return P, J_shared, J_sep