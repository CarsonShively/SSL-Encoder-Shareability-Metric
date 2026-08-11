import numpy as np

def ssl_shareability(current, future):
        current = np.array(current)
        future = np.array(future)
        
        if current.shape != future.shape:
            raise ValueError("Shape mismatch")
        
        M_cross_covariance = (future.T @ current) / current.shape[0]
        
        S_symmetric_cross_covariance = (M_cross_covariance + M_cross_covariance.T) / 2
        
        symmetric_eigenvalues = np.linalg.eigvalsh(S_symmetric_cross_covariance)
        
        J_shared = np.max(np.abs(symmetric_eigenvalues))
        
        J_sep = np.linalg.svd(M_cross_covariance, compute_uv=False)[0]
        
        P = J_shared / J_sep
        
        return P, J_shared, J_sep