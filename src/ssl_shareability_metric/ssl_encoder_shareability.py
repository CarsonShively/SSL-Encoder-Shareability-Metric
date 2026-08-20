import numpy as np

def ssl_encoder_shareability(observation_1, observation_2):
        observation_1 = np.array(observation_1)
        observation_2 = np.array(observation_2)
        
        if observation_1.shape != observation_2.shape:
            raise ValueError("Shape mismatch")
        
        M = (observation_2.T @ observation_1) / observation_1.shape[0]
        
        S = (M + M.T) / 2
        
        S_eigenvalues = np.linalg.eigvalsh(S)
        
        J_shared = np.max(np.abs(S_eigenvalues))
        
        J_sep = np.linalg.svd(M, compute_uv=False)[0]
        
        P = J_shared / J_sep
        
        return P