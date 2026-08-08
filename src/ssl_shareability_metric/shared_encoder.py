import torch

class SharedEncoder(torch.nn.Module):
    def __init__(self, vector_size):
        super().__init__()
        self.shared = torch.nn.Linear(vector_size, 1, bias=False)
        
    def __call__(self, x):
        return self.shared(x)