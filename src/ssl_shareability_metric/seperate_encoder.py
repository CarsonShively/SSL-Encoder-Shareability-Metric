import torch

class SeperateEncoder(torch.nn.Module):
    def __init__(self, vector_size):
        super().__init__()
        
        self.current = torch.nn.Linear(vector_size, 1, bias=False)
        self.future = torch.nn.Linear(vector_size, 1, bias=False)
        
    def forward(self, x, y):
        return self.current(x), self.future(y)