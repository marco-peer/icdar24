import torch
import torch.nn as nn

class GMP(nn.Module):
    """ Generalized Max Pooling
    """
    def __init__(self, lamb):
        super().__init__()
        self.lamb = nn.Parameter(lamb * torch.ones(1))
        #self.inv_lamb = nn.Parameter((1./lamb) * torch.ones(1))

    def forward(self, x):
        B, D, _ = x.shape
        identity = torch.eye(D).cuda()
        # reshape x, s.t. we can use the gmp formulation as a global pooling operation
        # x = x.view(B, D, N)
        # x = x.permute(0, 2, 1)
        # compute the linear kernel
        K = torch.bmm(x, x.permute(0, 2, 1))
        # solve the linear system (K + lambda * I) * alpha = ones
        A = K + self.lamb * identity
        o = torch.ones(B, D, 1).cuda()
        #alphas, _ = torch.gesv(o, A) # tested using pytorch 1.0.1
        alphas = torch.linalg.solve(A, o) # tested using pytorch 1.2.0
        alphas = alphas.view(B, 1, -1)        
        xi = torch.bmm(alphas, x)
        xi = xi.view(B, -1)
        # L2 normalization
        # xi = nn.functional.normalize(xi)
        return xi

if __name__ == '__main__':
    gmp = GMP(1e3).cuda()
    print(gmp(torch.rand(1, 65, 384).cuda()).shape)
