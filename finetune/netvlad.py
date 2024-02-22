
import torch
from torch import nn
import torch.nn.functional as F

class Wrapper(nn.Module):

    def __init__(self, backbone, args):
        super(Wrapper, self).__init__()
        self.backbone = backbone
        self.args = args
        embed_dim = self.embed_dim = args.get('model_options', {}).get('in_dim', -1)
        self.embed_fc = nn.Linear(backbone.num_features, embed_dim, bias=False) if embed_dim != -1 else nn.Identity()
        self.dropout = nn.Dropout(0.1) if embed_dim != -1 else nn.Identity()

        args['netvlad']['dim'] = backbone.num_features if embed_dim == -1 else embed_dim
        print(args['netvlad'])
        self.nv = NetVLAD(**args['netvlad'])

    def forward(self, x):

        embs = self.embed_fc(self.dropout(self.forward_features(x)))

        if self.args.get('netvlad_pooling', False):
            embs = self.backbone.unpatchify(embs)
        else:
            embs = embs.unsqueeze(-1).unsqueeze(-1)

        
        feats = self.nv(embs)
        return F.normalize(feats)
    
    def forward_features(self, x):
        if getattr(self.backbone, 'extract_features', None):
            return self.backbone.extract_features(x)
        else:
            return self.backbone(x)


class NetVLAD(nn.Module):
    """Net(R)VLAD layer implementation"""

    def __init__(self, num_clusters=100, dim=64, alpha=100.0, random=False, normalize=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            random : bool
                enables NetRVLAD, removes alpha-init and normalization

        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.random = random
        self.normalize = normalize
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)

        centroids = torch.rand(num_clusters, dim) 
        self.centroids = nn.Parameter(F.normalize(centroids))
        self._init_params()

    def _init_params(self, clusters=None):
        if clusters is not None:
            assert clusters.shape[0] == self.num_clusters
            self.centroids = nn.Parameter(torch.concat([clusters]))

        if not self.random:
            self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)
            )

    def get_assignments(self, x):
        N, C = x.shape[:2]
        if not self.random or self.normalize:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim
        
        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        return soft_assign
    
    def calc_residuals(self, x, soft_assign):
        N, C = x.shape[:2]

        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        
        residual *= soft_assign.unsqueeze(2)
        return residual
    
    def build_vlad(self, residual):
        vlad = residual.sum(dim=-1)

        if not self.random:
            vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization

        vlad = vlad.view(residual.size(0), -1)  # flatten

        if not self.random:
            vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
            
        return vlad
    
    def forward(self, x):
        soft_assign = self.get_assignments(x)
        residual = self.calc_residuals(x, soft_assign)
        vlad = self.build_vlad(residual)
        return vlad
    