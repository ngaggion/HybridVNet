import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ChebConv, Pool, residualBlock3D, residualBlock
import numpy as np

class Encoder3D(nn.Module):
    def __init__(self, latents = 64, h = 224, w = 224, slices = 20):
        super(Encoder3D, self).__init__()
               
        self.residual1 = residualBlock3D(in_channels=1, out_channels=8)
        self.residual2 = residualBlock3D(in_channels=8, out_channels=16)
        self.residual3 = residualBlock3D(in_channels=16, out_channels=32)
        self.residual4 = residualBlock3D(in_channels=32, out_channels=64)
        self.residual5 = residualBlock3D(in_channels=64, out_channels=128)
        self.residual6 = residualBlock3D(in_channels=128, out_channels=128)

        # Input shape is slices x h x w
        self.maxpool = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.maxpool_2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        h2 = h  // 32
        w2 = w  // 32
        s2 = slices  // 2

        self.mu = nn.Linear(128*h2*w2*s2, latents)
        self.sigma = nn.Linear(128*h2*w2*s2, latents)

    def forward(self, x):

        x = self.residual1(x)
        x = self.maxpool(x)
        x = self.residual2(x)
        x = self.maxpool(x)
        l3 = self.residual3(x)
        x = self.maxpool_2(l3)
        l4 = self.residual4(x)
        x = self.maxpool(l4)
        l5 = self.residual5(x)
        x = self.maxpool(l5)
        l6 = self.residual6(x)

        x = x.view(l6.size(0), -1)
        mu = self.mu(x)
        sigma = self.sigma(x)

        return mu, sigma, [l6, l5, l4, l3]


class GConv(nn.Module):
    def __init__(self, in_channels, out_channels, K = 6):
        super(GConv, self).__init__()

        self.gConv= ChebConv(in_channels, out_channels, K)
        self.norm = torch.nn.InstanceNorm1d(out_channels)

    def forward(self, x, adj_indices):
        x = self.gConv(x, adj_indices)
        x = self.norm(x)
        x = F.relu(x)

        return x


class IGSC_3D(nn.Module):
    def __init__(self, in_filters, in_conv_channels = 128, device = None, do_skip = True, grad_prob = 0.5):
        super(IGSC_3D, self).__init__()

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.grad_prob = grad_prob
        self.GConv = ChebConv(in_filters, 3, 1, bias = False) 
        self.do_skip = do_skip
    
    def lookup(self, positions, conv_layer):
        # Batch size
        B = positions.shape[0] # equivalent to volume.shape[0]
        # Number of nodes of the graph
        N = positions.shape[1]
        # Number of channels of the input image tensor
        C = conv_layer.shape[1]
        
        # Updates positions from (0, 1) interval to (-1, 1) interval
        positions = 2 * positions - 1
        
        # Creates a grid for the lookup
        # Uses the Z dimension for saving the different node positions
        # Grid shape is [BATCH, Z, desired height, desired width, positions]
        grid = torch.zeros(B, N, 1, 1, 3).float().to(self.device)

        for node in range(N):
            for batch in range(B):
                grid[batch, node, 0, 0, 0] = positions[batch, node, 0]
                grid[batch, node, 0, 0, 1] = positions[batch, node, 1]
                grid[batch, node, 0, 0, 2] = positions[batch, node, 2]

        # Performs the lookup, the output has shape [BATCH, C, N, 1, 1]
        out = F.grid_sample(conv_layer, grid, mode = 'bilinear', padding_mode = 'zeros', align_corners = True).squeeze(3).squeeze(3)
        
        return out.permute(0, 2, 1)
    

    def forward(self, x, adj, conv_layer):
        positions = self.GConv(x, adj)
        if self.do_skip:
            coin = np.random.uniform(0, 1)
            if coin < self.grad_prob:
                skip = self.lookup(positions, conv_layer)
            else:
                with torch.no_grad():
                    skip = self.lookup(positions, conv_layer)
            x = torch.cat([x, skip, positions], dim = 2)
        else:
            x = torch.cat([x, positions], dim = 2)

        return x, positions
    

class HybridGNet3D(nn.Module):
    def __init__(self, config, downsample_matrices, upsample_matrices, adjacency_matrices, do_skip = None):
        super(HybridGNet3D, self).__init__()
        
        self.device = config['device']
        self.latents = config['latents3D']
        self.latents2D = config['latents2D']
        self.encoder = Encoder3D(latents = self.latents, h = config['h'], w = config['w'], slices = config['slices'])

        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices
        self.kld_weight = config['kld_weight']
        self.grad_prob = config['grad_prob']
                
        n_nodes = config['n_nodes']

        # Filters of Graph convolutional layers
        self.filters = config['filters'] 
        
        self.K = config['K'] # orden del polinomio
        
        # Fully connected decoder
        self.linear_decoder = torch.nn.Linear(self.latents, self.filters[5] * n_nodes[-1])
        torch.nn.init.normal_(self.linear_decoder.weight, 0, 0.1) 

        # Graph convolutional decoder
        self.unpool = Pool()

        if do_skip is None:
            do_skip = [False, False, False, False]
        elif len(do_skip) == 1:
            do_skip = [do_skip, do_skip, do_skip, do_skip]
        elif len(do_skip) != 4:
            raise ValueError('do_skip must be either None, a boolean or a list of 4 booleans')
        
        self.GC1 = GConv(self.filters[5], self.filters[4], self.K)
        self.IGSC_1 = IGSC_3D(self.filters[4], in_conv_channels = 128, device=self.device, do_skip = do_skip[0], grad_prob = self.grad_prob)

        self.GC2 = GConv(self.filters[4] + 3 + 128 * do_skip[0], self.filters[3], self.K)
        self.IGSC_2 = IGSC_3D(self.filters[3], in_conv_channels = 128, device=self.device, do_skip = do_skip[1], grad_prob = self.grad_prob)

        self.GC3 = GConv(self.filters[3] + 3 + 128 * do_skip[1], self.filters[2], self.K)
        self.IGSC_3 = IGSC_3D(self.filters[2], in_conv_channels = 64, device=self.device, do_skip = do_skip[2], grad_prob = self.grad_prob)

        self.GC4 = GConv(self.filters[2] + 3 + 64 * do_skip[2], self.filters[1], self.K)
        self.IGSC_4 = IGSC_3D(self.filters[1], in_conv_channels = 32, device=self.device, do_skip = do_skip[3], grad_prob = self.grad_prob)

        self.GC5 = GConv(self.filters[1] + 3 + 32 * do_skip[3], self.filters[1], self.K)
        self.GCout = ChebConv(self.filters[1], self.filters[0], self.K, bias = False) # Out layer: No bias, normalization nor relu
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) 

    def forward(self, sax):
        self.mu, self.log_var, layers = self.encoder(sax)

        if self.training:
            z = self.sampling(self.mu, self.log_var)
        else:
            z = self.mu

        x = self.linear_decoder(z)
        x = F.relu(x)
        
        x = x.reshape(x.shape[0], -1, self.filters[5])
                
        x = self.GC1(x, self.adjacency_matrices[4]._indices())
        x, ds_1 = self.IGSC_1(x, self.adjacency_matrices[4]._indices(), layers[0])
        x = self.unpool(x, self.upsample_matrices[3])

        x = self.GC2(x, self.adjacency_matrices[3]._indices())
        x, ds_2 = self.IGSC_2(x, self.adjacency_matrices[3]._indices(), layers[1])
        x = self.unpool(x, self.upsample_matrices[2])

        x = self.GC3(x, self.adjacency_matrices[2]._indices())
        x, ds_3 = self.IGSC_3(x, self.adjacency_matrices[2]._indices(), layers[2])
        x = self.unpool(x, self.upsample_matrices[1])
        
        x = self.GC4(x, self.adjacency_matrices[1]._indices())
        x, ds_4 = self.IGSC_4(x, self.adjacency_matrices[1]._indices(), layers[3])
        x = self.unpool(x, self.upsample_matrices[0])
        
        x = self.GC5(x, self.adjacency_matrices[0]._indices()) 
        x = self.GCout(x, self.adjacency_matrices[0]._indices()) # Sin relu y sin bias
        
        return x, [ds_4, ds_3, ds_2, ds_1]
    

class CenterDetector(nn.Module):
    def __init__(self, config):
        super(CenterDetector, self).__init__()
        
        self.latents = config['latents3D']
        self.encoder = Encoder3D(latents = self.latents, h = config['h'], w = config['w'], slices = config['slices'])
        self.fc1 = nn.Linear(self.latents, 128)
        self.fcout = nn.Linear(128, 2)

    def forward(self, sax):
        self.mu, _, _ = self.encoder(sax)
        x = self.fc1(self.mu)
        out = self.fcout(x)
        
        return out