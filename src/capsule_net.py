# https://www.kaggle.com/code/ankitjha/hacker-s-guide-to-capsule-networks#Architecture
import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None, padding=0):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(num_route_nodes, num_capsules, out_channels, in_channels))

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        # x: [batch_size, in_channels, num_route_nodes] -> [batch_size, num_capsules, out_channels]
        u_hat = torch.matmul(self.W, x)

        # routing
        b = torch.zeros_like(u_hat)
        for i in range(self.num_routing_iterations):
            c = F.softmax(b, dim=2)
            s = (c * u_hat).sum(dim=2)
            v = self.squash(s)
            b = b + torch.matmul(u_hat, v.unsqueeze(-1)).squeeze(-1)

        return v

class CapsuleNetwork(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, num_routing_iterations):
        super(CapsuleNetwork, self).__init__()

        self.capsule_layer = CapsuleLayer(num_capsules, num_route_nodes, in_channels, out_channels, num_routing_iterations)
        self.tanh = nn.Tanh()

    def forward(self, x):
        capsule_output = self.capsule_layer(x)
        capsule_output = self.tanh(capsule_output)
        # capsule level attention
        capsule_attention = F.softmax(capsule_output, dim=-1)
        capsule_attention = capsule_attention.unsqueeze(-1).unsqueeze(-1)
        capsule_attention = capsule_attention.repeat(1, 1, capsule_output.size(2), 1)

        capsule_output = capsule_output * capsule_attention

        # final capsule
        final_capsule = capsule_output.sum(dim=1)
        final_capsule = self.capsule_layer.squash(final_capsule)

        return final_capsule