import torch
import torch.nn.functional as F

class CapsuleLayer(torch.nn.Module):
    def __init__(self, num_capsules, num_routes, in_dim, out_dim, num_iterations=3):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_iterations = num_iterations

        self.W = torch.nn.Parameter(
            torch.randn(1, num_routes, num_capsules, out_dim, in_dim)
        )

    def forward(self, ui):
        batch_size = ui.size(0)
        num_time_steps = ui.size(1)
        
        # Expand ui to match dimensions for the multiplication
        ui = ui.unsqueeze(2).unsqueeze(2)  # (batch_size, num_time_steps, 1, 1, hidden_dim)
        ui = ui.expand(batch_size, num_time_steps, self.num_routes, self.num_capsules, self.in_dim)
        
        # Expand W to match batch size
        W = self.W.expand(batch_size, self.num_routes, self.num_capsules, self.out_dim, self.in_dim)
        
        # Perform matrix multiplication
        u_hat = torch.einsum('brcdo,btrci->btrco', W, ui)  # (batch_size, num_time_steps, num_routes, num_capsules, out_dim)
        
        b_ij = torch.zeros(batch_size, self.num_routes, self.num_capsules).to(ui.device)
        
        for r in range(self.num_iterations):
            c_ij = F.softmax(b_ij, dim=2)  # (batch_size, num_routes, num_capsules)
            c_ij = c_ij.unsqueeze(1).unsqueeze(4)  # (batch_size, 1, num_routes, num_capsules, 1)
            
            s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)  # (batch_size, num_time_steps, 1, num_capsules, out_dim)
            v_j = self.squash(s_j)  # (batch_size, num_time_steps, 1, num_capsules, out_dim)
            
            if r < self.num_iterations - 1:
                a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True).sum(dim=1)  # (batch_size, num_routes, num_capsules, 1)
                b_ij = b_ij + a_ij.squeeze(-1).squeeze(-1)
        
        return v_j.squeeze(2)

    @staticmethod
    def squash(s, epsilon=1e-9):
        s_squared_norm = (s ** 2).sum(dim=-1, keepdim=True)
        scale = s_squared_norm / (1 + s_squared_norm) / torch.sqrt(s_squared_norm + epsilon)
        return scale * s

class CapsNet(torch.nn.Module):
    def __init__(self, num_capsules, num_routes, in_dim, out_dim, num_iterations=3):
        super(CapsNet, self).__init__()
        self.capsule_layer = CapsuleLayer(num_capsules, num_routes, in_dim, out_dim, num_iterations)
        self.W_lambda = torch.nn.Parameter(torch.randn(out_dim, out_dim))  # Adjusted size for concatenation
        self.b_w = torch.nn.Parameter(torch.randn(out_dim))
    
    def forward(self, ui):
        lambda_j = self.capsule_layer(ui)
        # print(v_j.shape)
        # print(lambda_j.shape)
        V = lambda_j.view(lambda_j.size(0), -1,self.capsule_layer.out_dim)
        
        g_j = torch.tanh(torch.matmul(V, self.W_lambda) + self.b_w)
        e_j = F.softmax(g_j, dim=1)
        # print(e_j.shape)
        # print(V.shape)
        f_caps = (e_j * V).sum(dim=1)
        return f_caps

# batch_size = 32
# num_time_steps = 10
# hidden_dim = 16
# num_capsules = 8
# num_routes = 6
# out_dim = 16
# num_iterations = 3

# ui = torch.randn(batch_size, num_time_steps, hidden_dim)
# capsnet = CapsNet(num_capsules, num_routes, hidden_dim, out_dim, num_iterations)
# output = capsnet(ui)
# print(output.shape)  # Should print (batch_size, out_dim)
