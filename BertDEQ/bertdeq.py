import torch.nn as nn
from torchdyn.numerics.root import BroydenBad

class DEQModel(nn.Module):
    def __init__(self):
        self.f = Layer(...)
        self.solver = BroydenBad
        
        # dimensionality of the encoder and pooler layers (d_model)
        self.hidden_size = 768
        
    
    def forward(self, x, ..., **kwargs):
        batch_size, seq_len = x.size()
        z0 = torch.zeros(batch_size, self.hidden_size, seq_len)

        # Forward pass
        with torch.no_grad():
            z_star = self.solver(lambda z: self.f(z, x, *args), z0, threshold=f_thres)['result']   # See step 2 above
            new_z_star = z_star

        # (Prepare for) Backward pass, see step 3 above
        if self.training:
            new_z_star = self.f(z_star.requires_grad_(), x, *args)
            
            # Jacobian-related computations, see additional step above. For instance:
            jac_loss = jac_loss_estimate(new_z_star, z_star, vecs=1)

            def backward_hook(grad):
                if self.hook is not None:
                    self.hook.remove()
                    torch.cuda.synchronize()   # To avoid infinite recursion
                # Compute the fixed point of yJ + grad, where J=J_f is the Jacobian of f at z_star
                new_grad = self.solver(lambda y: autograd.grad(new_z_star, z_star, y, retain_graph=True)[0] + grad,
                                       torch.zeros_like(grad), threshold=b_thres)['result']
                return new_grad

            self.hook = new_z_star.register_hook(backward_hook)
        return new_z_star, ...