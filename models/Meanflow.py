import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanFlowModel(nn.Module):
    """
    Implements the MeanFlow model based on arXiv:2505.13447v1.
    This model wraps a neural network (U-Net) and handles the specific loss
    computation involving Jacobian-Vector Products (JVP).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model # This will be the modified U-Net
        self.out_dim = model.out_dim

    def forward(self, x, cond_img=None):
        """
        Performs a training step for MeanFlow as described in Algorithm 1 of the paper.
        Args:
            x (torch.Tensor): The target data (in our case, the residual).
            cond_img (torch.Tensor, optional): Conditioning information.
        """
        b, *_, device = *x.shape, x.device

        # 1. Sample t and r from U(0, 1) [cite: 264, 265]
        t_rand = torch.rand(b, device=device).type_as(x)
        r_rand = torch.rand(b, device=device).type_as(x)
        
        # Enforce t > r [cite: 266]
        t = torch.max(t_rand, r_rand)
        r = torch.min(t_rand, r_rand)
        
        # 2. Sample noise and construct the path and velocity
        eps = torch.randn_like(x)
        
        # Reshape time variables for broadcasting
        t_b = t.view(b, 1, 1, 1)
        r_b = r.view(b, 1, 1, 1)

        # Path z_t and conditional velocity v_t [cite: 201]
        z_t = (1 - t_b) * x + t_b * eps
        v_t = eps - x
        
        # 3. Define the function for JVP. It must take (z, r, t) as input.
        #    This function will be differentiated by the JVP engine.
        def u_theta_fn(z, r_val, t_val):
            return self.model(z, r_val, t_val, cond_img)

        # 4. Compute Jacobian-Vector Product (JVP) for the total derivative d/dt u_theta
        #    The tangent vector is (v_t, 0, 1) for (z, r, t) respectively[cite: 164].
        u_theta, dudt = torch.func.jvp(
            u_theta_fn, 
            (z_t, r, t), 
            (v_t, torch.zeros_like(r), torch.ones_like(t))
        )
        
        # 5. Compute the regression target u_tgt using the MeanFlow Identity [cite: 172, 183]
        u_tgt = v_t - (t_b - r_b) * dudt

        # 6. Compute the loss, applying stop-gradient to the target [cite: 171, 178]
        loss = F.mse_loss(u_theta, u_tgt.detach())
        
        # For compatibility with RateDistortionLoss, return a predicted residual
        # A reasonable estimate is x_hat = eps - u_theta
        pred_residual = eps - u_theta

        return loss, pred_residual

    @torch.inference_mode()
    def sample(self, cond_img):
        """
        Performs one-step sampling as described in Algorithm 2[cite: 195].
        """
        b, *_, h, w = cond_img.shape
        shape = (b, self.out_dim, h, w)
        device = cond_img.device

        # Start from prior z_1 = eps [cite: 195]
        eps = torch.randn(shape, device=device)

        # Set r=0 and t=1 for one-step generation [cite: 195]
        r = torch.zeros(b, device=device)
        t = torch.ones(b, device=device)
        
        # The network predicts the average velocity u(z_1, 0, 1)
        u_pred = self.model(eps, t, r, cond_img)
        
        # The final sample is z_0 = z_1 - u(z_1, 0, 1) [cite: 195]
        x_pred = eps - u_pred
        
        return x_pred