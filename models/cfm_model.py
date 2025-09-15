import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from tqdm.auto import tqdm

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)

def Upsample(dim, dim_out=None):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(dim, default(dim_out, dim), 3, padding=1))

def Downsample(dim, dim_out=None):
    return nn.Sequential(Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2), nn.Conv2d(dim * 4, default(dim_out, dim), 1))

class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0.0):
        super().__init__()
        self.mlp = (nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None)
        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads), qkv)
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class Unet(nn.Module):
    def __init__(self, dim, out_dim=None, dim_mults=(1, 2, 4, 8), channels=3):
        super().__init__()
        self.channels = channels
        self.out_dim = default(out_dim, channels)
        init_dim = dim
        
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim), 
            nn.Linear(dim, time_dim), 
            nn.GELU(), 
            nn.Linear(time_dim, time_dim)
        )
        self.time_mlp_r = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.time_fusion_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                Attention(dim_in),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Attention(mid_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Attention(dim_out),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))
            
        self.final_res_block = ResnetBlock(init_dim * 2, init_dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    def forward(self, x, t, r=None, cond_img=None):
        if r is None:
            r = t

        if exists(cond_img):
            x = torch.cat((x, cond_img), dim=1)

        x = self.init_conv(x)
        r_clone = x.clone()

        t_emb = self.time_mlp(t)
        r_emb = self.time_mlp_r(r)
        time_emb = self.time_fusion_mlp(t_emb + r_emb)

        h = []
        
        for block1, attn, downsample in self.downs:
            x = block1(x, time_emb)
            x = attn(x) + x
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, time_emb)

        for block1, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, time_emb)
            x = attn(x) + x
            x = upsample(x)

        x = torch.cat((x, r_clone), dim=1)
        x = self.final_res_block(x, time_emb)
        return self.final_conv(x)

class FlowMatchingModel(nn.Module):
    def __init__(self, model, *, matcher_type='base', sigma=0.1, num_sampling_steps=100):
        super().__init__()
        self.model = model # This will be the Unet
        self.sigma = sigma
        self.num_sampling_steps = num_sampling_steps
        
        print(f"[INFO] Selected CFM Matcher: {matcher_type}")
        if matcher_type == 'base':
            self.cfm_matcher = ConditionalFlowMatcher(sigma=self.sigma)
        elif matcher_type == 'ot':
            self.cfm_matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=self.sigma)
        elif matcher_type == 'sb':
            self.cfm_matcher = SchrodingerBridgeConditionalFlowMatcher(sigma=self.sigma)
        elif matcher_type == 'vp':
            self.cfm_matcher = VariancePreservingConditionalFlowMatcher(sigma=self.sigma)
        elif matcher_type == 'target':
            self.cfm_matcher = TargetConditionalFlowMatcher(sigma=self.sigma)
        else:
            raise ValueError(f"Unknown matcher_type: {matcher_type}")
        
        
        # Expose out_dim for ResDiff to know the output channels
        self.out_dim = model.out_dim

    def forward(self, x1, cond_img=None):
        """ Training step """
        x0 = torch.randn_like(x1) # Sample source (noise)
        
        # Use the torchcfm-like interface to get a training pair
        t, xt, ut = self.cfm_matcher.sample_location_and_conditional_flow(x0, x1)
        
        # Predict the velocity
        pred_v = self.model(xt, t, cond_img)
        
        # Compute the loss
        loss = F.mse_loss(pred_v, ut)
        
        # For compatibility with ResDiff's forward pass, we need to return a predicted residual.
        # Since v_pred is an estimate for u_t = x1 - x0, then v_pred + x0 is an estimate for x1.
        pred_residual = pred_v + x0
        
        return loss, pred_residual

    @torch.inference_mode()
    def sample(self, cond_img):
        """ Inference/sampling step using an ODE solver """
        b, _, h, w = cond_img.shape
        device = cond_img.device
        
        # Start from noise
        x_t = torch.randn((b, self.out_dim, h, w), device=device)
        
        # Time steps for the ODE solver
        time_steps = torch.linspace(0, 1, self.num_sampling_steps + 1, device=device)
        dt = time_steps[1] - time_steps[0]
        
        # Simple Euler method for ODE solving
        for i in tqdm(range(self.num_sampling_steps), desc="CFM Sampling", leave=False):
            t = time_steps[i].expand(b)
            v = self.model(x_t, t, cond_img)
            x_t = x_t + v * dt
            
        return x_t