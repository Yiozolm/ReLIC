from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from compressai.models import MeanScaleHyperprior as Mbt
from compressai.models import Elic2022Chandelier
from compressai.models.utils import deconv
from compressai.ops import quantize_ste

from .cfm_model import FlowMatchingModel, Unet
from .Meanflow import MeanFlowModel
from torch.autograd import Function


class SmoothOperator(Function):
    @staticmethod
    def forward(ctx, i):
        return i.clamp_(0, 1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def smooth(i:torch.Tensor):
    return SmoothOperator.apply(i)

class Interpolater:
    def __init__(self, rate:float):
        assert rate >= 0 and rate <= 1, 'Invalid Rate'
        self.rate = rate
    
    def __call__(self, start:torch.Tensor, target:torch.Tensor):
        return start + (target - start) * self.rate


class ReLIC_mbt(Mbt):
    def __init__(self, 
                 N: int, M: int,
                 lazy_init: bool = False,
                 init_threshold: int = 0,
                 cfm_model_class=Unet,
                 cfm_model_kwargs: Dict[str, Any] = None,
                 cfm_wrapper_kwargs: Dict[str, Any] = None,
                 **kwargs: Dict[str, Any]):
        super().__init__(N=N, M=M)

        self.M = M
        self.N = N
        self.lazy_init = lazy_init
        self.init_threshold = init_threshold

        self._cfm_model_class = cfm_model_class
        self._cfm_model_kwargs = cfm_model_kwargs if cfm_model_kwargs is not None else {}
        self._cfm_wrapper_kwargs = cfm_wrapper_kwargs if cfm_wrapper_kwargs is not None else {}
        

        # Components that can be lazily initialized
        self.model_type = self._cfm_wrapper_kwargs.get("matcher_type", "ot")
        self.z_upsampler = None
        self.cfm = None
        self.use_norm = kwargs.get("use_norm", False)
        if self.use_norm:
            self.cond_norm = nn.InstanceNorm2d(M * 2) 

        if not self.lazy_init:
            self._initialize_cfm_components()
            print("ResDiff model created with all components initialized eagerly.")

    def _initialize_cfm_components(self):
        """Initializes the Conditional Flow Matching components."""
        if self.cfm is not None:
            return
        
        print("\n[INFO] Initializing Conditional Flow Matching (CFM) components...")

        self.z_upsampler = nn.Sequential(
            deconv(self.N, (self.N + self.M) // 2, kernel_size=5, stride=2),
            nn.GELU(),
            deconv((self.N + self.M) // 2, self.M, kernel_size=5, stride=2),
        )
        
        print(f"--> Instantiating core model: {self._cfm_model_class.__name__}")
        core_model = self._cfm_model_class(**self._cfm_model_kwargs)
        
        if self.model_type == "meanflow":
            print("--> Instantiating wrapper: MeanFlowModel")
            self.cfm = MeanFlowModel(model=core_model)
        else:
            print("--> Instantiating CFM wrapper: FlowMatchingModel")
            self.cfm = FlowMatchingModel(model=core_model, **self._cfm_wrapper_kwargs)
            
        # Move components to the correct device, assuming the base model is already there
        device = next(self.parameters()).device
        self.z_upsampler.to(device)
        self.cfm.to(device)
        
        print("[INFO] CFM components initialized successfully.\n")

    def forward(self, x: torch.Tensor, epoch: Optional[int] = None):
        # Eagerly initialize if needed
        if self.lazy_init and self.cfm is None and epoch is not None and epoch >= self.init_threshold:
            self._initialize_cfm_components()
            
        y = self.g_a(x)
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = quantize_ste(z - z_offset) + z_offset
        
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = quantize_ste(y - means_hat) + means_hat

        # Handle the case where CFM is not active yet (either lazy or during early epochs)
        if self.cfm is None or not self.training and self.lazy_init:
            x_hat = self.g_s(y_hat)
            return {"x_hat": x_hat, "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}, "CFMLoss": None}

        # --- CFM Path ---
        upsampled_z = self.z_upsampler(z_hat)
        condition = torch.cat([y_hat, upsampled_z], dim=1)
        if self.use_norm:
            condition = self.cond_norm(condition)

        if self.training:
            # For CFM training, the residual should not propagate gradients back to the VAE
            residual = (y - y_hat) #.detach()
            if self.use_norm:
                residual = residual * 2.0
            cfm_loss, pred_residual  = self.cfm(residual, cond_img=condition) #.detach()
            if self.use_norm:
                pred_residual = pred_residual / 2.0
            y_tilde = y_hat + pred_residual
            x_hat = self.g_s(y_tilde)
            return {"x_hat": x_hat, "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}, "CFMLoss": cfm_loss}
        else: # Inference
            pred_residual  = self.cfm.sample(cond_img=condition)
            if self.use_norm:
                pred_residual = pred_residual / 2.0
            y_tilde = y_hat + pred_residual
            x_hat = self.g_s(y_tilde)
            return {"x_hat": x_hat, "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}, "CFMLoss": None}

    def decompress(self, strings, shape):
        # Eagerly initialize if this is called on a lazy model before training
        if self.cfm is None:
            self._initialize_cfm_components()
            
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        
        upsampled_z = self.z_upsampler(z_hat)
        condition = torch.cat([y_hat, upsampled_z], dim=1)
        if self.use_norm:
            condition = self.cond_norm(condition)

        pred_residual = self.cfm.sample(cond_img=condition)
        if self.use_norm:
            pred_residual = pred_residual / 2.0
        y_tilde = y_hat + pred_residual
        x_hat = self.g_s(y_tilde).clamp_(0, 1)
        return {"x_hat": x_hat}

    def get_trainable_parameters(self):
        """Separates parameters for different optimizers."""
        vae_params, cfm_params, aux_params = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith(".quantiles"):
                aux_params.append(p)
            elif name.startswith("cfm.") or name.startswith("z_upsampler."):
                cfm_params.append(p)
            else:
                vae_params.append(p)
        return vae_params, cfm_params, aux_params


class ReLIC_ELIC(Elic2022Chandelier):
    def __init__(self, 
                 N: int, M: int,
                 lazy_init: bool = False,
                 init_threshold: int = 0,
                 cfm_model_class=Unet,
                 cfm_model_kwargs: Dict[str, Any] = None,
                 cfm_wrapper_kwargs: Dict[str, Any] = None,
                 **kwargs: Dict[str, Any]):
        super().__init__(N=N, M=M)

        self.M = M
        self.N = N
        self.lazy_init = lazy_init
        self.init_threshold = init_threshold

        self._cfm_model_class = cfm_model_class
        self._cfm_model_kwargs = cfm_model_kwargs if cfm_model_kwargs is not None else {}
        self._cfm_wrapper_kwargs = cfm_wrapper_kwargs if cfm_wrapper_kwargs is not None else {}

        self.interpolater = Interpolater(kwargs.pop('rate', 0.5))
        self.cfm = None
        self.g_s_perc = None
        if not self.lazy_init:
            self._initialize_cfm_components()
            print("ResDiff model created with all components initialized eagerly.")
        else:
            raise NotImplementedError("Lazy initialization of CFM components is not implemented.")

    def _initialize_cfm_components(self):
        if self.cfm is not None:
            return
        
        print("\n[INFO] Initializing Conditional Flow Matching (CFM) components...")
      
        print(f"--> Instantiating core model: {self._cfm_model_class.__name__}")
        core_model = self._cfm_model_class(**self._cfm_model_kwargs)
        
        print("--> Instantiating CFM wrapper: FlowMatchingModel")
        self.cfm = FlowMatchingModel(model=core_model, **self._cfm_wrapper_kwargs)
        
        self.g_s_perc = nn.Sequential(
            deconv(self.M, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N),
            GDN(self.N, inverse=True),
            deconv(self.N, 3),
        )
        device = next(self.parameters()).device
        self.cfm.to(device)
        
        print("[INFO] CFM components initialized successfully.\n")

    def forward(self, x: torch.Tensor, epoch: Optional[int] = None):
        if self.lazy_init and self.cfm is None and epoch is not None and epoch >= self.init_threshold:
            self._initialize_cfm_components()

        y = self.g_a(x)
        y_out = self.latent_codec(y)
        Condition = y_out["y_hat"]
        
        if self.training:
            cfm_loss, y_pred = self.cfm(y, cond_img=Condition.detach())
            x_hat = self.g_s(Condition)
            x_hat_perc = self.g_s_perc(y_pred)
            x_out = self.interpolater(x_hat, x_hat_perc)
            return {"x_hat": x_out, 
            "likelihoods": y_out["likelihoods"], 
            "CFMLoss": cfm_loss}
        else:
            y_pred = self.cfm.sample(cond_img=Condition)
            x_hat = self.g_s(Condition)
            x_hat_perc = self.g_s_perc(y_pred)
            x_out = self.interpolater(x_hat, x_hat_perc)
            return {"x_hat": x_out,
            "likelihoods": y_out["likelihoods"], 
            "CFMLoss": None}

    def decompress(self, *args, **kwargs):
        if self.cfm is None:
            self._initialize_cfm_components()
            
        y_out = self.latent_codec.decompress(*args, **kwargs)
        Condition = y_out["y_hat"]
        y_pred = self.cfm.sample(cond_img=Condition)
        x_hat = self.g_s(Condition)
        x_hat_perc = self.g_s_perc(y_pred)
        x_out = self.interpolater(x_hat, x_hat_perc).clamp_(0, 1)

        return {"x_hat": x_out}

    def get_trainable_parameters(self):
        """Separates parameters for different optimizers."""
        vae_params, cfm_params, aux_params = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith(".quantiles"):
                aux_params.append(p)
            elif name.startswith("cfm.") or name.startswith("g_s_perc."):
                cfm_params.append(p)
            else:
                vae_params.append(p)
        return vae_params, cfm_params, aux_params