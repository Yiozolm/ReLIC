from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from typing import Union

# Make sure you have geoopt installed: pip install geoopt
# And POT for the OT version: pip install pot
try:
    import geoopt
    import ot
except ImportError:
    warnings.warn(
        "To use the ManifoldFlowMatcher with geoopt, please install it with "
        "'pip install geoopt pot'"
    )


class RectifiedFlowMatcher(ConditionalFlowMatcher):
    """Implements Rectified Flow, a special case of Conditional Flow Matching where the
    probability path is a deterministic linear interpolation between x0 and x1. This
    corresponds to the base ConditionalFlowMatcher with sigma=0.

    The probability path is $p_t(x) = \delta(x - ((1-t)x_0 + t x_1))$, a Dirac delta function.
    The corresponding vector field is $u_t(x_t|x_0, x_1) = x_1 - x_0$.

    This formulation is based on the work in [1].

    References
    ----------
    [1] Rectified Flow: A Marginal Preserving Approach to Optimal Transport, ICLR, Liu et al.
    """

    def __init__(self, sigma: Union[float, int] = 0.0):
        r"""Initialize the RectifiedFlowMatcher class.

        Note that for a pure Rectified Flow, sigma should be 0. A non-zero sigma
        introduces a diffusion term, making it equivalent to Independent Conditional
        Flow Matching (I-CFM).

        Parameters
        ----------
        sigma : Union[float, int]
            The standard deviation of the Gaussian noise. Defaults to 0.0.
        """
        if sigma != 0.0:
            warnings.warn(
                f"RectifiedFlowMatcher is typically used with sigma=0, but got sigma={sigma}. "
                "This is equivalent to using the base ConditionalFlowMatcher."
            )
        super().__init__(sigma)


class GeoOptManifoldFlowMatcher(ConditionalFlowMatcher):
    """
    Implements Flow Matching on a Riemannian Manifold using the `geoopt` library.

    This class is an adaptation of the generic manifold flow matching logic, specifically
    tailored to the API provided by the `geoopt` library, which is popular for deep
    learning on manifolds and was used in the original Riemannian Flow Matching paper.

    References
    ----------
    [1] Riemannian Flow Matching, TMLR, C. Chen et al.
    """

    def __init__(self, manifold: "geoopt.Manifold", sigma: float = 0.0):
        """
        Initializes the GeoOptManifoldFlowMatcher.

        Parameters
        ----------
        manifold : geoopt.Manifold
            An instance of a manifold from the `geoopt` library (e.g., geoopt.PoincareBall()).
        sigma : float, default=0.0
            The level of noise for the stochastic interpolation. sigma=0 corresponds to
            deterministic geodesic flow matching.
        """
        super().__init__(sigma)
        if not issubclass(type(manifold), geoopt.Manifold):
            raise TypeError(
                f"The manifold must be an instance of a geoopt.Manifold class, but got {type(manifold)}."
            )
        self.manifold = manifold

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Samples a point on the geodesic path and computes the target vector field
        using `geoopt` for all manifold operations.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            The source minibatch of points on the manifold.
        x1 : Tensor, shape (bs, *dim)
            The target minibatch of points on the manifold.
        t : Tensor, shape (bs), optional
            Time levels. If None, drawn from U[0, 1].
        return_noise : bool
            Whether to return the sampled noise vector.

        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            The sampled points on the manifold.
        ut : Tensor, shape (bs, *dim)
            The target tangent vectors at `xt`.
        eps : Tensor, shape (bs, *dim), optional
            The sampled noise in the tangent space (if `return_noise` is True).
        """
        if t is None:
            t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)

        # 1. Compute intermediate point `mu_t` on the geodesic using geoopt's method
        # We need to unsqueeze t for broadcasting with the point dimensions
        t_reshaped = t.view(-1, *([1] * (x0.dim() - 1)))
        mu_t = self.manifold.geodesic(t_reshaped, x0, x1)

        # 2. Add noise if sigma > 0
        sigma_t = self.compute_sigma_t(t)
        xt = mu_t
        eps = None
        if self.sigma > 0:
            # Sample a random vector in the tangent space of each point in mu_t
            noise = self.manifold.random_tangent(mu_t)
            sigma_t_reshaped = sigma_t.view(-1, *([1] * (noise.dim() - 1)))
            scaled_noise = sigma_t_reshaped * noise
            # Project the noisy tangent vector back onto the manifold
            xt = self.manifold.expmap(mu_t, scaled_noise)
            if return_noise:
                eps = noise

        # 3. Compute the conditional flow vector at xt
        # First, find the initial velocity at x0 that points to x1
        v0 = self.manifold.logmap(x0, x1)
        # Then, parallel transport this vector from T_{x0}M to T_{xt}M
        ut = self.manifold.transp(x0, xt, v0)

        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut


class ExactOptimalTransportGeoOptManifoldFlowMatcher(GeoOptManifoldFlowMatcher):
    """
    Implements Flow Matching on a `geoopt` Manifold with Optimal Transport coupling.

    This class extends `GeoOptManifoldFlowMatcher` by first computing an optimal
    transport plan between the mini-batches `x0` and `x1`. The cost function
    for the transport plan is the squared geodesic distance on the manifold,
    provided by the `geoopt` manifold's `dist2` method.
    """

    def __init__(self, manifold: "geoopt.Manifold", sigma: float = 0.0):
        """
        Initializes the ExactOptimalTransportGeoOptManifoldFlowMatcher.

        Parameters
        ----------
        manifold : geoopt.Manifold
            An instance of a manifold from the `geoopt` library.
        sigma : float, default=0.0
            The level of noise for the stochastic interpolation.
        """
        super().__init__(manifold, sigma)
        self.ot_solver = ot.emd

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Computes the OT plan using `geoopt`'s distance and then samples points and flows.
        """
        n, m = x0.shape[0], x1.shape[0]

        # 1. Compute the cost matrix using squared geodesic distance from geoopt
        cost_matrix = self.manifold.dist2(x0.unsqueeze(1), x1.unsqueeze(0))

        # 2. Solve the exact OT problem.
        a = torch.ones(n, device=x0.device) / n
        b = torch.ones(m, device=x1.device) / m
        pi = self.ot_solver(a, b, cost_matrix.detach()) # Use detach for OT solver

        # 3. Sample from the transport plan to get indices for matching x1 to x0.
        if not isinstance(pi, torch.Tensor):
             pi = torch.tensor(pi, device=x0.device, dtype=x0.dtype)
        
        # Ensure pi is on the correct device for multinomial
        pi = pi.to(x0.device)
        
        # OT plan `pi` has shape (n, m), scale it to be a probability distribution for each row
        row_sums = pi.sum(dim=1, keepdim=True)
        # Avoid division by zero for rows that have no transport mass
        row_sums[row_sums == 0] = 1
        pi_prob = pi / row_sums

        coupling_indices = torch.multinomial(pi_prob, num_samples=1).squeeze(-1)
        matched_x1 = x1[coupling_indices]
        
        # 4. Call the parent method with the optimally coupled pairs.
        return super().sample_location_and_conditional_flow(x0, matched_x1, t, return_noise)