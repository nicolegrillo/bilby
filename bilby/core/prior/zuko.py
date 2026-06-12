import numpy as np
import torch
from bilby.core.prior.joint import BaseJointPriorDist

class ZukoFlowPriorDist(BaseJointPriorDist):
    """
            A class defining ZukoFlowPriorDist based on JointPriorDist


            Parameters
            ==========
            names: list (required)
                A list of the parameter names in the JointPriorDist. The
                listed parameters must have the same order that they appear in
                the lists of statistical parameters that may be passed in child class
            bounds: list (optional)
                A list of bounds on each parameter. The defaults are for bounds at
                +/- infinity.
            """

    def __init__(self, names, flow, context=None, bounds=None, device="cpu", dtype=torch.float32):
        super().__init__(names=names, bounds=bounds) # used to initialize the base class BaseJointPriorDist with necessary params

        self.flow = flow.to(device).eval()
        self.context = None if context is None else torch.as_tensor(context, device=device, dtype=dtype) # converts data into a tensor, faster for large pre-existing datasets
        self.device = device # type of device "cpu", for gpu use "cuda"
        self.dtype = dtype # type of data
        self.ndim = len(names) # names of variables to use

    def _distribution(self):
        with torch.no_grad():
            if self.context is None:
                return self.flow()
            return self.flow(self.context)

    def _as_tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=self.dtype)

    def _sample(self, size=1):
        with torch.no_grad():
            x = self._distribution().sample((size,))
        return x.cpu().numpy()

    def _ln_prob(self, value):
        x = self._as_tensor(value)
        if x.ndim == 1:
            x = x.unsqueeze(0)

        with torch.no_grad():
            lp = self._distribution().log_prob(x)

        return lp.cpu().numpy()

    def _rescale(self, value):
        u = self._as_tensor(value).clamp(1e-12, 1 - 1e-12)
        if u.ndim == 1:
            u = u.unsqueeze(0)

        dist = self._distribution()

        # For common Zuko flows with standard-normal / diagonal-normal base.
        base = dist.base
        z = base.icdf(u)

        # Zuko convention: transform maps x -> z, so sampling uses inverse.
        x = dist.transform.inv(z)

        return x.detach().cpu().numpy()