"""
Compatibility shim for legacy imports.

This module re-exports the lightweight SampleLocationAndConditionalFlow
from the new `diffusion.utils` package so older imports continue to work.
"""

from inspect import isfunction
from typing import Callable, List, Optional, Sequence, TypeVar, Union, Dict, Tuple
from typing_extensions import TypeGuard
import collections.abc
from itertools import repeat


T = TypeVar("T")


class SampleLocationAndConditionalFlow:
    """
    Simplified sampling helper

    Expose static methods for backward compatibility
    """

    @staticmethod
    def run(matcher, x0, x1, t=None):
        """
        Sample xt and ut from ConditionalFlowMatcher

        Example usage:
            fm = ConditionalFlowMatcher(sigma=0.1)
            noise = torch.randn(16, 32)
            x_real = torch.randn(16, 32)

            t, xt, ut = SampleLocationAndConditionalFlow.run(
                fm, x0=noise, x1=x_real
            )

        Parameters
        ----------
        matcher : ConditionalFlowMatcher
            Flow matcher instance
        x0, x1 : Tensor
            Source and target sample batches
        t : Tensor or None
            Time step; sample from Uniform(0,1) if omitted

        Returns
        -------
        t : Tensor, shape (bs,)
            Sampled time steps
        xt : Tensor, shape (bs, *dim)
            Intermediate state
        ut : Tensor, shape (bs, *dim)
            Conditional flow field
        """
        return matcher.sample_location_and_conditional_flow(x0, x1, t)
