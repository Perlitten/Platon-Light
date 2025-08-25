#!/usr/bin/env python
"""
Optimized technical indicators for Platon Light.

This module provides optimized implementations of technical indicators
for improved performance.
"""

try:
    from .numba_accelerated import NumbaAcceleratedRSI
    from .cached_indicator import CachedIndicator

    __all__ = ["NumbaAcceleratedRSI", "CachedIndicator"]
except ImportError:
    # Numba might not be installed
    __all__ = []
