"""
deepMask: Accurate Brain Segmentation in Malformations of Cortical Development

A PyTorch implementation using V-net variant of Fully Convolutional Neural Networks
for brain segmentation in cortical malformations.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__author__ = "Ravnoor Gill, Benoit Caldairou, Neda Bernasconi, Andrea Bernasconi"
__email__ = "ravnoor.gill@mail.mcgill.ca"
__license__ = "BSD-3-Clause"