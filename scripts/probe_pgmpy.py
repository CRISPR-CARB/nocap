"""Probe installed pgmpy version and LGBN method signatures."""

import inspect

import pgmpy
from pgmpy.models import LinearGaussianBayesianNetwork as LGBN

print("pgmpy version:", pgmpy.__version__)
print("fit signature:", inspect.signature(LGBN.fit))
print("simulate signature:", inspect.signature(LGBN.simulate))
