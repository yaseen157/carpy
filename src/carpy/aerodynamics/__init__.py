"""Methods for assessing aerodynamics of various vehicular components."""
# Analytical techniques in 2D
from ._soln2d_potentialflow import *
from ._soln2d_thinaerofoils import *
from ._soln2d_vortexsource import *

# Analytical techniques in 3D
from ._soln3d_liftingline import *

# Empirical methods
from ._solns_gudmundsson import *
from ._solns_hoerner import *
from ._solns_raymer import *
