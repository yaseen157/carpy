"""Module that provides access to all the equations of state declared public, as well as any shorthand aliases."""
from carpy.physicalchem._equations_of_state import *

__author__ = "Yaseen Reza"

# Define aliases for equations of state
VdW = VanderWaals
RK = RedlichKwong
SRK = SoaveRedlichKwong
PR = PengRobinson
BH2 = BaigangH2

SRKmP = SRKmodPeneloux
PRmP = PRmodPeneloux
PRmM = PRmodMathias
