# TODO: Port content from gaskinetics/_gases.py and tightly integrate with Equation of State to consider liquid-vapour
#   transitions. Presumably this means writing something that prevents users from using molar volumes that are
#   unphysical (sitting between the upper bound of liquid molar volume and lower bound of gas molar volume. Also needs
#   to have an intelligent way to instantiate a number of properties like latent heat, critical temperature, and
#   standard molar entropy...

# In most cases, I should just be able to scale up Vm output arrays to shape (2, *input.shape) and then record the
#   maximum (vapour) solution and the minimum (liquid) solution
