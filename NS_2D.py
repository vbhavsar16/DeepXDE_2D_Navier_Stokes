import tensorflow as tf
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

rho = 1
mu = 1
u_in = 1
D = 1
L = 2

geom = dde.geometry.Rectangle(xmin=[-L/2,-D/2], xmax=[L/2,D/2])                       # Rectangle xmin = the minimum point in for both the direction. xmax is the maximum for both direction.

def boundary_wall(X, on_boundary):                                                    # This function takesin points X = (x,y) and whether it is on boundary or not (on_boundary is a dde built-in. When the random points are generated, dde mark the boundary points.).
    print("X", X)                                                                     # So, we know if they are on boundaries, but not which one. We don't want the points on inlet and outlet boundaries. Only on the wall boundaries.
    print("on_boundary", on_boundary)
    
    on_wall = np.logical_and                                                            # Logical function gives either true or false, on_wall = 1/0, 1= on wall boundary, 0 = not (Boundary point but not on wall).
    
    '''
    np.logical_and is a (&& - and kind) of condition (both argument should satisfy), np.logical_or is a (|| - or kind) of condition (Either of them satisfy then true).
    In a and argument ( or argument (either on top wall or bottom wall = 1), points on boundary = 1), if those both are 1 then on_wall = 1.
    '''
    
    return on_wall


bc_wall_u = dde.DirichletBC(geom, lambda X:0., boundary_wall, component= 0)           # Component = 0 meaning out of the unknowns (u,v,p), it is u. Lambda function says that the u is 0 (No slip) on boundary walls, which is previously defined by a function.
bc_wall_u = dde.DirichletBC(geom, lambda X:0., boundary_wall, component= 1)           # Component = 1 meaning out of the unknowns (u,v,p), it is v. Lambda function says that the v is 0 (No slip) on boundary walls, which is previously defined by a function.