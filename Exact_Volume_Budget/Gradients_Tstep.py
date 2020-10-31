# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:26:00 2020

Tools to compute gradients within a control volume

@author: Jasen
"""

import numpy as np
import GridShift_3D as GridShift

def x_grad_u(RomsNC, var, dx) :
    """
    compute x-gradient on u points 
    """
    #gradient in x direction [u points]
    dvar_u = np.diff(var, n = 1, axis = 2)
    
    #dx on u points
    dx_u = GridShift.Rho_to_Upt(dx)
    
    #gradient
    dvar_dx = dvar_u/dx_u
    
    return dvar_dx


def x_grad_rho(RomsNC, var, dx) :
    """
    Compute x-gradient on rho points with rho dimension
    """
    #gradient in x direction on u points
    dvar_u = np.diff(var, n = 1, axis = 2)
    
    #shift u points to rho points
    dvar_rho = GridShift.Upt_to_Rho(dvar_u)
    
    #compute gradient
    dvar_dx = dvar_rho/dx
    
    return dvar_dx

def y_grad_rho(RomsNC, var, dy):
    """
    Compute y-gradient on rho points
    """
    #compute difference [v points]
    dvar_v = np.diff(var, n = 1, axis = 1)
    
    #shift v points to rho points
    dvar_rho = GridShift.Vpt_to_Rho(dvar_v)
    
    #compute gradient
    dvar_dy = dvar_rho/dy
    
    return dvar_dy

def y_grad_v(RomsNC, var, dy):
    """
    Compute y-gradient on v points that are shifted to rho grid
    """
    #gradient on v points
    dvar_v = np.diff(var, n = 1, axis = 1)
    
    #shift V points to rho grid
    dy_v = GridShift.Rho_to_Vpt(dy)
       
    #compute gradient on v
    dvar_dy = dvar_v/dy_v
    
    return dvar_dy
    

def z_grad(tstep, var, dz) :
    """
    Compute z-gradient on rho points
    """
    #vertical difference of variable
    dvar_w = np.diff(var, n = 1, axis = 0)
    
    #shift w points to rho points
    dvar_rho = GridShift.Wpt_to_Rho_pad(dvar_w)
    
    #compute gradient
    dvar_dz = dvar_rho/dz
    
    return dvar_dz
