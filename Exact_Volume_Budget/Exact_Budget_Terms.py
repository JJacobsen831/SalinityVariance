# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:07:37 2020

Exact budget subroutines compute each term at time step

@author: Jasen
"""
import numpy as np
import obs_depth_JJ as dep
import GridShift_3D as GridShift
import PolyMask as mt
import Differential_Tstep as dff
import Gradients_Tstep as gr

def CreateMasks(RomsNC, Vertices) :
    """
    define masks on rho, U, V points and along boundaries
    returns dictionary of masks
    """
    #lats and lons
    lats = RomsNC.variables['lat_rho'][:]
    lons = RomsNC.variables['lon_rho'][:]
    nz = RomsNC.variables['salt'][0,:,0,0].size    
    
    #mask on Rho, U, V points repeated over depth space
    Rmask = np.repeat(mt.RhoMask(lats, lons, Vertices)[np.newaxis,:,:], nz, axis = 0)
    Umask = np.array(GridShift.Rho_to_Upt(Rmask), dtype = bool)
    Vmask = np.array(GridShift.Rho_to_Vpt(Rmask), dtype = bool)
    
    #face masks
    NFace, WFace, SFace, EFace = mt.FaceMask(Rmask)
    
    NFace = GridShift.Rho_to_Vpt(NFace)
    SFace = GridShift.Rho_to_Vpt(SFace)
    WFace = GridShift.Rho_to_Upt(WFace)
    EFace = GridShift.Rho_to_Upt(EFace)
    
    Masks = {
            'RhoMask':Rmask,\
            'U_Mask' : Umask,\
            'V_Mask' : Vmask,\
            'NFace' : NFace, \
            'WFace' : WFace, \
            'SFace' : SFace, \
            'EFace' : EFace, 
            }
    
    return Masks

def CellAreas(tstep, dz, RomsNC, Masks) :
    """
    Compute cell areas
    """
    #area of upward normal faces
    Axy = dff.dA_top(RomsNC)*Masks['RhoMask']
    
    #Areas of all cell faces
    Ax_norm, Ay_norm = dff.dA(tstep, RomsNC, dz)
    
    #subset areas to faces of CV
    Areas = {
            'Axy' : Axy, \
            'North_Ay' : Ay_norm*Masks['NFace'], \
            'West_Ax' : Ax_norm*Masks['WFace'], \
            'South_Ay' : Ay_norm*Masks['SFace'], \
            'East_Ax' : Ax_norm*Masks['EFace']
            }
    
    return Areas


def TimeDeriv(tstep, vprime2, Hist, HistFile, Avg, AvgFile, Diag, dA_xy, Masks):
    """
    Exact volume time derivative of variance squared
    """
    #change in vertical thickness of cell at average point
    deltaA = np.diff(dep._set_depth(AvgFile, None, 'w', \
                                             Avg.variables['h'][:], \
                                             Avg.variables['zeta'][tstep,:,:]),
                              n = 1, axis = 0)*Masks['RhoMask']
    
    #diagnostic rate
    var_rate = Diag.variables['salt_rate'][tstep, :, :, :]*Masks['RhoMask']
    
    #change in vertical cell thickness at history points
    deltaH0 = np.diff(dep._set_depth(HistFile, None, 'w', \
                                             Hist.variables['h'][:], \
                                             Hist.variables['zeta'][tstep,:,:]),
                              n = 1, axis = 0)*Masks['RhoMask']
    
    deltaH1 = np.diff(dep._set_depth(HistFile, None, 'w', \
                                             Hist.variables['h'][:], \
                                             Hist.variables['zeta'][tstep+1,:,:]),
                              n = 1, axis = 0)*Masks['RhoMask']
    
    dtH = Hist.variables['ocean_time'][tstep+1] - Hist.variables['ocean_time'][tstep]
    
    #time derivative of cell volume on avg points
    dDelta_dt = (deltaH1 - deltaH0)/dtH
    
    #compute volume
    dV = dA_xy*deltaA
    
    salt = Avg.variables['salt'][tstep, :, :, :]*Masks['RhoMask']
    
    Int_Sprime = np.sum(2*vprime2*(var_rate - salt/deltaA*dDelta_dt)*dV)
    
    return Int_Sprime
    
def Adv_Flux(tstep, vprime2, Avg, Areas, Masks):
    """
    Compute advective fluxes through boundaries
    """
    #Shift from rho to u and v points & apply face mask
    West_var = GridShift.Rho_to_Upt(vprime2)*Masks['WFace']
    East_var = GridShift.Rho_to_Upt(vprime2)*Masks['EFace']
    North_var = GridShift.Rho_to_Vpt(vprime2)*Masks['NFace']
    South_var = GridShift.Rho_to_Vpt(vprime2)*Masks['SFace']
    
    #velocities at faces oriented into CV
    North_V = -1*Avg.variables['v'][tstep, :, :, :]*Masks['NFace']
    West_U = Avg.variables['u'][tstep, :, :, :]*Masks['WFace']
    South_V = Avg.variables['v'][tstep, :, :, :]*Masks['SFace']
    East_U = -1*Avg.variables['u'][tstep, :, :, :]*Masks['EFace']
    
    #Advective Fluxes through each boundary
    North = North_var*North_V*Areas['North_Ay']
    West = West_var*West_U*Areas['West_Ax']
    South = South_var*South_V*Areas['South_Ay']
    East = East_var*East_U*Areas['East_Ax']
    
    #sum (integrate) time step
    AdvFlux = np.sum(North) + np.sum(West) + np.sum(South) + np.sum(East)
    
    return AdvFlux
    
def Diff_Flux(Avg, vprime2, dx, dy, Areas, Masks) :
    """
    Diffusive flux across CV boundaries
    """
    #gradients of variance squared
    dprime_u_dx = gr.x_grad_u(Avg, vprime2, dx, Masks)
    dprime_v_dy = gr.y_grad_v(Avg, vprime2, dy, Masks)
    
    #apply face masks to gradients
    North_grad = dprime_u_dx*Masks['NFace']
    West_grad = dprime_v_dy*Masks['WFace']
    South_grad = dprime_v_dy*Masks['SFace']
    East_grad = dprime_u_dx*Masks['EFace']
    
    #horizontal diffustion coef
    Ks = Avg.variables['nl_tnu2'][0]
    
    #diffusive fluxes through each boundary
    North = -1*Ks*North_grad*Areas['North_Ay']
    West = Ks*West_grad*Areas['West_Ax']
    South = Ks*South_grad*Areas['South_Ay']
    East = -1*Ks*East_grad*Areas['East_Ax']
    
    DifFlux = np.sum(North) + np.sum(West) + np.sum(South) + np.sum(East)
    
    return DifFlux

def Int_Mixing(tstep, vprime, Avg, dx, dy, dz, Masks) :
    """
    Internal Mixing Term
    """
    xgrad = gr.x_grad_rho(Avg, vprime, dx, Masks)**2
    ygrad = gr.x_grad_rho(Avg, vprime, dy, Masks)**2
    zgrad = gr.z_grad(tstep, vprime, dz, Masks)**2
    
    kvw = Avg.variables['AKs'][tstep, :, :, :]
    
    kv = GridShift.Wpt_to_Rho(kvw)
    kh = Avg.variables['nl_tnu2'][0]
    
    dV = dx*dy*dz*Masks['RhoMask']
    
    xmix = 2*kh*xgrad
    ymix = 2*kh*ygrad
    zmix = 2*kv*zgrad
    
    mixing = (np.sum(xmix) + np.sum(ymix) + np.sum(zmix))*dV
    
    return mixing