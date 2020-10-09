# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:07:37 2020

Exact budget subroutines compute each term at time step

@author: Jasen
"""
import numpy as np
import numpy.ma as ma
import obs_depth_JJ as dep
import GridShift_3D as GridShift
import Mask_Tools as mt
import Differential_Tstep as dff
import Gradients_Tstep as gr

def Flux_Masks(AvgFile, Avg, latbounds, lonbounds, precision):
    """
    Masks for fluxes across boundaries
    """
    ndepth = Avg.variables['salt'].shape[1]
    
    #U and V Masks
    U_Mask = np.repeat(mt.UMask(Avg, latbounds, lonbounds, \
                        precision)[np.newaxis, :, :], ndepth, axis = 0)
    V_Mask = np.repeat(mt.VMask(Avg, latbounds, lonbounds, \
                        precision)[np.newaxis, :, :], ndepth, axis = 0)
    
    #Face Masks
    NFace, WFace, SFace, EFace = mt.FaceMask(Avg, latbounds, lonbounds, precision)
    NorthFace = np.repeat(NFace, ndepth, axis = 0)
    SouthFace = np.repeat(SFace, ndepth, axis = 0)
    WestFace = np.repeat(WFace, ndepth, axis = 0)
    EastFace = np.repeat(EFace, ndepth, axis = 0)
    
    #Control Volume Mask on Rho points
    RhoMask = np.repeat(mt.RhoMask(Avg, latbounds, lonbounds)[np.newaxis, :, :], \
                        ndepth, axis = 0)
    
    Masks = {
            'RhoMask':RhoMask,\
            'U_Mask' : U_Mask,\
            'V_Mask' : V_Mask,\
            'NFace' : NorthFace, \
            'WFace' : WestFace, \
            'SFace' : SouthFace, \
            'EFace' : EastFace, 
            }
    
    return Masks

def CellAreas(tstep, AvgFile, Avg, Masks) :
    """
    Compute cell areas
    """
    #area of upward normal faces
    Axy = ma.array(dff.dA_top(Avg), mask = Masks['RhoMask'])
    
    #Areas of all cell faces
    Ax_norm, Ay_norm = dff.dA(tstep, AvgFile)
    
    #subset areas to faces of CV
    Areas = {
            'Axy' : Axy, \
            'North_Ay' : ma.array(Ay_norm, mask = Masks['NFace']), \
            'West_Ax' : ma.array(Ax_norm, mask = Masks['WFace']), \
            'South_Ay' : ma.array(Ay_norm, mask = Masks['SFace']), \
            'East_Ax' : ma.array(Ax_norm, mask = Masks['EFace'])
            }
    
    return Areas


def TimeDeriv(tstep, vprime2, Hist, HistFile, Avg, AvgFile, Diag, dA_xy, Masks):
    """
    Exact volume time derivative of variance squared
    """
    #change in vertical thickness of cell at average point
    deltaA = ma.array(ma.diff(dep._set_depth(AvgFile, None, 'w', \
                                             Avg.variables['h'][:], \
                                             Avg.variables['zeta'][tstep,:,:]),
                              n = 1, axis = 0), \
                      mask = Masks['RhoMask'])
    
    #diagnostic rate
    var_rate = ma.array(Diag.variables['salt_rate'][tstep, :, :, :], \
                         mask = Masks['RhoMask'])
    
    #change in vertical cell thickness at history points
    deltaH0 = ma.array(ma.diff(dep._set_depth(HistFile, None, 'w', \
                                             Hist.variables['h'][:], \
                                             Hist.variables['zeta'][tstep,:,:]),
                              n = 1, axis = 0), \
                      mask = Masks['RhoMask'])
    
    deltaH1 = ma.array(ma.diff(dep._set_depth(HistFile, None, 'w', \
                                             Hist.variables['h'][:], \
                                             Hist.variables['zeta'][tstep+1,:,:]),
                              n = 1, axis = 0), \
                      mask = Masks['RhoMask'])
    
    dtH = Hist.variables['ocean_time'][tstep+1] - Hist.variables['ocean_time'][tstep]
    
    #time derivative of cell volume on avg points
    dDelta_dt = (deltaH1 - deltaH0)/dtH
    
    #compute volume
    dV = dA_xy*deltaA
    
    salt = ma.array(Avg.variables['salt'][tstep, :, :, :], mask = Masks['RhoMask'])
    
    Int_Sprime = ma.sum(2*vprime2*(var_rate - salt/deltaA*dDelta_dt)*dV)
    
    return Int_Sprime
    
def Adv_Flux(tstep, vprime2, Avg, Areas, Masks):
    """
    Compute advective fluxes through boundaries
    """
    #Shift from rho to u and v points
    _var_u = GridShift.Rho_to_Upt(vprime2)
    _var_v = GridShift.Rho_to_Vpt(vprime2)
    
    #apply mask that is shifted to u and v points to variance squared
    var2_u = ma.array(_var_u, mask = Masks['U_Mask'])
    var2_v = ma.array(_var_v, mask = Masks['V_Mask'])
    
    #apply face masks to variance squared
    North_var = ma.array(var2_v, mask = Masks['NFace'])
    South_var = ma.array(var2_v, mask = Masks['SFace'])
    West_var = ma.array(var2_u, mask = Masks['WFace'])
    East_var = ma.array(var2_u, mask = Masks['EFace'])
    
    #velocities at faces oriented into CV
    North_V = ma.array(-1*Avg.variables['v'][tstep, :, :, :], \
                       mask =  Masks['NFace'])
    West_U = ma.array(Avg.variables['u'][tstep, :, :, :], \
                      mask = Masks['WFace'])
    South_V = ma.array(Avg.variables['v'][tstep, :, :, :], \
                       mask = Masks['SFace'])
    East_U = ma.array(-1*Avg.variables['u'][tstep, :, :, :], \
                      mask = Masks['EFace'])
    
    #Advective Fluxes through each boundary
    North = North_var*North_V*Areas['North_Ay']
    West = West_var*West_U*Areas['West_Ax']
    South = South_var*South_V*Areas['South_Ay']
    East = East_var*East_U*Areas['East_Ax']
    
    #sum (integrate) time step
    AdvFlux = ma.sum(North) + ma.sum(West) + ma.sum(South) + ma.sum(East)
    
    return AdvFlux
    
def Diff_Flux(tstep, vprime2, Avg, Masks, Areas) :
    """
    Diffusive flux across CV boundaries
    """
    #gradients of variance squared
    dprime_u_dx = gr.x_grad_u(Avg, vprime2)
    dprime_v_dy = gr.y_grad_v(Avg, vprime2)
    
    #apply face masks to gradients
    North_grad = ma.array(dprime_u_dx, mask = Masks['NFace'])
    West_grad = ma.array(dprime_v_dy, mask = Masks['WFace'])
    South_grad = ma.array(dprime_v_dy, mask = Masks['SFace'])
    East_grad = ma.array(dprime_u_dx, mask = Masks['EFace'])
    
    #horizontal diffustion coef
    Ks = Avg.variables['nl_tnu2'][0]
    
    #diffusive fluxes through each boundary
    North = -1*Ks*North_grad*Areas['North_Ay']
    West = Ks*West_grad*Areas['West_Ax']
    South = Ks*South_grad*Areas['South_Ay']
    East = -1*Ks*East_grad*Areas['East_Ax']
    
    DifFlux = ma.sum(North) + ma.sum(West) + ma.sum(South) + ma.sum(East)
    
    return DifFlux

def Int_Mixing(tstep, vprime, Avg, AvgFile, Masks) :
    """
    Internal Mixing Term
    """
    xgrad = gr.x_grad_rho(Avg, vprime)**2
    ygrad = gr.x_grad_rho(Avg, vprime)**2
    zgrad = gr.z_grad(tstep, Avg, vprime)**2
    
    kvw = Avg.variables['Aks'][tstep, :, :, :]
    
    kv = ma.array(GridShift.Wpt_to_Rho(kvw), mask = Masks['RhoMask'])
    kh = Avg.variables['nl_tnu2'][0]
    
    dV = ma.array(dff.dV(tstep, AvgFile), mask = Masks['RhoMask'])
    
    xmix = 2*kh*xgrad
    ymix = 2*kh*ygrad
    zmix = 2*kv*zgrad
    
    mixing = (ma.sum(xmix) + ma.sum(ymix) + ma.sum(zmix))*dV
    
    return mixing