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
import Gradients as gr

def Flux_Masks(AvgFile, Avg, latbounds, lonbounds, precision):
    """
    Masks for fluxes across boundaries
    """
    ndepth = Avg.variables['salt'].shape[1]
    
    #define masks for u and v points
    U_Mask = np.repeat(mt.UMask(AvgFile, latbounds, lonbounds, precision)[np.newaxis, :, :], \
                       ndepth, axis = 0)
        
    V_Mask = np.repeat(mt.VMask(AvgFile, latbounds, lonbounds, precision)[np.newaxis, :, :], \
                       ndepth, axis = 0)
        
    #Face Masks
    NFace, WFace, SFace, EFace = mt.FaceMask(AvgFile, latbounds, lonbounds)
    NorthFace = np.repeat(NFace, ndepth, axis = 0)
    SouthFace = np.repeat(SFace, ndepth, axis = 0)
    WestFace = np.repeat(WFace, ndepth, axis = 0)
    EastFace = np.repeat(EFace, ndepth, axis = 0)
    
    #Control Volume Mask on Rho points
    RhoMask = np.repeat(mt.RhoMask(Avg, latbounds, lonbounds)[np.newaxis, :, :], \
                        ndepth, axis = 0)
    
    Masks = {
            'RhoMask':RhoMask,\
            'Umask' : U_Mask, \
            'Vmask' : V_Mask, \
            'NFace' : NorthFace, \
            'WFace' : WestFace, \
            'SFace' : SouthFace, \
            'EFace' : EastFace, 
            }
    
    return Masks

def CellAreas(AvgFile, Avg, GridFile, Masks) :
    """
    Compute cell areas
    """
    #area of upward normal faces
    Axy = dff.dA_top(Avg)
    
    #Areas of all cell faces
    Ax_norm, Ay_norm = dff.dA(AvgFile, GridFile)
    
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
    
    Int_Sprime = ma.sum(2*vprime2*(var_rate - var/deltaA*dDelta_dt)*dV)
    
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
    
def Diff_Flux(tstep, vprime2, Avg, AvgFile, GridFile, Masks, Areas) :
    """
    Diffusive flux across CV boundaries
    """
    #Shift from rho to u and v points
    _var_u = GridShift.Rho_to_Upt(vprime2)
    _var_v = GridShift.Rho_to_Vpt(vprime2)
    
    #apply mask that is shifted to u and v points to variance squared
    var2_u = ma.array(_var_u, mask = Masks['U_Mask'])
    var2_v = ma.array(_var_v, mask = Masks['V_Mask'])
    
    #gradients of variance squared
    dprime_u_dx = gr.x_grad_u(AvgFile, GridFile, var2_u)
    dprime_v_dy = gr.y_grad_v(AvgFile, GridFile, var2_v)
    
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

def Int_Mixing(tstep, vprime2, Avg, AvgFile, GridFile, Masks) :
    """
    Internal mixing within a control volume
    """
    #compute gradients squared
    xgrad = ma.array(gr.x_grad_rho(AvgFile, GridFile, vprime2), \
                     mask = Masks['RhoMask'])
    ygrad = ma.array(gr.y_grad_rho(AvgFile, GridFile, vprime2), \
                     mask = Masks['RhoMask'])
    zgrad = ma.array(gr.z_grad_rho(AvgFile, GridFile, vprime2), \
                     mask = Masks['RhoMask'])
    
    #vertical viscosity on w points
    _Kv_w = Avg.variables['Aks'][tstep, :, :, :]
    
    #shift to rho points
    Kv = ma.array(GridShift.Wpt_to_Rho(_Kv_w), mask = Masks['RhoMask']
    
    #horizontal viscosity
    Kh = Avg.variables['nl_tnu2'][0]
    
    #differential volume
    dV = ma.array(df.dV(AvgFile), mask = Masks['RhoMask'])
    
    #integrad
    x_m = 2*Kh*xgrad
    y_m = 2*Kh*ygrad
    z_m = 2*Kv*zgrad
    
    mixing = (ma.sum(x_m) + ma.sum(y_m) + ma.sum(z_m))*dV
    
    return mixing
    