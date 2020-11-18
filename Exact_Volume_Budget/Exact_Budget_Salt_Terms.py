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
import Gradients_TstepMask as gr

def CreateMasks(RomsNC, Vertices) :
    """
    define masks on rho, U, V points and along boundaries
    returns dictionary of masks
    """
    #lats and lons
    lats = RomsNC.variables['lat_rho'][:]
    lons = RomsNC.variables['lon_rho'][:]
    
    #variable
    salt = RomsNC.variables['salt'][0,:,:,:]
    nz = salt.shape[0]
    
    #land mask
    land = np.invert(np.ma.getmask(salt))
    
    
    #mask on Rho, U, V points repeated over depth space
    Rmask = np.array(np.repeat(mt.RhoMask(lats, lons, Vertices)[np.newaxis,:,:], nz, axis = 0), dtype = bool)
    Umask = np.array(GridShift.Bool_Rho_to_Upt(Rmask), dtype = bool)
    Vmask = np.array(GridShift.Bool_Rho_to_Vpt(Rmask), dtype = bool)
    
    #face masks
    NFace, WFace, SFace, EFace = mt.FaceMask_shift3D(Rmask)
    
    #intersection of land mask and face mask
    NFace = np.logical_and(NFace, land)
    WFace = np.logical_and(WFace, land)
    SFace = np.logical_and(SFace, land)
    EFace = np.logical_and(EFace, land)
    
    #shift to U or V
    NFace = np.array(GridShift.Bool_Rho_to_Vpt(NFace), dtype = bool)
    SFace = np.array(GridShift.Bool_Rho_to_Vpt(SFace), dtype = bool)
    WFace = np.array(GridShift.Bool_Rho_to_Upt(WFace), dtype = bool)
    EFace = np.array(GridShift.Bool_Rho_to_Upt(EFace), dtype = bool)
    
    #find intersection between land mask and rho mask
    Rmask = np.logical_and(Rmask, land)
        
    Masks = {
            'RhoMask':Rmask,\
            'U_Mask' : Umask,\
            'V_Mask' : Vmask,\
            'NFace' : NFace, \
            'WFace' : WFace, \
            'SFace' : SFace, \
            'EFace' : EFace, \
            'LandMask': land
            }
    
    return Masks

def CellAreas(dx, dy, dz, Masks) :
    """
    Compute cell areas
    """
    #area of upward normal faces
    dx_mask = dx*Masks['RhoMask']
    dy_mask = dy*Masks['RhoMask']
    Axy =dx_mask*dy_mask
    
    #Areas of all cell faces
    Ax_norm, Ay_norm = dff.dA_norm(dx, dy, dz)
    
    #subset areas to faces of CV
    Areas = {
            'Axy' : Axy, \
            'North_Ay' : Ay_norm*Masks['NFace'], \
            'West_Ax' : Ax_norm*Masks['WFace'], \
            'South_Ay' : Ay_norm*Masks['SFace'], \
            'East_Ax' : Ax_norm*Masks['EFace']
            }
    
    return Areas


def TimeDeriv(tstep,salt, Hist, HistFile, Avg, AvgFile, Diag, dA_xy, Masks):
    """
    Exact volume time derivative of salt
    """
    #change in vertical thickness of cell at average point
    deltaA = np.diff(dep._set_depth(AvgFile, None, 'w', \
                                             Avg.variables['h'][:], \
                                             Avg.variables['zeta'][tstep,:,:]),
                              n = 1, axis = 0)
    
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
    
    salt = salt*Masks['RhoMask']
    
        
    Int_Sprime = np.sum((var_rate - (salt/deltaA)*dDelta_dt)*dV)
    
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
    North_V = Avg.variables['v'][tstep, :, :, :]*Masks['NFace'] #-1
    West_U = -1*Avg.variables['u'][tstep, :, :, :]*Masks['WFace']
    South_V = -1*Avg.variables['v'][tstep, :, :, :]*Masks['SFace']
    East_U = Avg.variables['u'][tstep, :, :, :]*Masks['EFace'] #-1
    
    #Advective Fluxes through each boundary
    North = North_var*North_V*Areas['North_Ay']
    West = West_var*West_U*Areas['West_Ax']
    South = South_var*South_V*Areas['South_Ay']
    East = East_var*East_U*Areas['East_Ax']
    
    #sum (integrate) time step
    AdvFlux = np.sum(North) + np.sum(West) + np.sum(South) + np.sum(East)
    
    return AdvFlux
    
def Adv_Flux_west(tstep, var, Avg, Areas, Masks):
    """
    Advective flux through western boundary only
    """
    #shift to u points
    West_var = GridShift.Rho_to_Upt(var)*Masks['WFace']
    
    #Velocity at western face
    West_U = -1*Avg.variables['u'][tstep, :, :, :]*Masks['WFace']
    
    #Advective flux through western boundary
    West = West_var*West_U*Areas['West_Ax']
    AdvFluxWest = np.sum(West)
    
    return AdvFluxWest    
    
def Adv_Flux_west_diag(tstep, var, Diag, dx,dy,dz, Masks):
    """
    
    """
    #shift to u points
    West_var = var[Masks['RhoMask']]
    
    #x advection
    xadv = Diag.variables['salt_xadv'][tstep,:,:,:])[Masks['RhoMask']]
    
    #cell volume
    dV =GridShift.Rho_to_Upt(dx*dy*dz)[Masks['U_Mask']]
    
    #advective flux rough western boundary
    AdvFluxWest = np.sum(West_var*xadv*dV)
    
    return AdvFluxWest
    
def Ad_Flux_div(tstep, var, Avg, dx, dy, dz, Masks):
    """
    Advective flux with divergence method
    """
    #gradients of sprime2
    xgrad = gr.x_grad_rho(var, dx, Masks)[Masks['RhoMask']]
    ygrad = gr.y_grad_rho(var, dy, Masks)[Masks['RhoMask']]
    zgrad = gr.z_grad(var, dz)[Masks['RhoMask']]
    
    u = np.ma.array(GridShift.Upt_to_Rho(Avg.variables['u'][tstep, :,:, :]), \
                    mask = np.invert(Masks['LandMask']))
    ur = GridShift.Upt_to_Rho(uu)
    
    dV = (dx*dy*dz)[Masks['RhoMask']]
    
    #horizontal divergence
    div_i = Avg.variables['u'][tstep, :,:,:][Masks['U_Mask']]*xgrad
    div_j = Avg.variables['v'][tstep, :,:,:][Masks['V_Mask']]*ygrad
    
    #vertical divergence
    div_k = GridShift.Wpt_to_Rho(Avg.variables['w'][tstep, :,:,:])[Masks['RhoMask']]*zgrad

  
    
    div = (div_i + div_j + div_k)*dV
    
    AdvFlux = np.sum(div)
    
    return AdvFlux

    
def Diff_Flux(Avg, vprime2, dx, dy, Areas, Masks) :
    """
    Diffusive flux across CV boundaries
    """
    #gradients of variance squared
    dprime_u_dx = gr.x_grad_u(vprime2, dx, Masks)
    dprime_v_dy = gr.y_grad_v(vprime2, dy, Masks)
    
    #apply face masks to gradients
    North_grad = dprime_v_dy*Masks['NFace']
    West_grad = dprime_u_dx*Masks['WFace']
    
    South_grad = dprime_v_dy*Masks['SFace']
    East_grad = dprime_u_dx*Masks['EFace']
    
    #horizontal diffustion coef
    Ks = Avg.variables['nl_tnu2'][0]
    
    #diffusive fluxes through each boundary
    North = Ks*North_grad*Areas['North_Ay']
    West = -1*Ks*West_grad*Areas['West_Ax']
    South = -1*Ks*South_grad*Areas['South_Ay']
    East = Ks*East_grad*Areas['East_Ax']
    
    DifFlux = np.sum(North) + np.sum(West) + np.sum(South) + np.sum(East)
    
    return DifFlux
    
def Diff_Flux_west(Avg, salt, dx, Areas, Masks) :
    """
    Diffusive flux through western boundary only
    """
    #zonal gradient
    dprime_u_dx = gr.x_grad_u(salt, dx, Masks)
    
    #subset west face
    West_grad = dprime_u_dx*Masks['WFace']
    
    #horizontal diffusion coef
    Ks = Avg.variables['nl_tnu2'][0]
    
    #diffusive flux
    West = -1*Ks*West_grad*Areas['West_Ax']
    DifFlux_west = np.sum(West)
    
    return DifFlux_west
    
    

def Int_Mixing(tstep, var, Avg, dx, dy, dz, Masks) :
    """
    Internal Mixing Term 
    """
    #gradient of salt
    xgrad = gr.x_grad_rho(var, dx, Masks)
    ygrad = gr.y_grad_rho(var, dy, Masks)
    zgrad = gr.z_grad(var, dz)
    
    #eddy viscosity coefs
    kv = GridShift.Wpt_to_Rho(Avg.variables['AKs'][tstep, :, :, :])
    kh = Avg.variables['nl_tnu2'][0]
    
    #differential volume
    dV = dx*dy*dz
    
    #diverence
    xdiv = gr.x_grad_rho(kh*xgrad, dx, Masks)[Masks['RhoMask']]
    ydiv = gr.y_grad_rho(kh*ygrad, dy, Masks)[Masks['RhoMask']]
    zdiv = gr.z_grad(kv*zgrad,dz)[Masks['RhoMask']]
    
   
    
    div = xdiv + ydiv + zdiv
    
    #integrate
    mixing = np.sum(div*dV[Masks['RhoMask']])
        
    return mixing