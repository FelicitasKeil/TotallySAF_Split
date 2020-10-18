
##################################################################################
#Author: S. Yahia-Cherif
#This script computes the WL Fisher
##################################################################################

#Modules import.
import sys
import numpy as np
import scipy.integrate as pyint
import os
from os import path
import glob
import scipy
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
import multiprocessing as mp
from multiprocessing import Pool
import scipy.ndimage
from numba import jit, njit
from numba import autojit, prange
from scipy import integrate

paramo = 0
curvature = 'F'
gamma_MG = 'N'
zcut = 'N'
if curvature == 'F' and paramo == 4:
    sys.exit()
if gamma_MG == 'N' and paramo == 8:
    sys.exit()
if zcut == 'Y' and paramo > 16:
    sys.exit()


Pk_Path = "WP_Pk/"
pre_CC_path = ["Cl_GG/", "Cl_LL/", "Cl_GL/"]

#Fiducial, l, fsky
Omega_b, h, Omega_m, ns, Omega_DE, w0, wa, sigma_8, gamma = 0.05, 0.67*np.ones(5), 0.32*np.ones(5), 0.96, 0.68*np.ones(5), -1.*np.ones(5), 0.*np.ones(5), 0.81553388, 6./11*np.ones(5)
C_IA, A_IA, n_IA, B_IA = 0.0134, 1.72*np.ones(5), -0.41*np.ones(5), 2.17*np.ones(5)
eps_wb, eps_h, eps_wm, eps_ns, eps_wde, eps_w0, eps_wa, eps_s8, eps_gamma, eps_b = 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04
eps_A_IA, eps_n_IA, eps_B_IA = 0.04, 0.04, 0.04
fold_path = [Pk_Path+"fid", Pk_Path+"fid", Pk_Path+"fid", Pk_Path+"fid", Pk_Path+"fid"]

#BIAS
def bias(z_ind):
    return np.sqrt(1.+z_ind)

zrange = np.array([0.2095, 0.489, 0.619, 0.7335, 0.8445, 0.9595, 1.087, 1.2395, 1.45, 2.038])
BX = np.zeros((10,5))
i,j=0,0
while i < len(BX):
    while j < len(BX[0]):
        BX[i][j] = bias(zrange[i])
        if zcut == 'Y':
            if i > 4:
                BX[i][j] = 0.
        j=j+1
    j=0
    i=i+1
i,j=0,0

if paramo == 0:
    fold_path = [Pk_Path+"wb_up", Pk_Path+"wb_up2", Pk_Path+"fid", Pk_Path+"wb_dw", Pk_Path+"wb_dw2"]
    CC_path = ["C_wb_up", "C_wb_up2", "C_fid", "C_wb_dw", "C_wb_dw2"]
elif paramo == 1:
    h = np.array([h[0]*(1.+eps_h), h[1]*(1.+2*eps_h), h[2], h[3]*(1.-eps_h), h[4]*(1.-2*eps_h)])
    fold_path = [Pk_Path+"h_up", Pk_Path+"h_up2", Pk_Path+"fid", Pk_Path+"h_dw", Pk_Path+"h_dw2"]
    CC_path = ["C_h_up", "C_h_up2", "C_fid", "C_h_dw", "C_h_dw2"]
elif paramo == 2:
    Omega_m = np.array([Omega_m[0]*(1.+eps_wm), Omega_m[1]*(1.+2*eps_wm), Omega_m[2], Omega_m[3]*(1.-eps_wm), Omega_m[4]*(1.-2*eps_wm)])
    fold_path = [Pk_Path+"wm_up", Pk_Path+"wm_up2", Pk_Path+"fid", Pk_Path+"wm_dw", Pk_Path+"wm_dw2"]
    CC_path = ["C_wm_up", "C_wm_up2", "C_fid", "C_wm_dw", "C_wm_dw2"]
elif paramo == 3:
    fold_path = [Pk_Path+"ns_up", Pk_Path+"ns_up2", Pk_Path+"fid", Pk_Path+"ns_dw", Pk_Path+"ns_dw2"]
    CC_path = ["C_ns_up", "C_ns_up2", "C_fid", "C_ns_dw", "C_ns_dw2"] 
elif paramo == 4:
    Omega_DE = np.array([Omega_DE[0]*(1.+eps_wde), Omega_DE[1]*(1.+2*eps_wde), Omega_DE[2], Omega_DE[3]*(1.-eps_wde), Omega_DE[4]*(1.-2*eps_wde)])  
    fold_path = [Pk_Path+"wde_up", Pk_Path+"wde_up2", Pk_Path+"fid", Pk_Path+"wde_dw", Pk_Path+"wde_dw2"]
    CC_path = ["C_wde_up", "C_wde_up2", "C_fid", "C_wde_dw", "C_wde_dw2"]  
elif paramo == 5:
    w0 = np.array([w0[0]*(1.+eps_w0), w0[1]*(1.+2*eps_w0), w0[2], w0[3]*(1.-eps_w0), w0[4]*(1.-2*eps_w0)])
    fold_path = [Pk_Path+"w0_up", Pk_Path+"w0_up2", Pk_Path+"fid", Pk_Path+"w0_dw", Pk_Path+"w0_dw2"]
    CC_path = ["C_w0_up", "C_w0_up2", "C_fid", "C_w0_dw", "C_w0_dw2"]
elif paramo == 6:
    wa = np.array([wa[0]+eps_wa, wa[1]+2*eps_wa, wa[2], wa[3]-eps_wa, wa[4]-2*eps_wa])
    fold_path = [Pk_Path+"wa_up", Pk_Path+"wa_up2", Pk_Path+"fid", Pk_Path+"wa_dw", Pk_Path+"wa_dw2"]
    CC_path = ["C_wa_up", "C_wa_up2", "C_fid", "C_wa_dw", "C_wa_dw2"]
elif paramo == 7:
    fold_path = [Pk_Path+"s8_up", Pk_Path+"s8_up2", Pk_Path+"fid", Pk_Path+"s8_dw", Pk_Path+"s8_dw2"]
    CC_path = ["C_s8_up", "C_s8_up2", "C_fid", "C_s8_dw", "C_s8_dw2"]
elif paramo == 8:
    gamma = np.array([gamma[0]*(1.+eps_gamma), gamma[1]*(1.+2*eps_gamma), gamma[2], gamma[3]*(1.-eps_gamma), gamma[4]*(1.-2*eps_gamma)])
    CC_path = ["C_gamma_up", "C_gamma_up2", "C_fid", "C_gamma_dw", "C_gamma_dw2"]
elif paramo == 9:
    A_IA = np.array([A_IA[0]*(1.+eps_A_IA), A_IA[1]*(1.+2*eps_A_IA), A_IA[2], A_IA[3]*(1.-eps_A_IA), A_IA[4]*(1.-2*eps_A_IA)])
    CC_path = ["C_A_IA_up", "C_A_IA_up2", "C_fid", "C_A_IA_dw", "C_A_IA_dw2"]
elif paramo == 10:
    n_IA = np.array([n_IA[0]*(1.+eps_n_IA), n_IA[1]*(1.+2*eps_n_IA), n_IA[2], n_IA[3]*(1.-eps_n_IA), n_IA[4]*(1.-2*eps_n_IA)])
    CC_path = ["C_n_IA_up", "C_n_IA_up2", "C_fid", "C_n_IA_dw", "C_n_IA_dw2"]
elif paramo == 11:
    B_IA = np.array([B_IA[0]*(1.+eps_B_IA), B_IA[1]*(1.+2*eps_B_IA), B_IA[2], B_IA[3]*(1.-eps_B_IA), B_IA[4]*(1.-2*eps_B_IA)])
    CC_path = ["C_B_IA_up", "C_B_IA_up2", "C_fid", "C_B_IA_dw", "C_B_IA_dw2"]
elif  paramo > 11:
    CC_path = ["C_b"+str(paramo-11)+"_up", "C_b"+str(paramo-11)+"_up2", "C_fid", "C_b"+str(paramo-11)+"_dw", "C_b"+str(paramo-11)+"_dw2"]
    BX[paramo-12] = [BX[paramo-12][0]*(1.+eps_b), BX[paramo-12][1]*(1.+2*eps_b), BX[paramo-12][2], BX[paramo-12][3]*(1.-eps_b), BX[paramo-12][4]*(1.-2*eps_b)]

files_In = [glob.glob(fold_path[0] + '/*'), glob.glob(fold_path[1] + '/*'), glob.glob(fold_path[2] + '/*'), glob.glob(fold_path[3] + '/*'), glob.glob(fold_path[4] + '/*')]

i=0
while i < len(CC_path):
    if not os.path.exists(pre_CC_path[0]+CC_path[i]):
        os.makedirs(pre_CC_path[0]+CC_path[i])
    if not os.path.exists(pre_CC_path[1]+CC_path[i]):
        os.makedirs(pre_CC_path[1]+CC_path[i])
    if not os.path.exists(pre_CC_path[2]+CC_path[i]):
        os.makedirs(pre_CC_path[2]+CC_path[i])
    i=i+1
i=0

l_edge = np.linspace(10, 5000, 61)
l = np.zeros(len(l_edge)-1)
i=1
while i < len(l_edge):
    l[i-1] = (l_edge[i] + l_edge[i-1])/2
    i=i+1
delta_l = l[1]-l[0]
fsky = 0.3636099192785979
c = 299792.458

NNN = 600
MMM = 60
BBB = 1000
PPP = 10000
WWW = 600
#Redshift and n(z)    
zi_p = np.array([0.418, 0.560, 0.678, 0.789, 0.9, 1.019, 1.155, 1.324, 1.576, 3.731])
zi_m = np.array([0.001, 0.418, 0.560, 0.678, 0.789, 0.9, 1.019, 1.155, 1.324, 1.576])
zmax = np.max(zi_p)
zmin = np.min(zi_m)
zpm = np.linspace(zmin,zmax, NNN)
zsec = np.linspace(zmin,zmax, MMM)
zbkgnd = np.linspace(zmin,zmax, BBB)

z_win_x_y = np.linspace(zmin,zmax, WWW)
z_win_3_1 = 1./4*(3*z_win_x_y[1:]+z_win_x_y[:-1])
z_win_1_3 = 1./4*(z_win_x_y[1:]+3*z_win_x_y[:-1])
z_win_1_2 = 1./2*(z_win_x_y[1:]+z_win_x_y[:-1])
z_win = np.zeros(len(z_win_x_y)+len(z_win_3_1)+len(z_win_1_2)+len(z_win_1_3))
i,j=0,0
while i < len(z_win):
    if i == 0:
        z_win[i] = z_win_x_y[j]
        i=i+1
        j=j+1
    elif i > 0 and i < len(z_win):
        z_win[i] = z_win_1_3[j-1]
        z_win[i+1] = z_win_1_2[j-1]
        z_win[i+2] = z_win_3_1[j-1]
        z_win[i+3] = z_win_x_y[j]
        i=i+4
        j=j+1

z500 = np.copy(zsec)
z_diff = zmax - zmin
delta_zpm = zpm[1] - zpm[0]
delta_zsec = zsec[1] - zsec[0]
delta_z500 = z500[1] - z500[0]
zrange_0 = 0.9/np.sqrt(2)
ng = 354543085.80106884
#Photometric distribution of sources
cb = 1.0
zb = 0.
sig_b = 0.05
c0 = 1.
z0 = 0.1
sig_0 = 0.05
f_out = 0.1

#Shot noise
sig_epsilon = 0.3

#Luminosity table
zvals, Lumos = np.loadtxt("scaledmeanlum-E2Sa.dat", usecols=(0,1), unpack="True")
Lum = interp1d(zvals,Lumos,kind="linear")

#DENSITY, SHOT NOISE
def nz(z):
    return (z/zrange_0)**2*np.exp(-(z/zrange_0)**1.5)

def P_phot(z, zp):
    return (1. -  f_out)/(np.sqrt(2.*np.pi)*sig_b*(1+z))*np.exp(-0.5*((z - cb*zp - zb)/(sig_b*(1.+z)))**2) + f_out/(np.sqrt(2.*np.pi)*sig_0*(1+z))*np.exp(-0.5*((z - c0*zp - z0)/(sig_0*(1.+z)))**2)

@njit
def D_INT(TZ, zp, z500, delta_zp, TTZ_I):
    for sss in range(len(TZ)):
        TTZ_I[sss] = delta_zp/90*np.sum(7*(TZ[sss, 1:] + TZ[sss, :-1]) + 32*((TZ[sss, 1:] + (1./4*(3*z500[1:]+z500[:-1]) - z500[1:])*(TZ[sss, :-1]-TZ[sss, 1:])/(z500[:-1]-z500[1:])) + (TZ[sss, 1:] + (1./4*(z500[1:]+3*z500[:-1]) - z500[1:])*(TZ[sss, :-1]-TZ[sss, 1:])/(z500[:-1]-z500[1:]))) + 12*(TZ[sss, 1:] + (1./2*(z500[1:]+z500[:-1]) - z500[1:])*(TZ[sss, :-1]-TZ[sss, 1:])/(z500[:-1]-z500[1:])) )
        sss=sss+1
    return TTZ_I

def Prnd(z, z_ind):
    i_search = np.where((z_ind >= zi_m) & (z_ind <= zi_p))
    z_p = zi_p[i_search[0][0]]
    z_m = zi_m[i_search[0][0]]
    zp = np.linspace(z_m, z_p, 10000)
    delta_zp = zp[1]-zp[0]
    
    ZZPM, ZZP = np.meshgrid(z500, zp)

    Trap = delta_zp/90*nz(z)*np.sum(7*(P_phot(z, zp[1:]) + P_phot(z, zp[:-1])) + 32*(P_phot(z, 1./4*(3*zp[1:]+zp[:-1])) + P_phot(z, 1./4*(zp[1:]+3*zp[:-1]))) + 12*P_phot(z, 1./2*(zp[1:]+zp[:-1])))
    TZ = P_phot(ZZPM[:, :], ZZP[:, :])*nz(ZZPM[:, :])
    TTZ_I = np.zeros(len(zp))
    TTZ = D_INT(TZ, zp, z500, delta_zp, TTZ_I)
    Trap_Z = delta_z500/90*np.sum(7*(TTZ[1:] + TTZ[:-1]) +32*( ((TTZ[1:] + (1./4*(3*zp[1:]+zp[:-1]) - zp[1:])*(TTZ[:-1]-TTZ[1:])/(zp[:-1]-zp[1:]))) + ((TTZ[1:] + (1./4*(zp[1:]+3*zp[:-1]) - zp[1:])*(TTZ[:-1]-TTZ[1:])/(zp[:-1]-zp[1:])))) +12*(TTZ[1:] + (1./2*(zp[1:]+zp[:-1]) - zp[1:])*(TTZ[:-1]-TTZ[1:])/(zp[:-1]-zp[1:])))
    return Trap_Z

def P_shot_WL(zpi, zpj):
    if zpi == zpj:
        return sig_epsilon**2/(ng/10.)
    else:
        return 0.

def P_shot_GC(zpi, zpj):
    if zpi == zpj:
        return 1./(ng/10.)
    else:
        return 0.

#BCKGND    
def b_new(z, w0_p, wa_p):
    return pyint.quad(b_aux, 0, z, args = (w0_p, wa_p))[0]

def b_aux(z, w0_p, wa_p):
    return 1./(1+z)*(1+w0_p+wa_p*z/(1+z))

def E(z, h_p, wm_p, wDE_p, w0_p, wa_p): 
    if curvature == 'NF':
        return np.sqrt( wm_p*(1+z)**3 + wDE_p*np.exp(3*b_new(z, w0_p, wa_p)) + (1.-wm_p-wDE_p)*(1+z)**2)
    elif curvature == 'F':
        return np.sqrt( wm_p*(1+z)**3 + (1.-wm_p)*np.exp(3*b_new(z, w0_p, wa_p)))
    
def chi_aux_R(z, h_p, wm_p, wDE_p, w0_p, wa_p):
    return c/(100.*h_p*E(z, h_p, wm_p, wDE_p, w0_p, wa_p))
    
def chi_R(z, h_p, wm_p, wDE_p, w0_p, wa_p):  
    return pyint.quad(chi_aux_R, 0, z, args = (h_p, wm_p, wDE_p, w0_p, wa_p))[0]

def chi_R_tilde(z, h_p, wm_p, wDE_p, w0_p, wa_p):
    return chi_R(z, h_p, wm_p, wDE_p, w0_p, wa_p)/(c/(100*h_p))

def OM_M(z, h_p, wm_p, wDE_p, w0_p, wa_p, gamma_p):
    return (wm_p*(1.+z)**3/(E(z, h_p, wm_p, wDE_p, w0_p, wa_p)**2))**gamma_p/(1.+z)
    
def DGrowth(z, h_p, wm_p, wDE_p, w0_p, wa_p, gamma_p):
    return np.exp(-pyint.quad(OM_M, 0, z, args = (h_p, wm_p, wDE_p, w0_p, wa_p, gamma_p))[0])

#WEIGHT
def Weight_P(z, z_ind, h_p, wm_p, wDE_p, w0_p, wa_p, PT_T, E_T, ind_step):
    i_s = np.where((z >= zi_m) & (z <= zi_p))
    return 100.*h_p/c*BX[i_s[0][0]][ind_step]*PT_T(z)*E_T

def Weight_F(z, z_ind, h_p, wm_p, wDE_p, w0_p, wa_p, RT_T, PT_T):

    z_act_max = np.linspace(z, zmax, int(np.ceil((zmax-z)/delta_zsec*4)))
    if len(z_act_max) < 2:
        return 0.
    delta_z_act_max = z_act_max[1] - z_act_max[0]
    return 1.5*100.*h_p*wm_p*(1.+z)/c*RT_T(z)*delta_z_act_max/90*np.sum(7*(PT_T(z_act_max[1:])*(1. - RT_T(z)/RT_T(z_act_max[1:])) + (PT_T(z_act_max[:-1])*(1. - RT_T(z)/RT_T(z_act_max[:-1])))) + 32*(PT_T(1./4*(3*z_act_max[1:]+z_act_max[:-1]))*(1. - RT_T(z)/RT_T(1./4*(3*z_act_max[1:]+z_act_max[:-1]))) + PT_T(1./4*(z_act_max[1:]+3*z_act_max[:-1]))*(1. - RT_T(z)/RT_T(1./4*(z_act_max[1:]+3*z_act_max[:-1])))) + 12*(PT_T(1./2*(z_act_max[1:]+z_act_max[:-1]))*(1. - RT_T(z)/RT_T(1./2*(z_act_max[1:]+z_act_max[:-1])))))

def Weight_IA(z, z_ind, h_p, wm_p, wDE_p, w0_p, wa_p, E_T, PT_T):
    return 100.*h_p/c*PT_T*E_T

def F_IA(z, h_p, wm_p, wDE_p, w0_p, wa_p, n_IAp, B_IAp):
    return (1.+z)**n_IAp*Lum(z)**B_IAp

#GG
def F_dd_GG(z, zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, E_T, R_T, WGT_a, WGT_b, DG_T, P_dd):
    return WGT_a*WGT_b/(E_T*R_T**2)*P_dd

#WL
def P_dI(z, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, DG_T, P_dd):
    return - A_IAp*C_IAp*wm_p*F_IA(z, h_p, wm_p, wDE_p, w0_p, wa_p, n_IAp, B_IAp)/DG_T*P_dd
    
def P_II(z, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, DG_T, P_dd):
    return (-A_IAp*C_IAp*wm_p*F_IA(z, h_p, wm_p, wDE_p, w0_p, wa_p, n_IAp, B_IAp)/DG_T)**2*P_dd
    
def F_dd_LL(z, zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, E_T, R_T, WT_a, WT_b, DG_T, P_dd):
    return WT_a*WT_b/(E_T*R_T**2)*P_dd
    
def F_IA_d(z, zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, E_T, R_T, DG_T, WT_a, WT_b, WIAT_a, WIAT_b, P_dd):
    return (WT_a*WIAT_b + WIAT_a*WT_b)/(E_T*R_T**2)*P_dI(z, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, DG_T, P_dd)
    
def F_IAIA(z, zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, E_T, R_T, DG_T, WIAT_a, WIAT_b, P_dd):
    return WIAT_a*WIAT_b/(E_T*R_T**2)*P_II(z, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, DG_T, P_dd)

#XC  
def P_WL(z, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, DG_T, WT_b, WIAT_b):
    return WT_b - WIAT_b*A_IAp*C_IAp*wm_p*F_IA(z, h_p, wm_p, wDE_p, w0_p, wa_p, n_IAp, B_IAp)/DG_T

def F_dd_GL(z, zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, E_T, R_T, DG_T, WGT_a, WT_b, WIAT_b, P_dd):
    return WGT_a*P_WL(z, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, DG_T, WT_b, WIAT_b)/(E_T*R_T**2)*P_dd

def parainterp(P_dd_spec, cl, RT500, P_dd_C):
    P_dd_ok = np.zeros(len(P_dd_C))
    for cc in range(len(P_dd_C)): 
        if ((cl+0.5)/RT500[cc] < 35 and (cl+0.5)/RT500[cc] > 0.0005):
            P_dd_ok[cc] = P_dd_spec[cc]((cl+0.5)/RT500[cc])
    return P_dd_ok

def Pobs_C(z, zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, E_T, R_T, DG_T_fid, DG_T, WGT_T, WT_T, WIAT_T, cl, P_dd_INT, RT500):
    if paramo != 9 or paramo != 10 or paramo != 11:
        C_gg = c/(100.*h_p)*delta_zpm/90*np.sum(7*(F_dd_GG(z[1:], zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, E_T(z[1:]), R_T(z[1:]), WGT_T[aa][0:len(WGT_T[aa])-1:4], WGT_T[bb][0:len(WGT_T[bb])-1:4], DG_T(z[1:]), P_dd_INT[0:len(P_dd_INT)-1:4]) + F_dd_GG(z[:-1], zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, E_T(z[:-1]), R_T(z[:-1]), WGT_T[aa][4:len(WGT_T[aa]):4], WGT_T[bb][4:len(WGT_T[bb]):4], DG_T(z[:-1]), P_dd_INT[4:len(P_dd_INT):4])) + 32*(F_dd_GG(1./4*(3*z[1:]+z[:-1]), zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, E_T(1./4*(3*z[1:]+z[:-1])), R_T(1./4*(3*z[1:]+z[:-1])), WGT_T[aa][1::4], WGT_T[bb][1::4], DG_T(1./4*(3*z[1:]+z[:-1])), P_dd_INT[1::4]) + F_dd_GG(1./4*(z[1:]+3*z[:-1]), zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, E_T(1./4*(z[1:]+3*z[:-1])), R_T(1./4*(z[1:]+3*z[:-1])), WGT_T[aa][3::4], WGT_T[bb][3::4], DG_T(1./4*(z[1:]+3*z[:-1])), P_dd_INT[3::4])) + 12*F_dd_GG(1./2*(z[1:]+z[:-1]), zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, E_T(1./2*(z[1:]+z[:-1])), R_T(1./2*(z[1:]+z[:-1])), WGT_T[aa][2::4], WGT_T[bb][2::4], DG_T(1./2*(z[1:]+z[:-1])), P_dd_INT[2::4])) + P_shot_GC(zi, zj)
    else:
        C_gg = 0.
    if paramo < 12:
        C_ee = c/(100.*h_p)*delta_zpm/90*(np.sum(7*(F_dd_LL(z[1:], zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, E_T(z[1:]), R_T(z[1:]), WT_T[aa][0:len(WT_T[aa])-1:4], WT_T[bb][0:len(WT_T[aa])-1:4], DG_T(z[1:]), P_dd_INT[0:len(P_dd_INT)-1:4]) + F_dd_LL(z[:-1], zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, E_T(z[:-1]), R_T(z[:-1]), WT_T[aa][4:len(WT_T[aa]):4], WT_T[bb][4:len(WT_T[aa]):4], DG_T(z[:-1]), P_dd_INT[4:len(P_dd_INT):4])) + 32*(F_dd_LL(1./4*(3*z[1:]+z[:-1]), zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, E_T(1./4*(3*z[1:]+z[:-1])), R_T(1./4*(3*z[1:]+z[:-1])), WT_T[aa][1::4], WT_T[bb][1::4], DG_T(1./4*(3*z[1:]+z[:-1])), P_dd_INT[1::4]) + F_dd_LL(1./4*(z[1:]+3*z[:-1]), zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, E_T(1./4*(z[1:]+3*z[:-1])), R_T(1./4*(z[1:]+3*z[:-1])), WT_T[aa][3::4], WT_T[bb][3::4], DG_T(1./4*(z[1:]+3*z[:-1])), P_dd_INT[3::4])) + 12*F_dd_LL(1./2*(z[1:]+z[:-1]), zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, E_T(1./2*(z[1:]+z[:-1])), R_T(1./2*(z[1:]+z[:-1])), WT_T[aa][2::4], WT_T[bb][2::4], DG_T(1./2*(z[1:]+z[:-1])), P_dd_INT[2::4]) ) + np.sum(7*(F_IA_d(z[1:], zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, E_T(z[1:]), R_T(z[1:]), DG_T(z[1:]), WT_T[aa][0:len(WT_T[aa])-1:4], WT_T[bb][0:len(WT_T[aa])-1:4], WIAT_T[aa][0:len(WIAT_T[aa])-1:4], WIAT_T[bb][0:len(WIAT_T[aa])-1:4], P_dd_INT[0:len(P_dd_INT)-1:4]) + F_IA_d(z[:-1], zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, E_T(z[:-1]), R_T(z[:-1]), DG_T(z[:-1]), WT_T[aa][4:len(WT_T[aa]):4], WT_T[bb][4:len(WT_T[aa]):4], WIAT_T[aa][4:len(WIAT_T[aa]):4], WIAT_T[bb][4:len(WIAT_T[aa]):4], P_dd_INT[4:len(P_dd_INT):4])) + 32*(F_IA_d(1./4*(3*z[1:]+z[:-1]), zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, E_T(1./4*(3*z[1:]+z[:-1])), R_T(1./4*(3*z[1:]+z[:-1])), DG_T(1./4*(3*z[1:]+z[:-1])), WT_T[aa][1::4], WT_T[bb][1::4], WIAT_T[aa][1::4], WIAT_T[bb][1::4], P_dd_INT[1::4]) + F_IA_d(1./4*(z[1:]+3*z[:-1]), zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, E_T(1./4*(z[1:]+3*z[:-1])), R_T(1./4*(z[1:]+3*z[:-1])), DG_T(1./4*(z[1:]+3*z[:-1])), WT_T[aa][3::4], WT_T[bb][3::4], WIAT_T[aa][3::4], WIAT_T[bb][3::4], P_dd_INT[3::4])) + 12*F_IA_d(1./2*(z[1:]+z[:-1]), zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, E_T(1./2*(z[1:]+z[:-1])), R_T(1./2*(z[1:]+z[:-1])), DG_T(1./2*(z[1:]+z[:-1])), WT_T[aa][2::4], WT_T[bb][2::4], WIAT_T[aa][2::4], WIAT_T[bb][2::4], P_dd_INT[2::4])) + np.sum(7*(F_IAIA(z[1:], zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, E_T(z[1:]), R_T(z[1:]), DG_T(z[1:]), WIAT_T[aa][0:len(WIAT_T[aa])-1:4], WIAT_T[bb][0:len(WIAT_T[aa])-1:4], P_dd_INT[0:len(P_dd_INT)-1:4]) + F_IAIA(z[:-1], zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, E_T(z[:-1]), R_T(z[:-1]), DG_T(z[:-1]), WIAT_T[aa][4:len(WIAT_T[aa]):4], WIAT_T[bb][4:len(WIAT_T[aa]):4], P_dd_INT[4:len(P_dd_INT):4])) + 32*(F_IAIA(1./4*(3*z[1:]+z[:-1]), zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, E_T(1./4*(3*z[1:]+z[:-1])), R_T(1./4*(3*z[1:]+z[:-1])), DG_T(1./4*(3*z[1:]+z[:-1])), WIAT_T[aa][1::4], WIAT_T[bb][1::4], P_dd_INT[1::4]) + F_IAIA(1./4*(z[1:]+3*z[:-1]), zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, E_T(1./4*(z[1:]+3*z[:-1])), R_T(1./4*(z[1:]+3*z[:-1])), DG_T(1./4*(z[1:]+3*z[:-1])), WIAT_T[aa][3::4], WIAT_T[bb][3::4], P_dd_INT[3::4])) + 12*F_IAIA(1./2*(z[1:]+z[:-1]), zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, E_T(1./2*(z[1:]+z[:-1])), R_T(1./2*(z[1:]+z[:-1])), DG_T(1./2*(z[1:]+z[:-1])), WIAT_T[aa][2::4], WIAT_T[bb][2::4], P_dd_INT[2::4]))) + P_shot_WL(zi, zj)
    else:
        C_ee = 0.
    C_gl = c/(100.*h_p)*delta_zpm/90*np.sum(7*(F_dd_GL(z[1:], zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, E_T(z[1:]), R_T(z[1:]), DG_T(z[1:]), WGT_T[aa][0:len(WGT_T[aa])-1:4], WT_T[bb][0:len(WGT_T[aa])-1:4], WIAT_T[bb][0:len(WIAT_T[aa])-1:4], P_dd_INT[0:len(P_dd_INT)-1:4]) + F_dd_GL(z[:-1], zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, E_T(z[:-1]), R_T(z[:-1]), DG_T(z[:-1]), WGT_T[aa][4:len(WGT_T[aa]):4], WT_T[bb][4:len(WGT_T[aa]):4], WIAT_T[bb][4:len(WIAT_T[aa]):4], P_dd_INT[4:len(P_dd_INT):4])) + 32*(F_dd_GL(1./4*(3*z[1:]+z[:-1]), zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, E_T(1./4*(3*z[1:]+z[:-1])), R_T(1./4*(3*z[1:]+z[:-1])), DG_T(1./4*(3*z[1:]+z[:-1])), WGT_T[aa][1::4], WT_T[bb][1::4], WIAT_T[bb][1::4], P_dd_INT[1::4]) + F_dd_GL(1./4*(z[1:]+3*z[:-1]), zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, E_T(1./4*(z[1:]+3*z[:-1])), R_T(1./4*(z[1:]+3*z[:-1])), DG_T(1./4*(z[1:]+3*z[:-1])), WGT_T[aa][3::4], WT_T[bb][3::4], WIAT_T[bb][3::4], P_dd_INT[3::4])) + 12*F_dd_GL(1./2*(z[1:]+z[:-1]), zi, zj, h_p, wm_p, wDE_p, w0_p, wa_p, C_IAp, A_IAp, n_IAp, B_IAp, E_T(1./2*(z[1:]+z[:-1])), R_T(1./2*(z[1:]+z[:-1])), DG_T(1./2*(z[1:]+z[:-1])), WGT_T[aa][2::4], WT_T[bb][2::4], WIAT_T[bb][2::4], P_dd_INT[2::4]) )
    return C_gg, C_ee, C_gl

############# __MAIN__ #############    

#Build BCKGND Tables    
E_tab, R_tab, RT_tab, DG_tab = np.zeros(BBB), np.zeros(BBB), np.zeros(BBB), np.zeros(BBB)
E_tab_up, R_tab_up, RT_tab_up, DG_tab_up = np.zeros(BBB), np.zeros(BBB), np.zeros(BBB), np.zeros(BBB)
E_tab_dw, R_tab_dw, RT_tab_dw, DG_tab_dw = np.zeros(BBB), np.zeros(BBB), np.zeros(BBB), np.zeros(BBB)

#E_tab, E_tab_up, E_tab_dw, R_tab, R_tab_up, R_tab_dw, RT_tab, RT_tab_up, RT_tab_dw, DG_tab, DG_tab_up, DG_tab_dw = np.loadtxt("Background_tables", usecols=(1,2,3,4,5,6,7,8,9,10,11,12), unpack=True)

i=0
while i < BBB:
    E_tab[i] = E(zbkgnd[i], h[2], Omega_m[2], Omega_DE[2], w0[2], wa[2])
    R_tab[i] = chi_R(zbkgnd[i], h[2], Omega_m[2], Omega_DE[2], w0[2], wa[2])
    RT_tab[i] = chi_R_tilde(zbkgnd[i], h[2], Omega_m[2], Omega_DE[2], w0[2], wa[2])
    DG_tab[i] = DGrowth(zbkgnd[i], h[2], Omega_m[2], Omega_DE[2], w0[2], wa[2], gamma[2])
    
    E_tab_up[i] = E(zbkgnd[i], h[0], Omega_m[0], Omega_DE[0], w0[0], wa[0])
    R_tab_up[i] = chi_R(zbkgnd[i], h[0], Omega_m[0], Omega_DE[0], w0[0], wa[0])
    RT_tab_up[i] = chi_R_tilde(zbkgnd[i], h[0], Omega_m[0], Omega_DE[0], w0[0], wa[0])
    DG_tab_up[i] = DGrowth(zbkgnd[i], h[0], Omega_m[0], Omega_DE[0], w0[0], wa[0], gamma[0])
    
    E_tab_dw[i] = E(zbkgnd[i], h[3], Omega_m[3], Omega_DE[3], w0[3], wa[3])
    R_tab_dw[i] = chi_R(zbkgnd[i], h[3], Omega_m[3], Omega_DE[3], w0[3], wa[3])
    RT_tab_dw[i] = chi_R_tilde(zbkgnd[i], h[3], Omega_m[3], Omega_DE[3], w0[3], wa[3])
    DG_tab_dw[i] = DGrowth(zbkgnd[i], h[3], Omega_m[3], Omega_DE[3], w0[3], wa[3], gamma[3])

    i=i+1 
    
E_tab, E_tab_up, E_tab_dw = CubicSpline(zbkgnd, E_tab), CubicSpline(zbkgnd, E_tab_up), CubicSpline(zbkgnd, E_tab_dw)
R_tab, R_tab_up, R_tab_dw = CubicSpline(zbkgnd, R_tab), CubicSpline(zbkgnd, R_tab_up), CubicSpline(zbkgnd, R_tab_dw)
RT_T, RT_T_up, RT_T_dw = CubicSpline(zbkgnd, RT_tab), CubicSpline(zbkgnd, RT_tab_up), CubicSpline(zbkgnd, RT_tab_dw)
DG_tab, DG_tab_up, DG_tab_dw = CubicSpline(zbkgnd, DG_tab), CubicSpline(zbkgnd, DG_tab_up), CubicSpline(zbkgnd, DG_tab_dw)

z_alot, Photoz_tab = np.linspace(zmin, zmax, PPP), np.zeros((PPP,len(zrange)))
#os.system("rm Pz_array")
def PZ_tab(i): 
    global j
    j=0
    while j < len(zrange):
        Photoz_tab[i][j] = Prnd(z_alot[i], zrange[j])
        j=j+1
    j=0

    outF=open("Pz_array",'a')
    while j < len(Photoz_tab[0]):
        if j == 0:
            outF.write(str("%.16e" % i))
            outF.write(str(' '))

        outF.write(str("%.16e" % Photoz_tab[i][j]))
        outF.write(str(' '))
        
        j=j+1
    outF.write(str('\n'))
    outF.close()

    i=i+1  

"""
i = range(PPP)    
if __name__ == '__main__':          
    pool = mp.Pool(10)
    pool.map(PZ_tab, i)
"""

Photoz_tab = np.loadtxt("Pz_array_10000_10000")
Photoz_tab = sorted(Photoz_tab, key=lambda a_entry: a_entry[0])
Photoz_tab = np.transpose(Photoz_tab)
Photoz_tab = np.delete(Photoz_tab, 0, axis=0)

Photoz_T = []
i=0
while i < len(Photoz_tab):
    Photoz_T = np.append(Photoz_T, interp1d(z_alot, Photoz_tab[i]))
    i=i+1

#Build W_ind tables

os.system("rm Windows_phot/*")
os.system("rm Windows/*")
os.system("rm Windows_IA/*")

WG_tab, WG_tab_up, WG_tab_dw = np.zeros((len(z_win),len(zrange))), np.zeros((len(z_win),len(zrange))), np.zeros((len(z_win),len(zrange)))
WG_path = "Windows_phot/"

W_tab, W_tab_up, W_tab_dw = np.zeros((len(z_win),len(zrange))), np.zeros((len(z_win),len(zrange))), np.zeros((len(z_win),len(zrange)))
WIA_tab, WIA_tab_up, WIA_tab_dw = np.zeros((len(z_win),len(zrange))), np.zeros((len(z_win),len(zrange))), np.zeros((len(z_win),len(zrange)))
W_path = "Windows/"
WIA_path = "Windows_IA/"

def WG_counts(i):
    global j
    j=0
    while j < len(zrange):
        WG_tab[i][j] = Weight_P(z_win[i], zrange[j], h[2], Omega_m[2], Omega_DE[2], w0[2], wa[2], Photoz_T[j], E_tab(z_win[i]), 2)
        WG_tab_up[i][j] = Weight_P(z_win[i], zrange[j], h[0], Omega_m[0], Omega_DE[0], w0[0], wa[0], Photoz_T[j], E_tab_up(z_win[i]), 0)
        WG_tab_dw[i][j] = Weight_P(z_win[i], zrange[j], h[3], Omega_m[3], Omega_DE[3], w0[3], wa[3], Photoz_T[j], E_tab_dw(z_win[i]), 3)
        j=j+1
    j=0
    
    outF=open(WG_path+"W_array",'a')
    outF_up=open(WG_path+"W_array_up",'a')
    outF_dw=open(WG_path+"W_array_dw",'a')
    while j < len(W_tab[0]):
        if j == 0:
            outF.write(str("%.16e" % i))
            outF.write(str(' '))
            outF_up.write(str("%.16e" % i))
            outF_up.write(str(' '))
            outF_dw.write(str("%.16e" % i))
            outF_dw.write(str(' '))
            
        outF.write(str("%.16e" % WG_tab[i][j]))
        outF.write(str(' '))
        outF_up.write(str("%.16e" % WG_tab_up[i][j]))
        outF_up.write(str(' '))
        outF_dw.write(str("%.16e" % WG_tab_dw[i][j]))
        outF_dw.write(str(' '))
        
        j=j+1
    outF.write(str('\n'))
    outF.close()
    outF_up.write(str('\n'))
    outF_up.close()
    outF_dw.write(str('\n'))
    outF_dw.close()
    i=i+1
    

i = range(len(z_win))    
if __name__ == '__main__':          
    pool = mp.Pool(10)
    pool.map(WG_counts, i)

def W_counts(i):
    global j
    j=0
    while j < len(zrange):
        W_tab[i][j] = Weight_F(z_win[i], zrange[j], h[2], Omega_m[2], Omega_DE[2], w0[2], wa[2], RT_T, Photoz_T[j])
        WIA_tab[i][j] = Weight_IA(z_win[i], zrange[j], h[2], Omega_m[2], Omega_DE[2], w0[2], wa[2], E_tab(z_win[i]), Photoz_T[j](z_win[i]))
        
        W_tab_up[i][j] = Weight_F(z_win[i], zrange[j], h[0], Omega_m[0], Omega_DE[0], w0[0], wa[0], RT_T_up, Photoz_T[j])
        WIA_tab_up[i][j] = Weight_IA(z_win[i], zrange[j], h[0], Omega_m[0], Omega_DE[0], w0[0], wa[0], E_tab_up(z_win[i]), Photoz_T[j](z_win[i]))
        
        W_tab_dw[i][j] = Weight_F(z_win[i], zrange[j], h[3], Omega_m[3], Omega_DE[3], w0[3], wa[3], RT_T_dw, Photoz_T[j])
        WIA_tab_dw[i][j] = Weight_IA(z_win[i], zrange[j], h[3], Omega_m[3], Omega_DE[3], w0[3], wa[3], E_tab_dw(z_win[i]), Photoz_T[j](z_win[i]))
        
        j=j+1
    j=0
    outF=open(W_path+"W_array",'a')
    outG=open(WIA_path+"WIA_array",'a')
    outF_up=open(W_path+"W_array_up",'a')
    outG_up=open(WIA_path+"WIA_array_up",'a')
    outF_dw=open(W_path+"W_array_dw",'a')
    outG_dw=open(WIA_path+"WIA_array_dw",'a')
    
    while j < len(W_tab[0]):
        if j == 0:
            outF.write(str("%.16e" % i))
            outF.write(str(' '))
            outG.write(str("%.16e" % i))
            outG.write(str(' '))
            outF_up.write(str("%.16e" % i))
            outF_up.write(str(' '))
            outG_up.write(str("%.16e" % i))
            outG_up.write(str(' '))
            outF_dw.write(str("%.16e" % i))
            outF_dw.write(str(' '))
            outG_dw.write(str("%.16e" % i))
            outG_dw.write(str(' '))
            
        outF.write(str("%.16e" % W_tab[i][j]))
        outF.write(str(' '))
        outG.write(str("%.16e" % WIA_tab[i][j]))
        outG.write(str(' '))
        outF_up.write(str("%.16e" % W_tab_up[i][j]))
        outF_up.write(str(' '))
        outG_up.write(str("%.16e" % WIA_tab_up[i][j]))
        outG_up.write(str(' '))
        outF_dw.write(str("%.16e" % W_tab_dw[i][j]))
        outF_dw.write(str(' '))
        outG_dw.write(str("%.16e" % WIA_tab_dw[i][j]))
        outG_dw.write(str(' '))
        
        j=j+1
    outF.write(str('\n'))
    outG.write(str('\n'))
    outF.close()
    outG.close()
    outF_up.write(str('\n'))
    outG_up.write(str('\n'))
    outF_up.close()
    outG_up.close()
    outF_dw.write(str('\n'))
    outG_dw.write(str('\n'))
    outF_dw.close()
    outG_dw.close()
    i=i+1 


i = range(len(z_win))    
if __name__ == '__main__':          
    pool = mp.Pool(10)
    pool.map(W_counts, i)
    
WG_tab = np.loadtxt(WG_path+"W_array")
WG_tab = sorted(WG_tab, key=lambda a_entry: a_entry[0])
WG_tab = np.transpose(WG_tab)
WG_tab = np.delete(WG_tab, 0, axis=0)

WG_tab_up = np.loadtxt(WG_path+"W_array_up")
WG_tab_up = sorted(WG_tab_up, key=lambda a_entry: a_entry[0])
WG_tab_up = np.transpose(WG_tab_up)
WG_tab_up = np.delete(WG_tab_up, 0, axis=0)
WG_tab_dw = np.loadtxt(WG_path+"W_array_dw")
WG_tab_dw = sorted(WG_tab_dw, key=lambda a_entry: a_entry[0])
WG_tab_dw = np.transpose(WG_tab_dw)
WG_tab_dw = np.delete(WG_tab_dw, 0, axis=0)

W_tab = np.loadtxt(W_path+"W_array")
W_tab = sorted(W_tab, key=lambda a_entry: a_entry[0])
W_tab = np.transpose(W_tab)
W_tab = np.delete(W_tab, 0, axis=0)

W_tab_up = np.loadtxt(W_path+"W_array_up")
W_tab_up = sorted(W_tab_up, key=lambda a_entry: a_entry[0])
W_tab_up = np.transpose(W_tab_up)
W_tab_up = np.delete(W_tab_up, 0, axis=0)
W_tab_dw = np.loadtxt(W_path+"W_array_dw")
W_tab_dw = sorted(W_tab_dw, key=lambda a_entry: a_entry[0])
W_tab_dw = np.transpose(W_tab_dw)
W_tab_dw = np.delete(W_tab_dw, 0, axis=0)

WIA_tab = np.loadtxt(WIA_path+"WIA_array")
WIA_tab = sorted(WIA_tab, key=lambda a_entry: a_entry[0])
WIA_tab = np.transpose(WIA_tab)
WIA_tab = np.delete(WIA_tab, 0, axis=0)

WIA_tab_up = np.loadtxt(WIA_path+"WIA_array_up")
WIA_tab_up = sorted(WIA_tab_up, key=lambda a_entry: a_entry[0])
WIA_tab_up = np.transpose(WIA_tab_up)
WIA_tab_up = np.delete(WIA_tab_up, 0, axis=0)
WIA_tab_dw = np.loadtxt(WIA_path+"WIA_array_dw")
WIA_tab_dw = sorted(WIA_tab_dw, key=lambda a_entry: a_entry[0])
WIA_tab_dw = np.transpose(WIA_tab_dw)
WIA_tab_dw = np.delete(WIA_tab_dw, 0, axis=0)

P_dd_C, P_dd_C_up, P_dd_C_dw = [], [], []
i=0
while i < len(files_In[2]):
    if i == 0:
        k_current = np.genfromtxt(fold_path[2]+"/Pks8sqRatio_ist_LogSplineInterpPk_iz_"+str(i)+".dat", skip_header=3, usecols=(0), unpack = True)*h[2]
    P_dd_current = np.genfromtxt(fold_path[2]+"/Pks8sqRatio_ist_LogSplineInterpPk_iz_"+str(i)+".dat", skip_header=3, usecols=(1), unpack=True)/(h[2]**3)
    P_dd_C = np.append(P_dd_C, CubicSpline(k_current, P_dd_current))
    i=i+1    

i=0
while i < len(files_In[2]):
    if paramo == 1:
        if i == 0:
            k_current = np.genfromtxt(fold_path[0]+"/Pks8sqRatio_ist_LogSplineInterpPk_iz_"+str(i)+".dat", delimiter = " ", skip_header=3, usecols=(0), unpack = True)*h[0]
        P_dd_current_up = np.genfromtxt(fold_path[0]+"/Pks8sqRatio_ist_LogSplineInterpPk_iz_"+str(i)+".dat", delimiter = " ", skip_header=3, usecols=(1), unpack=True)/(h[0]**3)
        P_dd_C_up = np.append(P_dd_C_up, CubicSpline(k_current, P_dd_current_up))

    else:
        if i == 0:
            k_current = np.genfromtxt(fold_path[0]+"/Pks8sqRatio_ist_LogSplineInterpPk_iz_"+str(i)+".dat", skip_header=3, usecols=(0), unpack=True)*h[2]
        P_dd_current_up = np.genfromtxt(fold_path[0]+"/Pks8sqRatio_ist_LogSplineInterpPk_iz_"+str(i)+".dat", skip_header=3, usecols=(1), unpack=True)/(h[2]**3)
        P_dd_C_up = np.append(P_dd_C_up, CubicSpline(k_current, P_dd_current_up))

    i=i+1

i=0
while i < len(files_In[2]):
    if paramo == 1:
        if i == 0:
            k_current = np.genfromtxt(fold_path[3]+"/Pks8sqRatio_ist_LogSplineInterpPk_iz_"+str(i)+".dat", skip_header=3, usecols=(0), unpack = True)*h[3]
        P_dd_current_dw = np.genfromtxt(fold_path[3]+"/Pks8sqRatio_ist_LogSplineInterpPk_iz_"+str(i)+".dat", skip_header=3, usecols=(1), unpack=True)/(h[3]**3)
        P_dd_C_dw = np.append(P_dd_C_dw, CubicSpline(k_current, P_dd_current_dw))

    else:
        if i == 0:
            k_current = np.genfromtxt(fold_path[3]+"/Pks8sqRatio_ist_LogSplineInterpPk_iz_"+str(i)+".dat", skip_header=3, usecols=(0), unpack=True)*h[2]
        P_dd_current_dw = np.genfromtxt(fold_path[3]+"/Pks8sqRatio_ist_LogSplineInterpPk_iz_"+str(i)+".dat", skip_header=3, usecols=(1), unpack=True)/(h[2]**3)
        P_dd_C_dw = np.append(P_dd_C_dw, CubicSpline(k_current, P_dd_current_dw))
    i=i+1
i=0

z_pk = np.linspace(zmin, zmax, len(P_dd_C))
           
#MAIN LOOP

def MAIN_LOOP(lll, seed=None):
    global aa
    global bb
    aa, bb = 0,0
    P_dd_ok = parainterp(P_dd_C, l[lll], R_tab(z_pk), P_dd_C)
    P_dd_ok_up = parainterp(P_dd_C_up, l[lll], R_tab_up(z_pk), P_dd_C_up)
    P_dd_ok_dw = parainterp(P_dd_C_dw, l[lll], R_tab_dw(z_pk), P_dd_C_dw)
    if paramo == 0:
        C_ij_GG, C_ij_LL, C_ij_GL = np.zeros((len(zrange), len(zrange))), np.zeros((len(zrange), len(zrange))), np.zeros((len(zrange), len(zrange)))
    C_ij_GG_up, C_ij_LL_up, C_ij_GL_up = np.zeros((len(zrange), len(zrange))), np.zeros((len(zrange), len(zrange))), np.zeros((len(zrange), len(zrange)))
    C_ij_GG_dw, C_ij_LL_dw, C_ij_GL_dw = np.zeros((len(zrange), len(zrange))), np.zeros((len(zrange), len(zrange))), np.zeros((len(zrange), len(zrange)))
    while aa < len(zrange):
        while bb < len(zrange):
            if paramo == 0:
                C_ij_GG[aa][bb], C_ij_LL[aa][bb], C_ij_GL[aa][bb] = Pobs_C(zpm, zrange[aa], zrange[bb], h[2], Omega_m[2], Omega_DE[2], w0[2], wa[2], C_IA, A_IA[2], n_IA[2], B_IA[2], E_tab, R_tab, DG_tab, DG_tab, WG_tab, W_tab, WIA_tab, l[lll], P_dd_ok, R_tab(z_pk)) 
            C_ij_GG_up[aa][bb], C_ij_LL_up[aa][bb], C_ij_GL_up[aa][bb] = Pobs_C(zpm, zrange[aa], zrange[bb], h[0], Omega_m[0], Omega_DE[0], w0[0], wa[0], C_IA, A_IA[0], n_IA[0], B_IA[0], E_tab_up, R_tab_up, DG_tab, DG_tab_up, WG_tab_up, W_tab_up, WIA_tab_up, l[lll], P_dd_ok_up, R_tab_up(z_pk))
            C_ij_GG_dw[aa][bb], C_ij_LL_dw[aa][bb], C_ij_GL_dw[aa][bb] = Pobs_C(zpm, zrange[aa], zrange[bb], h[3], Omega_m[3], Omega_DE[3], w0[3], wa[3], C_IA, A_IA[3], n_IA[3], B_IA[3], E_tab_dw, R_tab_dw, DG_tab, DG_tab_dw, WG_tab_dw, W_tab_dw, WIA_tab_dw, l[lll], P_dd_ok_dw, R_tab_dw(z_pk))
            bb=bb+1
        bb=0
        aa=aa+1
        
    if paramo == 0:
        aa, bb = 0,0 
        outGG=open(pre_CC_path[0]+CC_path[2]+"/COVAR_fid_"+str(l[lll]),'w')
        outLL=open(pre_CC_path[1]+CC_path[2]+"/COVAR_fid_"+str(l[lll]),'w')
        outGL=open(pre_CC_path[2]+CC_path[2]+"/COVAR_fid_"+str(l[lll]),'w')
        while aa < len(C_ij_GG):
            while bb < len(C_ij_GG):
                outGG.write(str("%.16e" % C_ij_GG[aa][bb]))
                outGG.write(str(' '))
                outLL.write(str("%.16e" % C_ij_LL[aa][bb]))
                outLL.write(str(' '))
                outGL.write(str("%.16e" % C_ij_GL[aa][bb]))
                outGL.write(str(' '))
                bb=bb+1
            outGG.write(str('\n'))
            outLL.write(str('\n'))
            outGL.write(str('\n'))
            bb=0
            aa=aa+1
        outGG.close()
        outLL.close()
        outGL.close()
    
    aa, bb = 0,0            
    outGGU=open(pre_CC_path[0]+CC_path[0]+"/COVAR_up_"+str(l[lll]),'w')
    outGGD=open(pre_CC_path[0]+CC_path[3]+"/COVAR_dw_"+str(l[lll]),'w')
    outLLU=open(pre_CC_path[1]+CC_path[0]+"/COVAR_up_"+str(l[lll]),'w')
    outLLD=open(pre_CC_path[1]+CC_path[3]+"/COVAR_dw_"+str(l[lll]),'w')
    outGLU=open(pre_CC_path[2]+CC_path[0]+"/COVAR_up_"+str(l[lll]),'w')
    outGLD=open(pre_CC_path[2]+CC_path[3]+"/COVAR_dw_"+str(l[lll]),'w')
    while aa < len(C_ij_GG_up):
        while bb < len(C_ij_GG_up):
            outGGU.write(str("%.16e" % C_ij_GG_up[aa][bb]))
            outGGU.write(str(' '))
            outGGD.write(str("%.16e" % C_ij_GG_dw[aa][bb]))
            outGGD.write(str(' '))
            outLLU.write(str("%.16e" % C_ij_LL_up[aa][bb]))
            outLLU.write(str(' '))
            outLLD.write(str("%.16e" % C_ij_LL_dw[aa][bb]))
            outLLD.write(str(' '))
            outGLU.write(str("%.16e" % C_ij_GL_up[aa][bb]))
            outGLU.write(str(' '))
            outGLD.write(str("%.16e" % C_ij_GL_dw[aa][bb]))
            outGLD.write(str(' '))
            bb=bb+1
        outGGU.write(str('\n'))
        outGGD.write(str('\n'))
        outLLU.write(str('\n'))
        outLLD.write(str('\n'))
        outGLU.write(str('\n'))
        outGLD.write(str('\n'))
        bb=0
        aa=aa+1
    outGGU.close()
    outGGD.close()
    outLLU.close()
    outLLD.close()
    outGLU.close()
    outGLD.close()
    lll=lll+1
    
lll = range(len(l))    
if __name__ == '__main__':          
    pool = mp.Pool(10)
    pool.map(MAIN_LOOP, lll)
