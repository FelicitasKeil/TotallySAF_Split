# %%
"""
Authors: S.Yahia-Cherif, I.Tutusaus, F.Dournac.
Last Update 29/06/2020.
This script computes the Big Fisher matrix and performs the projection into the new parameter space.
"""

# Modules import.
import sys
import numpy as np
import scipy.integrate as pyint
import os
from os import path
import glob
from scipy.interpolate import CubicSpline
import multiprocessing as mp
from multiprocessing import Pool
import warnings

print("")

# These lines clean the temp file
if os.listdir("tmp_F") == "":
    pass
else:
    os.system("rm tmp_F/*.txt")

print("Checking input parameters")

# Redshift bins, bias, dn3 loading
zrange_T, Delta_z_T, b_T, dn3_T, z_mean_choice_T = np.loadtxt(
    "input/bias_n_growth_baseline_base.dat",
    delimiter=" ",
    usecols=(0, 1, 2, 3, 5),
    unpack=True,
)

# Load the parameters files.
Codes_elts = np.loadtxt("../QTLauncher/Codes_W.txt", dtype="str")
Spec_elts = np.loadtxt("../QTLauncher/SpecSAF_W.txt", dtype="str")
Parameters_elts = np.loadtxt("../QTLauncher/Parameters_W.txt", dtype="str")

# Building the SpacSAF dictionary.
SpecSAF_elts = {}
for i in range(len(Codes_elts)):
    SpecSAF_elts[Codes_elts[i][0]] = Codes_elts[i][1]
for i in range(len(Spec_elts)):
    SpecSAF_elts[Spec_elts[i][0]] = Spec_elts[i][1]
for i in range(len(Parameters_elts)):
    SpecSAF_elts[Parameters_elts[i][0]] = Parameters_elts[i][1]


# Stocking the input parameters inside variables.
kMIN = float(SpecSAF_elts["VkminSP"])
kMAX = float(SpecSAF_elts["VkmaxSP"])
z_mean_index = int(SpecSAF_elts["VMean_binSP"]) - 1
zrange = np.linspace(
    float(SpecSAF_elts["VzminSP"]),
    float(SpecSAF_elts["VzmaxSP"]),
    int(SpecSAF_elts["redbinsSP"]) + 1,
)
Delta_z = zrange[1:] - zrange[:-1]
zrange = 0.5 * (zrange[1:] + zrange[:-1])
Vol_eff = float(SpecSAF_elts["VSurfAreaSP"])

b_int = CubicSpline(zrange_T, b_T, extrapolate=True)
dn3_int = CubicSpline(zrange_T, dn3_T, extrapolate=True)
b = b_int(zrange)
dn3 = dn3_int(zrange)

choice_L_NL = int(SpecSAF_elts["LNLcaseSP_ch"])
choice_SNL1_SNL2 = int(SpecSAF_elts["SNLXcaseSP_ch"])
proj_ODE_fit = int(SpecSAF_elts["ODEfitSP_ch"])
choice_F_NF = int(SpecSAF_elts["FNF_ch"])
B_code = int(SpecSAF_elts["Camb_ch"])
bias_choice = int(SpecSAF_elts["biasvslnbiasSP_ch"])
Allder_choice = int(SpecSAF_elts["DerMethodSP_ch"])

choices = np.array(["L", "SNL"])
choice_L_NL = choices[choice_L_NL]
choices = np.array(["SNL1", "SNL2"])
choice_SNL1_SNL2 = choices[choice_SNL1_SNL2]
choices = np.array(["ODE", "fit"])
proj_ODE_fit = choices[proj_ODE_fit]
choices = np.array(["F", "NF"])
choice_F_NF = choices[choice_F_NF]
choices = np.array(["Y", "N"])
B_code = choices[B_code]
choices = np.array(["Y", "N"])
bias_choice = choices[bias_choice]

eps_proj = float(SpecSAF_elts["VDerprojSP"])
ODE_pts = int(SpecSAF_elts["VODEPrecSP"])
spectroprec = float(SpecSAF_elts["VSpectroprecSP"])
integ_prec = int(SpecSAF_elts["VIntegprecSP"])
c = 299792.458

omega_b_fid = float(SpecSAF_elts["VFidOmegab"]) * float(SpecSAF_elts["VFidh"]) ** 2
h_fid = float(SpecSAF_elts["VFidh"])
omega_m_fid = float(SpecSAF_elts["VFidOmegam"]) * float(SpecSAF_elts["VFidh"]) ** 2
ns_fid = float(SpecSAF_elts["VFidns"])
w0 = float(SpecSAF_elts["VFidw0"])
wa = float(SpecSAF_elts["VFidwa"])
sig_8_new = float(SpecSAF_elts["VFidsigma8"])
omega_nu = float(SpecSAF_elts["VFidWnu"]) * float(SpecSAF_elts["VFidh"]) ** 2

eps_wb = float(SpecSAF_elts["VStepOmegab"])
eps_h = float(SpecSAF_elts["VSteph"])
eps_wm = float(SpecSAF_elts["VStepOmegam"])
eps_ns = float(SpecSAF_elts["VStepns"])
eps_sig_p = float(SpecSAF_elts["VStepsp"])
eps_sig_v = float(SpecSAF_elts["VStepsv"])
eps_Da = float(SpecSAF_elts["VSteplnDa"])
eps_H = float(SpecSAF_elts["VSteplnH"])
eps_fs8 = float(SpecSAF_elts["VSteplnfs8"])
eps_bs8 = float(SpecSAF_elts["VStepGCspecbias"])

choices = np.array(["3", "5", "7"])
choice_der_pts_shape = int(choices[Allder_choice])
choice_der_pts_sig_p = int(choices[Allder_choice])
choice_der_pts_sig_v = int(choices[Allder_choice])
choice_der_pts_Da = int(choices[Allder_choice])
choice_der_pts_H = int(choices[Allder_choice])
choice_der_pts_fs8 = int(choices[Allder_choice])
choice_der_pts_bs8 = int(choices[Allder_choice])

include_gamma = int(SpecSAF_elts["Usegamma_ch"])
choices = np.array(["Y", "N"])
include_gamma = choices[include_gamma]

# Checking the consistency of the input parameters
if kMAX < kMIN:
    sys.exit("Error: kmax has to be higher than kmin.")

if float(SpecSAF_elts["VzmaxSP"]) < float(SpecSAF_elts["VzminSP"]):
    sys.exit("Error: zmax has to be higher than zmin.")

if int(SpecSAF_elts["redbinsSP"]) < int(SpecSAF_elts["VMean_binSP"]):
    sys.exit("Error: The mean redshift bin has to be in the redshift range.")

if choice_F_NF == "F" and int(SpecSAF_elts["UseOmegaDE_ch"]) == 0:
    warnings.warn(
        "Warning: Omega_{DE} isn't estimated on the flat case. The code will still run without estimating this parameter."
    )

if int(SpecSAF_elts["ODEfitSP_ch"]) == 0 and int(SpecSAF_elts["Usegamma_ch"]) == 0:
    sys.exit("Error: ODE isn't compatible with gamma constraint estimation (Use fit)")

N_notRD_params, N_RD_params = 6, 5

# Loading Camb power spectrums names and Fisher names
CAMB_IN_L = "Pk_baseline_NBSAF"
FD_input = "Pk_baseline_NBSAF/fid"
FD_output = "Big_Fisher_matrix"

print("Checking ok!")

# Load gamma parameter.
gamma_new = float(SpecSAF_elts["VFidgamma"])

# Camb is called here, we switch here in Camb directory run Camb and go back to SpecSAF main directory.
if B_code == "Y":
    print("Calling Camb")
    os.chdir("CAMB-0.1.7")
    os.system("python Camb_launcher_SpecSAF.py")
    os.system("cp -r " + CAMB_IN_L + " ../input")
    os.chdir("../")

# The path towards the linear matter/no wiggle power spectrum is defined here.
path_spec = path.abspath("input/" + FD_input)
# The scale values are loaded from a fiducial power spectrum and converted into no h units.
k_cf = np.genfromtxt(
    path_spec
    + "/Pks8sqRatio_ist_LogSplineInterpPk_"
    + str(zrange[z_mean_index])
    + ".dat",
    skip_header=3,
    usecols=(0,),
    unpack=True,
)
k_cf = k_cf * h_fid

"""
The following part loads the fiducial linear power spectrum at zmean.
The power spectrum is converted in no h units and interpolated using a cubic spline
Then we compute sigma_p(zmean) using the integral of the linear power spectrum
"""
P_m_fid = np.genfromtxt(
    path_spec + "/Pks8sqRatio_ist_LogSplineInterpPk_" + str(zrange[z_mean_index]) + ".dat",
    skip_header=3,
    usecols=(1,),
    unpack=True,
)
# convert into Mpc^3 units (from Mpc^3/h^3)
P_m_fid = P_m_fid / (h_fid**3)
sig_8_fid = float(
    np.genfromtxt(
        path_spec
        + "/Pks8sqRatio_ist_LogSplineInterpPk_" + str(zrange[z_mean_index]) + ".dat",
        skip_header=3,
        usecols=(2),
        unpack=True,
        max_rows=1,
    )
)
# interpolate matter power spectrum in no h
P_m = CubicSpline(np.log10(k_cf), np.log10(P_m_fid))
# convert k into no h units with the fiducial h
k = np.geomspace(0.001 * h_fid, 5.0 * h_fid, 10000)
sp = 0
j = 1
while j < len(k):
    sp = sp + sig_8_fid**2 / (6 * np.pi**2) * (k[j] - k[j - 1]) / 6 * (
        10 ** P_m(np.log10(k[j]))
        + 10 ** P_m(np.log10((k[j] + k[j - 1]) / 2)) * 4
        + 10 ** P_m(np.log10(k[j - 1]))
    )
    j = j + 1
j = 0

sig_p_fid = np.zeros(len(zrange))
sig_p_fid[:] = np.sqrt(sp)
sig_v_fid = np.copy(sig_p_fid)

# Init the power spectrums path.
path_In = path.abspath("input/" + FD_input)
files_In = glob.glob(path_In + "/*")

# sigma_p is transformed into the new paramter : sig_p_fid = sig_p_fid/S8_mean.
S8_mean = float(
    np.genfromtxt(
        path_spec
        + "/Pks8sqRatio_ist_LogSplineInterpPk_"
        + str(zrange[z_mean_index])
        + ".dat",
        skip_header=3,
        usecols=(2),
        unpack=True,
        max_rows=1,
    )
)
if choice_SNL1_SNL2 == "SNL2":
    sig_p_fid = np.copy(sig_p_fid) / S8_mean

# For the linear case the peculiar velocities are equal to 0
if choice_L_NL == "L":
    sig_p_fid = np.zeros(len(zrange))
    sig_v_fid = np.copy(sig_p_fid)

# kref values
k_cf = np.genfromtxt(files_In[z_mean_index], skip_header=3, usecols=(0,), unpack=True)

# Indexes for which kref values are < kMAX.
ind = np.where(k_cf < kMAX)

# Adding radiation (default is 0).
# Og=(2.469*h_fid**(-2))*10.**(-5)
# Omega_r = Og*(1+0.2271*3.046)
Omega_r = 0.0
omega_r = Omega_r * h_fid**2
# omega_c
omega_c = omega_m_fid - omega_b_fid - omega_nu

# zrange for the ODE.
tn = np.linspace(0, 10, ODE_pts)


# Background quantities functions, AP effect, RSD effect, etc... In order : H(z), chi(z), angular distance, Omega_m(z), growth D(z).
def H(z):
    return (
        100
        * h_fid
        * np.sqrt(
            omega_m_fid / (h_fid**2) * (1 + z) ** 3
            + omega_r * (1 + z) ** 4 / (h_fid**2)
            + 1
            - (omega_c + omega_b_fid + omega_nu + omega_r) / (h_fid**2)
        )
    )


def H_ref(z):
    return (
        100
        * h_fid
        * np.sqrt(
            omega_m_fid / (h_fid**2) * (1 + z) ** 3
            + omega_r * (1 + z) ** 4 / (h_fid**2)
            + 1
            - (omega_c + omega_b_fid + omega_nu + omega_r) / (h_fid**2)
        )
    )


def chi_aux(z):
    return c / H(z)


def chi_aux_ref(z):
    return c / H_ref(z)


def chi(z):
    return pyint.quad(chi_aux, 0, z)[0]


def chi_ref(z):
    return pyint.quad(chi_aux_ref, 0, z)[0]


def D_A(z):
    return 1.0 / (1 + z) * chi(z)


def D_A_ref(z):
    return 1.0 / (1 + z) * chi_ref(z)


def OM_M(z):
    return (
        omega_m_fid / (h_fid**2) * (1.0 + z) ** 3 / ((H_ref(z) / (100.0 * h_fid)) ** 2)
    ) ** gamma_new / (1.0 + z)


def DGrowth(z):
    return np.exp(-pyint.quad(OM_M, 0, z)[0])


# Ordinary differential equation solver to compute f(z).
def eq_diff_ref(x):
    y = np.zeros(len(x))
    dy = np.zeros(len(y))
    dx = x[len(x) - 2] - x[len(x) - 1]
    y[len(x) - 1] = 1
    dy[len(x) - 1] = (
        y[len(x) - 1] ** 2 / (1 + x[len(x) - 1])
        + y[len(x) - 1]
        * (
            2.0 / (1 + x[len(x) - 1])
            - (H_ref(x[len(x) - 1] + 1e-6) - H_ref(x[len(x) - 1] - 1e-6))
            / (2 * 1e-6 * H_ref(x[len(x) - 1]))
        )
        - 1.5
        * omega_m_fid
        / (h_fid**2)
        * (h_fid**2)
        * (1 + x[len(x) - 1]) ** 2
        * (100) ** 2
        / (H_ref(x[len(x) - 1]) ** 2)
    )

    i = len(x) - 2
    while i >= 0:
        y[i] = y[i + 1] + dx * dy[i + 1]
        dy[i] = (
            y[i] ** 2 / (1 + x[i])
            + y[i]
            * (
                2.0 / (1 + x[i])
                - (H_ref(x[i] + 1e-6) - H_ref(x[i] - 1e-6)) / (2 * 1e-6 * H_ref(x[i]))
            )
            - 1.5
            * omega_m_fid
            / (h_fid**2)
            * (h_fid**2)
            * (1 + x[i]) ** 2
            * (100) ** 2
            / (H_ref(x[i]) ** 2)
        )
        i = i - 1
    return y


# f(z).
def f_ref(x):
    soln = eq_diff_ref(x)
    return soln


# Redshift precision sigma_r factor : 0.001 is the sigma_0 from Euclid.
def sigma_r(z, er_FH):
    return spectroprec * (1 + z) * c / er_FH


# Redshift precision Fz(z).
def Fz(
    z,
    k_ref,
    mu_ref,
    er_FH,
):
    return np.exp(-(k_ref**2) * mu_ref**2 * sigma_r(z, er_FH) ** 2)


# AP effect : q_perpendicular and q_parallel.
def q_per(z, er_FDA, er_FDA_ref):
    return er_FDA / er_FDA_ref


def q_par(z, er_FH, er_FH_ref):
    return er_FH_ref / er_FH


# Scale corrections for deviation from fiducial cosmology.
# See equ. 78 in Blanchard et al.
def k(z, mu_ref, k_ref, er_FH, er_FH_ref, er_FDA, er_FDA_ref):
    return (
        k_ref
        / q_per(z, er_FDA, er_FDA_ref)
        * np.sqrt(
            1 + mu_ref**2
            * (q_per(z, er_FDA, er_FDA_ref) ** 2 / (q_par(z, er_FH, er_FH_ref) ** 2)
                - 1)))


# mu corrections for deviation from fiducial cosmology.
def mu(z, mu_ref, er_FH, er_FH_ref, er_FDA, er_FDA_ref):
    return (
        mu_ref
        * q_per(z, er_FDA, er_FDA_ref)
        / q_par(z, er_FH, er_FH_ref)
        * (1 + mu_ref**2
            * (
                q_per(z, er_FDA, er_FDA_ref) ** 2 / (q_par(z, er_FH, er_FH_ref) ** 2)
                - 1)) ** (-0.5))


# BAO damping factor.
def gmu(z, mu_ref, er_FH, er_FH_ref, er_FDA, er_FDA_ref, sigP, sigV):
    return (sigV * DG_growth[i] / DG_growth[z_mean_index]) ** 2 * (
        1 - mu_ref**2 + mu_ref**2 * (1 + frate) ** 2
    )


# Fiducial shot noise.
def P_shot(z):
    return 0.0


# Observational power spectrum.
# The 1st one is the semi non linear power spectrum in the SNL1 (this is the oldest semi non linear power spectrum used by the IST).
# Linear case = fix both peculiar velocities = 0.
# The 2nd one is the semi non linear power spectrum in the SNL2 (this is the current non linear power spectrum used by the IST).
def P_obs(
    k_ref, mu_ref, z, PM, PM_NW, bias_s8, gf_s8, er_FH, er_FH_ref, er_FDA,
    er_FDA_ref, sigP, sigV):
    if (choice_L_NL == "L") or (choice_L_NL == "SNL" and choice_SNL1_SNL2 == "SNL1"):
        return 1.0 / (
            q_per(z, er_FDA, er_FDA_ref) ** 2 * q_par(z, er_FH, er_FH_ref)
        ) * (
            1.0
            / (
                1.0
                + (
                    frate * k_ref * mu_ref * sigP * DG_growth[i]
                    / DG_growth[z_mean_index])** 2)
        ) * (
            bias_s8 + gf_s8 * mu_ref**2
        ) ** 2 * (
            PM
            * np.exp(
                -gmu(z, mu_ref, er_FH, er_FH_ref, er_FDA, er_FDA_ref, sigP, sigV)
                * k_ref**2
            )
            + PM_NW
            * (
                1.0
                - np.exp(
                    -gmu(z, mu_ref, er_FH, er_FH_ref, er_FDA, er_FDA_ref, sigP, sigV)
                    * k_ref**2
                )
            )
        ) * Fz(z, k_ref, mu_ref, er_FH) + P_shot(z)

    elif choice_L_NL == "SNL" and choice_SNL1_SNL2 == "SNL2":
        return 1.0 / (
            q_per(z, er_FDA, er_FDA_ref) ** 2 * q_par(z, er_FH, er_FH_ref)
        ) * (1.0 / (1.0 + (gf_s8 * k_ref * mu_ref * sigP) ** 2)) * (
            bias_s8 + gf_s8 * mu_ref**2
        ) ** 2 * (
            PM
            * np.exp(
                -gmu(z, mu_ref, er_FH, er_FH_ref, er_FDA, er_FDA_ref, sigP, sigV)
                * k_ref**2
            )
            + PM_NW
            * (
                1.0
                - np.exp(
                    -gmu(z, mu_ref, er_FH, er_FH_ref, er_FDA, er_FDA_ref, sigP, sigV)
                    * k_ref**2
                )
            )
        ) * Fz(
            z, k_ref, mu_ref, er_FH
        ) + P_shot(
            z
        )


# ln of the observational power spectrum.
def lnP_obs(k_ref, mu_ref, z, PM, PM_NW, bias_s8, gf_s8, er_FH, er_FH_ref,
            er_FDA, er_FDA_ref, sigP, sigV):
    return np.log(
        P_obs(k_ref,mu_ref,z,PM,PM_NW,bias_s8,gf_s8,er_FH,er_FH_ref,er_FDA,
              er_FDA_ref,sigP,sigV))


# Derivative functions of the shape parameters.
def der_wb_3pts(k_ref,mu_ref,z,PM_up,PM_dw,PM_NW_up,PM_NW_dw,bias_s8,gf_s8,
                er_FH,er_FH_ref,er_FDA,er_FDA_ref,sigP,sigV):
    return (
        lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_up(np.log10(k_ref)),
            10 ** PM_NW_up(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - lnP_obs(k_ref,mu_ref,z,10 ** PM_dw(np.log10(k_ref)),
                  10 ** PM_NW_dw(np.log10(k_ref)),bias_s8,gf_s8,er_FH,er_FH_ref,
                  er_FDA,er_FDA_ref,sigP,sigV,
        )
    ) / (2 * omega_b_fid * eps_wb)


def der_wb_5pts(k_ref,mu_ref,z,PM_up,PM_up2,PM_dw,PM_dw2,PM_NW_up,PM_NW_up2,
                PM_NW_dw,PM_NW_dw2,bias_s8,gf_s8,er_FH,er_FH_ref,er_FDA,
                er_FDA_ref,sigP,sigV,
):
    return (
        lnP_obs(k_ref,mu_ref,z,10 ** PM_dw2(np.log10(k_ref)),
                10 ** PM_NW_dw2(np.log10(k_ref)),bias_s8,gf_s8,er_FH,er_FH_ref,
                er_FDA,er_FDA_ref,sigP,sigV,
        )
        - 8
        * lnP_obs(k_ref,mu_ref,z,10 ** PM_dw(np.log10(k_ref)),
                  10 ** PM_NW_dw(np.log10(k_ref)),bias_s8,gf_s8,er_FH,er_FH_ref,
                  er_FDA,er_FDA_ref,sigP,sigV,
        )
        + 8
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_up(np.log10(k_ref)),
            10 ** PM_NW_up(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_up2(np.log10(k_ref)),
            10 ** PM_NW_up2(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
    ) / (12 * omega_b_fid * eps_wb)


def der_wb_7pts(
    k_ref,
    mu_ref,
    z,
    PM_up,
    PM_up2,
    PM_up3,
    PM_dw,
    PM_dw2,
    PM_dw3,
    PM_NW_up,
    PM_NW_up2,
    PM_NW_up3,
    PM_NW_dw,
    PM_NW_dw2,
    PM_NW_dw3,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    return (
        -lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_dw3(np.log10(k_ref)),
            10 ** PM_NW_dw3(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + 9
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_dw2(np.log10(k_ref)),
            10 ** PM_NW_dw2(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - 45
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_dw(np.log10(k_ref)),
            10 ** PM_NW_dw(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + 45
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_up(np.log10(k_ref)),
            10 ** PM_NW_up(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - 9
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_up2(np.log10(k_ref)),
            10 ** PM_NW_up2(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_up3(np.log10(k_ref)),
            10 ** PM_NW_up3(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
    ) / (60 * omega_b_fid * eps_wb)


def der_h_3pts(
    k_ref,
    mu_ref,
    z,
    PM_up,
    PM_dw,
    PM_NW_up,
    PM_NW_dw,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    return (
        lnP_obs(
            k_ref * (1.0 + eps_h),
            mu_ref,
            z,
            10 ** PM_up(np.log10(k_ref * (1.0 + eps_h))),
            10 ** PM_NW_up(np.log10(k_ref * (1.0 + eps_h))),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - lnP_obs(
            k_ref * (1.0 - eps_h),
            mu_ref,
            z,
            10 ** PM_dw(np.log10(k_ref * (1.0 - eps_h))),
            10 ** PM_NW_dw(np.log10(k_ref * (1.0 - eps_h))),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
    ) / (2 * h_fid * eps_h)


def der_h_5pts(
    k_ref,
    mu_ref,
    z,
    PM_up,
    PM_up2,
    PM_dw,
    PM_dw2,
    PM_NW_up,
    PM_NW_up2,
    PM_NW_dw,
    PM_NW_dw2,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    return (
        lnP_obs(
            k_ref * (1.0 - 2 * eps_h),
            mu_ref,
            z,
            10 ** PM_dw2(np.log10(k_ref * (1 - 2 * eps_h))),
            10 ** PM_NW_dw2(np.log10(k_ref * (1 - 2 * eps_h))),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - 8
        * lnP_obs(
            k_ref * (1.0 - eps_h),
            mu_ref,
            z,
            10 ** PM_dw(np.log10(k_ref * (1 - eps_h))),
            10 ** PM_NW_dw(np.log10(k_ref * (1 - eps_h))),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + 8
        * lnP_obs(
            k_ref * (1.0 + eps_h),
            mu_ref,
            z,
            10 ** PM_up(np.log10(k_ref * (1 + eps_h))),
            10 ** PM_NW_up(np.log10(k_ref * (1 + eps_h))),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - lnP_obs(
            k_ref * (1.0 + 2 * eps_h),
            mu_ref,
            z,
            10 ** PM_up2(np.log10(k_ref * (1 + 2 * eps_h))),
            10 ** PM_NW_up2(np.log10(k_ref * (1 + 2 * eps_h))),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
    ) / (12 * h_fid * eps_h)


def der_h_7pts(
    k_ref,
    mu_ref,
    z,
    PM_up,
    PM_up2,
    PM_up3,
    PM_dw,
    PM_dw2,
    PM_dw3,
    PM_NW_up,
    PM_NW_up2,
    PM_NW_up3,
    PM_NW_dw,
    PM_NW_dw2,
    PM_NW_dw3,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    return (
        -lnP_obs(
            k_ref * (1.0 - 3 * eps_h),
            mu_ref,
            z,
            10 ** PM_dw3(np.log10(k_ref * (1.0 - 3 * eps_h))),
            10 ** PM_NW_dw3(np.log10(k_ref * (1.0 - 3 * eps_h))),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + 9
        * lnP_obs(
            k_ref * (1.0 - 2 * eps_h),
            mu_ref,
            z,
            10 ** PM_dw2(np.log10(k_ref * (1.0 - 2 * eps_h))),
            10 ** PM_NW_dw2(np.log10(k_ref * (1.0 - 2 * eps_h))),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - 45
        * lnP_obs(
            k_ref * (1.0 - eps_h),
            mu_ref,
            z,
            10 ** PM_dw(np.log10(k_ref * (1.0 - eps_h))),
            10 ** PM_NW_dw(np.log10(k_ref * (1.0 - eps_h))),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + 45
        * lnP_obs(
            k_ref * (1.0 + eps_h),
            mu_ref,
            z,
            10 ** PM_up(np.log10(k_ref * (1.0 + eps_h))),
            10 ** PM_NW_up(np.log10(k_ref * (1.0 + eps_h))),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - 9
        * lnP_obs(
            k_ref * (1.0 + 2 * eps_h),
            mu_ref,
            z,
            10 ** PM_up2(np.log10(k_ref * (1.0 + 2 * eps_h))),
            10 ** PM_NW_up2(np.log10(k_ref * (1.0 + 2 * eps_h))),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + lnP_obs(
            k_ref * (1.0 + 3 * eps_h),
            mu_ref,
            z,
            10 ** PM_up3(np.log10(k_ref * (1.0 + 3 * eps_h))),
            10 ** PM_NW_up3(np.log10(k_ref * (1.0 + 3 * eps_h))),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
    ) / (60 * h_fid * eps_h)


def der_wm_3pts(
    k_ref,
    mu_ref,
    z,
    PM_up,
    PM_dw,
    PM_NW_up,
    PM_NW_dw,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    return (
        lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_up(np.log10(k_ref)),
            10 ** PM_NW_up(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_dw(np.log10(k_ref)),
            10 ** PM_NW_dw(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
    ) / (2 * omega_m_fid * eps_wm)


def der_wm_5pts(
    k_ref,
    mu_ref,
    z,
    PM_up,
    PM_up2,
    PM_dw,
    PM_dw2,
    PM_NW_up,
    PM_NW_up2,
    PM_NW_dw,
    PM_NW_dw2,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    return (
        lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_dw2(np.log10(k_ref)),
            10 ** PM_NW_dw2(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - 8
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_dw(np.log10(k_ref)),
            10 ** PM_NW_dw(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + 8
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_up(np.log10(k_ref)),
            10 ** PM_NW_up(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_up2(np.log10(k_ref)),
            10 ** PM_NW_up2(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
    ) / (12 * omega_m_fid * eps_wm)


def der_wm_7pts(
    k_ref,
    mu_ref,
    z,
    PM_up,
    PM_up2,
    PM_up3,
    PM_dw,
    PM_dw2,
    PM_dw3,
    PM_NW_up,
    PM_NW_up2,
    PM_NW_up3,
    PM_NW_dw,
    PM_NW_dw2,
    PM_NW_dw3,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    return (
        -lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_dw3(np.log10(k_ref)),
            10 ** PM_NW_dw3(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + 9
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_dw2(np.log10(k_ref)),
            10 ** PM_NW_dw2(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - 45
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_dw(np.log10(k_ref)),
            10 ** PM_NW_dw(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + 45
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_up(np.log10(k_ref)),
            10 ** PM_NW_up(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - 9
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_up2(np.log10(k_ref)),
            10 ** PM_NW_up2(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_up3(np.log10(k_ref)),
            10 ** PM_NW_up3(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
    ) / (60 * omega_m_fid * eps_wm)


def der_ns_3pts(
    k_ref,
    mu_ref,
    z,
    PM_up,
    PM_dw,
    PM_NW_up,
    PM_NW_dw,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    return (
        lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_up(np.log10(k_ref)),
            10 ** PM_NW_up(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_dw(np.log10(k_ref)),
            10 ** PM_NW_dw(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
    ) / (2 * ns_fid * eps_ns)


def der_ns_5pts(
    k_ref,
    mu_ref,
    z,
    PM_up,
    PM_up2,
    PM_dw,
    PM_dw2,
    PM_NW_up,
    PM_NW_up2,
    PM_NW_dw,
    PM_NW_dw2,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    return (
        lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_dw2(np.log10(k_ref)),
            10 ** PM_NW_dw2(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - 8
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_dw(np.log10(k_ref)),
            10 ** PM_NW_dw(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + 8
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_up(np.log10(k_ref)),
            10 ** PM_NW_up(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_up2(np.log10(k_ref)),
            10 ** PM_NW_up2(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
    ) / (12 * ns_fid * eps_ns)


def der_ns_7pts(
    k_ref,
    mu_ref,
    z,
    PM_up,
    PM_up2,
    PM_up3,
    PM_dw,
    PM_dw2,
    PM_dw3,
    PM_NW_up,
    PM_NW_up2,
    PM_NW_up3,
    PM_NW_dw,
    PM_NW_dw2,
    PM_NW_dw3,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    return (
        -lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_dw3(np.log10(k_ref)),
            10 ** PM_NW_dw3(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + 9
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_dw2(np.log10(k_ref)),
            10 ** PM_NW_dw2(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - 45
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_dw(np.log10(k_ref)),
            10 ** PM_NW_dw(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + 45
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_up(np.log10(k_ref)),
            10 ** PM_NW_up(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - 9
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_up2(np.log10(k_ref)),
            10 ** PM_NW_up2(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM_up3(np.log10(k_ref)),
            10 ** PM_NW_up3(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
    ) / (60 * ns_fid * eps_ns)


# Derivative functions of the redshift dependent parameters.
def der_H_7pts(
    k_ref,
    mu_ref,
    z,
    PM,
    PM_NW,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):

    # Computing new k, mu coming from the AP effect (for the derivative of lnP_obs over lnH(z)).

    mu_H_up = mu(z, mu_ref, er_FH ** (1 + eps_H), er_FH_ref, er_FDA, er_FDA_ref)
    mu_H_up_2 = mu(z, mu_ref, er_FH ** (1 + 2 * eps_H), er_FH_ref, er_FDA, er_FDA_ref)
    mu_H_up_3 = mu(z, mu_ref, er_FH ** (1 + 3 * eps_H), er_FH_ref, er_FDA, er_FDA_ref)

    mu_H_dw = mu(z, mu_ref, er_FH ** (1 - eps_H), er_FH_ref, er_FDA, er_FDA_ref)
    mu_H_dw_2 = mu(z, mu_ref, er_FH ** (1 - 2 * eps_H), er_FH_ref, er_FDA, er_FDA_ref)
    mu_H_dw_3 = mu(z, mu_ref, er_FH ** (1 - 3 * eps_H), er_FH_ref, er_FDA, er_FDA_ref)

    return (
        -lnP_obs(
            k_ref,
            mu_H_dw_3,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH ** (1 - 3 * eps_H),
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + 9
        * lnP_obs(
            k_ref,
            mu_H_dw_2,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH ** (1 - 2 * eps_H),
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - 45
        * lnP_obs(
            k_ref,
            mu_H_dw,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH ** (1 - eps_H),
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + 45
        * lnP_obs(
            k_ref,
            mu_H_up,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH ** (1 + eps_H),
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - 9
        * lnP_obs(
            k_ref,
            mu_H_up_2,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH ** (1 + 2 * eps_H),
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + lnP_obs(
            k_ref,
            mu_H_up_3,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH ** (1 + 3 * eps_H),
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
    ) / (60 * eps_H * np.log(er_FH_ref))


def der_H_5pts(
    k_ref,
    mu_ref,
    z,
    PM,
    PM_NW,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):

    mu_H_up = mu(z, mu_ref, er_FH ** (1 + eps_H), er_FH_ref, er_FDA, er_FDA_ref)
    mu_H_up_2 = mu(z, mu_ref, er_FH ** (1 + 2 * eps_H), er_FH_ref, er_FDA, er_FDA_ref)

    mu_H_dw = mu(z, mu_ref, er_FH ** (1 - eps_H), er_FH_ref, er_FDA, er_FDA_ref)
    mu_H_dw_2 = mu(z, mu_ref, er_FH ** (1 - 2 * eps_H), er_FH_ref, er_FDA, er_FDA_ref)

    return (
        -lnP_obs(
            k_ref,
            mu_H_up_2,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH ** (1 + 2 * eps_H),
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + 8
        * lnP_obs(
            k_ref,
            mu_H_up,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH ** (1 + eps_H),
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - 8
        * lnP_obs(
            k_ref,
            mu_H_dw,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH ** (1 - eps_H),
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + lnP_obs(
            k_ref,
            mu_H_dw_2,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH ** (1 - 2 * eps_H),
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
    ) / (12 * eps_H * np.log(er_FH_ref))


def der_H_3pts(
    k_ref,
    mu_ref,
    z,
    PM,
    PM_NW,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):

    mu_H_up = mu(z, mu_ref, er_FH ** (1 + eps_H), er_FH_ref, er_FDA, er_FDA_ref)
    mu_H_dw = mu(z, mu_ref, er_FH ** (1 - eps_H), er_FH_ref, er_FDA, er_FDA_ref)

    return (
        lnP_obs(
            k_ref,
            mu_H_up,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH ** (1 + eps_H),
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - lnP_obs(
            k_ref,
            mu_H_dw,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8,
            er_FH ** (1 - eps_H),
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
    ) / (2 * eps_H * np.log(er_FH_ref))


def der_Da_7pts(
    k_ref,
    mu_ref,
    z,
    PM,
    PM_NW,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):

    # Computing new k, mu coming from the AP effect (for the derivative of lnP_obs over lnDa(z).
    k_D_up = k(z, mu_ref, k_ref, er_FH, er_FH_ref, er_FDA ** (1 + eps_Da), er_FDA_ref)
    k_D_up_2 = k(
        z, mu_ref, k_ref, er_FH, er_FH_ref, er_FDA ** (1 + 2 * eps_Da), er_FDA_ref
    )
    k_D_up_3 = k(
        z, mu_ref, k_ref, er_FH, er_FH_ref, er_FDA ** (1 + 3 * eps_Da), er_FDA_ref
    )
    mu_D_up = mu(z, mu_ref, er_FH, er_FH_ref, er_FDA ** (1 + eps_Da), er_FDA_ref)
    mu_D_up_2 = mu(z, mu_ref, er_FH, er_FH_ref, er_FDA ** (1 + 2 * eps_Da), er_FDA_ref)
    mu_D_up_3 = mu(z, mu_ref, er_FH, er_FH_ref, er_FDA ** (1 + 3 * eps_Da), er_FDA_ref)
    k_D_dw = k(z, mu_ref, k_ref, er_FH, er_FH_ref, er_FDA ** (1 - eps_Da), er_FDA_ref)
    k_D_dw_2 = k(
        z, mu_ref, k_ref, er_FH, er_FH_ref, er_FDA ** (1 - 2 * eps_Da), er_FDA_ref
    )
    k_D_dw_3 = k(
        z, mu_ref, k_ref, er_FH, er_FH_ref, er_FDA ** (1 - 3 * eps_Da), er_FDA_ref
    )
    mu_D_dw = mu(z, mu_ref, er_FH, er_FH_ref, er_FDA ** (1 - eps_Da), er_FDA_ref)
    mu_D_dw_2 = mu(z, mu_ref, er_FH, er_FH_ref, er_FDA ** (1 - 2 * eps_Da), er_FDA_ref)
    mu_D_dw_3 = mu(z, mu_ref, er_FH, er_FH_ref, er_FDA ** (1 - 3 * eps_Da), er_FDA_ref)

    return (
        -lnP_obs(
            k_D_dw_3,
            mu_D_dw_3,
            z,
            10 ** PM(np.log10(k_D_dw_3)),
            10 ** PM_NW(np.log10(k_D_dw_3)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA ** (1 - 3 * eps_Da),
            er_FDA_ref,
            sigP,
            sigV,
        )
        + 9
        * lnP_obs(
            k_D_dw_2,
            mu_D_dw_2,
            z,
            10 ** PM(np.log10(k_D_dw_2)),
            10 ** PM_NW(np.log10(k_D_dw_2)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA ** (1 - 2 * eps_Da),
            er_FDA_ref,
            sigP,
            sigV,
        )
        - 45
        * lnP_obs(
            k_D_dw,
            mu_D_dw,
            z,
            10 ** PM(np.log10(k_D_dw)),
            10 ** PM_NW(np.log10(k_D_dw)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA ** (1 - eps_Da),
            er_FDA_ref,
            sigP,
            sigV,
        )
        + 45
        * lnP_obs(
            k_D_up,
            mu_D_up,
            z,
            10 ** PM(np.log10(k_D_up)),
            10 ** PM_NW(np.log10(k_D_up)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA ** (1 + eps_Da),
            er_FDA_ref,
            sigP,
            sigV,
        )
        - 9
        * lnP_obs(
            k_D_up_2,
            mu_D_up_2,
            z,
            10 ** PM(np.log10(k_D_up_2)),
            10 ** PM_NW(np.log10(k_D_up_2)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA ** (1 + 2 * eps_Da),
            er_FDA_ref,
            sigP,
            sigV,
        )
        + lnP_obs(
            k_D_up_3,
            mu_D_up_3,
            z,
            10 ** PM(np.log10(k_D_up_3)),
            10 ** PM_NW(np.log10(k_D_up_3)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA ** (1 + 3 * eps_Da),
            er_FDA_ref,
            sigP,
            sigV,
        )
    ) / (60 * eps_Da * np.log(er_FDA_ref))


def der_Da_5pts(
    k_ref,
    mu_ref,
    z,
    PM,
    PM_NW,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):

    k_D_up = k(z, mu_ref, k_ref, er_FH, er_FH_ref, er_FDA ** (1 + eps_Da), er_FDA_ref)
    k_D_up_2 = k(
        z, mu_ref, k_ref, er_FH, er_FH_ref, er_FDA ** (1 + 2 * eps_Da), er_FDA_ref
    )
    mu_D_up = mu(z, mu_ref, er_FH, er_FH_ref, er_FDA ** (1 + eps_Da), er_FDA_ref)
    mu_D_up_2 = mu(z, mu_ref, er_FH, er_FH_ref, er_FDA ** (1 + 2 * eps_Da), er_FDA_ref)
    k_D_dw = k(z, mu_ref, k_ref, er_FH, er_FH_ref, er_FDA ** (1 - eps_Da), er_FDA_ref)
    k_D_dw_2 = k(
        z, mu_ref, k_ref, er_FH, er_FH_ref, er_FDA ** (1 - 2 * eps_Da), er_FDA_ref
    )
    mu_D_dw = mu(z, mu_ref, er_FH, er_FH_ref, er_FDA ** (1 - eps_Da), er_FDA_ref)
    mu_D_dw_2 = mu(z, mu_ref, er_FH, er_FH_ref, er_FDA ** (1 - 2 * eps_Da), er_FDA_ref)

    return (
        -lnP_obs(
            k_D_up_2,
            mu_D_up_2,
            z,
            10 ** PM(np.log10(k_D_up_2)),
            10 ** PM_NW(np.log10(k_D_up_2)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA ** (1 + 2 * eps_Da),
            er_FDA_ref,
            sigP,
            sigV,
        )
        + 8
        * lnP_obs(
            k_D_up,
            mu_D_up,
            z,
            10 ** PM(np.log10(k_D_up)),
            10 ** PM_NW(np.log10(k_D_up)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA ** (1 + eps_Da),
            er_FDA_ref,
            sigP,
            sigV,
        )
        - 8
        * lnP_obs(
            k_D_dw,
            mu_D_dw,
            z,
            10 ** PM(np.log10(k_D_dw)),
            10 ** PM_NW(np.log10(k_D_dw)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA ** (1 - eps_Da),
            er_FDA_ref,
            sigP,
            sigV,
        )
        + lnP_obs(
            k_D_dw_2,
            mu_D_dw_2,
            z,
            10 ** PM(np.log10(k_D_dw_2)),
            10 ** PM_NW(np.log10(k_D_dw_2)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA ** (1 - 2 * eps_Da),
            er_FDA_ref,
            sigP,
            sigV,
        )
    ) / (12 * eps_Da * np.log(er_FDA_ref))


def der_Da_3pts(
    k_ref,
    mu_ref,
    z,
    PM,
    PM_NW,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):

    k_D_up = k(z, mu_ref, k_ref, er_FH, er_FH_ref, er_FDA ** (1 + eps_Da), er_FDA_ref)
    mu_D_up = mu(z, mu_ref, er_FH, er_FH_ref, er_FDA ** (1 + eps_Da), er_FDA_ref)
    k_D_dw = k(z, mu_ref, k_ref, er_FH, er_FH_ref, er_FDA ** (1 - eps_Da), er_FDA_ref)
    mu_D_dw = mu(z, mu_ref, er_FH, er_FH_ref, er_FDA ** (1 - eps_Da), er_FDA_ref)

    return (
        lnP_obs(
            k_D_up,
            mu_D_up,
            z,
            10 ** PM(np.log10(k_D_up)),
            10 ** PM_NW(np.log10(k_D_up)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA ** (1 + eps_Da),
            er_FDA_ref,
            sigP,
            sigV,
        )
        - lnP_obs(
            k_D_dw,
            mu_D_dw,
            z,
            10 ** PM(np.log10(k_D_dw)),
            10 ** PM_NW(np.log10(k_D_dw)),
            bias_s8,
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA ** (1 - eps_Da),
            er_FDA_ref,
            sigP,
            sigV,
        )
    ) / (2 * eps_Da * np.log(er_FDA_ref))


def der_bs8_3pts(
    k_ref,
    mu_ref,
    z,
    PM,
    PM_NW,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    return (
        lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8 ** (1 + eps_bs8),
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8 ** (1 - eps_bs8),
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
    ) / (2 * eps_bs8 * np.log(bias_s8))


def der_bs8_5pts(
    k_ref,
    mu_ref,
    z,
    PM,
    PM_NW,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    return (
        -lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8 ** (1 + 2 * eps_bs8),
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + 8
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8 ** (1 + eps_bs8),
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - 8
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8 ** (1 - eps_bs8),
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8 ** (1 - 2 * eps_bs8),
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
    ) / (12 * eps_bs8 * np.log(bias_s8))


def der_bs8_7pts(
    k_ref,
    mu_ref,
    z,
    PM,
    PM_NW,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    return (
        -lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8 ** (1 - 3 * eps_bs8),
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + 9
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8 ** (1 - 2 * eps_bs8),
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - 45
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8 ** (1 - eps_bs8),
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + 45
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8 ** (1 + eps_bs8),
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - 9
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8 ** (1 + 2 * eps_bs8),
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8 ** (1 + 3 * eps_bs8),
            gf_s8,
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
    ) / (60 * eps_bs8 * np.log(bias_s8))


def der_fs8_3pts(
    k_ref,
    mu_ref,
    z,
    PM,
    PM_NW,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    return (
        lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8 ** (1 + eps_fs8),
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8 ** (1 - eps_fs8),
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
    ) / (2 * eps_fs8 * np.log(gf_s8))


def der_fs8_5pts(
    k_ref,
    mu_ref,
    z,
    PM,
    PM_NW,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    return (
        -lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8 ** (1 + 2 * eps_fs8),
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + 8
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8 ** (1 + eps_fs8),
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - 8
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8 ** (1 - eps_fs8),
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8 ** (1 - 2 * eps_fs8),
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
    ) / (12 * eps_fs8 * np.log(gf_s8))


def der_fs8_7pts(
    k_ref,
    mu_ref,
    z,
    PM,
    PM_NW,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    return (
        -lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8 ** (1 - 3 * eps_fs8),
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + 9
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8 ** (1 - 2 * eps_fs8),
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - 45
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8 ** (1 - eps_fs8),
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + 45
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8 ** (1 + eps_fs8),
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        - 9
        * lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8 ** (1 + 2 * eps_fs8),
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
        + lnP_obs(
            k_ref,
            mu_ref,
            z,
            10 ** PM(np.log10(k_ref)),
            10 ** PM_NW(np.log10(k_ref)),
            bias_s8,
            gf_s8 ** (1 + 3 * eps_fs8),
            er_FH,
            er_FH_ref,
            er_FDA,
            er_FDA_ref,
            sigP,
            sigV,
        )
    ) / (60 * eps_fs8 * np.log(gf_s8))


def der_P_shot(
    k_ref,
    mu_ref,
    z,
    PM,
    PM_NW,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    return 1.0 / P_obs(
        k_ref,
        mu_ref,
        z,
        PM,
        PM_NW,
        bias_s8,
        gf_s8,
        er_FH,
        er_FH_ref,
        er_FDA,
        er_FDA_ref,
        sigP,
        sigV,
    )


# Derivatives of the non linear nuisance parameters : the condition = 0 is added for the linear derivatives
def der_sp_3pts(
    k_ref,
    mu_ref,
    z,
    PM,
    PM_NW,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    if choice_L_NL == "SNL":
        return (
            lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP * (1 + eps_sig_p),
                sigV,
            )
            - lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP * (1 - eps_sig_p),
                sigV,
            )
        ) / (2 * eps_sig_p * sigP)
    else:
        return 0.0


def der_sp_5pts(
    k_ref,
    mu_ref,
    z,
    PM,
    PM_NW,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    if choice_L_NL == "SNL":
        return (
            -lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP * (1 + 2 * eps_sig_p),
                sigV,
            )
            + 8
            * lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP * (1 + eps_sig_p),
                sigV,
            )
            - 8
            * lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP * (1 - eps_sig_p),
                sigV,
            )
            + lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP * (1 - 2 * eps_sig_p),
                sigV,
            )
        ) / (12 * eps_sig_p * sigP)
    else:
        return 0.0


def der_sp_7pts(
    k_ref,
    mu_ref,
    z,
    PM,
    PM_NW,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    if choice_L_NL == "SNL":
        return (
            -lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP * (1 - 3 * eps_sig_p),
                sigV,
            )
            + 9
            * lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP * (1 - 2 * eps_sig_p),
                sigV,
            )
            - 45
            * lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP * (1 - eps_sig_p),
                sigV,
            )
            + 45
            * lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP * (1 + eps_sig_p),
                sigV,
            )
            - 9
            * lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP * (1 + 2 * eps_sig_p),
                sigV,
            )
            + lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP * (1 + 3 * eps_sig_p),
                sigV,
            )
        ) / (60 * eps_sig_p * sigP)
    else:
        return 0.0


def der_sv_3pts(
    k_ref,
    mu_ref,
    z,
    PM,
    PM_NW,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    if choice_L_NL == "SNL":
        return (
            lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP,
                sigV * (1 + eps_sig_v),
            )
            - lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP,
                sigV * (1 - eps_sig_v),
            )
        ) / (2 * eps_sig_v * sigV)
    else:
        return 0.0


def der_sv_5pts(
    k_ref,
    mu_ref,
    z,
    PM,
    PM_NW,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    if choice_L_NL == "SNL":
        return (
            -lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP,
                sigV * (1 + 2 * eps_sig_v),
            )
            + 8
            * lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP,
                sigV * (1 + eps_sig_v),
            )
            - 8
            * lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP,
                sigV * (1 - eps_sig_v),
            )
            + lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP,
                sigV * (1 - 2 * eps_sig_v),
            )
        ) / (12 * eps_sig_v * sigV)
    else:
        return 0.0


def der_sv_7pts(
    k_ref,
    mu_ref,
    z,
    PM,
    PM_NW,
    bias_s8,
    gf_s8,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    if choice_L_NL == "SNL":
        return (
            -lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP,
                sigV * (1 - 3 * eps_sig_v),
            )
            + 9
            * lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP,
                sigV * (1 - 2 * eps_sig_v),
            )
            - 45
            * lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP,
                sigV * (1 - eps_sig_v),
            )
            + 45
            * lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP,
                sigV * (1 + eps_sig_v),
            )
            - 9
            * lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP,
                sigV * (1 + 2 * eps_sig_v),
            )
            + lnP_obs(
                k_ref,
                mu_ref,
                z,
                10 ** PM(np.log10(k_ref)),
                10 ** PM_NW(np.log10(k_ref)),
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP,
                sigV * (1 + 3 * eps_sig_v),
            )
        ) / (60 * eps_sig_v * sigV)
    else:
        return 0.0


# Computing the effective volume of the survey.
def Veff(
    k_ref,
    mu_ref,
    z,
    PM,
    PM_NW,
    bias_s8,
    gf_s8,
    com_n,
    Vs,
    er_FH,
    er_FH_ref,
    er_FDA,
    er_FDA_ref,
    sigP,
    sigV,
):
    return (
        Vs
        * (
            com_n
            * P_obs(
                k_ref,
                mu_ref,
                z,
                PM,
                PM_NW,
                bias_s8,
                gf_s8,
                er_FH,
                er_FH_ref,
                er_FDA,
                er_FDA_ref,
                sigP,
                sigV,
            )
            / (
                com_n
                * P_obs(
                    k_ref,
                    mu_ref,
                    z,
                    PM,
                    PM_NW,
                    bias_s8,
                    gf_s8,
                    er_FH,
                    er_FH_ref,
                    er_FDA,
                    er_FDA_ref,
                    sigP,
                    sigV,
                )
                + 1
            )
        )
        ** 2
    )


# Increment values.
i, j, l = 0, 0, 0

# Computing f(z)
f_of_z = CubicSpline(tn, f_ref(tn))
growth_f = f_of_z(zrange)

# Save the new f(z) to use them in the derivative call file
outQ = open("input/bias_n_growth_baseline.dat", "w")
outQ.write(
    "#In order: z, delta_z, bias, dn3, f(z) mean_redshift (1 = Yes, 0 = No)" + "\n"
)
ooo = 0
while ooo < len(zrange):
    outQ.write(str("%.10e" % zrange[ooo]))
    outQ.write(" ")
    outQ.write(str("%.10e" % Delta_z[ooo]))
    outQ.write(" ")
    outQ.write(str("%.10e" % b[ooo]))
    outQ.write(" ")
    outQ.write(str("%.10e" % dn3[ooo]))
    outQ.write(" ")
    outQ.write(str("%.10e" % growth_f[ooo]))
    outQ.write(" ")
    outQ.write(str("%.10e" % z_mean_index))
    outQ.write("\n")
    ooo = ooo + 1
outQ.close()

# Other background quantities. In order: dVdz, volume inside single redshift bin, n(z),
# H(z), angular distance, s8(z), chi(z) and growth D(z).
dV_dz = np.zeros(len(zrange))
V_a = np.zeros(len(zrange))
n = np.zeros(len(zrange))
H_km_s_Mpc = np.zeros(len(zrange))
D_A_Mpc = np.zeros(len(zrange))
S_8 = np.zeros(len(zrange))
X = np.zeros(len(zrange))
DG_growth = np.zeros(len(zrange))

i = 0
while i < len(zrange):
    DG_growth[i] = DGrowth(zrange[i])
    H_km_s_Mpc[i] = H(zrange[i])
    X[i] = chi(zrange[i])
    D_A_Mpc[i] = D_A(zrange[i])
    dV_dz[i] = (
        4
        * np.pi
        / 3
        * (
            (1 + zrange[i] + Delta_z[i] / 2) ** 3 * D_A(zrange[i] + Delta_z[i] / 2) ** 3
            - (1 + zrange[i] - Delta_z[i] / 2) ** 3
            * D_A(zrange[i] - Delta_z[i] / 2) ** 3
        )
        / (4 * np.pi * (180 / np.pi) ** 2)
    )
    V_a[i] = dV_dz[i] * Vol_eff
    n[i] = dn3[i] / dV_dz[i] * Delta_z[i]
    S_8[i] = float(
        np.genfromtxt(
            path_spec + "/Pks8sqRatio_ist_LogSplineInterpPk_" + str(zrange[i]) + ".dat",
            skip_header=3,
            usecols=(2),
            unpack=True,
            max_rows=1,
        )
    )
    i = i + 1
i = 0

# Computing the effective volume of the survey Veff.
Delta_Omega = Vol_eff * np.pi**2 / 180.0**2

# Calling function definition for the big fisher derivatives
print("Initializing derivatives call functions")
os.chdir("input/derivatives_call")
os.system("python set_inputF_n4_NL.py")
os.chdir("../../")

# Loading the derivative functions call in order to be called automatically while integrating.
der_input_LU = np.genfromtxt(
    "input/derivatives_call/der_input_NB_LU", unpack=True, delimiter="\n", dtype="S1000"
)
der_input_RD = np.genfromtxt(
    "input/derivatives_call/der_input_NB_RD", unpack=True, delimiter="\n", dtype="S1000"
)
der_input_RU = np.genfromtxt(
    "input/derivatives_call/der_input_NB_all",
    unpack=True,
    delimiter="\n",
    dtype="S1000",
)

# B. Integral computing part
print("Begin integrals calculation")

# Variables for the 3 Fisher blocs
A, B, C = 0.0, 0.0, 0.0

# Set the (k,mu) 2D space for the integral calculation : 100 values is enough for mu.
x_var = np.linspace(kMIN, kMAX, integ_prec) * h_fid
y_var = np.linspace(-1, 1, 100)

# (k,mu) array meshgrid
ecs, way = np.meshgrid(x_var, y_var)

# Integral steps for k and mu.
delta_x = x_var[1] - x_var[0]
delta_y = y_var[1] - y_var[0]

# Bar progression init.
maxval = 594
iterbar = 3

print("Integrals calculation in progress :")

"""
Integral calculation using multiprocessing.
Here we consider 3 different functions which calculate all the upper triangle Fisher matrix elements by considering 3 blocs, the 0 values are not computed :
    - The shape parameters only (Left Up part of the matrix).
    - The redshift dependent parameters only (Right Down part of the matrix).
    - The cross therms between shape and redshift dependent parameters (Right Up part of the matrix). 
This part run in parallel mode and uses 10 processes for the LU part, 20 for the RU part and 15 for the RD part for each redshift bins.  
All the elements are saved into 3 temporar files which are used in the next part of the program to build the final Fisher matrix.
"""

# Prefix power spectrum files.
i = 0
pref = "/Pks8sqRatio_ist_LogSplineInterpPk_"
path_In = path_In[:-4]

# Path towards each parameters power spectrums.
path_fid = ["/fid"]
path_wb = ["/wb_up", "/wb_up2", "/wb_up3", "/wb_dw", "/wb_dw2", "/wb_dw3"]
path_h = ["/h_up", "/h_up2", "/h_up3", "/h_dw", "/h_dw2", "/h_dw3"]
path_wm = ["/wm_up", "/wm_up2", "/wm_up3", "/wm_dw", "/wm_dw2", "/wm_dw3"]
path_ns = ["/ns_up", "/ns_up2", "/ns_up3", "/ns_dw", "/ns_dw2", "/ns_dw3"]

while i < len(zrange):

    # Loading fiducial power spectrum & fiducial sigma_8.
    P_m_fid = np.genfromtxt(
        path_In + path_fid[0] + pref + str(zrange[i]) + ".dat",
        skip_header=3,
        usecols=(1,),
        unpack=True,
    ) / (h_fid**3)
    sig_8_fid = float(
        np.genfromtxt(
            path_In + path_fid[0] + pref + str(zrange[i]) + ".dat",
            skip_header=3,
            usecols=(2),
            unpack=True,
            max_rows=1,
        )
    )
    P_m_NW_fid = np.genfromtxt(
        path_In + path_fid[0] + pref + str(zrange[i]) + "_NW.dat",
        skip_header=3,
        usecols=(1,),
        unpack=True,
    ) / (h_fid**3)

    # Loading other power spectrum & sigma_8 (for the derivatives over the shape parameters).
    P_m_ob_up = np.genfromtxt(
        path_In + path_wb[0] + pref + str(zrange[i]) + ".dat",
        skip_header=3,
        usecols=(1,),
        unpack=True,
    ) / (h_fid**3)
    P_m_ob_dw = np.genfromtxt(
        path_In + path_wb[3] + pref + str(zrange[i]) + ".dat",
        skip_header=3,
        usecols=(1,),
        unpack=True,
    ) / (h_fid**3)

    P_m_h_up = np.genfromtxt(
        path_In + path_h[0] + pref + str(zrange[i]) + ".dat",
        skip_header=3,
        usecols=(1,),
        unpack=True,
    ) / ((h_fid * (1.0 + eps_h)) ** 3)
    P_m_h_dw = np.genfromtxt(
        path_In + path_h[3] + pref + str(zrange[i]) + ".dat",
        skip_header=3,
        usecols=(1,),
        unpack=True,
    ) / ((h_fid * (1.0 - eps_h)) ** 3)

    P_m_om_up = np.genfromtxt(
        path_In + path_wm[0] + pref + str(zrange[i]) + ".dat",
        skip_header=3,
        usecols=(1,),
        unpack=True,
    ) / (h_fid**3)
    P_m_om_dw = np.genfromtxt(
        path_In + path_wm[3] + pref + str(zrange[i]) + ".dat",
        skip_header=3,
        usecols=(1,),
        unpack=True,
    ) / (h_fid**3)

    P_m_ns_up = np.genfromtxt(
        path_In + path_ns[0] + pref + str(zrange[i]) + ".dat",
        skip_header=3,
        usecols=(1,),
        unpack=True,
    ) / (h_fid**3)
    P_m_ns_dw = np.genfromtxt(
        path_In + path_ns[3] + pref + str(zrange[i]) + ".dat",
        skip_header=3,
        usecols=(1,),
        unpack=True,
    ) / (h_fid**3)

    P_m_NW_ob_up = np.genfromtxt(
        path_In + path_wb[0] + pref + str(zrange[i]) + "_NW.dat",
        skip_header=3,
        usecols=(1,),
        unpack=True,
    ) / (h_fid**3)
    P_m_NW_ob_dw = np.genfromtxt(
        path_In + path_wb[3] + pref + str(zrange[i]) + "_NW.dat",
        skip_header=3,
        usecols=(1,),
        unpack=True,
    ) / (h_fid**3)

    P_m_NW_h_up = np.genfromtxt(
        path_In + path_h[0] + pref + str(zrange[i]) + "_NW.dat",
        skip_header=3,
        usecols=(1,),
        unpack=True,
    ) / ((h_fid * (1.0 + eps_h)) ** 3)
    P_m_NW_h_dw = np.genfromtxt(
        path_In + path_h[3] + pref + str(zrange[i]) + "_NW.dat",
        skip_header=3,
        usecols=(1,),
        unpack=True,
    ) / ((h_fid * (1.0 - eps_h)) ** 3)

    P_m_NW_om_up = np.genfromtxt(
        path_In + path_wm[0] + pref + str(zrange[i]) + "_NW.dat",
        skip_header=3,
        usecols=(1,),
        unpack=True,
    ) / (h_fid**3)
    P_m_NW_om_dw = np.genfromtxt(
        path_In + path_wm[3] + pref + str(zrange[i]) + "_NW.dat",
        skip_header=3,
        usecols=(1,),
        unpack=True,
    ) / (h_fid**3)

    P_m_NW_ns_up = np.genfromtxt(
        path_In + path_ns[0] + pref + str(zrange[i]) + "_NW.dat",
        skip_header=3,
        usecols=(1,),
        unpack=True,
    ) / (h_fid**3)
    P_m_NW_ns_dw = np.genfromtxt(
        path_In + path_ns[3] + pref + str(zrange[i]) + "_NW.dat",
        skip_header=3,
        usecols=(1,),
        unpack=True,
    ) / (h_fid**3)

    # Cubic interpolation in log10 basis.
    k_cf_ad = (
        np.genfromtxt(
            path_In + path_fid[0] + pref + str(zrange[i]) + ".dat",
            skip_header=3,
            usecols=(0,),
            unpack=True,
        )
        * h_fid
    )
    P_m = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_fid))

    k_cf_ad = (
        np.genfromtxt(
            path_In + path_wb[0] + pref + str(zrange[i]) + ".dat",
            skip_header=3,
            usecols=(0,),
            unpack=True,
        )
        * h_fid
    )
    P_m_wb_up = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_ob_up))
    k_cf_ad = (
        np.genfromtxt(
            path_In + path_wb[3] + pref + str(zrange[i]) + ".dat",
            skip_header=3,
            usecols=(0,),
            unpack=True,
        )
        * h_fid
    )
    P_m_wb_dw = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_ob_dw))

    k_cf_ad = (
        np.genfromtxt(
            path_In + path_h[0] + pref + str(zrange[i]) + ".dat",
            skip_header=3,
            usecols=(0,),
            unpack=True,
        )
        * h_fid
        * (1.0 + eps_h)
    )
    P_m_h_up = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_h_up))
    k_cf_ad = (
        np.genfromtxt(
            path_In + path_h[3] + pref + str(zrange[i]) + ".dat",
            skip_header=3,
            usecols=(0,),
            unpack=True,
        )
        * h_fid
        * (1.0 - eps_h)
    )
    P_m_h_dw = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_h_dw))

    k_cf_ad = (
        np.genfromtxt(
            path_In + path_wm[0] + pref + str(zrange[i]) + ".dat",
            skip_header=3,
            usecols=(0,),
            unpack=True,
        )
        * h_fid
    )
    P_m_wm_up = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_om_up))
    k_cf_ad = (
        np.genfromtxt(
            path_In + path_wm[3] + pref + str(zrange[i]) + ".dat",
            skip_header=3,
            usecols=(0,),
            unpack=True,
        )
        * h_fid
    )
    P_m_wm_dw = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_om_dw))

    k_cf_ad = (
        np.genfromtxt(
            path_In + path_ns[0] + pref + str(zrange[i]) + ".dat",
            skip_header=3,
            usecols=(0,),
            unpack=True,
        )
        * h_fid
    )
    P_m_ns_up = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_ns_up))
    k_cf_ad = (
        np.genfromtxt(
            path_In + path_ns[3] + pref + str(zrange[i]) + ".dat",
            skip_header=3,
            usecols=(0,),
            unpack=True,
        )
        * h_fid
    )
    P_m_ns_dw = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_ns_dw))

    k_cf_ad = (
        np.genfromtxt(
            path_In + path_fid[0] + pref + str(zrange[i]) + "_NW.dat",
            skip_header=3,
            usecols=(0,),
            unpack=True,
        )
        * h_fid
    )
    P_m_NW = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_fid))

    k_cf_ad = (
        np.genfromtxt(
            path_In + path_wb[0] + pref + str(zrange[i]) + "_NW.dat",
            skip_header=3,
            usecols=(0,),
            unpack=True,
        )
        * h_fid
    )
    P_m_NW_wb_up = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_ob_up))
    k_cf_ad = (
        np.genfromtxt(
            path_In + path_wb[3] + pref + str(zrange[i]) + "_NW.dat",
            skip_header=3,
            usecols=(0,),
            unpack=True,
        )
        * h_fid
    )
    P_m_NW_wb_dw = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_ob_dw))

    k_cf_ad = (
        np.genfromtxt(
            path_In + path_h[0] + pref + str(zrange[i]) + "_NW.dat",
            skip_header=3,
            usecols=(0,),
            unpack=True,
        )
        * h_fid
        * (1.0 + eps_h)
    )
    P_m_NW_h_up = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_h_up))
    k_cf_ad = (
        np.genfromtxt(
            path_In + path_h[3] + pref + str(zrange[i]) + "_NW.dat",
            skip_header=3,
            usecols=(0,),
            unpack=True,
        )
        * h_fid
        * (1.0 - eps_h)
    )
    P_m_NW_h_dw = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_h_dw))

    k_cf_ad = (
        np.genfromtxt(
            path_In + path_wm[0] + pref + str(zrange[i]) + "_NW.dat",
            skip_header=3,
            usecols=(0,),
            unpack=True,
        )
        * h_fid
    )
    P_m_NW_wm_up = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_om_up))
    k_cf_ad = (
        np.genfromtxt(
            path_In + path_wm[3] + pref + str(zrange[i]) + "_NW.dat",
            skip_header=3,
            usecols=(0,),
            unpack=True,
        )
        * h_fid
    )
    P_m_NW_wm_dw = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_om_dw))

    k_cf_ad = (
        np.genfromtxt(
            path_In + path_ns[0] + pref + str(zrange[i]) + "_NW.dat",
            skip_header=3,
            usecols=(0,),
            unpack=True,
        )
        * h_fid
    )
    P_m_NW_ns_up = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_ns_up))
    k_cf_ad = (
        np.genfromtxt(
            path_In + path_ns[3] + pref + str(zrange[i]) + "_NW.dat",
            skip_header=3,
            usecols=(0,),
            unpack=True,
        )
        * h_fid
    )
    P_m_NW_ns_dw = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_ns_dw))

    # Load and interpolate more files for the 5pts derivatives.
    if int(choice_der_pts_shape) >= 5:
        P_m_ob_up2 = np.genfromtxt(
            path_In + path_wb[1] + pref + str(zrange[i]) + ".dat",
            skip_header=3,
            usecols=(1,),
            unpack=True,
        ) / (h_fid**3)
        P_m_ob_dw2 = np.genfromtxt(
            path_In + path_wb[4] + pref + str(zrange[i]) + ".dat",
            skip_header=3,
            usecols=(1,),
            unpack=True,
        ) / (h_fid**3)

        P_m_h_up2 = np.genfromtxt(
            path_In + path_h[1] + pref + str(zrange[i]) + ".dat",
            skip_header=3,
            usecols=(1,),
            unpack=True,
        ) / ((h_fid * (1.0 + 2 * eps_h)) ** 3)
        P_m_h_dw2 = np.genfromtxt(
            path_In + path_h[4] + pref + str(zrange[i]) + ".dat",
            skip_header=3,
            usecols=(1,),
            unpack=True,
        ) / ((h_fid * (1.0 - 2 * eps_h)) ** 3)

        P_m_om_up2 = np.genfromtxt(
            path_In + path_wm[1] + pref + str(zrange[i]) + ".dat",
            skip_header=3,
            usecols=(1,),
            unpack=True,
        ) / (h_fid**3)
        P_m_om_dw2 = np.genfromtxt(
            path_In + path_wm[4] + pref + str(zrange[i]) + ".dat",
            skip_header=3,
            usecols=(1,),
            unpack=True,
        ) / (h_fid**3)

        P_m_ns_up2 = np.genfromtxt(
            path_In + path_ns[1] + pref + str(zrange[i]) + ".dat",
            skip_header=3,
            usecols=(1,),
            unpack=True,
        ) / (h_fid**3)
        P_m_ns_dw2 = np.genfromtxt(
            path_In + path_ns[4] + pref + str(zrange[i]) + ".dat",
            skip_header=3,
            usecols=(1,),
            unpack=True,
        ) / (h_fid**3)

        P_m_NW_ob_up2 = np.genfromtxt(
            path_In + path_wb[1] + pref + str(zrange[i]) + "_NW.dat",
            skip_header=3,
            usecols=(1,),
            unpack=True,
        ) / (h_fid**3)
        P_m_NW_ob_dw2 = np.genfromtxt(
            path_In + path_wb[4] + pref + str(zrange[i]) + "_NW.dat",
            skip_header=3,
            usecols=(1,),
            unpack=True,
        ) / (h_fid**3)

        P_m_NW_h_up2 = np.genfromtxt(
            path_In + path_h[1] + pref + str(zrange[i]) + "_NW.dat",
            skip_header=3,
            usecols=(1,),
            unpack=True,
        ) / ((h_fid * (1.0 + 2 * eps_h)) ** 3)
        P_m_NW_h_dw2 = np.genfromtxt(
            path_In + path_h[4] + pref + str(zrange[i]) + "_NW.dat",
            skip_header=3,
            usecols=(1,),
            unpack=True,
        ) / ((h_fid * (1.0 - 2 * eps_h)) ** 3)

        P_m_NW_om_up2 = np.genfromtxt(
            path_In + path_wm[1] + pref + str(zrange[i]) + "_NW.dat",
            skip_header=3,
            usecols=(1,),
            unpack=True,
        ) / (h_fid**3)
        P_m_NW_om_dw2 = np.genfromtxt(
            path_In + path_wm[4] + pref + str(zrange[i]) + "_NW.dat",
            skip_header=3,
            usecols=(1,),
            unpack=True,
        ) / (h_fid**3)

        P_m_NW_ns_up2 = np.genfromtxt(
            path_In + path_ns[1] + pref + str(zrange[i]) + "_NW.dat",
            skip_header=3,
            usecols=(1,),
            unpack=True,
        ) / (h_fid**3)
        P_m_NW_ns_dw2 = np.genfromtxt(
            path_In + path_ns[4] + pref + str(zrange[i]) + "_NW.dat",
            skip_header=3,
            usecols=(1,),
            unpack=True,
        ) / (h_fid**3)

        k_cf_ad = (
            np.genfromtxt(
                path_In + path_wb[1] + pref + str(zrange[i]) + ".dat",
                skip_header=3,
                usecols=(0,),
                unpack=True,
            )
            * h_fid
        )
        P_m_wb_up2 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_ob_up2))
        k_cf_ad = (
            np.genfromtxt(
                path_In + path_wb[4] + pref + str(zrange[i]) + ".dat",
                skip_header=3,
                usecols=(0,),
                unpack=True,
            )
            * h_fid
        )
        P_m_wb_dw2 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_ob_dw2))

        k_cf_ad = (
            np.genfromtxt(
                path_In + path_h[1] + pref + str(zrange[i]) + ".dat",
                skip_header=3,
                usecols=(0,),
                unpack=True,
            )
            * h_fid
            * (1.0 + 2 * eps_h)
        )
        P_m_h_up2 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_h_up2))
        k_cf_ad = (
            np.genfromtxt(
                path_In + path_h[4] + pref + str(zrange[i]) + ".dat",
                skip_header=3,
                usecols=(0,),
                unpack=True,
            )
            * h_fid
            * (1.0 - 2 * eps_h)
        )
        P_m_h_dw2 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_h_dw2))

        k_cf_ad = (
            np.genfromtxt(
                path_In + path_wm[1] + pref + str(zrange[i]) + ".dat",
                skip_header=3,
                usecols=(0,),
                unpack=True,
            )
            * h_fid
        )
        P_m_wm_up2 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_om_up2))
        k_cf_ad = (
            np.genfromtxt(
                path_In + path_wm[4] + pref + str(zrange[i]) + ".dat",
                skip_header=3,
                usecols=(0,),
                unpack=True,
            )
            * h_fid
        )
        P_m_wm_dw2 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_om_dw2))

        k_cf_ad = (
            np.genfromtxt(
                path_In + path_ns[1] + pref + str(zrange[i]) + ".dat",
                skip_header=3,
                usecols=(0,),
                unpack=True,
            )
            * h_fid
        )
        P_m_ns_up2 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_ns_up2))
        k_cf_ad = (
            np.genfromtxt(
                path_In + path_ns[4] + pref + str(zrange[i]) + ".dat",
                skip_header=3,
                usecols=(0,),
                unpack=True,
            )
            * h_fid
        )
        P_m_ns_dw2 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_ns_dw2))

        k_cf_ad = (
            np.genfromtxt(
                path_In + path_wb[1] + pref + str(zrange[i]) + "_NW.dat",
                skip_header=3,
                usecols=(0,),
                unpack=True,
            )
            * h_fid
        )
        P_m_NW_wb_up2 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_ob_up2))
        k_cf_ad = (
            np.genfromtxt(
                path_In + path_wb[4] + pref + str(zrange[i]) + "_NW.dat",
                skip_header=3,
                usecols=(0,),
                unpack=True,
            )
            * h_fid
        )
        P_m_NW_wb_dw2 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_ob_dw2))

        k_cf_ad = (
            np.genfromtxt(
                path_In + path_h[1] + pref + str(zrange[i]) + "_NW.dat",
                skip_header=3,
                usecols=(0,),
                unpack=True,
            )
            * h_fid
            * (1.0 + 2 * eps_h)
        )
        P_m_NW_h_up2 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_h_up2))
        k_cf_ad = (
            np.genfromtxt(
                path_In + path_h[4] + pref + str(zrange[i]) + "_NW.dat",
                skip_header=3,
                usecols=(0,),
                unpack=True,
            )
            * h_fid
            * (1.0 - 2 * eps_h)
        )
        P_m_NW_h_dw2 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_h_dw2))

        k_cf_ad = (
            np.genfromtxt(
                path_In + path_wm[1] + pref + str(zrange[i]) + "_NW.dat",
                skip_header=3,
                usecols=(0,),
                unpack=True,
            )
            * h_fid
        )
        P_m_NW_wm_up2 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_om_up2))
        k_cf_ad = (
            np.genfromtxt(
                path_In + path_wm[4] + pref + str(zrange[i]) + "_NW.dat",
                skip_header=3,
                usecols=(0,),
                unpack=True,
            )
            * h_fid
        )
        P_m_NW_wm_dw2 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_om_dw2))

        k_cf_ad = (
            np.genfromtxt(
                path_In + path_ns[1] + pref + str(zrange[i]) + "_NW.dat",
                skip_header=3,
                usecols=(0,),
                unpack=True,
            )
            * h_fid
        )
        P_m_NW_ns_up2 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_ns_up2))
        k_cf_ad = (
            np.genfromtxt(
                path_In + path_ns[4] + pref + str(zrange[i]) + "_NW.dat",
                skip_header=3,
                usecols=(0,),
                unpack=True,
            )
            * h_fid
        )
        P_m_NW_ns_dw2 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_ns_dw2))

        # Load and interpolate more files for the 7pts derivatives.
        if int(choice_der_pts_shape) == 7:
            P_m_ob_up3 = np.genfromtxt(
                path_In + path_wb[2] + pref + str(zrange[i]) + ".dat",
                skip_header=3,
                usecols=(1,),
                unpack=True,
            ) / (h_fid**3)
            P_m_ob_dw3 = np.genfromtxt(
                path_In + path_wb[5] + pref + str(zrange[i]) + ".dat",
                skip_header=3,
                usecols=(1,),
                unpack=True,
            ) / (h_fid**3)

            P_m_h_up3 = np.genfromtxt(
                path_In + path_h[2] + pref + str(zrange[i]) + ".dat",
                skip_header=3,
                usecols=(1,),
                unpack=True,
            ) / ((h_fid * (1.0 + 3 * eps_h)) ** 3)
            P_m_h_dw3 = np.genfromtxt(
                path_In + path_h[5] + pref + str(zrange[i]) + ".dat",
                skip_header=3,
                usecols=(1,),
                unpack=True,
            ) / ((h_fid * (1.0 - 3 * eps_h)) ** 3)

            P_m_om_up3 = np.genfromtxt(
                path_In + path_wm[2] + pref + str(zrange[i]) + ".dat",
                skip_header=3,
                usecols=(1,),
                unpack=True,
            ) / (h_fid**3)
            P_m_om_dw3 = np.genfromtxt(
                path_In + path_wm[5] + pref + str(zrange[i]) + ".dat",
                skip_header=3,
                usecols=(1,),
                unpack=True,
            ) / (h_fid**3)

            P_m_ns_up3 = np.genfromtxt(
                path_In + path_ns[2] + pref + str(zrange[i]) + ".dat",
                skip_header=3,
                usecols=(1,),
                unpack=True,
            ) / (h_fid**3)
            P_m_ns_dw3 = np.genfromtxt(
                path_In + path_ns[5] + pref + str(zrange[i]) + ".dat",
                skip_header=3,
                usecols=(1,),
                unpack=True,
            ) / (h_fid**3)

            P_m_NW_ob_up3 = np.genfromtxt(
                path_In + path_wb[2] + pref + str(zrange[i]) + "_NW.dat",
                skip_header=3,
                usecols=(1,),
                unpack=True,
            ) / (h_fid**3)
            P_m_NW_ob_dw3 = np.genfromtxt(
                path_In + path_wb[5] + pref + str(zrange[i]) + "_NW.dat",
                skip_header=3,
                usecols=(1,),
                unpack=True,
            ) / (h_fid**3)

            P_m_NW_h_up3 = np.genfromtxt(
                path_In + path_h[2] + pref + str(zrange[i]) + "_NW.dat",
                skip_header=3,
                usecols=(1,),
                unpack=True,
            ) / ((h_fid * (1.0 + 3 * eps_h)) ** 3)
            P_m_NW_h_dw3 = np.genfromtxt(
                path_In + path_h[5] + pref + str(zrange[i]) + "_NW.dat",
                skip_header=3,
                usecols=(1,),
                unpack=True,
            ) / ((h_fid * (1.0 - 3 * eps_h)) ** 3)

            P_m_NW_om_up3 = np.genfromtxt(
                path_In + path_wm[2] + pref + str(zrange[i]) + "_NW.dat",
                skip_header=3,
                usecols=(1,),
                unpack=True,
            ) / (h_fid**3)
            P_m_NW_om_dw3 = np.genfromtxt(
                path_In + path_wm[5] + pref + str(zrange[i]) + "_NW.dat",
                skip_header=3,
                usecols=(1,),
                unpack=True,
            ) / (h_fid**3)

            P_m_NW_ns_up3 = np.genfromtxt(
                path_In + path_ns[2] + pref + str(zrange[i]) + "_NW.dat",
                skip_header=3,
                usecols=(1,),
                unpack=True,
            ) / (h_fid**3)
            P_m_NW_ns_dw3 = np.genfromtxt(
                path_In + path_ns[5] + pref + str(zrange[i]) + "_NW.dat",
                skip_header=3,
                usecols=(1,),
                unpack=True,
            ) / (h_fid**3)

            k_cf_ad = (
                np.genfromtxt(
                    path_In + path_wb[2] + pref + str(zrange[i]) + ".dat",
                    skip_header=3,
                    usecols=(0,),
                    unpack=True,
                )
                * h_fid
            )
            P_m_wb_up3 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_ob_up3))
            k_cf_ad = (
                np.genfromtxt(
                    path_In + path_wb[5] + pref + str(zrange[i]) + ".dat",
                    skip_header=3,
                    usecols=(0,),
                    unpack=True,
                )
                * h_fid
            )
            P_m_wb_dw3 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_ob_dw3))

            k_cf_ad = (
                np.genfromtxt(
                    path_In + path_h[2] + pref + str(zrange[i]) + ".dat",
                    skip_header=3,
                    usecols=(0,),
                    unpack=True,
                )
                * h_fid
                * (1.0 + 3 * eps_h)
            )
            P_m_h_up3 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_h_up3))
            k_cf_ad = (
                np.genfromtxt(
                    path_In + path_h[5] + pref + str(zrange[i]) + ".dat",
                    skip_header=3,
                    usecols=(0,),
                    unpack=True,
                )
                * h_fid
                * (1.0 - 3 * eps_h)
            )
            P_m_h_dw3 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_h_dw3))

            k_cf_ad = (
                np.genfromtxt(
                    path_In + path_wm[2] + pref + str(zrange[i]) + ".dat",
                    skip_header=3,
                    usecols=(0,),
                    unpack=True,
                )
                * h_fid
            )
            P_m_wm_up3 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_om_up3))
            k_cf_ad = (
                np.genfromtxt(
                    path_In + path_wm[5] + pref + str(zrange[i]) + ".dat",
                    skip_header=3,
                    usecols=(0,),
                    unpack=True,
                )
                * h_fid
            )
            P_m_wm_dw3 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_om_dw3))

            k_cf_ad = (
                np.genfromtxt(
                    path_In + path_ns[2] + pref + str(zrange[i]) + ".dat",
                    skip_header=3,
                    usecols=(0,),
                    unpack=True,
                )
                * h_fid
            )
            P_m_ns_up3 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_ns_up3))
            k_cf_ad = (
                np.genfromtxt(
                    path_In + path_ns[5] + pref + str(zrange[i]) + ".dat",
                    skip_header=3,
                    usecols=(0,),
                    unpack=True,
                )
                * h_fid
            )
            P_m_ns_dw3 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_ns_dw3))

            k_cf_ad = (
                np.genfromtxt(
                    path_In + path_wb[2] + pref + str(zrange[i]) + "_NW.dat",
                    skip_header=3,
                    usecols=(0,),
                    unpack=True,
                )
                * h_fid
            )
            P_m_NW_wb_up3 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_ob_up3))
            k_cf_ad = (
                np.genfromtxt(
                    path_In + path_wb[5] + pref + str(zrange[i]) + "_NW.dat",
                    skip_header=3,
                    usecols=(0,),
                    unpack=True,
                )
                * h_fid
            )
            P_m_NW_wb_dw3 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_ob_dw3))

            k_cf_ad = (
                np.genfromtxt(
                    path_In + path_h[2] + pref + str(zrange[i]) + "_NW.dat",
                    skip_header=3,
                    usecols=(0,),
                    unpack=True,
                )
                * h_fid
                * (1.0 + 3 * eps_h)
            )
            P_m_NW_h_up3 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_h_up3))
            k_cf_ad = (
                np.genfromtxt(
                    path_In + path_h[5] + pref + str(zrange[i]) + "_NW.dat",
                    skip_header=3,
                    usecols=(0,),
                    unpack=True,
                )
                * h_fid
                * (1.0 - 3 * eps_h)
            )
            P_m_NW_h_dw3 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_h_dw3))

            k_cf_ad = (
                np.genfromtxt(
                    path_In + path_wm[2] + pref + str(zrange[i]) + "_NW.dat",
                    skip_header=3,
                    usecols=(0,),
                    unpack=True,
                )
                * h_fid
            )
            P_m_NW_wm_up3 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_om_up3))
            k_cf_ad = (
                np.genfromtxt(
                    path_In + path_wm[5] + pref + str(zrange[i]) + "_NW.dat",
                    skip_header=3,
                    usecols=(0,),
                    unpack=True,
                )
                * h_fid
            )
            P_m_NW_wm_dw3 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_om_dw3))

            k_cf_ad = (
                np.genfromtxt(
                    path_In + path_ns[2] + pref + str(zrange[i]) + "_NW.dat",
                    skip_header=3,
                    usecols=(0,),
                    unpack=True,
                )
                * h_fid
            )
            P_m_NW_ns_up3 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_ns_up3))
            k_cf_ad = (
                np.genfromtxt(
                    path_In + path_ns[5] + pref + str(zrange[i]) + "_NW.dat",
                    skip_header=3,
                    usecols=(0,),
                    unpack=True,
                )
                * h_fid
            )
            P_m_NW_ns_dw3 = CubicSpline(np.log10(k_cf_ad), np.log10(P_m_NW_ns_dw3))

    # Stock f(z) of the zth bin into a variable
    frate = growth_f[i]

    # Tegmarck formula for a (k,mu) cell (LU bloc)
    def aux_fun_LU(mu_ref, k_ref, j, l):

        return (
            1.0
            / (8 * np.pi**2)
            * k_ref**2
            * eval(der_input_LU[j])
            * eval(der_input_LU[l])
            * Veff(k_ref,mu_ref,zrange[i],10 ** P_m(np.log10(k_ref)),
                    10 ** P_m_NW(np.log10(k_ref)), b[i] * sig_8_fid,
                    growth_f[i] * sig_8_fid, n[i], V_a[i], H(zrange[i]),
                    H_ref(zrange[i]), D_A(zrange[i]), D_A_ref(zrange[i]),
                    sig_p_fid[i], sig_v_fid[i],
            )
        )

    def aux_fun_RU(mu_ref, k_ref, j, l):

        return (
            1.0
            / (8 * np.pi**2)
            * k_ref**2
            * eval(der_input_RU[j])
            * eval(der_input_RU[l])
            * Veff(
                k_ref,
                mu_ref,
                zrange[i],
                10 ** P_m(np.log10(k_ref)),
                10 ** P_m_NW(np.log10(k_ref)),
                b[i] * sig_8_fid,
                growth_f[i] * sig_8_fid,
                n[i],
                V_a[i],
                H(zrange[i]),
                H_ref(zrange[i]),
                D_A(zrange[i]),
                D_A_ref(zrange[i]),
                sig_p_fid[i],
                sig_v_fid[i],
            )
        )

    def aux_fun_RD(mu_ref, k_ref, j, l):

        return (
            1.0
            / (8 * np.pi**2)
            * k_ref**2
            * eval(der_input_RD[j])
            * eval(der_input_RD[l])
            * Veff(
                k_ref,
                mu_ref,
                zrange[i],
                10 ** P_m(np.log10(k_ref)),
                10 ** P_m_NW(np.log10(k_ref)),
                b[i] * sig_8_fid,
                growth_f[i] * sig_8_fid,
                n[i],
                V_a[i],
                H(zrange[i]),
                H_ref(zrange[i]),
                D_A(zrange[i]),
                D_A_ref(zrange[i]),
                sig_p_fid[i],
                sig_v_fid[i],
            )
        )

    # Tegmarck integral : in this method we sum other the right, lower and right-lower cell related to the one considered
    def integ(I1):
        # The Pobs(k,mu) is duplicated and rolled on the right, lower and right-lower directions to sum each elements
        function_A = aux_fun_LU(way, ecs, I1[0], I1[1]) * delta_x * delta_y
        function_A_10 = np.roll(function_A, -1, axis=1)
        function_A_01 = np.roll(function_A, -1, axis=0)
        function_A_11 = np.roll(function_A_01, -1, axis=1)

        function_A = function_A[0:-1, 0:-1]
        function_A_10 = function_A_10[0:-1, 0:-1]
        function_A_01 = function_A_01[0:-1, 0:-1]
        function_A_11 = function_A_11[0:-1, 0:-1]

        # Integral computation.
        integrale_A = (
            np.sum(function_A + function_A_10 + function_A_01 + function_A_11) / 4
        )

        # The integral is saved into the temporary files.
        file = open("tmp_F/LU_table_NL.txt", "a")
        file.write(
            str(I1[0] % N_notRD_params)
            + " "
            + str(I1[1] % N_notRD_params)
            + " "
            + str("%.12e" % integrale_A)
        )
        file.write(str("\n"))

        return A

    # Function that yields over two index (equivalent to double for loop) for parallel computing.
    def g():
        for j in range(N_notRD_params * i, N_notRD_params * i + N_notRD_params):
            for l in range(j, N_notRD_params * i + N_notRD_params):
                yield j, l

    # Pool map function for parallel computing.
    if __name__ == "__main__":
        pool = mp.Pool(12)
        pool.map(integ, g())
        pool.terminate()

    # Iteration bar progress.
    iterbar = iterbar + 21 * 9.0 / len(zrange)
    F_A = int(iterbar / 10)
    sys.stdout.write("\r")
    sys.stdout.write("[%-59s] %d%%" % (">" * F_A, iterbar * 100 / 594))
    sys.stdout.flush()

    # The next part concerns the two other blocks.
    def integ(I1):
        function_B = aux_fun_RU(way, ecs, I1[0], I1[1]) * delta_x * delta_y
        function_B_10 = np.roll(function_B, -1, axis=1)
        function_B_01 = np.roll(function_B, -1, axis=0)
        function_B_11 = np.roll(function_B_01, -1, axis=1)

        function_B = function_B[0:-1, 0:-1]
        function_B_10 = function_B_10[0:-1, 0:-1]
        function_B_01 = function_B_01[0:-1, 0:-1]
        function_B_11 = function_B_11[0:-1, 0:-1]

        integrale_B = (
            np.sum(function_B + function_B_10 + function_B_01 + function_B_11) / 4
        )

        file = open("tmp_F/RU_table_NL.txt", "a")
        file.write(
            str(I1[0] % (N_notRD_params + N_RD_params))
            + " "
            + str(I1[1])
            + " "
            + str("%.12e" % integrale_B)
        )
        file.write(str("\n"))

        return B

    def g():
        for j in range(
            (N_notRD_params + N_RD_params) * i,
            (N_notRD_params + N_RD_params) * i + N_notRD_params,
        ):
            for l in range(
                (N_notRD_params + N_RD_params) * i + N_notRD_params,
                (N_notRD_params + N_RD_params) * i + N_notRD_params + N_RD_params,
            ):
                yield j, l

    if __name__ == "__main__":
        pool = mp.Pool(20)
        pool.map(integ, g())
        pool.terminate()

    iterbar = iterbar + 30 * 9.0 / len(zrange)
    F_A = int(iterbar / 10)
    sys.stdout.write("\r")
    sys.stdout.write("[%-59s] %d%%" % (">" * F_A, iterbar * 100 / 594))
    sys.stdout.flush()

    def integ(I1):
        function_C = aux_fun_RD(way, ecs, I1[0], I1[1]) * delta_x * delta_y
        function_C_10 = np.roll(function_C, -1, axis=1)
        function_C_01 = np.roll(function_C, -1, axis=0)
        function_C_11 = np.roll(function_C_01, -1, axis=1)

        function_C = function_C[0:-1, 0:-1]
        function_C_10 = function_C_10[0:-1, 0:-1]
        function_C_01 = function_C_01[0:-1, 0:-1]
        function_C_11 = function_C_11[0:-1, 0:-1]

        integrale_C = (
            np.sum(function_C + function_C_10 + function_C_01 + function_C_11) / 4
        )

        file = open("tmp_F/RD_table_NL.txt", "a")
        file.write(str(I1[0]) + " " + str(I1[1]) + " " + str("%.12e" % integrale_C))
        file.write(str("\n"))

        return C

    def g():
        for j in range(N_RD_params * i, N_RD_params * i + N_RD_params):
            for l in range(j, N_RD_params * i + N_RD_params):
                yield j, l

    if __name__ == "__main__":
        pool = mp.Pool(15)
        pool.map(integ, g())
        pool.terminate()

    iterbar = iterbar + 15 * 9.0 / len(zrange)
    F_A = int(iterbar / 10)
    sys.stdout.write("\r")
    sys.stdout.write("[%-59s] %d%%" % (">" * F_A, iterbar * 100 / 594))
    sys.stdout.flush()

    i = i + 1

print("")
print("Integrals calculated")

# Fisher matrix build.
print("Begin Fisher build")

# Building the LU bloc.
ind1, ind2, FM = np.genfromtxt(
    "tmp_F/LU_table_NL.txt", delimiter=" ", usecols=(0, 1, 2), unpack=True
)

I1I2IF = np.column_stack((ind2, ind1, FM))
keys = (I1I2IF[:, 0], I1I2IF[:, 1])
indices = np.lexsort(keys)
FM = np.take(I1I2IF, indices, axis=0)

FM_tmp = np.zeros(len(FM))

i = 0
while i < len(FM_tmp):
    FM_tmp[i] = FM[i][2]
    i = i + 1

full_FM = np.zeros((N_notRD_params, N_notRD_params))  # z-independent parameters

i, j, l, o = 0, 0, 0, 0

while i < len(full_FM):
    while j < len(full_FM):
        while o < l + len(zrange):
            if i != j:
                full_FM[i][j] = full_FM[i][j] + FM_tmp[o]
                full_FM[j][i] = full_FM[i][j]
            else:
                full_FM[i][j] = full_FM[i][j] + FM_tmp[o]
            o = o + 1
        l = l + len(zrange)
        o = l
        j = j + 1
    i = i + 1
    j = i

# Building the RU bloc.
ind1, ind2, FM2 = np.genfromtxt(
    "tmp_F/RU_table_NL.txt", delimiter=" ", usecols=(0, 1, 2), unpack=True
)

I1I2IF = np.column_stack((ind2, ind1, FM2))
keys = (I1I2IF[:, 0], I1I2IF[:, 1])
indices = np.lexsort(keys)
FM2 = np.take(I1I2IF, indices, axis=0)

FIM = np.zeros((len(FM2)))

i = 0
while i < len(FIM):
    FIM[i] = FM2[i][2]
    i = i + 1

full_FM2 = np.reshape(FIM, (N_notRD_params, N_RD_params * len(zrange)))

# Building the RD bloc.
ind_a, ind_b, FM3 = np.genfromtxt(
    "tmp_F/RD_table_NL.txt", delimiter=" ", usecols=(0, 1, 2), unpack=True
)

full_FM3 = np.zeros((N_RD_params * len(zrange), N_RD_params * len(zrange)))

i, j, k = 0, 0, 0
while i < len(full_FM3):
    while j < len(full_FM3):
        while k < len(FM3):
            if (ind_a[k] == i and ind_b[k] == j) or (ind_a[k] == j and ind_b[k] == i):
                full_FM3[i][j] = FM3[k]
            k = k + 1
        k = 0
        j = j + 1
    j = 0
    i = i + 1

i, j, k = 0, 0, 0

# The 3 blocs are gathered together to form the final Fisher matrix.
full_DF_up = np.hstack((full_FM, full_FM2))
full_DF_dw = np.hstack((np.transpose(full_FM2), full_FM3))
full_F = np.vstack((full_DF_up, full_DF_dw))
full_F_save = np.copy(full_F)

# The peculiar velocities elements are suppressed for the linear case.
if choice_L_NL == "L":
    full_F_save = np.delete(full_F_save, [4, 5], 0)
    full_F_save = np.delete(full_F_save, [4, 5], 1)

# The Final fisher matrix is saved in an output file.
out_F = open("output/" + FD_output, "w")
while i < len(full_F_save):
    while j < len(full_F_save):
        out_F.write(str("%.12e" % full_F_save[i][j]))
        out_F.write(str(" "))
        j = j + 1
    out_F.write(str("\n"))
    j = 0
    i = i + 1
out_F.close()

# Remove temporar files.
os.system("rm tmp_F/*.txt")
print("Fisher matrix built")

# C. Projection to the new set of parameters.
print("Begin projection")

# Set the fiducial values of the new parameters.
Omega_b_new = omega_b_fid / (h_fid**2)
h_new = h_fid
Omega_m_new = omega_m_fid / (h_fid**2)
sig_p_new = sig_p_fid[0]
sig_v_new = sig_v_fid[0]
ns_new = ns_fid
fsig8 = np.zeros(len(zrange))
bsig8 = np.zeros(len(zrange))
Psho = np.zeros(len(zrange))

i = 0
while i < len(zrange):
    fsig8[i] = growth_f[i] * S_8[i]
    bsig8[i] = b[i] * S_8[i]
    Psho[i] = 0.0
    i = i + 1
i = 0

if choice_F_NF == "NF":
    Omega_L_new = 1 - Omega_m_new - Omega_r
else:
    Omega_L_new = 1 - Omega_m_new - Omega_r

"""        
Background quantities in order:
H(z) written with the big Omega : non flat cosmology, flat cosmology,
chi(z) : non flat cosmology, flat cosmology,
angular distance : non flat cosmology, flat cosmology.
"""


def H_new(z, n_num, ind):

    if ind == 0 or ind == 3:
        return 1

    elif ind == 1:
        return (
            100
            * n_num
            * np.sqrt(
                Omega_r * (1 + z) ** 4
                + Omega_m_new * (1 + z) ** 3
                + (1 - Omega_m_new - Omega_L_new - Omega_r) * (1 + z) ** 2
                + Omega_L_new * np.exp(3 * b_new(z, n_num, ind))
            )
        )

    elif ind == 2:
        return (
            100
            * h_new
            * np.sqrt(
                Omega_r * (1 + z) ** 4
                + n_num * (1 + z) ** 3
                + (1 - n_num - Omega_L_new - Omega_r) * (1 + z) ** 2
                + Omega_L_new * np.exp(3 * b_new(z, n_num, ind))
            )
        )

    elif ind == 4:
        return (
            100
            * h_new
            * np.sqrt(
                Omega_r * (1 + z) ** 4
                + Omega_m_new * (1 + z) ** 3
                + (1 - Omega_m_new - n_num - Omega_r) * (1 + z) ** 2
                + n_num * np.exp(3 * b_new(z, n_num, ind))
            )
        )

    elif ind == 5 or ind == 6:
        return (
            100
            * h_new
            * np.sqrt(
                Omega_r * (1 + z) ** 4
                + Omega_m_new * (1 + z) ** 3
                + (1 - Omega_m_new - Omega_L_new - Omega_r) * (1 + z) ** 2
                + Omega_L_new * np.exp(3 * b_new(z, n_num, ind))
            )
        )

    else:
        return (
            100
            * h_new
            * np.sqrt(
                Omega_r * (1 + z) ** 4
                + Omega_m_new * (1 + z) ** 3
                + (1 - Omega_m_new - Omega_L_new - Omega_r) * (1 + z) ** 2
                + Omega_L_new * np.exp(3 * b_new(z, n_num, ind))
            )
        )


def H_flat(z, n_num, ind):

    if ind == 1:
        return (
            100
            * n_num
            * np.sqrt(
                Omega_r * (1 + z) ** 4
                + Omega_m_new * (1 + z) ** 3
                + (1 - Omega_m_new - Omega_r) * np.exp(3 * b_new(z, n_num, ind))
            )
        )

    elif ind == 2:
        return (
            100
            * h_new
            * np.sqrt(
                Omega_r * (1 + z) ** 4
                + n_num * (1 + z) ** 3
                + (1 - n_num - Omega_r) * np.exp(3 * b_new(z, n_num, ind))
            )
        )

    elif ind == 5 or ind == 6:
        return (
            100
            * h_new
            * np.sqrt(
                Omega_r * (1 + z) ** 4
                + Omega_m_new * (1 + z) ** 3
                + (1 - Omega_m_new - Omega_r) * np.exp(3 * b_new(z, n_num, ind))
            )
        )

    else:
        return (
            100
            * h_new
            * np.sqrt(
                Omega_r * (1 + z) ** 4
                + Omega_m_new * (1 + z) ** 3
                + (1 - Omega_m_new - Omega_r) * np.exp(3 * b_new(z, n_num, ind))
            )
        )


def chi_aux_new(z, n_num, ind):
    if ind == 1:
        return 100 * n_num / H_new(z, n_num, ind)
    else:
        return 100 * h_new / H_new(z, n_num, ind)


def chi_aux_new_flat(z, n_num, ind):
    if ind == 1:
        return 100 * n_num / H_flat(z, n_num, ind)
    else:
        return 100 * h_new / H_flat(z, n_num, ind)


def chi_new(z, n_num, ind):
    return pyint.quad(chi_aux_new, 0, z, args=(n_num, ind))[0]


def chi_new_flat(z, n_num, ind):
    return pyint.quad(chi_aux_new_flat, 0, z, args=(n_num, ind))[0]


def Da_new(z, n_num, ind):

    if ind == 1:
        if 1 - Omega_m_new - Omega_L_new - Omega_r > 0:
            return (
                c
                / (
                    100
                    * n_num
                    * (1 + z)
                    * np.sqrt(1 - Omega_m_new - Omega_L_new - Omega_r)
                )
                * np.sinh(
                    np.sqrt(1 - Omega_m_new - Omega_L_new - Omega_r)
                    * chi_new(z, n_num, ind)
                )
            )
        elif 1 - Omega_m_new - Omega_L_new - Omega_r < 0:
            return (
                c
                / (
                    100
                    * n_num
                    * (1 + z)
                    * np.sqrt(np.fabs(1 - Omega_m_new - Omega_L_new - Omega_r))
                )
                * np.sin(
                    np.sqrt(np.fabs(1 - Omega_m_new - Omega_L_new - Omega_r))
                    * chi_new(z, n_num, ind)
                )
            )
        else:
            return c / (100 * n_num * (1 + z)) * chi_new(z, n_num, ind)

    elif ind == 2:
        if 1 - n_num - Omega_L_new - Omega_r > 0:
            return (
                c
                / (100 * h_new * (1 + z) * np.sqrt(1 - n_num - Omega_L_new - Omega_r))
                * np.sinh(
                    np.sqrt(1 - n_num - Omega_L_new - Omega_r) * chi_new(z, n_num, ind)
                )
            )
        elif 1 - n_num - Omega_L_new - Omega_r < 0:
            return (
                c
                / (
                    100
                    * h_new
                    * (1 + z)
                    * np.sqrt(np.fabs(1 - n_num - Omega_L_new - Omega_r))
                )
                * np.sin(
                    np.sqrt(np.fabs(1 - n_num - Omega_L_new - Omega_r))
                    * chi_new(z, n_num, ind)
                )
            )
        else:
            return c / (100 * h_new * (1 + z)) * chi_new(z, n_num, ind)

    elif ind == 4:
        if 1 - Omega_m_new - n_num - Omega_r > 0:
            return (
                c
                / (100 * h_new * (1 + z) * np.sqrt(1 - Omega_m_new - n_num - Omega_r))
                * np.sinh(
                    np.sqrt(1 - Omega_m_new - n_num - Omega_r) * chi_new(z, n_num, ind)
                )
            )
        elif 1 - Omega_m_new - n_num - Omega_r < 0:
            return (
                c
                / (
                    100
                    * h_new
                    * (1 + z)
                    * np.sqrt(np.fabs(1 - Omega_m_new - n_num - Omega_r))
                )
                * np.sin(
                    np.sqrt(np.fabs(1 - Omega_m_new - n_num - Omega_r))
                    * chi_new(z, n_num, ind)
                )
            )
        else:
            return c / (100 * h_new * (1 + z)) * chi_new(z, n_num, ind)

    elif ind == 5 or ind == 6:
        if 1 - Omega_m_new - Omega_L_new - Omega_r > 0:
            return (
                c
                / (
                    100
                    * h_new
                    * (1 + z)
                    * np.sqrt(1 - Omega_m_new - Omega_L_new - Omega_r)
                )
                * np.sinh(
                    np.sqrt(1 - Omega_m_new - Omega_L_new - Omega_r)
                    * chi_new(z, n_num, ind)
                )
            )
        elif 1 - Omega_m_new - Omega_L_new - Omega_r < 0:
            return (
                c
                / (
                    100
                    * h_new
                    * (1 + z)
                    * np.sqrt(np.fabs(1 - Omega_m_new - Omega_L_new - Omega_r))
                )
                * np.sin(
                    np.sqrt(np.fabs(1 - Omega_m_new - Omega_L_new - Omega_r))
                    * chi_new(z, n_num, ind)
                )
            )
        else:
            return c / (100 * h_new * (1 + z)) * chi_new(z, n_num, ind)

    else:
        return c / (100 * h_new * (1 + z)) * chi_new(z, n_num, ind)


def Da_flat(z, n_num, ind):

    if ind == 1:
        return c / (100 * n_num * (1 + z)) * chi_new_flat(z, n_num, ind)

    elif ind == 2:
        return c / (100 * h_new * (1 + z)) * chi_new_flat(z, n_num, ind)

    elif ind == 5 or ind == 6:
        return c / (100 * h_new * (1 + z)) * chi_new_flat(z, n_num, ind)

    else:
        return c / (100 * h_new * (1 + z)) * chi_new_flat(z, n_num, ind)


def b_new(z, n_num, ind):
    if choice_F_NF == "NF":
        return pyint.quad(b_aux, 0, z, args=(n_num, ind))[0]
    if choice_F_NF == "F":
        return pyint.quad(b_aux_flat, 0, z, args=(n_num, ind))[0]


def b_aux(z, n_num, ind):
    if ind == 5:
        return 1.0 / (1 + z) * (1 + n_num + wa * z / (1 + z))
    elif ind == 6:
        return 1.0 / (1 + z) * (1 + w0 + n_num * z / (1 + z))
    else:
        return 1.0 / (1 + z) * (1 + w0 + wa * z / (1 + z))


def b_aux_flat(z, n_num, ind):
    if ind == 5:
        return 1.0 / (1 + z) * (1 + n_num + wa * z / (1 + z))
    elif ind == 6:
        return 1.0 / (1 + z) * (1 + w0 + n_num * z / (1 + z))
    else:
        return 1.0 / (1 + z) * (1 + w0 + wa * z / (1 + z))


# ODE for fs8(z) with the new parameters : non flat and flat cosmology.
def eq_diff(x, n_num, ind):
    y = np.zeros(len(x))
    dy = np.zeros(len(y))
    dx = x[len(x) - 2] - x[len(x) - 1]
    y[len(x) - 1] = 1

    if choice_F_NF == "NF":
        if ind == 1:
            dy[len(x) - 1] = (
                y[len(x) - 1] ** 2 / (1 + x[len(x) - 1])
                + y[len(x) - 1]
                * (
                    2.0 / (1 + x[len(x) - 1])
                    - (
                        H_new(x[len(x) - 1] + 1e-6, n_num, ind)
                        - H_new(x[len(x) - 1] - 1e-6, n_num, ind)
                    )
                    / (2 * 1e-6 * H_new(x[len(x) - 1], n_num, ind))
                )
                - 1.5
                * Omega_m_new
                * (1 + x[len(x) - 1]) ** 2
                * (100 * n_num) ** 2
                / (H_new(x[len(x) - 1], n_num, ind) ** 2)
            )
        elif ind == 2:
            dy[len(x) - 1] = (
                y[len(x) - 1] ** 2 / (1 + x[len(x) - 1])
                + y[len(x) - 1]
                * (
                    2.0 / (1 + x[len(x) - 1])
                    - (
                        H_new(x[len(x) - 1] + 1e-6, n_num, ind)
                        - H_new(x[len(x) - 1] - 1e-6, n_num, ind)
                    )
                    / (2 * 1e-6 * H_new(x[len(x) - 1], n_num, ind))
                )
                - 1.5
                * n_num
                * (1 + x[len(x) - 1]) ** 2
                * (100 * h_new) ** 2
                / (H_new(x[len(x) - 1], n_num, ind) ** 2)
            )
        else:
            dy[len(x) - 1] = (
                y[len(x) - 1] ** 2 / (1 + x[len(x) - 1])
                + y[len(x) - 1]
                * (
                    2.0 / (1 + x[len(x) - 1])
                    - (
                        H_new(x[len(x) - 1] + 1e-6, n_num, ind)
                        - H_new(x[len(x) - 1] - 1e-6, n_num, ind)
                    )
                    / (2 * 1e-6 * H_new(x[len(x) - 1], n_num, ind))
                )
                - 1.5
                * Omega_m_new
                * (1 + x[len(x) - 1]) ** 2
                * (100 * h_new) ** 2
                / (H_new(x[len(x) - 1], n_num, ind) ** 2)
            )
        i = len(x) - 2
        while i >= 0:
            y[i] = y[i + 1] + dx * dy[i + 1]
            if ind == 1:
                dy[i] = (
                    y[i] ** 2 / (1 + x[i])
                    + y[i]
                    * (
                        2.0 / (1 + x[i])
                        - (
                            H_new(x[i] + 1e-6, n_num, ind)
                            - H_new(x[i] - 1e-6, n_num, ind)
                        )
                        / (2 * 1e-6 * H_new(x[i], n_num, ind))
                    )
                    - 1.5
                    * Omega_m_new
                    * (1 + x[i]) ** 2
                    * (100 * n_num) ** 2
                    / (H_new(x[i], n_num, ind) ** 2)
                )
            elif ind == 2:
                dy[i] = (
                    y[i] ** 2 / (1 + x[i])
                    + y[i]
                    * (
                        2.0 / (1 + x[i])
                        - (
                            H_new(x[i] + 1e-6, n_num, ind)
                            - H_new(x[i] - 1e-6, n_num, ind)
                        )
                        / (2 * 1e-6 * H_new(x[i], n_num, ind))
                    )
                    - 1.5
                    * n_num
                    * (1 + x[i]) ** 2
                    * (100 * h_new) ** 2
                    / (H_new(x[i], n_num, ind) ** 2)
                )
            else:
                dy[i] = (
                    y[i] ** 2 / (1 + x[i])
                    + y[i]
                    * (
                        2.0 / (1 + x[i])
                        - (
                            H_new(x[i] + 1e-6, n_num, ind)
                            - H_new(x[i] - 1e-6, n_num, ind)
                        )
                        / (2 * 1e-6 * H_new(x[i], n_num, ind))
                    )
                    - 1.5
                    * Omega_m_new
                    * (1 + x[i]) ** 2
                    * (100 * h_new) ** 2
                    / (H_new(x[i], n_num, ind) ** 2)
                )
            i = i - 1

    else:
        if ind == 1:
            dy[len(x) - 1] = (
                y[len(x) - 1] ** 2 / (1 + x[len(x) - 1])
                + y[len(x) - 1]
                * (
                    2.0 / (1 + x[len(x) - 1])
                    - (
                        H_flat(x[len(x) - 1] + 1e-6, n_num, ind)
                        - H_flat(x[len(x) - 1] - 1e-6, n_num, ind)
                    )
                    / (2 * 1e-6 * H_flat(x[len(x) - 1], n_num, ind))
                )
                - 1.5
                * Omega_m_new
                * (1 + x[len(x) - 1]) ** 2
                * (100 * n_num) ** 2
                / (H_flat(x[len(x) - 1], n_num, ind) ** 2)
            )
        elif ind == 2:
            dy[len(x) - 1] = (
                y[len(x) - 1] ** 2 / (1 + x[len(x) - 1])
                + y[len(x) - 1]
                * (
                    2.0 / (1 + x[len(x) - 1])
                    - (
                        H_flat(x[len(x) - 1] + 1e-6, n_num, ind)
                        - H_flat(x[len(x) - 1] - 1e-6, n_num, ind)
                    )
                    / (2 * 1e-6 * H_flat(x[len(x) - 1], n_num, ind))
                )
                - 1.5
                * n_num
                * (1 + x[len(x) - 1]) ** 2
                * (100 * h_new) ** 2
                / (H_flat(x[len(x) - 1], n_num, ind) ** 2)
            )
        else:
            dy[len(x) - 1] = (
                y[len(x) - 1] ** 2 / (1 + x[len(x) - 1])
                + y[len(x) - 1]
                * (
                    2.0 / (1 + x[len(x) - 1])
                    - (
                        H_flat(x[len(x) - 1] + 1e-6, n_num, ind)
                        - H_flat(x[len(x) - 1] - 1e-6, n_num, ind)
                    )
                    / (2 * 1e-6 * H_flat(x[len(x) - 1], n_num, ind))
                )
                - 1.5
                * Omega_m_new
                * (1 + x[len(x) - 1]) ** 2
                * (100 * h_new) ** 2
                / (H_flat(x[len(x) - 1], n_num, ind) ** 2)
            )
        i = len(x) - 2
        while i >= 0:
            y[i] = y[i + 1] + dx * dy[i + 1]
            if ind == 1:
                dy[i] = (
                    y[i] ** 2 / (1 + x[i])
                    + y[i]
                    * (
                        2.0 / (1 + x[i])
                        - (
                            H_flat(x[i] + 1e-6, n_num, ind)
                            - H_flat(x[i] - 1e-6, n_num, ind)
                        )
                        / (2 * 1e-6 * H_flat(x[i], n_num, ind))
                    )
                    - 1.5
                    * Omega_m_new
                    * (1 + x[i]) ** 2
                    * (100 * n_num) ** 2
                    / (H_flat(x[i], n_num, ind) ** 2)
                )
            elif ind == 2:
                dy[i] = (
                    y[i] ** 2 / (1 + x[i])
                    + y[i]
                    * (
                        2.0 / (1 + x[i])
                        - (
                            H_flat(x[i] + 1e-6, n_num, ind)
                            - H_flat(x[i] - 1e-6, n_num, ind)
                        )
                        / (2 * 1e-6 * H_flat(x[i], n_num, ind))
                    )
                    - 1.5
                    * n_num
                    * (1 + x[i]) ** 2
                    * (100 * h_new) ** 2
                    / (H_flat(x[i], n_num, ind) ** 2)
                )
            else:
                dy[i] = (
                    y[i] ** 2 / (1 + x[i])
                    + y[i]
                    * (
                        2.0 / (1 + x[i])
                        - (
                            H_flat(x[i] + 1e-6, n_num, ind)
                            - H_flat(x[i] - 1e-6, n_num, ind)
                        )
                        / (2 * 1e-6 * H_flat(x[i], n_num, ind))
                    )
                    - 1.5
                    * Omega_m_new
                    * (1 + x[i]) ** 2
                    * (100 * h_new) ** 2
                    / (H_flat(x[i], n_num, ind) ** 2)
                )
            i = i - 1

    return y


# Parametric equation to get fs8(z).
def param_eq(x, n_num, ind):
    y = np.zeros(len(x))
    i = 0
    while i < len(y):
        if choice_F_NF == "NF":
            if ind == 1:
                y[i] = (
                    Omega_m_new
                    * (1 + x[i]) ** 3
                    / ((H_new(x[i], n_num, ind) / 100 / h_new) ** 2)
                ) ** gamma_new
            elif ind == 2:
                y[i] = (
                    n_num
                    * (1 + x[i]) ** 3
                    / ((H_new(x[i], n_num, ind) / 100 / h_new) ** 2)
                ) ** gamma_new
            elif ind == 8:
                y[i] = (
                    Omega_m_new
                    * (1 + x[i]) ** 3
                    / ((H_new(x[i], n_num, ind) / 100 / h_new) ** 2)
                ) ** n_num
            else:
                y[i] = (
                    Omega_m_new
                    * (1 + x[i]) ** 3
                    / ((H_new(x[i], n_num, ind) / 100 / h_new) ** 2)
                ) ** gamma_new
        else:
            if ind == 1:
                y[i] = (
                    Omega_m_new
                    * (1 + x[i]) ** 3
                    / ((H_flat(x[i], n_num, ind) / 100 / h_new) ** 2)
                ) ** gamma_new
            elif ind == 2:
                y[i] = (
                    n_num
                    * (1 + x[i]) ** 3
                    / ((H_flat(x[i], n_num, ind) / 100 / h_new) ** 2)
                ) ** gamma_new
            elif ind == 8:
                y[i] = (
                    Omega_m_new
                    * (1 + x[i]) ** 3
                    / ((H_flat(x[i], n_num, ind) / 100 / h_new) ** 2)
                ) ** n_num
            else:
                y[i] = (
                    Omega_m_new
                    * (1 + x[i]) ** 3
                    / ((H_flat(x[i], n_num, ind) / 100 / h_new) ** 2)
                ) ** gamma_new
        i = i + 1
    return y


def G_aux(x, fctn):
    return (fctn - 1) / (1 + x)


def f_sig8(x, n_num, ind):
    if proj_ODE_fit == "ODE":
        soln = eq_diff(x, n_num, ind)
    if proj_ODE_fit == "fit":
        soln = param_eq(x, n_num, ind)

    delt = x[1] - x[0]

    somme = np.zeros(len(x))
    ind_up = np.where(x >= 1.0)
    somme[ind_up[0][0]] = 1
    i = ind_up[0][0] + 1
    while i < len(x):
        somme[i] = (
            somme[i - 1]
            - delt * (G_aux(x[i], soln[i]) + G_aux(x[i - 1], soln[i - 1])) / 2
        )
        i = i + 1

    ind_dw = np.where(x < 1.0)
    i = ind_up[0][0] - 1
    while i >= 0:
        somme[i] = (
            somme[i + 1]
            + delt * (G_aux(x[i], soln[i]) + G_aux(x[i + 1], soln[i + 1])) / 2
        )
        i = i - 1

    G_factor = np.exp(somme) / np.exp(somme[0])

    S8 = sig_8_new * G_factor / (1 + x)
    FS8 = np.zeros(len(x))

    i = 0
    while i < len(x):
        FS8[i] = S8[i] * soln[i]
        i = i + 1

    if ind == 7:
        return S8
    else:
        return FS8


# fs8 computing while all the new parameters are moved one by one.
print("fs8 computing")
if choice_F_NF == "F":
    fs8_Vname = np.array(
        [
            "f_s8_h_p",
            "f_s8_h_m",
            "f_s8_OM_p",
            "f_s8_OM_m",
            "f_s8_w0_p",
            "f_s8_w0_m",
            "f_s8_wa_p",
            "f_s8_wa_m",
            "s8_der",
        ]
    )
    fs8_Functions = np.array(
        [
            "f_sig8(tn, h_new+eps_proj, 1)",
            "f_sig8(tn, h_new-eps_proj, 1)",
            "f_sig8(tn, Omega_m_new+eps_proj, 2)",
            "f_sig8(tn, Omega_m_new-eps_proj, 2)",
            "f_sig8(tn, w0+eps_proj, 5)",
            "f_sig8(tn, w0-eps_proj, 5)",
            "f_sig8(tn, wa+eps_proj, 6)",
            "f_sig8(tn, wa-eps_proj, 6)",
            "f_sig8(tn, 7, 7)",
        ]
    )
if choice_F_NF == "NF":
    fs8_Vname = np.array(
        [
            "f_s8_h_p",
            "f_s8_h_m",
            "f_s8_OM_p",
            "f_s8_OM_m",
            "f_s8_w0_p",
            "f_s8_w0_m",
            "f_s8_wa_p",
            "f_s8_wa_m",
            "s8_der",
            "f_s8_OL_p",
            "f_s8_OL_m",
        ]
    )
    fs8_Functions = np.array(
        [
            "f_sig8(tn, h_new+eps_proj, 1)",
            "f_sig8(tn, h_new-eps_proj, 1)",
            "f_sig8(tn, Omega_m_new+eps_proj, 2)",
            "f_sig8(tn, Omega_m_new-eps_proj, 2)",
            "f_sig8(tn, w0+eps_proj, 5)",
            "f_sig8(tn, w0-eps_proj, 5)",
            "f_sig8(tn, wa+eps_proj, 6)",
            "f_sig8(tn, wa-eps_proj, 6)",
            "f_sig8(tn, 7, 7)",
            "f_sig8(tn, Omega_L_new+eps_proj, 4)",
            "f_sig8(tn, Omega_L_new-eps_proj, 4)",
        ]
    )


def fs8_call(I1, sed=None):
    A_fs8 = eval(fs8_Functions[I1])
    out_F = open("tmp_F/fs8_tmp_values_" + str(fs8_Vname[I1]), "w")
    i = 0
    while i < len(A_fs8):
        out_F.write(str("%.12e" % A_fs8[i]))
        out_F.write(str(" "))
        i = i + 1
    out_F.write(str("\n"))
    out_F.close()


if __name__ == "__main__":
    pool = mp.Pool(len(fs8_Functions))
    pool.map(fs8_call, range(len(fs8_Functions)))
    pool.terminate()

i = 0
while i < len(fs8_Vname):
    globals()[fs8_Vname[i]] = np.loadtxt(
        "tmp_F/fs8_tmp_values_" + str(fs8_Vname[i]), unpack=False
    )
    i = i + 1
i = 0
os.system("rm tmp_F/*")

# fs8 interpolation while all the new parameters are moved one by one.
f_s8_h_p_int = CubicSpline(tn, f_s8_h_p)
f_s8_h_m_int = CubicSpline(tn, f_s8_h_m)
f_s8_OM_p_int = CubicSpline(tn, f_s8_OM_p)
f_s8_OM_m_int = CubicSpline(tn, f_s8_OM_m)
f_s8_w0_p_int = CubicSpline(tn, f_s8_w0_p)
f_s8_w0_m_int = CubicSpline(tn, f_s8_w0_m)
f_s8_wa_p_int = CubicSpline(tn, f_s8_wa_p)
f_s8_wa_m_int = CubicSpline(tn, f_s8_wa_m)
s8_der_int = CubicSpline(tn, s8_der)

# fs8 computing and interpolation for Omega_DE : non flat case.
if choice_F_NF == "NF":
    f_s8_OL_p_int = CubicSpline(tn, f_s8_OL_p)
    f_s8_OL_m_int = CubicSpline(tn, f_s8_OL_m)

# fs8 computing and interpolation for gamma : fit is required.
if proj_ODE_fit == "fit":
    f_s8_gam_p = f_sig8(tn, gamma_new + eps_proj, 8)
    f_s8_gam_m = f_sig8(tn, gamma_new - eps_proj, 8)
    f_s8_gam_p_int = CubicSpline(tn, f_s8_gam_p)
    f_s8_gam_m_int = CubicSpline(tn, f_s8_gam_m)


# Jacobian functions : in order H(z), Da(z) and fs8(z) --> 3pts derivatives.
def J_lnH_3(z, num, step, ind):
    return (np.log(H_new(z, num + step, ind)) - np.log(H_new(z, num - step, ind))) / (
        2 * step
    )


def J_lnDa_3(z, num, step, ind):
    return (np.log(Da_new(z, num + step, ind)) - np.log(Da_new(z, num - step, ind))) / (
        2 * step
    )


if (proj_ODE_fit == "fit") and (include_gamma == "Y"):

    def J_lnfS8_3(z, ind, step):
        if ind == 1:
            return 0.0
        elif ind == 2:
            return (np.log(f_s8_OM_p_int(z)) - np.log(f_s8_OM_m_int(z))) / (2 * step)
        elif ind == 4:
            return (np.log(f_s8_OL_p_int(z)) - np.log(f_s8_OL_m_int(z))) / (2 * step)
        elif ind == 5:
            return (np.log(f_s8_w0_p_int(z)) - np.log(f_s8_w0_m_int(z))) / (2 * step)
        elif ind == 6:
            return (np.log(f_s8_wa_p_int(z)) - np.log(f_s8_wa_m_int(z))) / (2 * step)
        elif ind == 7:
            return 1.0 / s8_der_int(0)
        elif ind == 8:
            return (np.log(f_s8_gam_p_int(z)) - np.log(f_s8_gam_m_int(z))) / (2 * step)
        else:
            return 0.0

else:

    def J_lnfS8_3(z, ind, step):
        if ind == 1:
            return 0.0
        elif ind == 2:
            return (np.log(f_s8_OM_p_int(z)) - np.log(f_s8_OM_m_int(z))) / (2 * step)
        elif ind == 4:
            return (np.log(f_s8_OL_p_int(z)) - np.log(f_s8_OL_m_int(z))) / (2 * step)
        elif ind == 5:
            return (np.log(f_s8_w0_p_int(z)) - np.log(f_s8_w0_m_int(z))) / (2 * step)
        elif ind == 6:
            return (np.log(f_s8_wa_p_int(z)) - np.log(f_s8_wa_m_int(z))) / (2 * step)
        elif ind == 7:
            return 1.0 / s8_der_int(0)
        else:
            return 0.0


def J_lnH_3_flat(z, num, step, ind):
    return (np.log(H_flat(z, num + step, ind)) - np.log(H_flat(z, num - step, ind))) / (
        2 * step
    )


def J_lnDa_3_flat(z, num, step, ind):
    return (
        np.log(Da_flat(z, num + step, ind)) - np.log(Da_flat(z, num - step, ind))
    ) / (2 * step)


if proj_ODE_fit == "fit" and include_gamma == "Y":

    def J_lnfS8_3_flat(z, ind, step):
        if ind == 1:
            return 0.0
        elif ind == 2:
            return (np.log(f_s8_OM_p_int(z)) - np.log(f_s8_OM_m_int(z))) / (2 * step)
        elif ind == 5:
            return (np.log(f_s8_w0_p_int(z)) - np.log(f_s8_w0_m_int(z))) / (2 * step)
        elif ind == 6:
            return (np.log(f_s8_wa_p_int(z)) - np.log(f_s8_wa_m_int(z))) / (2 * step)
        elif ind == 7:
            return 1.0 / s8_der_int(0)
        elif ind == 8:
            return (np.log(f_s8_gam_p_int(z)) - np.log(f_s8_gam_m_int(z))) / (2 * step)
        else:
            return 0.0

else:

    def J_lnfS8_3_flat(z, ind, step):
        if ind == 1:
            return 0.0
        elif ind == 2:
            return (np.log(f_s8_OM_p_int(z)) - np.log(f_s8_OM_m_int(z))) / (2 * step)
        elif ind == 5:
            return (np.log(f_s8_w0_p_int(z)) - np.log(f_s8_w0_m_int(z))) / (2 * step)
        elif ind == 6:
            return (np.log(f_s8_wa_p_int(z)) - np.log(f_s8_wa_m_int(z))) / (2 * step)
        elif ind == 7:
            return 1.0 / s8_der_int(0)
        else:
            return 0.0


print("Jacobian computation")
# Jacobian computation for the shape parameters and the peculiar velocities. All derivatives are analytical.
if choice_F_NF == "NF":
    if proj_ODE_fit == "ODE" or include_gamma == "N":
        therms = np.array(
            [
                "Omega_b_new",
                "h_new",
                "Omega_m_new",
                "ns_new",
                "Omega_L_new",
                "w0",
                "wa",
                "sig_8_new",
                "sig_p_new",
                "sig_v_new",
            ]
        )
        j, l, spsv_index = len(therms), len(therms), len(therms) - 1
        while i < len(zrange):
            therms = np.insert(therms, len(therms), "bsig8[" + str(i) + "]")
            therms = np.insert(therms, len(therms), "Psho[" + str(i) + "]")
            i = i + 1

        # Jacobian creation
        J_wb = [h_new**2, 2 * h_new * Omega_b_new, 0, 0, 0, 0, 0, 0, 0, 0]
        J_h  = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        J_wm = [0, 2 * h_new * Omega_m_new, h_new**2, 0, 0, 0, 0, 0, 0, 0]
        J_ns = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        J_sp = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        J_sv = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        while j < len(therms):
            J_wb = np.insert(J_wb, len(J_wb), 0)
            J_h  = np.insert(J_h, len(J_h), 0)
            J_wm = np.insert(J_wm, len(J_wm), 0)
            J_ns = np.insert(J_ns, len(J_ns), 0)
            J_sp = np.insert(J_sp, len(J_sp), 0)
            J_sv = np.insert(J_sv, len(J_sv), 0)
            j = j + 1

    if (proj_ODE_fit == "fit") and (include_gamma == "Y"):
        therms = np.array(
            [
                "Omega_b_new",
                "h_new",
                "Omega_m_new",
                "ns_new",
                "Omega_L_new",
                "w0",
                "wa",
                "sig_8_new",
                "gamma_new",
                "sig_p_new",
                "sig_v_new",
            ]
        )
        j, l, spsv_index = len(therms), len(therms), len(therms) - 1
        while i < len(zrange):
            therms = np.insert(therms, len(therms), "bsig8[" + str(i) + "]")
            therms = np.insert(therms, len(therms), "Psho[" + str(i) + "]")
            i = i + 1

        J_wb = [h_new**2, 2 * h_new * Omega_b_new, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        J_h  = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        J_wm = [0, 2 * h_new * Omega_m_new, h_new**2, 0, 0, 0, 0, 0, 0, 0, 0]
        J_ns = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        J_sp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        J_sv = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        while j < len(therms):
            J_wb = np.insert(J_wb, len(J_wb), 0)
            J_h  = np.insert(J_h, len(J_h), 0)
            J_wm = np.insert(J_wm, len(J_wm), 0)
            J_ns = np.insert(J_ns, len(J_ns), 0)
            J_sp = np.insert(J_sp, len(J_sp), 0)
            J_sv = np.insert(J_sv, len(J_sv), 0)
            j = j + 1

# flat geometry case
else:
    if proj_ODE_fit == "ODE" or include_gamma == "N":
        therms = np.array(
            [
                "Omega_b_new",
                "h_new",
                "Omega_m_new",
                "ns_new",
                "w0",
                "wa",
                "sig_8_new",
                "sig_p_new",
                "sig_v_new",
            ]
        )
        j, l, spsv_index = len(therms), len(therms), len(therms) - 1
        while i < len(zrange):
            therms = np.insert(therms, len(therms), "bsig8[" + str(i) + "]")
            therms = np.insert(therms, len(therms), "Psho[" + str(i) + "]")
            i = i + 1
        # Jacobian creation
        J_wb = [h_new**2, 2 * h_new * Omega_b_new, 0, 0, 0, 0, 0, 0, 0]
        J_h  = [0, 1, 0, 0, 0, 0, 0, 0, 0]
        J_wm = [0, 2 * h_new * Omega_m_new, h_new**2, 0, 0, 0, 0, 0, 0]
        J_ns = [0, 0, 0, 1, 0, 0, 0, 0, 0]
        J_sp = [0, 0, 0, 0, 0, 0, 0, 1, 0]
        J_sv = [0, 0, 0, 0, 0, 0, 0, 0, 1]
        while j < len(therms):
            J_wb = np.insert(J_wb, len(J_wb), 0)
            J_h  = np.insert(J_h, len(J_h), 0)
            J_wm = np.insert(J_wm, len(J_wm), 0)
            J_ns = np.insert(J_ns, len(J_ns), 0)
            J_sp = np.insert(J_sp, len(J_sp), 0)
            J_sv = np.insert(J_sv, len(J_sv), 0)
            j = j + 1

    if (proj_ODE_fit == "fit") and (include_gamma == "Y"):
        therms = np.array(
            [
                "Omega_b_new",
                "h_new",
                "Omega_m_new",
                "ns_new",
                "w0",
                "wa",
                "sig_8_new",
                "gamma_new",
                "sig_p_new",
                "sig_v_new",
            ]
        )
        j, l, spsv_index = len(therms), len(therms), len(therms) - 1
        while i < len(zrange):
            therms = np.insert(therms, len(therms), "bsig8[" + str(i) + "]")
            therms = np.insert(therms, len(therms), "Psho[" + str(i) + "]")
            i = i + 1

        J_wb = [h_new**2, 2 * h_new * Omega_b_new, 0, 0, 0, 0, 0, 0, 0, 0]
        J_h  = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        J_wm = [0, 2 * h_new * Omega_m_new, h_new**2, 0, 0, 0, 0, 0, 0, 0]
        J_ns = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        J_sp = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        J_sv = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        while j < len(therms):
            J_wb = np.insert(J_wb, len(J_wb), 0)
            J_h  = np.insert(J_h, len(J_h), 0)
            J_wm = np.insert(J_wm, len(J_wm), 0)
            J_ns = np.insert(J_ns, len(J_ns), 0)
            J_sp = np.insert(J_sp, len(J_sp), 0)
            J_sv = np.insert(J_sv, len(J_sv), 0)
            j = j + 1

# Jacobian computation for the the bias and the shot noise. All derivatives are also analytical.
if bias_choice == "Y":
    i = 0
    while i < len(zrange):
        globals()["J_lnbS8_3_%d" % i] = np.zeros(len(therms))
        globals()["J_lnbS8_3_%d" % i][l + (2 * i)] = 1.0 / b[i]
        i = i + 1
else:
    i = 0
    while i < len(zrange):
        globals()["J_lnbS8_3_%d" % i] = np.zeros(len(therms))
        globals()["J_lnbS8_3_%d" % i][l + (2 * i)] = 1.0
        i = i + 1

i = 0
while i < len(zrange):
    globals()["J_lnPs_3_%d" % i] = np.zeros(len(therms))
    globals()["J_lnPs_3_%d" % i][l + (2 * i + 1)] = 1.0
    i = i + 1


# Jacobian computation for the redshift dependent parameters. Most of the derivatives are numerical.
J_Da_3 = np.zeros((len(therms), len(zrange)))
J_H_3 = np.zeros((len(therms), len(zrange)))
J_fs8_3 = np.zeros((len(therms), len(zrange)))

i, j = 0, 0

while i < len(therms):
    while j < len(zrange):
        if choice_F_NF == "NF":
            J_Da_3[i][j] = J_lnDa_3(zrange[j], eval(therms[i]), eps_proj, i)
            J_H_3[i][j] = J_lnH_3(zrange[j], eval(therms[i]), eps_proj, i)
            J_fs8_3[i][j] = J_lnfS8_3(zrange[j], i, eps_proj)
        else:
            if i < 4:
                J_Da_3[i][j] = J_lnDa_3_flat(zrange[j], eval(therms[i]), eps_proj, i)
                J_H_3[i][j] = J_lnH_3_flat(zrange[j], eval(therms[i]), eps_proj, i)
                J_fs8_3[i][j] = J_lnfS8_3_flat(zrange[j], i, eps_proj)
            else:
                J_Da_3[i][j] = J_lnDa_3_flat(
                    zrange[j], eval(therms[i]), eps_proj, i + 1
                )
                J_H_3[i][j] = J_lnH_3_flat(zrange[j], eval(therms[i]), eps_proj, i + 1)
                J_fs8_3[i][j] = J_lnfS8_3_flat(zrange[j], i + 1, eps_proj)
        j = j + 1
    j = 0
    i = i + 1


print("Projection into the new set of parameters")
# Load big Fisher matrix.
F_previous = np.copy(full_F)

i, j = 0, 0
# Creation of the projected Fisher matrix, and jacobian allocation.
F_new = np.zeros((len(therms), len(therms)))
Jacobian_m = np.zeros((len(F_previous), len(therms)))
i, j, l = 0, 0, 0
while i < len(Jacobian_m[0]):
    while j < len(Jacobian_m):
        if j == 0:
            Jacobian_m[j][i] = J_wb[i]
            Jacobian_m[j + 1][i] = J_h[i]
            Jacobian_m[j + 2][i] = J_wm[i]
            Jacobian_m[j + 3][i] = J_ns[i]
            Jacobian_m[j + 4][i] = J_sp[i]
            Jacobian_m[j + 5][i] = J_sv[i]
            j = j + 6
        else:
            Jacobian_m[j][i] = J_Da_3[i][l]
            Jacobian_m[j + 1][i] = J_H_3[i][l]
            Jacobian_m[j + 2][i] = J_fs8_3[i][l]
            Jacobian_m[j + 3][i] = eval("J_lnbS8_3_" + str(l) + "[i]")
            Jacobian_m[j + 4][i] = eval("J_lnPs_3_" + str(l) + "[i]")
            l = l + 1
            j = j + 5
    l = 0
    j = 0
    i = i + 1

F_new = np.dot(np.transpose(Jacobian_m), F_previous)
F_new = np.dot(F_new, Jacobian_m)

therms_elements = []
therms_use = []
therms_tmp = np.array(
    [
        "Omega_b_new",
        "h_new",
        "Omega_m_new",
        "ns_new",
        "Omega_L_new",
        "w0",
        "wa",
        "sig_8_new",
        "gamma_new",
        "sig_p_new",
        "sig_v_new",
    ]
)
for i in range(len(zrange)):
    therms_tmp = np.insert(therms_tmp, len(therms_tmp), "bsig8[" + str(i) + "]")
    therms_tmp = np.insert(therms_tmp, len(therms_tmp), "Psho[" + str(i) + "]")
for i in range(len(therms_tmp) - 2 * len(zrange) + 2):
    if Parameters_elts[i][0].startswith("Use"):
        therms_elements = np.insert(
            therms_elements, len(therms_elements), int(Parameters_elts[i][1])
        )
        therms_use.append(Parameters_elts[i][0])

# Erasing all the no needed parameters from the Fisher matrix. All parameters erased from the Fisher matrix are fixed.
if choice_L_NL == "L":
    if choice_F_NF == "NF" and (proj_ODE_fit == "ODE" or include_gamma == "N"):
        F_new = np.delete(F_new, [spsv_index - 1, spsv_index], 0)
        F_new = np.delete(F_new, [spsv_index - 1, spsv_index], 1)
        therms_elements = np.delete(therms_elements, [8, 9, 10])
        therms_use = np.delete(therms_use, [8, 9, 10])
    elif choice_F_NF == "F" and (proj_ODE_fit == "ODE" or include_gamma == "N"):
        F_new = np.delete(F_new, [spsv_index - 1, spsv_index], 0)
        F_new = np.delete(F_new, [spsv_index - 1, spsv_index], 1)
        therms_elements = np.delete(therms_elements, [1, 8, 9, 10])
        therms_use = np.delete(therms_use, [1, 8, 9, 10])
    elif choice_F_NF == "NF" and (proj_ODE_fit == "fit" and include_gamma == "Y"):
        F_new = np.delete(F_new, [spsv_index - 1, spsv_index], 0)
        F_new = np.delete(F_new, [spsv_index - 1, spsv_index], 1)
        therms_elements = np.delete(therms_elements, [9, 10])
        therms_use = np.delete(therms_use, [9, 10])
    elif choice_F_NF == "F" and (proj_ODE_fit == "fit" and include_gamma == "Y"):
        F_new = np.delete(F_new, [spsv_index - 1, spsv_index], 0)
        F_new = np.delete(F_new, [spsv_index - 1, spsv_index], 1)
        therms_elements = np.delete(therms_elements, [1, 9, 10])
        therms_use = np.delete(therms_use, [1, 9, 10])

if choice_L_NL == "SNL":
    if choice_F_NF == "NF" and (proj_ODE_fit == "ODE" or include_gamma == "N"):
        if SpecSAF_elts["Usesp_ch"] == "1" and SpecSAF_elts["Usesv_ch"] == "0":
            F_new = np.delete(F_new, [spsv_index - 1], 0)
            F_new = np.delete(F_new, [spsv_index - 1], 1)
            therms_elements = np.delete(therms_elements, [9])
            therms_use = np.delete(therms_use, [9])
        if SpecSAF_elts["Usesp_ch"] == "0" and SpecSAF_elts["Usesv_ch"] == "1":
            F_new = np.delete(F_new, [spsv_index], 0)
            F_new = np.delete(F_new, [spsv_index], 1)
            therms_elements = np.delete(therms_elements, [10])
            therms_use = np.delete(therms_use, [10])
        if SpecSAF_elts["Usesp_ch"] == "1" and SpecSAF_elts["Usesv_ch"] == "1":
            F_new = np.delete(F_new, [spsv_index - 1, spsv_index], 0)
            F_new = np.delete(F_new, [spsv_index - 1, spsv_index], 1)
            therms_elements = np.delete(therms_elements, [9, 10])
            therms_use = np.delete(therms_use, [9, 10])
        therms_elements = np.delete(therms_elements, [8])
        therms_use = np.delete(therms_use, [8])
    elif choice_F_NF == "F" and (proj_ODE_fit == "ODE" or include_gamma == "N"):
        if SpecSAF_elts["Usesp_ch"] == "1" and SpecSAF_elts["Usesv_ch"] == "0":
            F_new = np.delete(F_new, [spsv_index - 1], 0)
            F_new = np.delete(F_new, [spsv_index - 1], 1)
            therms_elements = np.delete(therms_elements, [9])
            therms_use = np.delete(therms_use, [9])
        if SpecSAF_elts["Usesp_ch"] == "0" and SpecSAF_elts["Usesv_ch"] == "1":
            F_new = np.delete(F_new, [spsv_index], 0)
            F_new = np.delete(F_new, [spsv_index], 1)
            therms_elements = np.delete(therms_elements, [10])
            therms_use = np.delete(therms_use, [10])
        if SpecSAF_elts["Usesp_ch"] == "1" and SpecSAF_elts["Usesv_ch"] == "1":
            F_new = np.delete(F_new, [spsv_index - 1, spsv_index], 0)
            F_new = np.delete(F_new, [spsv_index - 1, spsv_index], 1)
            therms_elements = np.delete(therms_elements, [9, 10])
            therms_use = np.delete(therms_use, [9, 10])
        therms_elements = np.delete(therms_elements, [8])
        therms_use = np.delete(therms_use, [8])
        therms_elements = np.delete(therms_elements, [1])
        therms_use = np.delete(therms_use, [1])
    elif choice_F_NF == "NF" and (proj_ODE_fit == "fit" and include_gamma == "Y"):
        if SpecSAF_elts["Usesp_ch"] == "1" and SpecSAF_elts["Usesv_ch"] == "0":
            F_new = np.delete(F_new, [spsv_index - 1], 0)
            F_new = np.delete(F_new, [spsv_index - 1], 1)
            therms_elements = np.delete(therms_elements, [9])
            therms_use = np.delete(therms_use, [9])
        if SpecSAF_elts["Usesp_ch"] == "0" and SpecSAF_elts["Usesv_ch"] == "1":
            F_new = np.delete(F_new, [spsv_index], 0)
            F_new = np.delete(F_new, [spsv_index], 1)
            therms_elements = np.delete(therms_elements, [10])
            therms_use = np.delete(therms_use, [10])
        if SpecSAF_elts["Usesp_ch"] == "1" and SpecSAF_elts["Usesv_ch"] == "1":
            F_new = np.delete(F_new, [spsv_index - 1, spsv_index], 0)
            F_new = np.delete(F_new, [spsv_index - 1, spsv_index], 1)
            therms_elements = np.delete(therms_elements, [9, 10])
            therms_use = np.delete(therms_use, [9, 10])
    elif choice_F_NF == "F" and (proj_ODE_fit == "fit" and include_gamma == "Y"):
        if SpecSAF_elts["Usesp_ch"] == "1" and SpecSAF_elts["Usesv_ch"] == "0":
            F_new = np.delete(F_new, [spsv_index - 1], 0)
            F_new = np.delete(F_new, [spsv_index - 1], 1)
            therms_elements = np.delete(therms_elements, [9])
            therms_use = np.delete(therms_use, [9])
        if SpecSAF_elts["Usesp_ch"] == "0" and SpecSAF_elts["Usesv_ch"] == "1":
            F_new = np.delete(F_new, [spsv_index], 0)
            F_new = np.delete(F_new, [spsv_index], 1)
            therms_elements = np.delete(therms_elements, [10])
            therms_use = np.delete(therms_use, [10])
        if SpecSAF_elts["Usesp_ch"] == "1" and SpecSAF_elts["Usesv_ch"] == "1":
            F_new = np.delete(F_new, [spsv_index - 1, spsv_index], 0)
            F_new = np.delete(F_new, [spsv_index - 1, spsv_index], 1)
            therms_elements = np.delete(therms_elements, [9, 10])
            therms_use = np.delete(therms_use, [9, 10])
        therms_elements = np.delete(therms_elements, [1])
        therms_use = np.delete(therms_use, [1])

if choice_F_NF == "F":
    permutation = [2, 0, 4, 5, 1, 3, 6]
    if proj_ODE_fit == "fit" and include_gamma == "Y":
        permutation = np.insert(permutation, len(permutation), len(permutation))
    if SpecSAF_elts["Usesp_ch"] == "0":
        permutation = np.insert(permutation, len(permutation), len(permutation))
    if SpecSAF_elts["Usesv_ch"] == "0":
        permutation = np.insert(permutation, len(permutation), len(permutation))
    tmp_per = permutation[len(permutation) - 1]
    for i in range(len(zrange)):
        permutation = np.insert(permutation, len(permutation), tmp_per + (2 * i + 1))
    for i in range(len(zrange)):
        permutation = np.insert(permutation, len(permutation), tmp_per + (2 * i + 2))
if choice_F_NF == "NF":
    permutation = [2, 4, 0, 5, 6, 1, 3, 7]
    if proj_ODE_fit == "fit" and include_gamma == "Y":
        permutation = np.insert(permutation, len(permutation), len(permutation))
    if SpecSAF_elts["Usesp_ch"] == "0":
        permutation = np.insert(permutation, len(permutation), len(permutation))
    if SpecSAF_elts["Usesv_ch"] == "0":
        permutation = np.insert(permutation, len(permutation), len(permutation))
    tmp_per = permutation[len(permutation) - 1]
    for i in range(len(zrange)):
        permutation = np.insert(permutation, len(permutation), tmp_per + (2 * i + 1))
    for i in range(len(zrange)):
        permutation = np.insert(permutation, len(permutation), tmp_per + (2 * i + 2))

F_new = F_new[:, permutation]
F_new = F_new[permutation, :]

for i in range(len(zrange) - 1):
    therms_elements = np.insert(
        therms_elements,
        len(therms_elements) - 1,
        therms_elements[len(therms_elements) - 2],
    )
    therms_use = np.insert(
        therms_use, len(therms_use) - 1, therms_use[len(therms_use) - 2]
    )
for i in range(len(zrange) - 1):
    therms_elements = np.insert(
        therms_elements, len(therms_elements), therms_elements[len(therms_elements) - 1]
    )
    therms_use = np.insert(therms_use, len(therms_use), therms_use[len(therms_use) - 1])

ind = np.where(therms_elements == 0)
therms_use = therms_use[ind]

F_new = F_new[ind[0]]
F_new = np.transpose(F_new)
F_new = F_new[ind[0]]
F_new = np.transpose(F_new)

# Saving the spectroscopic Fisher matrix.
out_F = open("output/Fisher_GCs_SpecSAF", "w")
i, j = 0, 0
while i < len(F_new):
    while j < len(F_new):
        out_F.write(str("%.10e" % F_new[i][j]))
        out_F.write(" ")
        j = j + 1
    out_F.write("\n")
    j = 0
    i = i + 1
out_F.close()

print("Projection done")

# %%
