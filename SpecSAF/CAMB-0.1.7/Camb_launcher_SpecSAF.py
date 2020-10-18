###################################################################################################
#Authors: S. Yahia-Cherif
#Last update: 29/06/2020
#This script call Camb to get the matter power spectrums and computes the no wiggle power spectrum
###################################################################################################

import os
import numpy as np
import time
import sys, platform
from scipy.interpolate import CubicSpline

#Loading the needed parameters files.
Codes_elts = np.loadtxt("../../QTLauncher/Codes_W.txt", dtype='str')
Spec_elts = np.loadtxt("../../QTLauncher/SpecSAF_W.txt", dtype='str')
Parameters_elts = np.loadtxt("../../QTLauncher/Parameters_W.txt", dtype='str')

#Saving all the entries in a dctionary
SpecSAF_elts = {}
for i in range(len(Spec_elts)):
    SpecSAF_elts[Spec_elts[i][0]] = Spec_elts[i][1]
for i in range(len(Parameters_elts)):
    SpecSAF_elts[Parameters_elts[i][0]] = Parameters_elts[i][1]

#Initialize all the variables.
TCMB = float(SpecSAF_elts["VCMB_TSP"])
Neff = float(SpecSAF_elts["VN_speciesSP"])

omega_b_fid = float(SpecSAF_elts["VFidOmegab"])*float(SpecSAF_elts["VFidh"])**2
h_fid = float(SpecSAF_elts["VFidh"])
omega_m_fid = float(SpecSAF_elts["VFidOmegam"])*float(SpecSAF_elts["VFidh"])**2
ns_fid = float(SpecSAF_elts["VFidns"])
w_neu = float(SpecSAF_elts["VFidWnu"])*float(SpecSAF_elts["VFidh"])**2
fid = np.array([omega_b_fid, h_fid, omega_m_fid, ns_fid])

eps_wb = float(SpecSAF_elts["VStepOmegab"])
eps_h = float(SpecSAF_elts["VSteph"])
eps_wm = float(SpecSAF_elts["VStepOmegam"])
eps_ns = float(SpecSAF_elts["VStepns"])

#Radiation if needed
#Og=(2.469*67**(-2))*10.**(-5)
#Omega_r = Og*(1+0.2271*3.046)
Omega_r = 0.
omega_r = Omega_r*fid[1]**2

fold_path = "Pk_baseline_NBSAF"

zrange_T = np.loadtxt("../input/bias_n_growth_baseline_base.dat", usecols=(0), unpack=True)
zrange = np.linspace(float(SpecSAF_elts["VzminSP"]), float(SpecSAF_elts["VzmaxSP"]), int(SpecSAF_elts["redbinsSP"])+1)
Delta_z = zrange[1:] - zrange[:-1]
zrange = 0.5*(zrange[1:]+zrange[:-1])

Allder_choice = int(SpecSAF_elts["DerMethodSP_ch"])
choices = np.array(["3", "5", "7"])
d_step = int(choices[Allder_choice])

i,l,paramo,t=0,0,0,0 

wb_new = fid[0]
h_new = fid[1]
wm_new = fid[2]
ns_new = fid[3]

#Set the power spectrums paths
fold_path_fid = [fold_path+"/fid"]
fold_path_wb = [fold_path+"/wb_up", fold_path+"/wb_up2", fold_path+"/wb_up3", fold_path+"/wb_dw", fold_path+"/wb_dw2", fold_path+"/wb_dw3"]
fold_path_h = [fold_path+"/h_up", fold_path+"/h_up2", fold_path+"/h_up3", fold_path+"/h_dw", fold_path+"/h_dw2", fold_path+"/h_dw3"]
fold_path_wm = [fold_path+"/wm_up", fold_path+"/wm_up2", fold_path+"/wm_up3", fold_path+"/wm_dw", fold_path+"/wm_dw2", fold_path+"/wm_dw3"]
fold_path_ns = [fold_path+"/ns_up", fold_path+"/ns_up2", fold_path+"/ns_up3", fold_path+"/ns_dw", fold_path+"/ns_dw2", fold_path+"/ns_dw3"]

#Create the power spectrums paths if they don't exist
if not os.path.exists(fold_path_fid[0]):
    os.makedirs(fold_path_fid[0])
i=0
while i < len(fold_path_wb):
    if not os.path.exists(fold_path_wb[i]):
        os.makedirs(fold_path_wb[i])
    if not os.path.exists(fold_path_h[i]):
        os.makedirs(fold_path_h[i])
    if not os.path.exists(fold_path_wm[i]):
        os.makedirs(fold_path_wm[i])
    if not os.path.exists(fold_path_ns[i]):
        os.makedirs(fold_path_ns[i])
    i=i+1

i,l,paramo,t=0,0,0,0 
while paramo < len(fid)+1:
    
    st=(d_step-1)/2
            
    if paramo == 0:
        wb_new = fid[0]
        h_new = fid[1]
        wm_new = fid[2]
        ns_new = fid[3]
                
        #Camb init modification : write the fiducial values of the shape parameters + the redshift values
        with open('params_base.ini', 'r') as file :
            filedata = file.readlines()
            filedata[35] = "ombh2          = "+str(wb_new) + "\n"
            filedata[36] = "omch2          = "+str(wm_new-wb_new-w_neu) + "\n"
            filedata[39] = "hubble         = "+str(h_new*100) + "\n"
            filedata[87] = "scalar_spectral_index(1)  = "+str(ns_new) + "\n"
            filedata[167] = "transfer_num_redshifts  = "+ str(len(zrange)+1) + "\n"
            
            aaa, bbb = 169, 0
            while bbb < len(zrange)+1:
                if bbb == len(zrange):
                    filedata = np.insert(filedata, aaa+bbb, "transfer_redshift("+str(bbb+1)+")    =  0." + "\n")
                else:
                    filedata = np.insert(filedata, aaa+bbb, "transfer_redshift("+str(bbb+1)+")    =  "+str(zrange[len(zrange)-1-bbb]) + "\n")
                bbb=bbb+1
   
            aaa = aaa + bbb
            bbb = 0
            while bbb < len(zrange)+1:
                filedata = np.insert(filedata, aaa+bbb, "transfer_filename("+str(bbb+1)+")    =  transfer_out"+str(bbb+1)+".dat" + "\n")
                bbb=bbb+1
            
        flin=0
        with open('params.ini', 'w') as file:
            while flin < len(filedata):
                file.write(filedata[flin])
                flin=flin+1
            flin=0
        
        #Call Camb
        os.system('./camb params.ini')  
        
        #Load the linear power spectrums
        aaa = len(zrange)
        while aaa >= 0:
            if aaa == len(zrange):
                kh, pk = np.loadtxt("test_matterpower_"+str(aaa)+".dat", usecols=(0,1,), unpack=True)
            elif aaa > 0:
                kh1, pk1 = np.loadtxt("test_matterpower_"+str(aaa)+".dat", usecols=(0,1,), unpack=True)
                kh = np.vstack((kh,kh1))
                pk = np.vstack((pk,pk1))
            else:
                kh1, pk1 = np.loadtxt("test_matterpower_"+str(len(zrange)+1)+".dat", usecols=(0,1,), unpack=True)
                kh = np.vstack((kh,kh1))
                pk = np.vstack((pk,pk1))
            aaa = aaa-1  
        
        #Integral computation for s8(z)
        ooo,ppp = 0,0
        integrale = np.zeros((len(pk)))
        while ppp < len(integrale):
            P_m = CubicSpline(np.log10(kh[ppp]),np.log10(pk[ppp]))
            while ooo < len(kh[0])-1:
                integrale[ppp] = integrale[ppp] + 1./(2*np.pi**2) * (kh[ppp][ooo+1]-kh[ppp][ooo])/6. * ( (kh[ppp][ooo]**2*(3*(np.sin(8*kh[ppp][ooo]) - 8*kh[ppp][ooo]*np.cos(8*kh[ppp][ooo]))/(8*kh[ppp][ooo])**3)**2*pk[ppp][ooo]) + (kh[ppp][ooo+1]**2*(3*(np.sin(8*kh[ppp][ooo+1]) - 8*kh[ppp][ooo+1]*np.cos(8*kh[ppp][ooo+1]))/(8*kh[ppp][ooo+1])**3)**2*pk[ppp][ooo+1]) + 4.*( ((kh[ppp][ooo+1]+kh[ppp][ooo])/2.)**2*(3*(np.sin(8*(kh[ppp][ooo+1]+kh[ppp][ooo])/2.) - 8*(kh[ppp][ooo+1]+kh[ppp][ooo])/2*np.cos(8*(kh[ppp][ooo+1]+kh[ppp][ooo])/2.))/(8*(kh[ppp][ooo+1]+kh[ppp][ooo])/2.)**3)**2*10**P_m(np.log10((kh[ppp][ooo]+kh[ppp][ooo+1])/2))) )
                ooo=ooo+1
            ooo=0
            ppp=ppp+1
            
        #####################################
        #############   NW PS   #############
        #####################################
        
        h_fid = h_new
        omega_b_fid = wb_new
        omega_m_fid = wm_new
        omega_c = omega_m_fid - omega_b_fid - w_neu
        ns_fid = ns_new
        
        omega_L_fid = 1.-omega_m_fid
        Omega_L_fid = omega_L_fid/(h_fid**2)

        k_cf = kh[0]

        theta27 = TCMB/2.7
        zeq = 2.5*1e4*omega_m_fid*theta27**(-4)

        q_h = k_cf*theta27**2/omega_m_fid

        b1 = 0.313*omega_m_fid**(-0.419)*(1.+0.607*omega_m_fid**0.674)
        b2 = 0.238*omega_m_fid**0.223 

        zd = 1291.*(omega_m_fid**0.251)/(1+0.659*omega_m_fid**0.828)*(1.+b1*omega_b_fid**b2)

        yd = (1. + zeq)/(1. + zd)
        s = 44.5*np.log(9.83/omega_m_fid)/(np.sqrt(1. + 10*omega_b_fid**0.75))

        fc = omega_c/omega_m_fid
        fcb = (omega_c + omega_b_fid)/omega_m_fid
        fnu = w_neu/omega_m_fid
        fnub = (omega_b_fid + w_neu)/omega_m_fid

        pc = 0.25*(5. - np.sqrt(1.+24.*fc))
        pcb = 0.25*(5. - np.sqrt(1.+24.*fcb))

        alpha_nu = fc/fcb*(5. - 2*(pc+pcb))/(5.-4*pcb)*(1. - 0.553*fnub + 0.126*fnub**3)/(1. - 0.193*np.sqrt(fnu*Neff) + 0.169*fnu*Neff**0.2)*(1. + yd)**(pcb-pc) * (1. + (pc - pcb)/2*(1.+1./((3. - 4*pc)*(7. - 4*pcb)))/(1. + yd))

        gamma_eff = omega_m_fid*(np.sqrt(alpha_nu) + (1. - np.sqrt(alpha_nu))/(1. + (0.43*k_cf*s)**4))
        qeff = k_cf*theta27**2/gamma_eff

        beta_c = 1./(1. - 0.949*fnub)
        L = np.log(np.exp(1) + 1.84*beta_c*np.sqrt(alpha_nu)*qeff)
        C = 14.4 + 325./(1. + 60.5*qeff**1.08)
    
        Omega_MZ = np.zeros(len(zrange))
        Omega_LZ = np.zeros(len(zrange))
        D1 = np.zeros(len(zrange))
        Dcb = np.zeros(len(zrange))
        Dcb_nu = np.zeros((len(zrange), len(q_h)))
        
        yfs = 17.2*fnu*(1.+0.488*fnu**(-7./6))*(Neff*q_h/fnu)**2
        qnu = 3.92*q_h*np.sqrt(Neff/fnu)
        
        i,j=0,0
        while i < len(zrange):
            while j < len(q_h):
                Omega_MZ[i] = omega_m_fid*(1+zrange[i])**3/(omega_L_fid + omega_r*(1+zrange[i])**2 + omega_m_fid*(1+zrange[i])**3)
                Omega_LZ[i] = omega_L_fid/(omega_L_fid + omega_r*(1+zrange[i])**2 + omega_m_fid*(1+zrange[i])**3)
    
                D1[i] = (1.+zeq)/(1.+zrange[i])*2.5*Omega_MZ[i]/(Omega_MZ[i]**(4./7) -  Omega_LZ[i] + (1. + 0.5*Omega_MZ[i])*(1. + Omega_LZ[i]/70))
                Dcb_nu[i][j] = (fcb**(0.7/pcb) + (D1[i]/(1+yfs[j]))**0.7)**(pcb/0.7)*D1[i]**(1.-pcb)

                j=j+1
            j=0
            i=i+1

        B = 1. + (1.24*fnu**0.64*Neff**(0.3+0.6*fnu))/(qnu**(-1.6) + qnu**0.8)

        TEH = np.zeros((len(zrange), len(k_cf)))
        PEH = np.copy(TEH)
        P_NW = np.copy(TEH)
        kh_NW = np.copy(kh)
        sig_82 = np.zeros(len(zrange))

        i=0
        while i < len(zrange):
            TEH[i] = L/(L+C*qeff**2)*(Dcb_nu[i]/D1[i])*B
            TEH_I = CubicSpline(np.log10(kh_NW[i]/h_fid),np.log10(TEH[i]))
            TEH[i] = 10**TEH_I(np.log10(kh_NW[i]))
            PEH[i] = TEH[i]**2*kh_NW[i]**ns_fid
            i=i+1
        
        i,j=0,0    
        while i < len(zrange):    
            while j < len(k_cf)-1:
                sig_82[i] = sig_82[i] + (kh[i][j+1] - kh[i][j]) * ( kh[i][j+1]**2/(2*np.pi**2)*PEH[i][j+1]*((3./(8.*kh[i][j+1])**3) * (np.sin(8.*kh[i][j+1]) - 8.*kh[i][j+1]*np.cos(8.*kh[i][j+1])))**2 + kh[i][j]**2/(2*np.pi**2)*PEH[i][j]*((3./(8.*kh[i][j])**3) * (np.sin(8.*kh[i][j]) - kh[i][j]*8.*np.cos(8.*kh[i][j])))**2 )/2
                j=j+1
            j=0
            P_NW[i] = PEH[i]/sig_82[i]
            i=i+1
        
        pknum=0
        while pknum < len(zrange):
            pk[pknum] = pk[pknum]/(integrale[pknum])
            pknum=pknum+1
        integrale = np.sqrt(integrale)
        
        kh = np.delete(kh, len(kh)-1, axis=0)
        pk = np.delete(pk, len(pk)-1, axis=0)
        
        #The fiducial power spectrums are saved.
        files,flines=0,0
        while files < len(zrange):
            outP = open(fold_path_fid[0]+"/Pks8sqRatio_ist_LogSplineInterpPk_"+str(zrange[files])+".dat",'w')
            outP.write("z(iz=00)            wb(ip=00)           h(ip=01)           wm(ip=02)            ns(ip=03) \n")
            outP.write(str("%.10e" % zrange[files]) + "        "+str(wb_new)+"        "+str(h_new)+"        "+str(wm_new)+"        "+str(ns_new)+"\n")
            outP.write("   k(h/Mpc)      Pk/s8^2(Mpc/h)^3       s8 \n")
                    
            while flines < len(kh[files]):
               outP.write(str("%.10e" % kh[files][flines])+" "+str("%.10e" % pk[files][flines])+" "+str("%.10e" % integrale[files])+"\n")
               flines=flines+1
            outP.close()
            flines=0
            
            outP = open(fold_path_fid[0]+"/Pks8sqRatio_ist_LogSplineInterpPk_"+str(zrange[files])+"_NW.dat",'w')
            outP.write("z(iz=00)            wb(ip=00)           h(ip=01)           wm(ip=02)            ns(ip=03) \n")
            outP.write(str("%.10e" % zrange[files]) + "        "+str(wb_new)+"        "+str(h_new)+"        "+str(wm_new)+"        "+str(ns_new)+"\n")
            outP.write("   k(h/Mpc)      Pk/s8^2(Mpc/h)^3       s8 \n")
                    
            while flines < len(kh[files]):
               outP.write(str("%.10e" % kh_NW[files][flines])+" "+str("%.10e" % P_NW[files][flines])+" "+str("%.10e" % integrale[files])+"\n")
               flines=flines+1
            outP.close()
            flines=0
            
            files=files+1
        paramo=paramo+1
        continue
    
    #Next iterations for the non fiducial power spectrums.
    while l < d_step:
        #Each parameters are changed one by one according to the derivatives.
        if (l == st):
            l=l+1
        if paramo == 1:
            FPA = fold_path_wb
            wb_new = fid[0]*(1 + (l-st)*eps_wb)
            h_new = fid[1]
            wm_new = fid[2]
            ns_new = fid[3]
        elif paramo == 2:
            FPA = fold_path_h
            wb_new = fid[0]
            h_new = fid[1]*(1 + (l-st)*eps_h)
            wm_new = fid[2]
            ns_new = fid[3]
        elif paramo == 3:
            FPA = fold_path_wm
            wb_new = fid[0]
            h_new = fid[1]
            wm_new = fid[2]*(1 + (l-st)*eps_wm)
            ns_new = fid[3]
        elif paramo == 4:
            FPA = fold_path_ns
            wb_new = fid[0]
            h_new = fid[1]
            wm_new = fid[2]
            ns_new = fid[3]*(1 + (l-st)*eps_ns)

        #This part reproduces exactly the same scheme as for the fiducial part .         
        with open('params.ini', 'r') as file :
            filedata = file.readlines()
            filedata[35] = "ombh2          = "+str(wb_new) + "\n"
            filedata[36] = "omch2          = "+str(wm_new-wb_new-w_neu) + "\n"
            filedata[39] = "hubble         = "+str(h_new*100) + "\n"
            filedata[87] = "scalar_spectral_index(1)  = "+str(ns_new) + "\n"
            filedata[167] = "transfer_num_redshifts  = "+ str(len(zrange)+1) + "\n"
            
            aaa, bbb = 169, 0
            while bbb < len(zrange)+1:
                if bbb == len(zrange):
                    filedata[aaa+bbb] = "transfer_redshift("+str(bbb+1)+")    =  0." + "\n"
                else:
                    filedata[aaa+bbb] = "transfer_redshift("+str(bbb+1)+")    =  "+str(zrange[len(zrange)-1-bbb]) + "\n"
                bbb=bbb+1
   
            aaa = aaa + bbb
            bbb = 0
            while bbb < len(zrange)+1:
                filedata[aaa+bbb] = "transfer_filename("+str(bbb+1)+")    =  transfer_out"+str(bbb+1)+".dat" + "\n"
                bbb=bbb+1
            
        flin=0
        with open('params.ini', 'w') as file:
            while flin < len(filedata):
                file.write(filedata[flin])
                flin=flin+1
            flin=0
        
        os.system('./camb params.ini')    
        
        aaa = len(zrange)
        while aaa >= 0:
            if aaa == len(zrange):
                kh, pk = np.loadtxt("test_matterpower_"+str(aaa)+".dat", usecols=(0,1,), unpack=True)
            elif aaa > 0:
                kh1, pk1 = np.loadtxt("test_matterpower_"+str(aaa)+".dat", usecols=(0,1,), unpack=True)
                kh = np.vstack((kh,kh1))
                pk = np.vstack((pk,pk1))
            else:
                kh1, pk1 = np.loadtxt("test_matterpower_"+str(len(zrange)+1)+".dat", usecols=(0,1,), unpack=True)
                kh = np.vstack((kh,kh1))
                pk = np.vstack((pk,pk1))
            aaa = aaa-1  
        
        ooo,ppp = 0,0
        integrale = np.zeros((len(pk)))
        while ppp < len(integrale):
            P_m = CubicSpline(np.log10(kh[ppp]),np.log10(pk[ppp]))
            while ooo < len(kh[0])-1:
                integrale[ppp] = integrale[ppp] + 1./(2*np.pi**2) * (kh[ppp][ooo+1]-kh[ppp][ooo])/6. * ( (kh[ppp][ooo]**2*(3*(np.sin(8*kh[ppp][ooo]) - 8*kh[ppp][ooo]*np.cos(8*kh[ppp][ooo]))/(8*kh[ppp][ooo])**3)**2*pk[ppp][ooo]) + (kh[ppp][ooo+1]**2*(3*(np.sin(8*kh[ppp][ooo+1]) - 8*kh[ppp][ooo+1]*np.cos(8*kh[ppp][ooo+1]))/(8*kh[ppp][ooo+1])**3)**2*pk[ppp][ooo+1]) + 4.*( ((kh[ppp][ooo+1]+kh[ppp][ooo])/2.)**2*(3*(np.sin(8*(kh[ppp][ooo+1]+kh[ppp][ooo])/2.) - 8*(kh[ppp][ooo+1]+kh[ppp][ooo])/2*np.cos(8*(kh[ppp][ooo+1]+kh[ppp][ooo])/2.))/(8*(kh[ppp][ooo+1]+kh[ppp][ooo])/2.)**3)**2*10**P_m(np.log10((kh[ppp][ooo]+kh[ppp][ooo+1])/2))) )
                ooo=ooo+1
            ooo=0
            ppp=ppp+1
            
        #####################################
        #############   NW PS   #############
        #####################################
        
        h_fid = h_new
        omega_b_fid = wb_new
        omega_m_fid = wm_new
        omega_c = omega_m_fid - omega_b_fid - w_neu
        ns_fid = ns_new
        
        omega_L_fid = 1.-omega_m_fid
        Omega_L_fid = omega_L_fid/(h_fid**2)

        k_cf = kh[0]

        theta27 = TCMB/2.7
        zeq = 2.5*1e4*omega_m_fid*theta27**(-4)

        q_h = k_cf*theta27**2/omega_m_fid

        b1 = 0.313*omega_m_fid**(-0.419)*(1.+0.607*omega_m_fid**0.674)
        b2 = 0.238*omega_m_fid**0.223 

        zd = 1291.*(omega_m_fid**0.251)/(1+0.659*omega_m_fid**0.828)*(1.+b1*omega_b_fid**b2)

        yd = (1. + zeq)/(1. + zd)
        s = 44.5*np.log(9.83/omega_m_fid)/(np.sqrt(1. + 10*omega_b_fid**0.75))

        fc = omega_c/omega_m_fid
        fcb = (omega_c + omega_b_fid)/omega_m_fid
        fnu = w_neu/omega_m_fid
        fnub = (omega_b_fid + w_neu)/omega_m_fid

        pc = 0.25*(5. - np.sqrt(1.+24.*fc))
        pcb = 0.25*(5. - np.sqrt(1.+24.*fcb))

        alpha_nu = fc/fcb*(5. - 2*(pc+pcb))/(5.-4*pcb)*(1. - 0.553*fnub + 0.126*fnub**3)/(1. - 0.193*np.sqrt(fnu*Neff) + 0.169*fnu*Neff**0.2)*(1. + yd)**(pcb-pc) * (1. + (pc - pcb)/2*(1.+1./((3. - 4*pc)*(7. - 4*pcb)))/(1. + yd))

        gamma_eff = omega_m_fid*(np.sqrt(alpha_nu) + (1. - np.sqrt(alpha_nu))/(1. + (0.43*k_cf*s)**4))
        qeff = k_cf*theta27**2/gamma_eff

        beta_c = 1./(1. - 0.949*fnub)
        L = np.log(np.exp(1) + 1.84*beta_c*np.sqrt(alpha_nu)*qeff)
        C = 14.4 + 325./(1. + 60.5*qeff**1.08)
    
        Omega_MZ = np.zeros(len(zrange))
        Omega_LZ = np.zeros(len(zrange))
        D1 = np.zeros(len(zrange))
        Dcb = np.zeros(len(zrange))
        Dcb_nu = np.zeros((len(zrange), len(q_h)))
        
        yfs = 17.2*fnu*(1.+0.488*fnu**(-7./6))*(Neff*q_h/fnu)**2
        qnu = 3.92*q_h*np.sqrt(Neff/fnu)
        
        i,j=0,0
        while i < len(zrange):
            while j < len(q_h):
                Omega_MZ[i] = omega_m_fid*(1+zrange[i])**3/(omega_L_fid + omega_r*(1+zrange[i])**2 + omega_m_fid*(1+zrange[i])**3)
                Omega_LZ[i] = omega_L_fid/(omega_L_fid + omega_r*(1+zrange[i])**2 + omega_m_fid*(1+zrange[i])**3)
    
                D1[i] = (1.+zeq)/(1.+zrange[i])*2.5*Omega_MZ[i]/(Omega_MZ[i]**(4./7) -  Omega_LZ[i] + (1. + 0.5*Omega_MZ[i])*(1. + Omega_LZ[i]/70))
                Dcb_nu[i][j] = (fcb**(0.7/pcb) + (D1[i]/(1+yfs[j]))**0.7)**(pcb/0.7)*D1[i]**(1.-pcb)
                j=j+1
            j=0
            i=i+1

        B = 1. + (1.24*fnu**0.64*Neff**(0.3+0.6*fnu))/(qnu**(-1.6) + qnu**0.8)

        TEH = np.zeros((len(zrange), len(k_cf)))
        PEH = np.copy(TEH)
        P_NW = np.copy(TEH)
        kh_NW = np.copy(kh)
        sig_82 = np.zeros(len(zrange))

        i=0
        while i < len(zrange):
            TEH[i] = L/(L+C*qeff**2)*(Dcb_nu[i]/D1[i])*B
            TEH_I = CubicSpline(np.log10(kh_NW[i]/h_fid),np.log10(TEH[i]))
            TEH[i] = 10**TEH_I(np.log10(kh_NW[i]))
            PEH[i] = TEH[i]**2*kh_NW[i]**ns_fid
            i=i+1
        
        i,j=0,0    
        while i < len(zrange):    
            while j < len(k_cf)-1:
                sig_82[i] = sig_82[i] + (kh[i][j+1] - kh[i][j]) * ( kh[i][j+1]**2/(2*np.pi**2)*PEH[i][j+1]*((3./(8.*kh[i][j+1])**3) * (np.sin(8.*kh[i][j+1]) - 8.*kh[i][j+1]*np.cos(8.*kh[i][j+1])))**2 + kh[i][j]**2/(2*np.pi**2)*PEH[i][j]*((3./(8.*kh[i][j])**3) * (np.sin(8.*kh[i][j]) - kh[i][j]*8.*np.cos(8.*kh[i][j])))**2 )/2
                j=j+1
            j=0
            P_NW[i] = PEH[i]/sig_82[i]
            i=i+1
            
        pknum=0
        while pknum < len(zrange):
            pk[pknum] = pk[pknum]/(integrale[pknum])
            pknum=pknum+1
        integrale = np.sqrt(integrale)
        
        #Save the non fiducial power spectrums.
        files,flines=0,0
        while files < len(zrange):
            if (l == st+1):
                outP = open(FPA[0]+"/Pks8sqRatio_ist_LogSplineInterpPk_"+str(zrange[files])+".dat",'w')
                outP_NW = open(FPA[0]+"/Pks8sqRatio_ist_LogSplineInterpPk_"+str(zrange[files])+"_NW.dat",'w')
            elif (l == st+2):
                outP = open(FPA[1]+"/Pks8sqRatio_ist_LogSplineInterpPk_"+str(zrange[files])+".dat",'w')
                outP_NW = open(FPA[1]+"/Pks8sqRatio_ist_LogSplineInterpPk_"+str(zrange[files])+"_NW.dat",'w')
            elif (l == st+3):
                outP = open(FPA[2]+"/Pks8sqRatio_ist_LogSplineInterpPk_"+str(zrange[files])+".dat",'w')
                outP_NW = open(FPA[2]+"/Pks8sqRatio_ist_LogSplineInterpPk_"+str(zrange[files])+"_NW.dat",'w')
            elif (l == st-1):
                outP = open(FPA[3]+"/Pks8sqRatio_ist_LogSplineInterpPk_"+str(zrange[files])+".dat",'w')
                outP_NW = open(FPA[3]+"/Pks8sqRatio_ist_LogSplineInterpPk_"+str(zrange[files])+"_NW.dat",'w')
            elif (l == st-2):
                outP = open(FPA[4]+"/Pks8sqRatio_ist_LogSplineInterpPk_"+str(zrange[files])+".dat",'w')
                outP_NW = open(FPA[4]+"/Pks8sqRatio_ist_LogSplineInterpPk_"+str(zrange[files])+"_NW.dat",'w')
            elif (l == st-3):
                outP = open(FPA[5]+"/Pks8sqRatio_ist_LogSplineInterpPk_"+str(zrange[files])+".dat",'w')
                outP_NW = open(FPA[5]+"/Pks8sqRatio_ist_LogSplineInterpPk_"+str(zrange[files])+"_NW.dat",'w')

            outP.write("z(iz=00)            wb(ip=00)           h(ip=01)           wm(ip=02)            ns(ip=03) \n")
            outP.write(str("%.10e" % zrange[files]) + "        "+str(wb_new)+"        "+str(h_new)+"        "+str(wm_new)+"        "+str(ns_new)+"\n")
            outP.write("   k(h/Mpc)      Pk/s8^2(Mpc/h)^3 \n")
            while flines < len(kh[files]):
                outP.write(str("%.10e" % kh[files][flines])+" "+str("%.10e" % pk[files][flines])+"\n")
                flines=flines+1
            flines=0
            outP.close()

            outP_NW.write("z(iz=00)            wb(ip=00)           h(ip=01)           wm(ip=02)            ns(ip=03) \n")
            outP_NW.write(str("%.10e" % zrange[files]) + "        "+str(wb_new)+"        "+str(h_new)+"        "+str(wm_new)+"        "+str(ns_new)+"\n")
            outP_NW.write("   k(h/Mpc)      Pk/s8^2(Mpc/h)^3 \n")
            while flines < len(kh[files]):
                outP_NW.write(str("%.10e" % kh_NW[files][flines])+" "+str("%.10e" % P_NW[files][flines])+"\n")
                flines=flines+1
            flines=0
            outP_NW.close()
                        
            flines=0
            files=files+1
        l=l+1
    l=0
    paramo=paramo+1
paramo=0

#Erase all the temporar files.
os.system("rm -r test_transfer_out*")
os.system("rm -r test_matterpower_*")
os.system("rm -r test_matterpower*")
