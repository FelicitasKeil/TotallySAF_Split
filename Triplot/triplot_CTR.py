##################################################################################
#Authors: I. Tutusaus & S. Yahia-Cherif                                          #
#This script plot automatically the contours figure                              #
##################################################################################

#Modules import.
import numpy as np
from numpy import newaxis
import math
import matplotlib.pyplot as plt
import os
from os import path
import glob
import scipy
import re
from matplotlib.patches import Ellipse
from matplotlib.ticker import FormatStrFormatter
import matplotlib.cm as cm

print(" ")
print("Checking input parameters")
#Empty list creation in order to stock the input parameters.
input_param = []

#Saving the input parameters inside a list.
with open ("init_CTR.txt", "r") as myfile:
    for line in myfile:
        li=line.strip()
        if not li.startswith("#"):
            if not li=="":
                input_param.append(line.strip())

#Stocking the input parameters in variables
tick_size = int(input_param[0])
font_size = int(input_param[1])
lg_size = int(input_param[2])
lgx = float(input_param[3])
lgy = float(input_param[4])
lg_bckgnd = str(input_param[5])
lg_frame_color = str(input_param[6])
lg_text_color = str(input_param[7])
lg_frame_size = float(input_param[8])

#Stocking the input parameters in variables.
x = np.copy(input_param)
i,param_N,pmin,pmax=0,0,0,0
while i < len(x):
    if str(x[i][0]) == '$':
        param_N = param_N+1
    if param_N == 1:
        pmin=i
    
    i=i+1
i=0
pmax = pmin+param_N

#Saving the fiducials in a file.
x = x[pmin:pmax]
out_F = open('fiducial','w')
i=0
while i < len(x):
    out_F.write(x[i])
    out_F.write("\n")
    i=i+1
out_F.close()
i=0

#Building the input files paths.
path_In = ("F_input")
files_In = glob.glob(path_In + '/*')

NF = len(files_In)

#Building the legend into a variable.
if NF == 1:
    leg_N = input_param[len(input_param)-1]
else:
    leg_N = np.chararray(NF, itemsize=1000)
    while i < NF:
        leg_N[i] = str(input_param[len(input_param)-i-1])
        i=i+1
    i=0

leg_N = np.sort(leg_N)
for i in range(len(leg_N)):
    leg_tmp = leg_N[i]
    leg_tmp = leg_tmp.split(b"|")
    leg_N[i] = leg_tmp[1]

#Saving the legend in a file.
out_F = open('legendtxt','bw')
i=0
if NF == 1:
    out_F.write(leg_N)
    out_F.close()
else:
    while i < len(leg_N):
        out_F.write(leg_N[i])
        out_F.write(b"\n")
        i=i+1
    out_F.close()
    i=0  

v = 0
#Sorting the Fisher matrix paths into alphabetical order and loading the Fisher matrix into a 3D matrix.
files_In = np.sort(files_In)
while v < NF:
    F_matrix_T = np.loadtxt(files_In[v])
    F_matrix_T = np.linalg.inv(F_matrix_T)[0:7,0:7]
    F_matrix_T = np.linalg.inv(F_matrix_T)
    
    if v == 0:
        F_matrix = np.copy(F_matrix_T)
    else:
        F_matrix = np.dstack((F_matrix, F_matrix_T))
    v=v+1

if len(F_matrix.shape) == 3:
    F_matrix = np.swapaxes(F_matrix, 0, 2)
else:    
    F_matrix = F_matrix[newaxis, :, :]

#Loading the legend
leg = np.genfromtxt("legendtxt", dtype='S100', delimiter='\t')
leg = leg.astype('U1000')

#Loading the fiducial values in a variable.
x = np.genfromtxt("fiducial", delimiter=' ', usecols=(0), unpack=True, dtype="S1000")
x = x.astype('U1000')
fid_values = np.genfromtxt("fiducial", delimiter=' ', usecols=(1), unpack=True, dtype="f16")

C_matrix = np.copy(F_matrix)
#Inverting the Fisher matrix.
v=0
while v < NF:
    C_matrix[v] = np.linalg.inv(F_matrix[v])
    v=v+1
       
C_matrix_I = np.copy(C_matrix)

print("Plot all likelihoods")
#Computing the likelihood.
def Lik(x,mu,sigma):
    return 1./(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

#Loading the colors and stocking it into a variable.
couleur = np.genfromtxt("colors", dtype='S100', delimiter=' ', unpack=True)
couleur = couleur.astype('U1000')
couleur_I = np.copy(couleur)
xax = np.zeros((len(C_matrix), len(C_matrix[0]), 10000))

#Building the parameters space area.
i,j=0,0
while i < len(C_matrix[0]):
    while j < len(C_matrix):
        xax[j][i] = np.linspace(fid_values[i] - 3*np.sqrt(C_matrix[j][i][i]), fid_values[i] + 3*np.sqrt(C_matrix[j][i][i]),10000)
        j=j+1
    j=0
    i=i+1
xax_I = np.copy(xax)
 
lik = np.zeros((len(C_matrix), len(C_matrix[0]), 10000))

#Calling the likelihood plot functions.
i,j=0,0
while j < len(C_matrix):
    while i < len(C_matrix[0]):
        lik[j][i] = Lik(xax[j][i],(min(xax[j][i])+max(xax[j][i]))/2,np.sqrt(C_matrix[j][i][i]))/max(Lik(xax[j][i],(min(xax[j][i])+max(xax[j][i]))/2,np.sqrt(C_matrix[j][i][i])))
        i=i+1
    i=0
    j=j+1
lik_I = np.copy(lik)

y = '$\mathcal{P}/ \mathcal{P}_{max}$'
#Building the figure.
fig = plt.figure(num=None, figsize=(24, 24), facecolor='w', edgecolor='k')
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1)
plt.rc('xtick', labelsize=tick_size)
plt.rc('ytick', labelsize=tick_size)

#This function plots the 1D likelihoods.
def plot_Lik(x_val,y_val,x_lab,y_lab,x_min,x_max,y_min,y_max,tab,tab_I,center,chainza,chainzb,chainzc,ind,coloris):

    ax = fig.add_subplot(chainza,chainzb,chainzc)

    plt.xlabel(x_lab,fontsize=font_size)
    plt.ylabel(y_lab,fontsize=font_size)
    plt.yticks(np.linspace(0, 1, 3))
    
    if l == 0:
        plt.xticks(np.array([fid_values[i] - 2.48*np.sqrt(C_matrix[l][i][i]), fid_values[i], fid_values[i] + 2.48*np.sqrt(C_matrix[l][i][i])]))
    
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        for tick in ax.get_yticklabels():
            tick.set_rotation(45)   

    plt.fill_between(x_val,y_val,where=np.fabs(x_val-center) <= 1.52*np.sqrt(tab), facecolor=coloris, alpha=0.8)
    plt.fill_between(x_val,y_val,where=(np.fabs(x_val-center) > 1.52*np.sqrt(tab)) & (np.fabs(x_val-center) <= 2.48*np.sqrt(tab)), facecolor=coloris, alpha=0.2)
    plt.plot(x_val,y_val,coloris,linewidth=0.5)
    
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max) 
    
    if (ind == 0):
        plt.tick_params(labelbottom='off')
    elif(ind < tri_len-1):
        plt.tick_params(labelbottom='off')
        plt.tick_params(labelleft='off')
    else:
        plt.tick_params(labelleft='off')

i,j,l = 0,1,0
tri_len = len(F_matrix[0])
tri_chain = len(F_matrix[0])/2*(len(F_matrix[0])+1)     

minx = np.copy(fid_values)
maxx = np.copy(fid_values)

while i < tri_len:
    while l < len(F_matrix):
        if (min(xax[l][i]) < minx[i]):
            minx[i] = min(xax[l][i])
        if (max(xax[l][i]) > maxx[i]):
            maxx[i] = max(xax[l][i])
        l=l+1
    l=0
    i=i+1

i,j,l,s = 0,1,0,0    
Mat_order = np.zeros(len(C_matrix)) 

#Calling the contour plot likelihood and organizing the matrix order. The matrix are ordered from the largest to the narrower.
while i < tri_len:
    while s < len(C_matrix):
        Mat_order[s] = C_matrix[s][i][i] 
        s=s+1
    s=0
    SSS = Mat_order.argsort()
    SSS = SSS[::-1]
    Mat_order = Mat_order[SSS]
    C_matrix = C_matrix[SSS]
    couleur = couleur[SSS]
    lik = lik[SSS]
    xax = xax[SSS]
    leg = leg[SSS]
    while l < len(F_matrix):
        if i == 0:
            plot_Lik(xax[l][i],lik[l][i],"",y,minx[i],maxx[i],0.,1.1,C_matrix[l][i][i],C_matrix[l][i][i],(min(xax[l][i])+max(xax[l][i]))/2,tri_len,tri_len,j,i,couleur[l])
        elif i < tri_len-1:
            plot_Lik(xax[l][i],lik[l][i],"","",minx[i],maxx[i],0.,1.1,C_matrix[l][i][i],C_matrix[l][i][i],(min(xax[l][i])+max(xax[l][i]))/2,tri_len,tri_len,j,i,couleur[l])
        else:
            plot_Lik(xax[l][i],lik[l][i],x[i],"",minx[i],maxx[i],0.,1.1,C_matrix[l][i][i],C_matrix[l][i][i],(min(xax[l][i])+max(xax[l][i]))/2,tri_len,tri_len,j,i,couleur[l])
        l=l+1
    l=0
    i=i+1
    j=j+tri_len+1  
    
couleur = np.copy(couleur_I)
C_matrix = np.copy(C_matrix_I)
lik = np.copy(lik_I)
xax = np.copy(xax_I)
 

print("Plot all contours")   
#This function plots the inner contours.
def plt_ell1(sx2,sy2,sxy,pos,P,edge,face,alpha,x,x_lab,y_lab,x_min,x_max,y_min,y_max,chainza,chainzb,chainzc,ind_i,ind_j):
    a2=(sx2+sy2)/2+np.sqrt((sx2-sy2)**2/4+sxy**2)
    b2=(sx2+sy2)/2-np.sqrt((sx2-sy2)**2/4+sxy**2)
    theta=0.5*np.arctan(2*sxy/(sx2-sy2))
    
    U, s , Vh = np.linalg.svd(P)
    orient = math.atan2(U[1,0],U[0,0])*180/np.pi
    
    fig.add_subplot(chainza,chainzb,chainzc)
    
    ellipsePlot = Ellipse(xy=pos, width=2.0*alpha*math.sqrt(a2),height=2.0*alpha*math.sqrt(b2), angle=orient,facecolor=face,alpha=x,edgecolor='None')
    ax = plt.gca()
    ax.add_patch(ellipsePlot);
    
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    plt.xlabel(x_lab,fontsize=font_size)
    plt.ylabel(y_lab,fontsize=font_size)
    
    if (ind_i != len(F_matrix[0])-1):
        plt.tick_params(labelbottom='off')
    if (ind_j != 0):
        plt.tick_params(labelleft='off')
        
    return ellipsePlot;
   
#This function plots the outer contours. 
def plt_ell2(sx2,sy2,sxy,pos,P,edge,face,alpha,x,x_lab,y_lab,x_min,x_max,y_min,y_max,chainza,chainzb,chainzc,ind_i,ind_j):
    a2=(sx2+sy2)/2+np.sqrt((sx2-sy2)**2/4+sxy**2)
    b2=(sx2+sy2)/2-np.sqrt((sx2-sy2)**2/4+sxy**2)
    theta=0.5*np.arctan(2*sxy/(sx2-sy2))
    
    U, s , Vh = np.linalg.svd(P)
    orient = math.atan2(U[1,0],U[0,0])*180/np.pi
    
    fig.add_subplot(chainza,chainzb,chainzc)
    
    ellipsePlot = Ellipse(xy=pos, width=2.0*alpha*math.sqrt(a2),height=2.0*alpha*math.sqrt(b2), angle=orient,facecolor=face,alpha=x,edgecolor=couleur[l],linewidth=1.0,fill=False)
    ax = plt.gca()
    ax.add_patch(ellipsePlot)
    
    if l == 0:
        plt.xticks(np.array([fid_values[j] - 2.48*np.sqrt(C_matrix[l][j][j]), fid_values[j], fid_values[j] + 2.48*np.sqrt(C_matrix[l][j][j])]))
        plt.yticks(np.array([fid_values[i] - 2.48*np.sqrt(C_matrix[l][i][i]), fid_values[i], fid_values[i] + 2.48*np.sqrt(C_matrix[l][i][i])]))
        plt.xlim(x_min,x_max)
        plt.ylim(y_min,y_max)
        plt.xlabel(x_lab,fontsize=font_size)
        plt.ylabel(y_lab,fontsize=font_size)
    
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        for tick in ax.get_yticklabels():
            tick.set_rotation(45)
    
    if (ind_i != len(F_matrix[0])-1):
        plt.tick_params(labelbottom='off')
    if (ind_j != 0):
        plt.tick_params(labelleft='off')
    
    return ellipsePlot
       
i,j,l = 1,0,0
#Plotting the inner contour.
while i < tri_len:
    while j < i:
        while s < len(C_matrix):
            Mat_order[s] = 1./np.sqrt(np.linalg.det(np.array([[C_matrix[s][j][j],C_matrix[s][i][j]], [C_matrix[s][j][i],C_matrix[s][i][i]]])))
            s=s+1
        s=0
        SSS = Mat_order.argsort()
        Mat_order = Mat_order[SSS]
        C_matrix = C_matrix[SSS]
        couleur = couleur[SSS]
        leg = leg[SSS]
        while l < len(F_matrix):
            sq_tab_A = np.array([[C_matrix[l][j][j],C_matrix[l][i][j]], [C_matrix[l][j][i],C_matrix[l][i][i]]])
            sq_tab_B = np.array([[C_matrix[l][j][j],C_matrix[l][i][j]], [C_matrix[l][j][i],C_matrix[l][i][i]]])
            if (i != len(F_matrix[0])-1 and j!=0):
                ellipsePlot_A=plt_ell1(C_matrix[l][j][j],C_matrix[l][i][i],C_matrix[l][i][j],[fid_values[j],fid_values[i]],sq_tab_A,'0.5',couleur[l],1.52,0.8,"","",minx[j],maxx[j],minx[i],maxx[i],tri_len,tri_len,tri_len*i+(j+1),i,j)
            
            elif (i != len(F_matrix[0])-1):
                ellipsePlot_A=plt_ell1(C_matrix[l][j][j],C_matrix[l][i][i],C_matrix[l][i][j],[fid_values[j],fid_values[i]],sq_tab_A,'0.5',couleur[l],1.52,0.8,"",x[i],minx[j],maxx[j],minx[i],maxx[i],tri_len,tri_len,tri_len*i+(j+1),i,j)
            
            elif (j != 0):
                ellipsePlot_A=plt_ell1(C_matrix[l][j][j],C_matrix[l][i][i],C_matrix[l][i][j],[fid_values[j],fid_values[i]],sq_tab_A,'0.5',couleur[l],1.52,0.8,x[j],"",minx[j],maxx[j],minx[i],maxx[i],tri_len,tri_len,tri_len*i+(j+1),i,j)
        
            else:
                ellipsePlot_A=plt_ell1(C_matrix[l][j][j],C_matrix[l][i][i],C_matrix[l][i][j],[fid_values[j],fid_values[i]],sq_tab_A,'0.5',couleur[l],1.52,0.8,x[j],x[i],minx[j],maxx[j],minx[i],maxx[i],tri_len,tri_len,tri_len*i+(j+1),i,j)
                
            l=l+1
        l=0
        j=j+1
    i=i+1
    j=0

couleur = np.copy(couleur_I)
C_matrix = np.copy(C_matrix_I)
lik = np.copy(lik_I)
xax = np.copy(xax_I)

#Plotting the outer contour.
i,j,l = 1,0,0
while i < tri_len:
    while j < i:
        while s < len(C_matrix):
            Mat_order[s] = 1./np.sqrt(np.linalg.det(np.array([[C_matrix[s][j][j],C_matrix[s][i][j]], [C_matrix[s][j][i],C_matrix[s][i][i]]])))
            s=s+1
        s=0
        SSS = Mat_order.argsort()
        Mat_order = Mat_order[SSS]
        C_matrix = C_matrix[SSS]
        couleur = couleur[SSS]
        leg = leg[SSS]
        while l < len(F_matrix):
            sq_tab_A = np.array([[C_matrix[l][j][j],C_matrix[l][i][j]], [C_matrix[l][j][i],C_matrix[l][i][i]]])
            sq_tab_B = np.array([[C_matrix[l][j][j],C_matrix[l][i][j]], [C_matrix[l][j][i],C_matrix[l][i][i]]])
            if (i != len(F_matrix[0])-1 and j!=0):
                ellipsePlot_A=plt_ell2(C_matrix[l][j][j],C_matrix[l][i][i],C_matrix[l][i][j],[fid_values[j],fid_values[i]],sq_tab_A,'0.8',couleur[l],2.48,0.8,"","",minx[j],maxx[j],minx[i],maxx[i],tri_len,tri_len,tri_len*i+(j+1),i,j)
            
            elif (i != len(F_matrix[0])-1):
                ellipsePlot_A=plt_ell2(C_matrix[l][j][j],C_matrix[l][i][i],C_matrix[l][i][j],[fid_values[j],fid_values[i]],sq_tab_A,'0.8',couleur[l],2.48,0.8,"",x[i],minx[j],maxx[j],minx[i],maxx[i],tri_len,tri_len,tri_len*i+(j+1),i,j)
            
            elif (j != 0):
                ellipsePlot_A=plt_ell2(C_matrix[l][j][j],C_matrix[l][i][i],C_matrix[l][i][j],[fid_values[j],fid_values[i]],sq_tab_A,'0.8',couleur[l],2.48,0.8,x[j],"",minx[j],maxx[j],minx[i],maxx[i],tri_len,tri_len,tri_len*i+(j+1),i,j)
        
            else:
                ellipsePlot_A=plt_ell2(C_matrix[l][j][j],C_matrix[l][i][i],C_matrix[l][i][j],[fid_values[j],fid_values[i]],sq_tab_A,'0.8',couleur[l],2.48,0.8,x[j],x[i],minx[j],maxx[j],minx[i],maxx[i],tri_len,tri_len,tri_len*i+(j+1),i,j)
                
            l=l+1
        l=0
        j=j+1
    i=i+1
    j=0

lg = plt.legend(leg, bbox_to_anchor=(lgx, len(F_matrix[0])+lgy), borderpad=2, fontsize=lg_size)
lg.get_frame().set_linewidth(lg_frame_size)
lg.get_frame().set_edgecolor(lg_frame_color)
lg.get_frame().set_facecolor(lg_bckgnd)
for text in lg.get_texts():
    text.set_color(lg_text_color)

#Adjusting the contours windows size.
plt.subplots_adjust(hspace = 0.1)
plt.subplots_adjust(wspace = 0.1)
#Saving the final contours.
plt.savefig("tri_output/contours.pdf")
plt.savefig("tri_output/contours.png")

print("Plot done")