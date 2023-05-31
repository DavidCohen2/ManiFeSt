# -*- coding: utf-8 -*-
"""
Created on 

@author:
"""

# import sys

import scipy.io

import numpy as np

from sklearn.metrics.pairwise import euclidean_distances


def construct_kernel(X,y,kernel_scale_factor):

    label = list(set(y))
    x_1= X[np.where(y==label[0])]
    x_2= X[np.where(y==label[1])]
    
    K1_dis=euclidean_distances(np.transpose(x_1))                    
    K2_dis=euclidean_distances(np.transpose(x_2))      
    
    epsilon1 = kernel_scale_factor*np.median(K1_dis[~np.eye(K1_dis.shape[0],dtype=bool)])
    epsilon2 = kernel_scale_factor*np.median(K2_dis[~np.eye(K2_dis.shape[0],dtype=bool)])
        
    K1 = np.exp(-(K1_dis**2)/(2*epsilon1**2))     
    K2 = np.exp(-(K2_dis**2)/(2*epsilon2**2)) 
    
    return K1,K2


def calc_tol(matrix,var_type='float64',energy_tol = 0 ):     
        tol = np.max(matrix) * len(matrix) * np.core.finfo(var_type).eps
        tol2 = np.sqrt(np.sum(matrix**2)*energy_tol)
        tol = np.max([tol,tol2])
        
        return tol

def spsd_geodesics(G1,G2,p=0.5,r=None,eigVecG1=None,eigValG1=None,eigVecG2=None,eigValG2=None):
                    
    if eigVecG1 is None:
        eigValG1,eigVecG1 =np.linalg.eigh(G1)    
    if eigVecG2 is None:
        eigValG2,eigVecG2 =np.linalg.eigh(G2)   
    
    if r is None:
        tol = calc_tol(eigValG1)
        rank_G1 = len(np.abs(eigValG1)[np.abs(eigValG1)>tol])
        
        tol = calc_tol(eigValG2)
        rank_G2 = len(np.abs(eigValG2)[np.abs(eigValG2)>tol])

        r = min(rank_G1,rank_G2)  
   
    maxIndciesG1 = np.flip(np.argsort(np.abs(eigValG1))[-r:],[0])
    V1 = eigVecG1[:,maxIndciesG1]
    lambda1 = eigValG1[maxIndciesG1]
    
    maxIndciesG2 = np.flip(np.argsort(np.abs(eigValG2))[-r:],[0])
    V2 = eigVecG2[:,maxIndciesG2]
    lambda2 = eigValG2[maxIndciesG2]
    
    #lapack_driver='gesvd' is more stable while lapack_driver='gesdd' is more fast        
    try:
        O2,sigma,O1T =np.linalg.svd(V2.T@V1)   
    except:
        O2,sigma,O1T =scipy.linalg.svd(V2.T@V1,lapack_driver='gesvd')   
                 
    O1 = O1T.T
    
    sigma[sigma<-1] =-1
    sigma[sigma>1] =1
    theta = np.arccos(sigma)
   
    U1 = V1@O1
    R1 = O1.T@np.diag(lambda1)@O1
    
    U2 = V2@O2
    R2 = O2.T@np.diag(lambda2)@O2
    
    tol = calc_tol(sigma)
    valid_ind = np.where(np.abs(sigma-1)>tol)
    pinv_sin_theta = np.zeros(theta.shape)    
    pinv_sin_theta[valid_ind]=1/np.sin(theta[valid_ind])
    
    UG1G2 = U1@np.diag(np.cos(theta*p))+(np.eye(G1.shape[0])-U1@U1.T)@U2@np.diag(pinv_sin_theta)@np.diag(np.sin(theta*p))
  
    return UG1G2,R1,R2,O1,lambda1


def get_operators(K1,K2,use_spsd=True):
          
    if  use_spsd:
        eigValK1,eigVecK1 =np.linalg.eigh(K1)
        tol = calc_tol(eigValK1)
        rank_K1 = len(np.abs(eigValK1)[np.abs(eigValK1)>tol])
        
        eigValK2,eigVecK2 =np.linalg.eigh(K2)
        tol = calc_tol(eigValK2)
        rank_K2 = len(np.abs(eigValK2)[np.abs(eigValK2)>tol])
                  
        # create SPSD Mean operator M
        min_rank = min(rank_K1,rank_K2)        
        UK1K2,RK1,RK2,OK1,lambdaK1 = spsd_geodesics(K1,K2,p=0.5,r=min_rank,eigVecG1=eigVecK1,eigValG1=eigValK1,eigVecG2=eigVecK2,eigValG2=eigValK2)
        
        RK1PowerHalf = OK1.T@np.diag(np.sqrt(lambdaK1))@OK1
        RK1PowerMinusHalf = OK1.T@np.diag(1/np.sqrt(lambdaK1))@OK1        
        e, v = np.linalg.eigh(RK1PowerMinusHalf@RK2@RK1PowerMinusHalf)
        e[e<0]=0
        RK1K2 = RK1PowerHalf@v@np.diag(np.sqrt(e))@v.T@RK1PowerHalf
                
        M = UK1K2@RK1K2@UK1K2.T

        eigValM,eigVecM =np.linalg.eigh(M)
        tol = calc_tol(eigValM)
        rank_M = len(np.abs(eigValM)[np.abs(eigValM)>tol])
            
        # create SPSD Difference operator D
        min_rank = min(rank_K1,rank_M)     
        UMK1,RM,RK1,OM,lambdaM = spsd_geodesics(M,K1,p=1,r=min_rank,eigVecG1=eigVecM,eigValG1=eigValM,eigVecG2=eigVecK1,eigValG2=eigValK1)
        
        RMPowerHalf = OM.T@np.diag(np.sqrt(lambdaM))@OM
        RMPowerMinusHalf = OM.T@np.diag(1/np.sqrt(lambdaM))@OM      
        e,v = np.linalg.eigh(RMPowerMinusHalf@RK1@RMPowerMinusHalf)
        tol = calc_tol(e)
        e[e<tol]=tol
        logarithmic = RMPowerHalf@v@np.diag(np.log(e))@v.T@RMPowerHalf
            
        D= UMK1@logarithmic@UMK1.T
           
    else:   #SPD form
        # create SPD Mean operator M
        K1 = K1 +np.eye(K1.shape[0])*np.core.finfo('float64').eps*2
        K2 = K2 +np.eye(K2.shape[0])*np.core.finfo('float64').eps*2
		
        eigValK1,eigVecK1 =np.linalg.eigh(K1)
        tol = calc_tol(eigValK1)
        rank_K1 = len(np.abs(eigValK1)[np.abs(eigValK1)>tol])
        
        K1PowerHalf =eigVecK1@np.diag(np.sqrt(eigValK1))@eigVecK1.T
        K1PowerMinusHalf =eigVecK1@np.diag(1/np.sqrt(eigValK1))@eigVecK1.T      
        e, v = np.linalg.eigh(K1PowerMinusHalf@K2@K1PowerMinusHalf)
        e[e<0]=0
        M = K1PowerHalf@v@np.diag(np.sqrt(e))@v.T@K1PowerHalf
                
        eigValM,eigVecM =np.linalg.eigh(M)
        tol = calc_tol(eigValM)
        rank_M = len(np.abs(eigValM)[np.abs(eigValM)>tol])
        
        # create SPD Difference operator D
        MPowerHalf =eigVecM@np.diag(np.sqrt(eigValM))@eigVecM.T
        MPowerMinusHalf =eigVecM@np.diag(1/np.sqrt(eigValM))@eigVecM.T      
        e,v = np.linalg.eigh(MPowerMinusHalf@K1@MPowerMinusHalf)
        tol = calc_tol(e)
        e[e<tol]=tol
        D = MPowerHalf@v@np.diag(np.log(e))@v.T@MPowerHalf
        
    return M,D
    
def compute_manifest_score(D):

    eigValD,eigVecD =np.linalg.eigh(D)
            
    eigVec_norm = eigVecD**2
    score = eigVec_norm@np.abs(eigValD)
            
    return score

          
def ManiFeSt(X,y,kernel_scale_factor=1,use_spsd=True):
    
    K1,K2 = construct_kernel(X,y,kernel_scale_factor)
    
    M,D = get_operators(K1,K2,use_spsd=use_spsd)
    
    score = compute_manifest_score(D)
    idx = np.argsort(score, 0)[::-1]    
    
    ##eig_vecs
    eigValM,eigVecM =np.linalg.eigh(M)
    eigValD,eigVecD =np.linalg.eigh(D)
        
    sorted_indexes = np.argsort(np.abs(eigValM))[::-1]
    eigVecM = eigVecM[:,sorted_indexes]
    eigValM = eigValM[sorted_indexes]    
    sorted_indexes = np.argsort(np.abs(eigValD))[::-1]
    eigVecD = eigVecD[:,sorted_indexes]
    eigValD = eigValD[sorted_indexes]    
        
    eig_vecs = (eigVecD,eigValD,eigVecM,eigValM)
    
    return score, idx, eig_vecs
    
    
    
    
   
 