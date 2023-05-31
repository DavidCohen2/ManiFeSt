# -*- coding: utf-8 -*-
"""
Created on 

@author: 
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras.datasets import mnist
from sklearn.model_selection  import train_test_split


from ManiFeSt import ManiFeSt


#%%  ManiFeSt Score              
        
# General Params
random_state=40
n_samples_each_class =2000

#load MNIST dataset 4 and 9 digits
(X1, y1), (X2, y2) = mnist.load_data()
(X,y)=(np.concatenate((X1,X2)),np.concatenate((y1,y2)))

#extract 4 and 9 digits
X=np.concatenate((X[(y==4),:][:n_samples_each_class,:,:],X[(y==9),:][:n_samples_each_class,:,:]))
y=np.concatenate((y[(y==4)][:n_samples_each_class],y[(y==9)][:n_samples_each_class]))
X=X.reshape(X.shape[0],-1)/255

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify = y, random_state=random_state)

# ManiFeSt Score
use_spsd = True  #False - use SPD form  - default is SPSD, MNIST is SPSD since there are blank pixels
kernel_scale_factor = 1 # The RBF kernel scale is typically set to the median of the Euclidean distances up to some scalar defiend by kernel_scale_factor ,  default value 1
score, idx, eig_vecs = ManiFeSt(X_train,y_train,kernel_scale_factor=kernel_scale_factor,use_spsd=use_spsd)  #use_spsd=use_spsd


#%% Plot Score
label = list(set(y_train))
x_train_9= X_train[np.where(y_train==9)]
x_train_4= X_train[np.where(y_train==4)]
(eigVecD,eigValD,eigVecM,eigValM) = eig_vecs 

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

fig = plt.figure(figsize=(6.75, 3.53),constrained_layout=False, facecolor='0.9',dpi=500)
gs = fig.add_gridspec(nrows=22, ncols=42, left=0, right=1,top=1, bottom=0,hspace=0, wspace=0)

# plot samples from each class
ax =  fig.add_subplot(gs[1:6,0:5])
ax.set_xticklabels([])
ax.set_yticklabels([])  
ax.set_xticks([])
ax.set_yticks([])         
ax.set_title(r'Class 1 - $\boldsymbol{X}^{(\text{1})}$',fontsize=7,y=0.92)

inner_grid = gridspec.GridSpecFromSubplotSpec(4, 4,
        subplot_spec=gs[1:6,0:5], wspace=0.0, hspace=0.0)
for j in range(16):            
    ax = plt.Subplot(fig, inner_grid[j])
    im = ax.imshow(abs(x_train_9[j,:].reshape((28,28))),cmap=plt.get_cmap('gray'))        
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)

ax =  fig.add_subplot(gs[17:22,0:5])
ax.set_xticklabels([])
ax.set_yticklabels([])  
ax.set_xticks([])
ax.set_yticks([])         
ax.set_title(r'Class 2 - $\boldsymbol{X}^{(\text{2})}$',fontsize=7,y=0.92)    
    
inner_grid = gridspec.GridSpecFromSubplotSpec(4, 4,
        subplot_spec=gs[17:22,0:5], wspace=0.0, hspace=0.0)
for j in range(16):            
    ax = plt.Subplot(fig, inner_grid[j])
    im = ax.imshow(abs(x_train_4[j,:].reshape((28,28))),cmap=plt.get_cmap('gray'))        
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)
    

# plot eigenvectors Mean operator M
ax =  fig.add_subplot(gs[7:11,0:4])
im = ax.imshow(abs(eigVecM[:,0].reshape((28,28))))
ax.set_xticklabels([])
ax.set_yticklabels([])  
ax.set_xticks([])
ax.set_yticks([])         
ax.set_title(r'$\boldsymbol{\phi}^{(\boldsymbol{M})}_\text{1}$',fontsize=7,y=0.94)    

ax =  fig.add_subplot(gs[12:16,0:4])
im = ax.imshow(abs(eigVecM[:,1].reshape((28,28))))
ax.set_xticklabels([])
ax.set_yticklabels([])           
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'$\boldsymbol{\phi}^{(\boldsymbol{M})}_\text{2}$',fontsize=7,y=0.94)    


# plot eigenvectors of Difference operator D
ax =  fig.add_subplot(gs[2:6,18:22])
im = ax.imshow(abs(eigVecD[:,0].reshape((28,28))))
ax.set_xticklabels([])
ax.set_yticklabels([])  
ax.set_xticks([])
ax.set_yticks([])         
ax.set_title(r'$\boldsymbol{\phi}^{(\boldsymbol{D})}_\text{1}$',fontsize=7,y=0.94)    

ax =  fig.add_subplot(gs[7:11,18:22])
im = ax.imshow(abs(eigVecD[:,1].reshape((28,28))))
ax.set_xticklabels([])
ax.set_yticklabels([])           
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'$\boldsymbol{\phi}^{(\boldsymbol{D})}_\text{2}$',fontsize=7,y=0.94)    

ax =  fig.add_subplot(gs[12:16,18:22])
im = ax.imshow(abs(eigVecD[:,2].reshape((28,28))))
ax.set_xticklabels([])
ax.set_yticklabels([])           
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'$\boldsymbol{\phi}^{(\boldsymbol{D} )}_\text{3}$',fontsize=7,y=0.94)    

ax =  fig.add_subplot(gs[17:21,18:22])
im = ax.imshow(abs(eigVecD[:,3].reshape((28,28))))
ax.set_xticklabels([])
ax.set_yticklabels([])           
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'$\boldsymbol{\phi}^{(\boldsymbol{D})}_\text{4}$',fontsize=7,y=0.94)    


#plot ManiFeSt Score
ax =  fig.add_subplot(gs[7:16,33:42])
im = ax.imshow(abs(score.reshape((28,28))))
ax.set_xticklabels([])
ax.set_yticklabels([])           
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'ManiFeSt Score - $\boldsymbol{r}$',fontsize=12,y=0.97)    

plt.show()      

plt.rc('text', usetex=False)          
   
            