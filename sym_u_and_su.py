#####################################################################################
#
# Discovering Sparse Representations of Lie Groups
#
# Author: Roy Forestano
#
# Date of Completion: 18 February 2023
#
#####################################################################################
# Standard Imports Needed

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import os
import copy
from tqdm import tqdm
from time import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torch import linalg
from torchvision.transforms import ToTensor
from complexPyTorch.complexFunctions import complex_relu
torch.set_default_dtype(torch.float64)

plt.rcParams["font.family"] = 'sans-serif'
np.set_printoptions(formatter={'float_kind':'{:f}'.format}) 

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#####################################################################################


def run_model(n, n_dim, n_gen, n_com, eps, lr, epochs, oracle, include_sc):
    #####################################################################################
    # Initialize general set up

    # initialiaze data
    data    = torch.randn(n,n_dim,dtype=torch.cfloat).to(device) #Can also use torch.complex128 # Ceate n number of n-dim vectors
    # initialize generators
    initialize_matrices = torch.stack([ torch.randn(n_dim,n_dim,dtype=torch.cfloat) for i in range(n_gen) ], dim=0).to(device)
    # initialize structure constants
    initialize_struc_const = torch.randn(n_com,n_gen,dtype=torch.cfloat).to(device)
    # Lie Bracket or Commutator
    def bracket(M, N):
        return M@N - N@M


    #####################################################################################
    # Set up model paramters
    
    class complex_activation(nn.Module):
        def forward(self, x):
            return nn.ReLU()(x.real) + 1.j * nn.ReLU()(x.imag)
    
    # Define model
    class find_generators(nn.Module):
        def __init__(self,n_dim,n_gen,n_com):
            super(find_generators,self).__init__()

            G = [ nn.Sequential( nn.Linear(in_features = n_dim*n_dim, out_features = n_dim*n_dim, bias = True, dtype=torch.cfloat), #torch.complex128
                                 complex_activation(),
                                 nn.Linear(in_features = n_dim*n_dim, out_features = n_dim*n_dim, bias = True, dtype=torch.cfloat),
                                 complex_activation(),
                                 nn.Linear(in_features = n_dim*n_dim, out_features = n_dim*n_dim, bias = True, dtype=torch.cfloat) )  for _ in range(n_gen)]


            self.gens = nn.ModuleList(G)

            if include_sc:
                C = [ nn.Sequential( nn.Linear(in_features = n_gen, out_features = n_gen, bias = True, dtype=torch.cfloat),
                                 complex_activation(),
                                 nn.Linear(in_features = n_gen, out_features = n_gen, bias = True, dtype=torch.cfloat),
                                 complex_activation(),
                                 nn.Linear(in_features = n_gen, out_features = n_gen, bias = True, dtype=torch.cfloat) ) for _ in range(n_com) ]


                self.struct_const = nn.ModuleList(C)

            self.n_gen = n_gen
            self.n_dim = n_dim
            self.n_com = n_com

        def forward(self, x, c):
            generators = []
            for i in range(self.n_gen):
                generators.append( ( self.gens[i](x[i].flatten()) ).reshape(self.n_dim,self.n_dim)  )

            structure_constants = torch.zeros((self.n_com,self.n_gen),dtype=torch.cfloat)

            if include_sc:
                structure_constants = torch.empty((self.n_com,self.n_gen),dtype=torch.cfloat)
                for i in range(self.n_com):
                    structure_constants[i,:] = ( self.struct_const[i](c[i].flatten()) ).reshape(1,self.n_gen)

            return generators , structure_constants
    
    # Initialize Model
    model = find_generators(n_dim,n_gen,n_com).to(device)
    
    # Loss function
    def loss_fn(data,
                generators,
                struc_const,
                eps,
                ainv=1., anorm=1., aorth=1., aclos=1., asp = 1. ):

        upper_elements = int(n_dim*(n_dim-1)/2)
        lossi = 0.
        lossn = 0.
        losso = 0.
        lossc = 0.
        losssp = 0.
        comm_index = 0
        struc_const = struc_const.to(device)
        indcs_upper = np.triu_indices(n_dim)
        indices_upper_offset = np.triu_indices_from(generators[0], k=1)
        indcs_lower = np.tril_indices(n_dim)
        indices_lower_offset = np.tril_indices_from(generators[0], k=1)
        identity = torch.eye(generators[0].shape[0],dtype=torch.cfloat).to(device)

        for i,G in enumerate(generators): 
            transform = torch.transpose((identity + 1.j*eps*G)@torch.transpose(data,dim0=1,dim1=0), dim0=1, dim1=0 )
            lossi  += torch.mean( ( oracle(transform) - oracle(data) )**2)**2/ eps**2
            #lossi  += torch.mean( ( G2(transform) - G2(data) ).abs()**2 ) / eps**2
            lossn  += ((torch.view_as_real(G).flatten()**2).sum() - 2)**2 #torch.conj(G)
            lossn  += (G-G.conj().T).abs().sum()**2

            losssp += (torch.outer(G.real.flatten(),G.real.flatten())**2 - torch.eye(G.real.flatten().shape[0])*torch.outer(G.real.flatten(),G.real.flatten())**2).sum()**2
            losssp += (torch.outer(G.imag.flatten(),G.imag.flatten())**2 - torch.eye(G.real.flatten().shape[0])*torch.outer(G.imag.flatten(),G.imag.flatten())**2).sum()**2

            losssp += (torch.outer(G[indcs_upper].abs(),G[indcs_upper].abs())**2 - torch.eye(G[indcs_upper].shape[0],dtype=torch.cfloat).abs()*torch.outer(G[indcs_upper].abs(),G[indcs_upper].abs())**2).sum()**2
            losssp += (torch.outer(G[indcs_lower].abs(),G[indcs_lower].abs())**2 - torch.eye(G[indcs_lower].shape[0],dtype=torch.cfloat).abs()*torch.outer(G[indcs_lower].abs(),G[indcs_lower].abs())**2).sum()**2
            losssp += (torch.outer(G.real.flatten(),G.imag.flatten())**2).sum()**2

            losso += ((G@G).trace().abs()-2)**2
            for j, H in enumerate(generators):
                if i < j:
                    losso += (G@H).trace().abs()**2
                    losso += (G*H).sum().abs()**2

                    C1 = torch.view_as_real(bracket(G,H))
                    C2 = 0
                    for k,K in enumerate(generators):
                        C2 += struc_const[comm_index,k]*K
                    C = C1-torch.view_as_real(C2)
                    lossc += torch.sum(C**2)**2
                    comm_index +=1

        components = [ ainv*lossi,  
                    anorm*lossn,  
                    aorth*losso,  
                    aclos*lossc,
                    asp*losssp ]

        L = ainv*lossi + anorm*lossn + aorth*losso + aclos*lossc + asp*losssp
        return  L.to(device), components
    
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training function
    def train(initial_matrices, 
              initial_struc_const,  
              data, 
              model, 
              loss_fn, 
              epochs, 
              optimizer, 
              eps):

        history = {'train_loss': [],
                   'components_loss':[]} 

        best_val_loss = float('inf') #torch.inf #float('inf')
        start = time()

        ainv  = 1.
        anorm = 1.
        aorth = 10.
        aclos = 0.
        if include_sc:
            aclos = 10.
        asp   = 1e-1


        X = initial_matrices.to(device)
        Y = initial_struc_const.to(device)
        size = X.shape[0]

        for i in range(epochs):
            train_loss = 0.
            model.train()
            gens, struc_const = model(X,Y)

            loss, comp_loss = loss_fn( data         = data,
                            generators   = gens,
                            struc_const  = struc_const,
                            eps          = eps,
                            ainv         = ainv,
                            anorm        = anorm,
                            aorth        = aorth,
                            aclos        = aclos,
                            asp          = asp )

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()
            comp_loss_for_epoch = []

            for j in range(len(comp_loss)):
                if torch.is_tensor(comp_loss[j]):
                    comp_loss_for_epoch.append(comp_loss[j].data.item())
                else:
                    comp_loss_for_epoch.append(comp_loss[j])

            history['train_loss'].append(train_loss)
            history['components_loss'].append(comp_loss_for_epoch)

            if i%1==0:
                print(f"Epoch {i+1}   |  Train Loss: {train_loss}",end='\r') #{train_loss:>8f}
            if i==epochs-1:
                print(f"Epoch {i+1}   |  Train Loss: {train_loss}")

            if train_loss*1e25 < 1:
                print()
                print('Reached Near Machine Zero')
                break

            if train_loss < best_val_loss:
                best_val_loss = train_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                best_model = copy.deepcopy(model)
                torch.save(model.state_dict(),'best_complex_U6.pth')

        model = best_model   
        end = time()
        total_time = end-start
        print(f'Total Time: {total_time:>.8f}')
        print("Complete.")
        return {'history': history}
    
    

    training = train( initial_matrices    = initialize_matrices, 
                      initial_struc_const = initialize_struc_const,
                      data                = data,
                      model               = model, 
                      loss_fn             = loss_fn,
                      epochs              = epochs,
                      optimizer           = optimizer,
                      eps                 = eps  )
                
    if n_gen>1:
        train_loss = np.array(training['history']['train_loss'])
        comp_loss = np.array(training['history']['components_loss'])
    else:
        train_loss = np.array(training['history']['train_loss'])
        comp_loss = np.empty( ( train_loss.shape[0],len(training['history']['components_loss']) ) )
        for i,comp in enumerate(training['history']['components_loss']):
            for j,term in enumerate(comp):
                if torch.is_tensor(term) and term.requires_grad:
                    comp_loss[i,j] = term.detach().numpy()
                else:
                    comp_loss[i,j] = term

    N=train_loss.shape[0]
    plt.figure(figsize=(6,4)) #, dpi=100)
    plt.plot( train_loss[:N], linewidth=1, linestyle='-',  color = 'r', label='Total')
    plt.plot(comp_loss[:N,0], linewidth=1, linestyle=':',  color='b',   label='Invariance')
    plt.plot(comp_loss[:N,1], linewidth=1, linestyle='--', color='g',   label='Normalization')
    plt.plot(comp_loss[:N,2], linewidth=1, linestyle='-.', color='magenta', label='Orthogonality')
    plt.plot(comp_loss[:N,3], linewidth=1, linestyle='-.', color='cyan', label='Closure')
    plt.plot(comp_loss[:N,4], linewidth=1, linestyle='--', color='black', label='Sparsity')
    plt.legend()

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Components of Loss')

    plt.show()
    
    # Evaluate Model
    model.eval()

    with torch.no_grad():
        gens_pred, struc_pred = model(initialize_matrices,initialize_struc_const)
        
    return gens_pred, struc_pred


#####################################################################################
# Visualize Generators

def visualize_generators(figsize, n_dim, n_gen, eps, gens_pred, rows, cols):
    # Create labels for matrix rows and columns
    ticks_gen_im =[]
    ticks_gen_im_label = []
    for i in range(n_dim):
        ticks_gen_im.append(i)
        ticks_gen_im_label.append(str(i+1))

    fig,axes = plt.subplots(rows,cols,figsize=figsize)
    for i,GEN in enumerate(gens_pred):
        plt.subplot(rows,cols,i*2+1)
        #if n_gen<10:
        #    print(f'Generator {i+1}: \n {GEN} \n')
        im = plt.imshow(GEN.real.detach().numpy(), cmap='RdBu', vmin=-np.sqrt(2), vmax=np.sqrt(2)) # use ax_GEN[0] with axes
        if i<9:
            plt.ylabel(r'$\mathbb{J}$'+f'$_{i+1}$',fontsize=20)
        else:
            number = str(i+1)
            plt.ylabel(r'$\mathbb{J}$'+f'$_{number[0]}$'+f'$_{number[1]}$',fontsize=20)
        if i*2<cols:
            plt.title('$\mathfrak{R}$',fontsize=20)
        if rows>1:
            if i*2>=(rows-1)*cols:
                plt.xticks(ticks=ticks_gen_im, labels=ticks_gen_im_label)
            else:
                plt.xticks([])
        else:
            plt.xticks(ticks=ticks_gen_im, labels=ticks_gen_im_label)
            
        if i%(cols/2)==0:
            plt.yticks(ticks=ticks_gen_im, labels=ticks_gen_im_label)
        else:
            plt.yticks([])
            
            
        plt.subplot(rows,cols,i*2+2)
        im = plt.imshow(GEN.imag.detach().numpy(), cmap='RdBu', vmin=-np.sqrt(2), vmax=np.sqrt(2))
        if i*2<cols:
            plt.title('$\mathfrak{I}$',fontsize=20)
#         if n_gen<7:
#             det = np.linalg.det(np.eye(GEN.shape[0]) + eps * GEN.detach().numpy()) #ax_GEN[1]
#             plt.title(f'det = {det}')
#         plt.axis('off')
        if rows>1:
            if i*2>=(rows-1)*cols:
                plt.xticks(ticks=ticks_gen_im, labels=ticks_gen_im_label)
            else:   #(rows*cols/2-rows-1):
                plt.xticks([])
        else:
            plt.xticks(ticks=ticks_gen_im, labels=ticks_gen_im_label)
        plt.yticks([])

    if (len(gens_pred)%2==1 and n_dim%2==0) or (len(gens_pred)%2==0 and n_dim%2==1):
        plt.subplot(rows,cols,len(gens_pred)*2+1)
        plt.axis('off')
        plt.subplot(rows,cols,len(gens_pred)*2+2)
        plt.axis('off')
            
        plt.subplots_adjust(right=0.8)
    cbar = fig.colorbar(im,ax=axes.ravel().tolist(), ticks=[-np.sqrt(2), 0, np.sqrt(2)])
    cbar.ax.set_yticklabels(['-$\sqrt{2}$', '0', '$\sqrt{2}$']) 

            
#####################################################################################
# Visualize Structure Constants
        
def visualize_structure_constants(figsize, n_gen, n_com, struc_pred):
    if n_gen==3:
        X = torch.tensor(struc_pred.numpy())
        struc_cyclic = X
        struc_cyclic[1] = -X[1]
        
    commutator_labels = []
    if n_com==3:
        # Make the commutations cyclic for 3 generators
        for i in range(n_gen):
             for j in range(n_gen):
                    if i<j:
                        if (j-i)==2:
                            commutator_labels.append(str(j+1)+str(i+1))
                        else:
                            commutator_labels.append(str(i+1)+str(j+1))
    else:
        for i in range(n_gen):
            for j in range(n_gen):
                if i<j:
                    if n_gen>9:
                        if i<9 and j<9:
                            commutator_labels.append(str(0)+str(i+1)+str(0)+str(j+1))
                        elif i<9 and j>=9:
                            commutator_labels.append(str(0)+str(i+1)+str(j+1))
                        else:
                            commutator_labels.append(str(i+1)+str(j+1))
                    else:
                        commutator_labels.append(str(i+1)+str(j+1))
        
    ticks_com = []
    for i in range(n_com):
        ticks_com.append(i)

    ticks_gen = []
    generator_labels = []
    for i in range(n_gen):
        ticks_gen.append(i)
        generator_labels.append(str(i+1))
    
    fig,axes = plt.subplots(1,2,figsize=figsize)
    if n_com==3:
        plt.subplot(1,2,1)
        plt.imshow(struc_cyclic.real.detach().numpy(), cmap='RdBu', vmin=-2.,vmax=2.)#norm=mpl.colors.CenteredNorm())
        plt.xticks(ticks=ticks_gen,labels=generator_labels)
        plt.xlabel('$\mathbb{J}_\gamma$',fontsize=20)
        plt.yticks(ticks=ticks_com,labels=commutator_labels)
        plt.ylabel(r'$ [\mathbb{J}_\alpha$,$\mathbb{J}_\beta ] $',fontsize=20)
        plt.title(r'$\mathfrak{R}(a_{[\alpha\beta]\gamma})$',fontsize=20)
        
        plt.subplot(1,2,2)
        im = plt.imshow(struc_cyclic.imag.detach().numpy(), cmap='RdBu', vmin=-2.,vmax=2.)
        plt.xticks(ticks=ticks_gen,labels=generator_labels)
        plt.xlabel('$\mathbb{J}_\gamma$',fontsize=20)
        plt.yticks([])
        plt.title(r'$\mathfrak{I}(a_{[\alpha\beta]\gamma})$',fontsize=20)
        
    else:
        plt.subplot(1,2,1)
        plt.imshow(struc_pred.real.detach().numpy(), cmap='RdBu', vmin=-2.,vmax=2.)#norm=mpl.colors.CenteredNorm())
        plt.xticks(ticks=ticks_gen,labels=generator_labels)
        plt.xlabel('$\mathbb{J}_\gamma$',fontsize=20)
        plt.yticks(ticks=ticks_com,labels=commutator_labels)
        plt.ylabel(r'$ [\mathbb{J}_\alpha$,$\mathbb{J}_\beta ] $',fontsize=20)
        plt.title(r'$\mathfrak{R}(a_{[\alpha\beta]\gamma})$',fontsize=20)
        
        plt.subplot(1,2,2)
        im = plt.imshow(struc_pred.imag.detach().numpy(), cmap='RdBu', vmin=-2.,vmax=2.)
        plt.xticks(ticks=ticks_gen,labels=generator_labels)
        plt.xlabel('$\mathbb{J}_\gamma$',fontsize=20)
        plt.yticks([])
        plt.title(r'$\mathfrak{I}(a_{[\alpha\beta]\gamma})$',fontsize=20)
        
    plt.colorbar(im, ax=axes.ravel().tolist())
    
    # add grid lines
    # for i in range(n_gen-1):
    #     plt.axvline(x=1/2+i, linewidth=1, color ='black')
    # for i in range(n_com-1):
    #     plt.axhline(y=1/2+i-0.01, linewidth=1, color ='black')


#####################################################################################
# Verify Commutations with Structure Constants

def verify_struc_constants(n_gen, struc_pred, gens_pred):
    # Lie Bracket or Commutator
    def bracket(A, B):
        return A @ B - B @ A
    
    if n_gen==3:
        X = torch.tensor(struc_pred.numpy())
        struc_cyclic = X
        struc_cyclic[1] = -X[1]

    comm_index = 0
    Cs = []
    for i,G in enumerate(gens_pred):
        for j,H in enumerate(gens_pred):
            if i<j and n_gen!=3:
                C1 = bracket(G,H)
                C2 = 0
                for k,K in enumerate(gens_pred):
                    C2 += struc_pred[comm_index,k]*K
                C = C1 - C2
                error = torch.mean(torch.abs(C))
                print(str(i+1)+str(j+1)+': \n Structure Constants = '+str(struc_pred[comm_index,:].detach().numpy())+'\n \n C = \n ',C.detach().numpy(),'\n')
                if error<1e-1:
                    print(f'The structure constants were found successfully with a mean absolute error (MAE) of {error}. \n \n')
                elif error>1e-1:
                    print(f'The structure constants were NOT found successfully with a mean absolute error (MAE) of {error}. \n \n')
                Cs.append(C)
                comm_index+=1
            # Make the cyclic commutators if n_gen = 3   
            elif i<j and n_gen==3:
                if (j-i)==2:
                    C1 = bracket(H,G)
                    C2 = 0
                    for k,K in enumerate(gens_pred):
                        C2 += struc_cyclic[comm_index,k]*K
                    C = C1 - C2
                    error = torch.mean(torch.abs(C))
                    print(str(j+1)+str(i+1)+': \n Structure Constants = '+str(struc_cyclic[comm_index,:].detach().numpy())+'\n \n C = \n ',C.detach().numpy(),'\n')
                    if error<1e-1:
                        print(f'The structure constants were found successfully with a mean absolute error (MAE) of {error}. \n \n')
                    elif error>1e-1:
                        print(f'The structure constants were NOT found successfully with a mean absolute error (MAE) of {error}. \n \n') 
                    Cs.append(C)
                    comm_index+=1
                else:
                    C1 = bracket(G,H)
                    C2 = 0
                    for k,K in enumerate(gens_pred):
                        C2 += struc_cyclic[comm_index,k]*K
                    C = C1 - C2
                    error = torch.mean(torch.abs(C))
                    print(str(i+1)+str(j+1)+': \n Structure Constants = '+str(struc_cyclic[comm_index,:].detach().numpy())+'\n \n C = \n ',C.detach().numpy(),'\n')
                    if error<1e-1:
                        print(f'The structure constants were found successfully with a mean absolute error (MAE) of {error}. \n \n')
                    elif error>1e-1:
                        print(f'The structure constants were NOT found successfully with a mean absolute error (MAE) of {error}. \n \n') 
                    Cs.append(C)
                    comm_index+=1
    
    
    # Calculate the total MSE in finding the structure constants
    tot_error = 0.
    for i,C in enumerate(Cs):
        tot_error+=torch.mean(torch.abs(C))
    print(f'Total MAE = {tot_error}')
    # if error < 1e-1:
    #     print(f'The structure constants were found successfully with a mean absolute error (MAE) of {error}.')
    # else:
    #     print(f'The structure constants were NOT found successfully with a mean absolute error (MAE) of {error}.')




#####################################################################################
# Verify Orthogonality


def verify_orthogonality(gens_pred):
    def get_angle(v, w):
        # Angle between vectors
        return np.arccos( (v @ w.conj()).real / ( torch.norm(v.abs()) * torch.norm(w.abs()) ) )

    def get_axis(M):
        # Finds the eigenvector with min(Imaginary(eigenvalue))
        # if the matrix is a rotation matrix or a generator of rotation,s then this vector is the axis of rotation  
        eig_vals, eig_vecs = torch.linalg.eig(M)
        # find the minimum arg of the minimum imaginary component
        # pass that to the transposed eigenvector array to pull the eigenvector
        axis = eig_vecs.T[torch.argmin(torch.abs(eig_vals.imag))]
        # Change to more positive than negative values in axis vector by multiplying by the net sign
        return torch.sign(torch.sum(axis).real)*axis
    
    for i,G in enumerate(gens_pred):
        for j,H in enumerate(gens_pred):
            if i<j:
                angle = get_angle(get_axis(G), get_axis(H))
                angle_deg = 180/np.pi*float(angle)
                print(f'Angle between generator {i+1} and {j+1}: {angle:>.10f} rad, {angle_deg:>.10f} deg')