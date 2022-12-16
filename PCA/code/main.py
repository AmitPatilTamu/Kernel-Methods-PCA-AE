import torch
from helper import load_data
from solution import PCA, AE, frobeniu_norm_error
import numpy as np
import os


def test_pca(A, p):
    pca = PCA(A, p)
    Ap, G = pca.get_reduced()
    A_re = pca.reconstruction(Ap)
    error = frobeniu_norm_error(A, A_re)
    print('PCA-Reconstruction error for {k} components is'.format(k=p), error)
    return G

def test_ae(A, p):
    model = AE(d_hidden_rep=p)
    model.train(A, A, 128, 300)
    A_re = model.reconstruction(A)
    final_w = model.get_params()
    error = frobeniu_norm_error(A, A_re)
    print('AE-Reconstruction error for {k}-dimensional hidden representation is'.format(k=p), error)
    return final_w

if __name__ == '__main__':
    dataloc = "../data/USPS.mat"
    A = load_data(dataloc)
    A = A.T
    ## Normalize A
    A = A/A.max()

    ### YOUR CODE HERE
    # Note: You are free to modify your code here for debugging and justifying your ideas for 5(f)

    #we need to normalize A by subtracting mean for PCA for minimum recontruction error as per notes, we can use same A_norm
    #for both AE and PCA for proper comparision
    size = A.shape[1]
    A_norm = (A - ((1/size)*A@np.ones((size,1)))@np.ones((1,size)))

    ps = [32, 64, 128]
    #ps = [64]
    for p in ps:
        G = test_pca(A_norm, p)
        final_w = test_ae(A_norm, p)

    #for relation between W and G
    ps = [32, 64, 128]
    for p in ps:
        G = test_pca(A_norm, p)
        W = test_ae(A_norm, p)
        print("G-W", frobeniu_norm_error(G, W))
        M = G.T @ W
        U, S, V = np.linalg.svd(M, full_matrices=True)
        Mp = U @ np.eye(U.shape[0], V.shape[0]) @ V
        Gp = G @ Mp
        print("Gp-W", frobeniu_norm_error(Gp, W))
        print('M-Mp', frobeniu_norm_error(M, Mp))
        A_re = Gp @ Gp.T @ A_norm
        print("A -A_Re:", frobeniu_norm_error(A_norm, A_re))

    ### END YOUR CODE 
