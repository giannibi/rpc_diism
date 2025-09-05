import matplotlib.pyplot as plt
from itertools import product
import slycot as sly
import numpy as np
import control as ct
import cvxpy as cp


def weightS(wb, M, e, n):
    '''
    Generate a typical sensitivity weight function
    We = weightS(wb,M,e)
    wb - design frequency (where |We| is approximately 1)
    M - high frequency gain of 1/We; should be > 1
    e - low frequency gain of 1/We; should be < 1
    n - order of the weight
    '''
    s = ct.tf('s')
    w1 = (s/pow(M,(1/n)) + wb) / (s + wb*pow(e,1/n))
    w = ct.ss([],[],[],1)
    for i in range(n):
        w = w * w1
    return w
    

def weightU(wbc, M, e, n):
    '''
    Generate a typical input sensitivity weight function
    Wu = weightU(wbc,M,e)
    wbc - design frequency (where |Wu| is approximately 1)
    M - low frequency gain of 1/Wu; should be > 1
    e - high frequency gain of 1/Wu; should be < 1
    n - order of the weight
    '''
    s = ct.tf('s')
    w1 = (s + wbc/pow(M,(1/n))) / (pow(e,1/n)*s + wbc)
    w = ct.ss([],[],[],1)
    for i in range(n):
        w = w * w1
    return w


def sigma(g, w):
    '''
    Custom function to compute singular values of a system given frequencies
    s = mysigma(g,w)
    g - LTI object, order n
    w - frequencies, length m
    '''
    m, p, _ = g.frequency_response(w)
    sjw = (m*np.exp(1j*p)).transpose(2, 0, 1)
    sv = np.linalg.svd(sjw, compute_uv=False)
    return sv


def invss(d):
    '''
    Compute the inverse of a state space system with nonsingular D matrix
    Used to compute D^-1 in D-K iteration
    '''
    assert d.D.shape[0] == d.D.shape[1], "D matrix must be square"
    assert np.linalg.matrix_rank(d.D) == d.D.shape[0], "D matrix must be nonsingular"
    dinv = np.linalg.inv(d.D)
    ainv = d.A - d.B @ dinv @ d.C
    binv = d.B @ dinv
    cinv = -dinv @ d.C
    return ct.StateSpace(ainv, binv, cinv, dinv)


def hinfsyn(G, nmeas, ncon, initgamma=1e6):
    '''
    Modified version of H_{inf} control synthesis included in python control

    Parameters
    ----------
    G: Plant in LFT form, from [w;u] to [z;y] (State-space sys)
    nmeas: number of measurements y (input to controller)
    ncon: number of control inputs u (output from controller)
    initgamma: initial gamma for optimization

    Returns
    -------
    K: controller 
    CL: closed loop system 
    gam: infinity norm of closed loop system
    rcond: 4-vector, reciprocal condition estimates of:
        1: control transformation matrix
        2: measurement transformation matrix
        3: X-Riccati equation
        4: Y-Riccati equation
    '''
    n = np.size(G.A, 0)
    m = np.size(G.B, 1)
    np_ = np.size(G.C, 0)
    # Call SLICOT routine
    out = sly.sb10ad(n, m, np_, ncon, nmeas, initgamma, G.A, G.B, G.C, G.D)
    gam = out[0]
    Ak = out[1]
    Bk = out[2]
    Ck = out[3]
    Dk = out[4]
    Ac = out[5]
    Bc = out[6]
    Cc = out[7]
    Dc = out[8]
    rcond = out[9]
    K = ct.StateSpace(Ak, Bk, Ck, Dk)
    CL = ct.StateSpace(Ac, Bc, Cc, Dc)
    return K, CL, gam, rcond


def mucomp(M, nblock, itype, omega):
    '''
    Computation of the upper bound of the structured singular value of system M,
    given the uncertainty structure. Returns the upper bound as a function of frequency
    and its maximum over frequency (nubar)

    mubar, nubar = mucomp(M, nblock, itype, omega)

    M:      LFT representation of the system "seen" from the uncertain block, i.e., from w_delta to z_delta
    nblock: uncertainty structure (vector of sizes of uncertain blocks)
    itype:  must be 2 for all nblock entries
    omega:  frequency vector to evaluate mu upper bound
    mubar:  mu upper bound as a function of frequency
    nubar: mu upper bound peak 
    '''
    mubar = []
    for w in omega:
        # Compute the upper bound to the structured singular value
        # of M at frequency w using SLICOT routine
        m,_,_,_ = sly.ab13md(M.horner(1j*w), nblock, itype)
        mubar.append(m)
    nubar = np.max(mubar)
    return mubar, nubar


def musyn(G, f, nblock, itype, omega, maxiter=10, minorder=4, reduce=0, initgamma=1e6, verbose=True):
      '''
      Perform mu synthesis using D-K iteration
      
      K, best_nubar, init_mubar, best_mubar, gamma 
               = musyn(G, f, nblock, itype, omega, maxiter=10, qutol=2, order=4, reduce=0, verbose=True)
               
      G:       LFT form of the system from [w_delta,u] to [z_delta,y]
      f:       controller input-output dimension
      nblock:  uncertainty structure (vector of sizes of uncertain blocks)
      itype:   must be 2 for each entry of nblock, other values not implemented
      omega:   frequency vector for D scaling computation
      maxiter: max number of iterations
      minorder:  minimum order of the scalings D(j*omega), increase it to try to get more accurate results (unlikely)
      reduce:  if > 0, do a model reduction on the closed loop at each iteration for the sake of computing 
               the scaling D; set it to something lower than the full order of cl0 if you run into numerical
               problems (may impact performance of the final controller)
      verbose: print iteration info
      K:       controller
      best_nubar:
               best achieved upper bound to mu norm of Tzw_delta (best achieved nubar) 
      init_mubar:
               mu upper bound at the first iteration (as function of frequency)
      best_mubar:
               achieved mu upper bound at the last iteration (as function of frequency)
      gamma:   closed loop norm achieved by initial Hinf controller
      '''
      # Initial K-step: compute an initial optimal Hinf controller without D scaling
      k, cl0, gamma, rcond = hinfsyn(G, f, f, initgamma)
      if verbose:
            print("Infinity norm of Tzw_delta with initial Hinfinity controller: ", gamma)

      # Start with a best mu norm upper bound
      # slightly higher than the achieved gamma without scaling
      best_nubar = gamma * 1.001
      i = 1
      order = minorder
      qutol = 1
      while(True):
            if verbose:
                print("Iteration #", i)
            # D-step: compute optimal scalings for the current closed loop
            # and the corresponding upper bound mubar vs. frequency
            # If numerical problems occur, try reducing the order of the closed loop cl0
            # for the sake of computing the scaling D. This may impact the performance of the final controller
            if reduce > 0:
                cl0 = ct.balred(cl0, reduce, method='truncate')
            _, _, _, _, _, _, D_A, D_B, D_C, D_D, mubar, _ = sly.sb10md(f, order, nblock, itype, qutol, cl0.A, cl0.B, cl0.C, cl0.D, omega)
            if i == 1:
                  # Save the mubar of the first iteration
                  initial_mubar = mubar
                  best_mubar = mubar
            # Get current value of the peak of mubar, i.e., the current upper bound
            # to the mu norm
            sup_mubar = np.max(mubar)
            if sup_mubar >= best_nubar:
                  if qutol < 4:
                      qutol = qutol +1
                  else:
                      qutol = 1 
                      order = order + 1
                  # The current iteration did not improve nubar over the previous ones
                  if verbose:
                      print("No better upper bound to mu norm of Tzw_delta found: trying D order ", order, "qutol ", qutol)
                  if i > 1:
                        mubar = best_mubar
            else:
                  # Save best upper bound so far
                  best_nubar = sup_mubar
                  qutol = 1
                  order = minorder
                  if verbose:
                      print("Best upper bound to mu norm of Tzw_delta: ", best_nubar)
                  # And the best mubar
                  best_mubar = mubar
                  if i > 1:
                        # And the best controller
                        k = kb
            i = i+1
            if i > maxiter:
                  break 

            D = ct.StateSpace(D_A, D_B, D_C, D_D)
            # Compute D*G*(inv(D))
            DGDInv =  ct.minreal(D * G * invss(D), verbose = False)

            # K-step: compute controller for current scaling
            try: 
                  kb, cl0, gamma, rcond = hinfsyn(DGDInv, f, f, 1.5*best_nubar)
            except:
                  # Something went wrong: keep last controller
                  kb = k

      return k, best_nubar, initial_mubar, best_mubar, gamma

