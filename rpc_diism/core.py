import matplotlib.pyplot as plt
from itertools import product
import slycot as sly
import numpy as np
import control as ct
import cvxpy as cp


'''
Robust Control
'''


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


def mysigma(g, w):
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


def myhinfsyn(G, nmeas, ncon, initgamma=1e6):
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
      k, cl0, gamma, rcond = myhinfsyn(G, f, f, initgamma)
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
                  kb, cl0, gamma, rcond = myhinfsyn(DGDInv, f, f, initgamma)
            except:
                  # Something went wrong: keep last controller
                  kb = k

      return k, best_nubar, initial_mubar, best_mubar, gamma



'''
MPC
'''


def sampled_data_controller(controller, plant_dt): 
    '''
    Create a discrete-time system that models the behaviour 
    of a digital controller. 
    
    The system that is returned models the behavior of a sampled-data 
    controller, including a sampler and a ZOH converter. 
    The returned system is discrete-time, and its timebase `plant_dt` is 
    much smaller than the sampling interval of the controller, 
    `controller.dt`, to insure that continuous-time dynamics of the plant 
    are accurately simulated. This system must be interconnected
    to a plant with the same dt. The controller's sampling period must be 
    greater than or equal to `plant_dt`, and an integral multiple of it. 
    The plant that is connected to it must be converted to a discrete-time 
    ZOH equivalent with a sampling interval that is also `plant_dt`. A 
    controller that is a pure gain must have its `dt` specified (not None).
    ''' 
    
    # the following is used to ensure the number before '%' is a bit larger 
    one_plus_eps = 1 + np.finfo(float).eps 
    assert np.isclose(0, controller.dt*one_plus_eps % plant_dt), \
        "plant_dt must be an integral multiple of the controller's dt"
    nsteps = int(round(controller.dt / plant_dt))
    step = 0
    y = np.zeros((controller.noutputs, 1))

    def updatefunction(t, x, u, params):  
        nonlocal step

        # Update the controller state only if it is time to sample
        if step == 0:
            x = controller._rhs(t, x, u)
        step += 1
        if step == nsteps:
            step = 0

        return x
           
    def outputfunction(t, x, u, params):
        nonlocal y
        
        # Compute controller action if it is time to sample
        if step == 0:
            y = controller._out(t, x, u)       
        return y

    # Return the controller system object
    return ct.ss(updatefunction, outputfunction, dt=plant_dt, 
                 name=controller.name, inputs=controller.input_labels, 
                 outputs=controller.output_labels, states=controller.state_labels)


def terminal_cost_set(p):
    '''
    Computes an ellipsoidal terminal set x^T * Qt * x <= 1 and an associated terminal cost
    x^T * P * x, on the basis of a nominal controller computed with the same cost weights as the MPC problem
    p: MPC controller parameters
     '''
    # Terminal controller, computed as an LQR
    K, P, _ = ct.dlqr(p.A, p.B, p.Q, p.R)
    K = -K
    
    # State constraints under the nominal controller
    Z = np.vstack((p.F,p.E @ K))
    z = np.vstack((p.f, p.e))

    (nz, _) = Z.shape
    (n, _) = p.A.shape

    # Compute the maximal sublevel set of the terminal cost x^T * P * x
    # that lies within the constraints for the terminal controller
    a = cp.Variable(nz)
    gamma = cp.Variable(1)
    epsi = np.finfo(float).eps

    constraints = [gamma >= epsi]
    for i in range(nz):
        constraints += [a[i] >= epsi]
        M2 = cp.bmat([[gamma*P, np.zeros([n,1])],[np.zeros([1,n]), [[-1]]]])
        M1 = cp.bmat([[np.zeros([n,n]), 0.5*Z[i,:].reshape(1,-1).T], [0.5*Z[i,:].reshape(1,-1), -z[i,:].reshape(1,-1)]])
        constraints += [-a[i]*M1 + M2 >> 0]

    problem = cp.Problem(cp.Minimize(gamma), constraints)
    problem.solve(solver='MOSEK')
    if problem.status != 'optimal':
        raise(ValueError("Infeasible problem in terminal set"))
    else:
        # Terminal set x^T * Qt * x <= 1
        Qt = gamma.value * P
    
        return P, Qt
    

def quadratic_mpc_problem(p):
    '''
    Generate a cvxpy problem for the quadratic MPC step
    p: Controller parameters
        A,B: Model
        N: Horizon length
        Q, R: State and input weights for stage cost
        P: Terminal cost weight
        Fx <= f: State constraints
        Eu <= e: Input constraints
        Qt: x^T * Qt * x <= 1 terminal constraint
    Parameters:
    x0: initial state
    returns cvxpy problem object with x0 as parameter
    '''
    # State and input dimension
    (nx, nu) = p.B.shape

    # Define initial state as parameter  
    x0 = cp.Parameter(nx, name='x0')

    # Define state and input variables
    x = cp.Variable((nx, p.N + 1), name='x') # From k=0 to k=N
    u = cp.Variable((nu, p.N), name='u')     # From k=0 to k=N-1

    # Initialize cost and constraints
    cost = 0.0
    constraints = []
    
    # Generate cost and constraints
    for k in range(p.N):
        
        # Stage cost
        cost +=  cp.quad_form(x[:, k], p.Q)
        cost +=  cp.quad_form(u[:, k], p.R)

        # Dynamics constraint
        constraints += [x[:, k + 1] == p.A @ x[:, k] + p.B @ u[:, k]]

        # State constraints
        if p.F is not None:
            constraints += [p.F @ x[:, k] <= p.f.reshape(-1)]
        
    # Add terminal cost
    cost +=  cp.quad_form(x[:, p.N], p.P)
    
    # Terminal constraint: if unspecified, fall back to trivial terminal constraint
    if p.Qt is not None:
        constraints += [cp.quad_form(x[:, p.N], p.Qt) <= 1]
    else:
        constraints += [x[:, p.N] == np.zeros_like(x0) ]
    
    # Input constraints
    if p.E is not None:
        constraints += [p.E @ u <= p.e]  

    # Initial state constraint
    constraints += [x[:, 0] == x0]

    # Generate cvx problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    
    # Return problem object
    return problem


def mpc_controller(parameters, T, **kwargs):
        '''
        Create an I/O system implementing an MPC controller
        parameters: Controller parameters
            A,B: Model
            N: Horizon length
            Q,R,P: Cost weights
        T: Sampling period
        '''
    
        p = parameters
        nx, nu = p.B.shape
        ux = np.zeros((nu, 1))
    
        # Set up the cvxpy problem to be solved at each step
        problem = quadratic_mpc_problem(p)
      
        # State x of the MPC controller is the current optimal sequence
        # while its input u is composed by dummy inputs and current plant state
    
        # State update function
        def _update(t, x, u, params={}):
            nonlocal nu
            nonlocal p
            nonlocal problem
        
           
            # Retrieve current plant state
            x0 = u[-nx:]
            # Pass it as parameter to cvxpy
            problem.param_dict['x0'].value = x0
            # Solve optimization problem
            problem.solve(solver='MOSEK', warm_start=True)
            if problem.status != 'optimal':
                raise(ValueError("Infeasible problem"))
            else:
                # Retrieve solution (optimal sequence) and return it
                res = problem.var_dict['u'].value
                return res.reshape(-1)
                
        
        def _output(t, x, u, params={}):
            nonlocal nu
            nonlocal p
            nonlocal problem
            
            # Compute controller output

            # Retrieve plant state
            x0 = u[-nx:]
            # Pass it as parameter
            problem.param_dict['x0'].value = x0
            # Solve optimization problem
            problem.solve(solver='MOSEK', warm_start=True)
            if problem.status != 'optimal':
                raise(ValueError("Infeasible problem"))
            else:
                # Retrieve optimal sequence and return first sample as output
                res = problem.var_dict['u'].value
                return res[:,0]
                
        # Number of states of controller
        kwargs['states'] = nu * p.N
    
        return ct.NonlinearIOSystem(
            _update, _output, dt=T, **kwargs)


