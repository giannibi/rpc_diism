import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import control as ct
import cvxpy as cp


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


