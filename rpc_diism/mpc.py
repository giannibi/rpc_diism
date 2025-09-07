import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import control as ct
import cvxpy as cp


""" GENERAL FUNCTIONS """

def sampled_data_controller(controller, plant_dt): 
    """
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
    """ 
    
    # The following is used to ensure the number before '%' is a bit larger 
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



""" MPC RELATED FUNCTIONS """

def terminal_cost_set(p):
    """
    Computes an ellipsoidal terminal set x^T * Qt * x <= 1 and an associated terminal cost
    x^T * P * x, on the basis of a nominal controller computed with the same cost weights as the MPC problem
    p: MPC controller parameters
     """
    
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
    

""" BASIC MPC IMPLEMENTATION TO REGULATE THE SYSTEM TO ZERO """

def quadratic_mpc_problem_basic(p):
    """
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
    """

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


def mpc_controller_basic(p, T, **kwargs):
        """
        Create an I/O system implementing an MPC controller
        p: Controller parameters
            A,B: Model
            N: Horizon length
            Q,R,P: Cost weights
        T: Sampling period
        """
    
        nx, nu = p.B.shape
        ux = np.zeros((nu, 1))
    
        # Set up the cvxpy problem to be solved at each step
        problem = quadratic_mpc_problem_basic(p)
      
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




""" MPC IMPLEMENTATION WITH CONSTANT REFERENCE TRACKING, NO CONSTRAINT RELAXATION """


def quadratic_mpc_problem(p):
    """
    Generate a cvxpy problem for the quadratic MPC step
    Terminal set and cost are computed internally as an LQR set/cost
    p: Controller parameters
        A,B: Model
        N: Horizon length
        Q, R: State and input weights for stage cost
        Fx <= f: State constraints
        Eu <= e: Input constraints
        Xf: if == None, use the trivial zero terminal constraint
            if == 'lqr', compute an ellipsoidal terminal set
            and quadratic terminal cost based on lqr with weights p.Q, p.R
    Parameters:
    x0: initial state
    Returns cvxpy problem object with x0 as parameter, to be solved at each MPC step
    using current state as parameter
    """
    
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
        
    # Terminal cost and constraint:
    # if unspecified, fall back to trivial terminal constraint
    if p.Xf == 'lqr': # Ellipsoidal/lqr set/cost
        P,Qt = terminal_cost_set(p)
        cost += cp.quad_form(x[:, p.N], P)
        constraints += [cp.quad_form(x[:, p.N], Qt) <= 1]
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


def mpc_controller(p, T, **kwargs):
        """
        Create an I/O system implementing an MPC controller
        p: Controller parameters
            ref: Constant reference to be tracked
            A,B: Model
            N: Horizon length
            Q,R: Cost weights
        T: Sampling period
        """

        nx, nu = p.B.shape
        no, _ = p.C.shape

        # Set up the cvxpy problem to be solved at each step

        # If there is a (constant) reference to track, compute
        # steady-state inputs/states and error system constraints
        if p.ref is not None:
            # Compute steady-state inputs and states
            Wsp = np.linalg.inv(np.block([ 
                      [np.eye(nx)-p.A, -p.B],
                      [p.C, np.zeros([no,nu]) ] ]))
            xusp = Wsp@np.block([[np.zeros([nx,1])] , [p.ref]])
            xsp = xusp[:nx]
            usp = xusp[-nu:]  
            # Compute constraints on error states and error inputs 
            p.f = p.f - p.F @ xsp
            p.e = p.e - p.E @ usp
        else:
            xsp = np.zeros([nx,1])
            usp = np.zeros([nu,1])
            
        # Generate cvxpy problem
        problem = quadratic_mpc_problem(p)
      
        # State x of the MPC controller is the current optimal sequence
        # while its input u is composed by current plant state
    
        # Controller state update function
        def _update(t, x, u, params={}):
            nonlocal nu, nx, p, problem, xsp, usp
 
            # Retrieve current plant state (take out dummy input)
            x0 = u[-nx:]
            
            # Pass it to cvxpy
            problem.param_dict['x0'].value = x0-xsp.reshape(-1)
                
            # Solve optimization problem
            problem.solve(solver='MOSEK', warm_start=True)
            if problem.status != 'optimal':
                raise(ValueError("Infeasible problem"))
            else:
                # Retrieve solution (optimal sequence) and return it
                res = problem.var_dict['u'].value
                return res.reshape(-1)

        # Controller output computation
        def _output(t, x, u, params={}):
            nonlocal nu, nx, p, problem, xsp, usp
            
            # Retrieve current plant state (take out dummy input)
            x0 = u[-nx:]
            
            # Pass state (minus offset if tracking) to cvxpy
            problem.param_dict['x0'].value = x0-xsp.reshape(-1)
                
            # Solve optimization problem
            problem.solve(solver='MOSEK', warm_start=True)
            if problem.status != 'optimal':
                raise(ValueError("Infeasible problem"))
            else:
                # Retrieve solution (optimal sequence)
                res = problem.var_dict['u'].value
                # Return first sample (plus offset if tracking)
                return res[:,0] + usp.reshape(-1)
               
        # Number of states of the controller    
        kwargs['states'] = nu * p.N
    
        return ct.NonlinearIOSystem(
            _update, _output, dt=T, **kwargs)



""" MPC IMPLEMENTATION WITH DYNAMIC (PIECEWISE CONSTANT) REFERENCE TRACKING AND CONSTRAINT RELAXATION """

def terminal_cost_set_with_relax(p):
    """
    Computes an ellipsoidal terminal set x^T * Qt * x <= 1 and an associated terminal cost
    x^T * P * x, on the basis of a nominal controller computed with the same cost weights as the MPC problem
    p: MPC controller parameters
    returns terminal cost matrix P, ellipsoidal terminal set matrix Qt and a
    status variable to check if the terminal set is empty or unbounded
     """
    # Terminal controller, computed as an LQR
    K, P, _ = ct.dlqr(p.A, p.B, p.Q, p.R)
    K = -K
    
    # State constraints under the nominal controller
    if p.alpharelax is None:
        # No state constraint relaxation: consider hard constraints in terminal set
        Z = np.vstack((p.F,p.E @ K))
        z = np.vstack((p.f, p.e))
    else:
        # State constraints are relaxed: only input constraints must be enforced
        Z = p.E @ K
        z = p.e
        
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
    if problem.status == 'unbounded':
        print("WARNING: terminal set is unbounded: is your problem even constrained?",
               "If not, use LQR :) ")
        Qt = np.zeros([n,n])
    elif problem.status == 'optimal':
        # Terminal set is defined by x^T * Qt * x <= 1
        Qt = gamma.value * P
    elif problem.status == 'infeasible':
        print("WARNING: terminal set is empty: probably the origin",
                "is in the infeasible region, is your problem overconstrained?")
        Qt = -1 * np.eye(n)
    else:
        raise(ValueError, "Solver error in computing terminal set")
    return P, Qt, problem.status


def quadratic_mpc_problem_with_relax(p):
    """
    Generate a cvxpy problem for the quadratic MPC step with possible soft state constraints
    p: Controller parameters
        A,B: Model
        N: Horizon length
        Q, R: State and input weights for stage cost
        Fx <= f: State constraints
        Eu <= e: Input constraints
        alpharelax: State constraint violation weight in cost function (None = no relaxation)
        Xf: if == None, use the trivial zero terminal constraint
            if == 'lqr', compute an ellipsoidal terminal set
            and quadratic terminal cost based on lqr with weights Q, R
    Parameters (cvxpy):
    x0: initial state
    returns cvxpy problem object
    """
    
    # State and input dimension
    (nx, nu) = p.B.shape
    # Number of state and input constraints
    nf, _ = p.F.shape
    ne, _ = p.E.shape

    # Define initial state as parameter  
    x0 = cp.Parameter(nx, name='x0')
    
    # Define state and input variables
    x = cp.Variable((nx, p.N + 1), name='x') # From k=0 to k=N
    u = cp.Variable((nu, p.N), name='u')     # From k=0 to k=N-1

    # If parameter alpharelax is a value, make the state constraints soft
    if p.alpharelax is not None:
        # State constraints are required to be soft
        if p.alpharelax <= 0:
            raise(ValueError, "p.alpharelax must be positive")
        # Introduce slack variables for soft constraints
        epsilonrelax = cp.Variable((nf, p.N), name='epsilon')

    # Initialize cost and constraints
    cost = 0.0
    constraints = []
    
    # Generate cost and constraints
    for k in range(p.N):
        
        # Stage cost
        cost += cp.quad_form(x[:, k], p.Q)
        cost += cp.quad_form(u[:, k], p.R)
        
        if p.alpharelax is not None:
            # Add slack variable penalization in cost if constraints are soft
            cost += p.alpharelax * cp.quad_form(epsilonrelax[:, k], np.eye(nf))

        # Dynamics constraint
        constraints += [x[:, k + 1] == p.A @ x[:, k] + p.B @ u[:, k]]

        # Input constraints
        if p.E is not None:
            constraints += [p.E @ u[:,k] <= p.e.reshape(-1)]  

        # State constraints
        if p.F is not None:
            if p.alpharelax is not None:
                # Make state constraints soft using slacks
                constraints += [p.F @ x[:, k] <= p.f.reshape(-1) + epsilonrelax[:, k] ]
            else:
                # Make state constraints hard
                constraints += [p.F @ x[:, k] <= p.f.reshape(-1)]
        
    # Terminal cost and constraint:
    # if unspecified, fall back to trivial terminal constraint
    if p.Xf == 'lqr': # Ellipsoidal/lqr set/cost
        # Add terminal cost
        cost += cp.quad_form(x[:, p.N], p.P)
        # Enforce terminal set
        constraints += [cp.quad_form(x[:, p.N], p.Qt) <= 1]
    else:
        constraints += [x[:, p.N] == np.zeros_like(x0) ]
    
    # Initial state constraint
    constraints += [x[:, 0] == x0]

    # Generate cvx problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    
    # Return problem object
    return problem


def mpc_controller_with_relax(parameters, T, **kwargs):
    """
        Create an I/O system implementing an MPC controller with tracking 
        State constraints get automatically relaxed in case of infeasibility
        parameters: Controller parameters
            A,B: Model
            N: Horizon length
            Q,R: Cost weights
        T: Sampling period
    """

    p = parameters
    # Dimensions
    nx, nu = p.B.shape
    no, _ = p.C.shape

    # Map from references to steady-state inputs and states for tracking
    Wsp = np.linalg.inv(np.block([ 
         [np.eye(nx)-p.A, -p.B],
         [p.C, np.zeros([no,nu]) ] ]))
    
    # This variable holds the last reference value
    lastref = np.zeros(no)

    # State and input at set point, initialize at zero
    xsp = np.zeros([nx,1])
    usp = np.zeros([nu,1])

    # Compute initial terminal set for zero reference (may change if reference changes)
    if p.Xf == 'lqr':
        p.P, p.Qt, status = terminal_cost_set_with_relax(p)
        if status == 'unbounded':
            dummy = 0 # Do nothing
        # Terminal set is unbounded: Qt=0
        elif status == 'infeasible':
            # Terminal set is empty: the constraint set is probably empty
            print("Terminal set is empty: falling back to zero terminal", 
                    "constraint but something is probably wrong with your",
                    "constraints, expect infeasibility")

    # Generate cvxpy problem
    problem = quadratic_mpc_problem_with_relax(p)

    # This function updates the mpc problem when reference changes
    def update_mpc_problem(ref, t):
        nonlocal p, nx, nu, no, Wsp, xsp, usp, problem

        # Compute state and input at new set point
        xusp = Wsp@np.vstack([np.zeros([nx,1]), ref.reshape([no,1])])
        xsp = xusp[:nx]
        usp = xusp[-nu:]  

        # Compute new constraints on error states and error inputs (p.fa and p.ea hold the initial f and e matrices)
        p.f = p.fa - p.F @ xsp
        p.e = p.ea - p.E @ usp
        
        # Update terminal set
        p.P, p.Qt, status = terminal_cost_set_with_relax(p)
        if status == 'unbounded':
            # Terminal set is unbounded: enforce no terminal constraint
            dummy = 0 # Do nothing       
        elif status == 'infeasible':
            # Terminal set for new reference is empty: trigger relaxation and recompute terminal set
            print(f"Relaxation triggered for empty terminal set at t={t}")
            p.alpharelax = p.alpharelaxe
            p.P, p.Qt, status = terminal_cost_set_with_relax(p)
            if status != 'optimal':
                # If we get here, the input violates the constraints at the set point
                # and hence the set point is impossible to reach, give up
                raise(ValueError("The terminal set with relaxed state constraints is empty",
                                 "It is likely that the input violates the constraints at set point, giving up"))
            
        # Generate updated problem
        problem = quadratic_mpc_problem_with_relax(p)

    
    # Controller state update function
    def _update(t, x, u, params={}):
        nonlocal xsp, usp, nu, nx, problem, lastref, p

        # On the first try at each step, never relax
        p.alpharelax=None
 
        # Retrieve current plant state (last nx components of u)
        x0 = u[-nx:]
            
        # Retrieve current reference (first no components of u)
        ref = u[:no]

        # See if the reference changed. If so, update problem
        if not ((lastref == ref).all()):
            lastref = ref
            update_mpc_problem(ref, t)
               
        # Pass parameters to cvxpy
        problem.param_dict['x0'].value = x0-xsp.reshape(-1)
                
        # Solve optimization problem
        problem.solve(solver='MOSEK', warm_start=True)
        
        if problem.status != 'optimal':
            # The optimization problem is infeasible: we try relaxing unless we are relaxing already
            if p.alpharelax is not None:
                # The relaxed problem is infeasible: give up 
                raise(ValueError("Infeasible relaxed problem: giving up"))
            else:
                # Generate and solve a relaxed problem with soft state constraints
                print(f"Relaxation triggered for infeasibility at t={t}")
                p.alpharelax = p.alpharelaxe
                p.P, p.Qt, status = terminal_cost_set_with_relax(p)
                problem = quadratic_mpc_problem_with_relax(p)
                # Solve again 
                problem.param_dict['x0'].value = x0-xsp.reshape(-1)
                problem.solve(solver='MOSEK', warm_start=True)
                if problem.status != 'optimal':
                    # We are infeasible on the relaxation: give up 
                    raise(ValueError("Infeasible relaxed problem: giving up"))
                    
    
        # Retrieve solution (optimal sequence)
        res = problem.var_dict['u'].value
        # New state is the optimal sequence: return it
        return res

        
    # Controller output computation
    def _output(t, x, u, params={}):
        nonlocal xsp, usp, nu, nx, problem, lastref, p

        # On the first try at each step, never relax
        p.alpharelax=None
 
        # Retrieve current plant state (last nx components of u)
        x0 = u[-nx:]
            
        # Retrieve current reference (first no components of u)
        ref = u[:no]

        # See if the reference changed. If so, update problem
        if not ((lastref == ref).all()):
            lastref = ref
            update_mpc_problem(ref, t)
               
        # Pass parameters to cvxpy
        problem.param_dict['x0'].value = x0-xsp.reshape(-1)
                
        # Solve optimization problem
        problem.solve(solver='MOSEK', warm_start=True)
        
        if problem.status != 'optimal':
            # The optimization problem is infeasible: we try relaxing unless we are relaxing already
            if p.alpharelax is not None:
                # The relaxed problem is infeasible: give up  
                raise(ValueError("Infeasible relaxed problem: giving up"))
            else:
                # Generate and solve a relaxed problem with soft state constraints
                print(f"Relaxation triggered for infeasibility at t={t}")
                p.alpharelax = p.alpharelaxe
                p.P, p.Qt, status = terminal_cost_set_with_relax(p)
                problem = quadratic_mpc_problem_with_relax(p)
                # Solve again 
                problem.param_dict['x0'].value = x0-xsp.reshape(-1)
                problem.solve(solver='MOSEK', warm_start=True)
                if problem.status != 'optimal':
                    # We are infeasible on the relaxation: give up 
                    raise(ValueError("Infeasible relaxed problem: giving up"))
                    
        # Retrieve solution (optimal sequence)
        res = problem.var_dict['u'].value
        # Return first sample, plus set point offset
        uout = res[:,0]+usp.reshape(-1)
        return uout
               
                
    # Number of states of the controller: last input sequence
    kwargs['states'] = nu * p.N
        
    return ct.NonlinearIOSystem(_update, _output, dt=T, **kwargs)


