import casadi as ca
import numpy as np

class MPCParameters:
    def __init__(self,
                 Q: ca.DM = None,
                 R: ca.DM = None,
                 N: int = 20,
                 stepHorizon: float = 0.05,
                 u_max: float = 0.5, v_max: float = 0.5, w_max: float = 0.5, r_max: float = 0.5,
                 u_ROC_max: float = 0.05, v_ROC_max: float = 0.05, w_ROC_max: float = 0.05, r_ROC_max: float = 0.05):
        """
        Args:
            Q: State error weight matrix.
            R: Control effort weight matrix.
            N: Prediction horizon.
            stepHorizon: Time between predictions
            u_max, v_max, w_max, r_max: Maximum control values (min are negative of these).
            u_ROC_max, v_ROC_max, w_ROC_max, r_ROC_max: Maximum rate-of-change (ROC) values (min are negatives).
        """
        # Set default matrices if not provided.
        if Q is None:
            Q = ca.diagcat(1, 1, 1, 1)
        if R is None:
            R = ca.diagcat(1, 1, 1, 1)

        self.Q = Q
        self.R = R
        self.N = N
        self.stepHorizon = stepHorizon

        self.u_max = u_max
        self.u_min = -u_max
        self.v_max = v_max
        self.v_min = -v_max
        self.w_max = w_max
        self.w_min = -w_max
        self.r_max = r_max
        self.r_min = -r_max

        self.u_ROC_max = u_ROC_max
        self.u_ROC_min = -u_ROC_max
        self.v_ROC_max = v_ROC_max
        self.v_ROC_min = -v_ROC_max
        self.w_ROC_max = w_ROC_max
        self.w_ROC_min = -w_ROC_max
        self.r_ROC_max = r_ROC_max
        self.r_ROC_min = -r_ROC_max

        self.updateParams = 0

class MPCController:
    def __init__(self, params: MPCParameters):
        """
        Initialize the controller controller.
        Args:
            params: controller Parameters.
        """

        self.target = ca.DM([0, 0, 0, 0])
        self.state = ca.DM([0, 0, 0, 0])

        self.n_states = 4
        self.n_controls = 4
        self.update_from_params(params)

        self.X0 = ca.DM.zeros(self.n_states, self.N + 1)
        self.u0 = ca.DM.zeros(self.n_controls, self.N)
        self.u_prev = ca.DM.zeros(self.n_controls, 1)



    def update_from_params(self, params: MPCParameters):
        self.Q = params.Q
        self.R = params.R
        self.N = params.N
        self.stepHorizon = params.stepHorizon

        self.u_max = params.u_max
        self.u_min = params.u_min
        self.v_max = params.v_max
        self.v_min = params.v_min
        self.w_max = params.w_max
        self.w_min = params.w_min
        self.r_max = params.r_max
        self.r_min = params.r_min

        self.u_ROC_max = params.u_ROC_max * self.stepHorizon
        self.u_ROC_min = params.u_ROC_min * self.stepHorizon
        self.v_ROC_max = params.v_ROC_max * self.stepHorizon
        self.v_ROC_min = params.v_ROC_min * self.stepHorizon
        self.w_ROC_max = params.w_ROC_max * self.stepHorizon
        self.w_ROC_min = params.w_ROC_min * self.stepHorizon
        self.r_ROC_max = params.r_ROC_max * self.stepHorizon
        self.r_ROC_min = params.r_ROC_min * self.stepHorizon

        self.build_solver()

    def build_solver(self):
        """
        Build the CasADi NLP solver.

        The cost function includes the tracking error weighted by Q and the control effort weighted by R.
        """
        # Define symbolic variables.
        states = ca.SX.sym('x', self.n_states)
        controls = ca.SX.sym('u', self.n_controls)

        # Decision variables for all steps.
        X = ca.SX.sym('X', self.n_states, self.N + 1)
        U = ca.SX.sym('U', self.n_controls, self.N)

        # Parameters: [current_state; target_state; previous_control]
        P = ca.SX.sym('P', self.n_states * 2 + self.n_controls)

        # Define a simple kinematic model.
        psi = states[3]
        rot = ca.vertcat(
            ca.horzcat(ca.cos(psi), -ca.sin(psi), 0, 0),
            ca.horzcat(ca.sin(psi), ca.cos(psi), 0, 0),
            ca.horzcat(0, 0, 1, 0),
            ca.horzcat(0, 0, 0, 1)
        )
        f = ca.Function('f', [states, controls], [rot @ controls])

        # Build the cost function and dynamics constraints.
        cost_fn = 0
        g = X[:, 0] - P[0:self.n_states]  # initial state constraint

        for k in range(self.N):
            st_k = X[:, k]
            con_k = U[:, k]
            cost_fn = cost_fn \
                      + (P[self.n_states: 2 * self.n_states] - st_k).T @ self.Q @ (P[self.n_states: 2 * self.n_states] - st_k) \
                      + con_k.T @ self.R @ con_k
            st_next = X[:, k + 1]
            k1 = f(st_k, con_k)
            k2 = f(st_k + self.stepHorizon / 2 * k1, con_k)
            k3 = f(st_k + self.stepHorizon / 2 * k2, con_k)
            k4 = f(st_k + self.stepHorizon * k3, con_k)
            st_next_RK4 = st_k + (self.stepHorizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            g = ca.vertcat(g, st_next - st_next_RK4)

        g = ca.vertcat(g, U[:, 0] - P[2 * self.n_states: 2 * self.n_states + self.n_controls])

        for k in range(1, self.N):
            g = ca.vertcat(g, U[:, k] - U[:, k - 1])

        # Decision variable vector.
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        # NLP problem.
        nlp = {'f': cost_fn,
               'x': opt_vars,
               'g': g,
               'p': P
               }

        opts = {
            'ipopt': {
                'max_iter': 10000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0
        }

        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        lbx = ca.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N, 1))
        ubx = ca.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N, 1))

        lbx[0: self.n_states * (self.N + 1): self.n_states] = -ca.inf  # X lower bound
        lbx[1: self.n_states * (self.N + 1): self.n_states] = -ca.inf  # Y lower bound
        lbx[2: self.n_states * (self.N + 1): self.n_states] = -ca.inf  # Z lower bound
        lbx[3: self.n_states * (self.N + 1): self.n_states] = -ca.inf  # Psi lower bound

        ubx[0: self.n_states * (self.N + 1): self.n_states] = ca.inf  # X upper bound
        ubx[1: self.n_states * (self.N + 1): self.n_states] = ca.inf  # Y upper bound
        ubx[2: self.n_states * (self.N + 1): self.n_states] = ca.inf  # Z upper bound
        ubx[3: self.n_states * (self.N + 1): self.n_states] = ca.inf  # Psi upper bound

        lbx[self.n_states * (self.N + 1) + 0: self.n_states * (self.N + 1) + self.n_controls * self.N: self.n_controls] = self.u_min  # u lower bound
        lbx[self.n_states * (self.N + 1) + 1: self.n_states * (self.N + 1) + self.n_controls * self.N: self.n_controls] = self.v_min  # v lower bound
        lbx[self.n_states * (self.N + 1) + 2: self.n_states * (self.N + 1) + self.n_controls * self.N: self.n_controls] = self.w_min  # w lower bound
        lbx[self.n_states * (self.N + 1) + 3: self.n_states * (self.N + 1) + self.n_controls * self.N: self.n_controls] = self.r_min  # r lower bound

        ubx[self.n_states * (self.N + 1) + 0: self.n_states * (self.N + 1) + self.n_controls * self.N: self.n_controls] = self.u_max  # u upper bound
        ubx[self.n_states * (self.N + 1) + 1: self.n_states * (self.N + 1) + self.n_controls * self.N: self.n_controls] = self.v_max  # v upper bound
        ubx[self.n_states * (self.N + 1) + 2: self.n_states * (self.N + 1) + self.n_controls * self.N: self.n_controls] = self.w_max  # w upper bound
        ubx[self.n_states * (self.N + 1) + 3: self.n_states * (self.N + 1) + self.n_controls * self.N: self.n_controls] = self.r_max  # r upper bound

        # Number of constraints from dynamics:
        n_state_con = self.n_states * (self.N + 1)

        # Create bounds for these dynamics constraints (equality constraints, so both lower and upper bounds are zero):
        lbg_state = ca.DM.zeros((n_state_con, 1))
        ubg_state = ca.DM.zeros((n_state_con, 1))

        # Now, the ROC constraints:

        # Define a DM vector with the bounds for each control's ROC:
        roc_lower = ca.DM([self.u_ROC_min, self.v_ROC_min, self.w_ROC_min, self.r_ROC_min]) # u,v,w,r
        roc_upper = ca.DM([self.u_ROC_max, self.u_ROC_max, self.u_ROC_max, self.u_ROC_max]) # u,v,w,r
        # Repeat these bounds for each time step (N-1 times)
        lbg_roc = ca.repmat(roc_lower, (self.N, 1))
        ubg_roc = ca.repmat(roc_upper, (self.N, 1))

        # Combine both parts:
        lbg_total = ca.vertcat(lbg_state, lbg_roc)
        ubg_total = ca.vertcat(ubg_state, ubg_roc)

        args = {
            'lbg': lbg_total,
            'ubg': ubg_total,
            'lbx': lbx,
            'ubx': ubx
        }

        # Save relevant data.
        self.f = f
        self.solver = solver
        self.args = args



def mpc_step(mpc: MPCController) -> tuple:
    """
    Compute the optimal control using the controller solver.

    Args:
        mpc: An instance of MPCController.

    Returns:
        The computed control (DM) for the current step.
    """
    # Pack parameters: current state, target state, and previous control.
    mpc.args['p'] = ca.vertcat(mpc.state, mpc.target, mpc.u_prev)
    # Pack the warm-start initial guess.
    mpc.args['x0'] = ca.vertcat(ca.reshape(mpc.X0, mpc.n_states*(mpc.N+1), 1),
                                ca.reshape(mpc.u0, mpc.n_controls*mpc.N, 1))

    # Solve the NLP.
    sol = mpc.solver(**mpc.args)

    # Extract control trajectory.
    U_opt = ca.reshape(sol['x'][mpc.n_states * (mpc.N + 1):], mpc.n_controls, -1)
    X_opt = ca.reshape(sol['x'][:mpc.n_states * (mpc.N + 1)], mpc.n_states, -1)

    # Select the first control input.
    u_apply = U_opt[:, 0]

    # Update warm-start guesses.
    mpc.u_prev = u_apply
    mpc.X0 = ca.horzcat(X_opt[:, 1:], X_opt[:, -1])
    mpc.u0 = ca.horzcat(U_opt[:, 1:], U_opt[:, -1])

    # Scale each control value by 1000 and convert to integer.
    control_values = u_apply.full().flatten() # converts the DM to a NumPy array.
    scaled_control = np.asarray((np.round(control_values * 1000, 0)),dtype="int")
    u_ref, v_ref, w_ref, r_ref = scaled_control

    return u_ref, v_ref, w_ref, r_ref