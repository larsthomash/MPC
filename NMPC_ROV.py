import casadi as ca
import numpy as np


class MPCTrajectory:
    def __init__(self):
        self.x = []
        self.y = []
        self.z = []
        self.psi = []

        self.resolution = 0.005  # A waypoint per X [m]

        self.n = 4  # Amount of turns in helix
        self.R = 1  # Radius for helix
        self.h = 2  # Height for helix


    def updateToWaypoints(self, initState, TrajVersion):
        self.x = []
        self.y = []
        self.z = []
        self.psi = []
        displacements = []

        pos0 = np.array([float(initState[0]),float(initState[1]),float(initState[2])])
        if TrajVersion == 0:  # Lawnmover pattern
            displacements = [
                np.array([4.0, 0.0, 0.0]),
                np.array([4.0, 2.0, 0.0]),
                np.array([0.0, 2.0, 0.0]),
                np.array([0.0, 4.0, 0.0]),
                np.array([4.0, 4.0, 0.0]),
                np.array([4.0, 6.0, 0.0]),
                np.array([0.0, 6.0, 0.0]),
            ]

        if TrajVersion == 1:  # Sawtooth
            displacements = [
                np.array([1, 1, 0.0]),
                np.array([2, -1, 0.0]),
                np.array([3, 1, 0.0]),
                np.array([4, -1, 0.0]),
                np.array([5, 1, 0.0]),
                np.array([6, -1, 0.0]),
                np.array([7, 1, 0.0]),
            ]

        WP = [pos0]
        for disp in displacements:
            WP.append(pos0 + disp)

        for i in range(len(WP)-1): # Calculates the length of each segment
            x_i, y_i, z_i = WP[i]
            x_e, y_e, z_e = WP[i+1]

            distance = ((x_e - x_i)**2 + (y_e - y_i)**2 + (z_e - z_i)**2) ** 0.5
            steps = int(np.ceil(distance / self.resolution))

            for j in range(steps + 1):  # Creates a point at each cm

                if i > 0 and j == 0:  # Avoid the same point twice
                    continue
                else:
                    k = j / steps
                    self.x.append(x_i + k * (x_e - x_i))
                    self.y.append(y_i + k * (y_e - y_i))
                    self.z.append(z_i + k * (z_e - z_i))
                    self.psi.append(np.atan2(y_e-y_i, x_e-x_i))

    def updateToHelix(self, initState):
        self.x = []
        self.y = []
        self.z = []
        self.psi = []


        distance = ( (2 * np.pi * self.n * self.R)**2 + self.h**2 ) ** 0.5
        steps = int(np.ceil(distance / self.resolution))

        for i in range(steps + 1):  # Creates a point at each cm
            k = 2*np.pi*self.n * i / steps
            self.x.append(float(initState[0]) + self.R * np.cos(k))
            self.y.append(float(initState[1]) + self.R * np.sin(k))
            self.z.append(float(initState[2]) + (self.h / (2*np.pi*self.n)) * k)

        for i in range(steps):
            self.psi.append(np.atan2(self.y[i+1] - self.y[i], self.x[i+1] - self.x[i]))
        self.psi.append(self.psi[-1]) # Add the last point


class MPCParameters_:
    def __init__(self,
                 Q: ca.DM = None,
                 R: ca.DM = None,
                 N: int = 20,
                 stepHorizon: float = 0.05,
                 desiredVelocity: float = 0.1,
                 tau: float = 0.05,
                 u_max: float = 0.5, v_max: float = 0.5, w_max: float = 0.5, r_max: float = 0.5,
                 u_ROC_max: float = 0.5, v_ROC_max: float = 0.5, w_ROC_max: float = 0.5, r_ROC_max: float = 0.5):
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
        self.desiredVelocity = desiredVelocity
        self.tau = tau

        self.u_max = u_max
        self.v_max = v_max
        self.w_max = w_max
        self.r_max = r_max

        self.u_ROC_max = u_ROC_max
        self.v_ROC_max = v_ROC_max
        self.w_ROC_max = w_ROC_max
        self.r_ROC_max = r_ROC_max

        self.manualMode = 1
        self.activeTrajectory = 0
        self.activeTarget = 0
        self.whichTrajectory = 0

        self.updateParams = 0

class MPCController_:
    def __init__(self, params: MPCParameters_):
        """
        Initialize the controller controller.
        Args:
            params: controller Parameters.
        """

        self.n_states = 4
        self.n_controls = 4
        self.update_from_params(params)

        self.X0 = ca.DM.zeros(self.n_states, self.N + 1)
        self.u0 = ca.DM.zeros(self.n_controls, self.N)
        self.u_prev = ca.DM.zeros(self.n_controls, 1)

        self.target = ca.DM([0, 0, 0, 0])
        self.state = ca.DM([0, 0, 0, 0])
        self.trajectory = ca.DM.zeros(self.n_states, self.N)
        self.elapsed_time_trajectory = 0


    def update_from_params(self, params: MPCParameters_):
        self.Q = params.Q
        self.R = params.R
        self.N = params.N
        self.stepHorizon = params.stepHorizon
        self.desiredVelocity = params.desiredVelocity
        self.tau = params.tau

        self.u_max = params.u_max
        self.u_min = -params.u_max
        self.v_max = params.v_max
        self.v_min = -params.v_max
        self.w_max = params.w_max
        self.w_min = -params.w_max
        self.r_max = params.r_max
        self.r_min = -params.r_max

        self.u_ROC_max = params.u_ROC_max * self.stepHorizon
        self.u_ROC_min = -params.u_ROC_max * self.stepHorizon
        self.v_ROC_max = params.v_ROC_max * self.stepHorizon
        self.v_ROC_min = -params.v_ROC_max * self.stepHorizon
        self.w_ROC_max = params.w_ROC_max * self.stepHorizon
        self.w_ROC_min = -params.w_ROC_max * self.stepHorizon
        self.r_ROC_max = params.r_ROC_max * self.stepHorizon
        self.r_ROC_min = -params.r_ROC_max * self.stepHorizon

        self.activeTrajectory = params.activeTrajectory
        self.activeTarget = params.activeTarget


        self.build_solver(self.activeTarget, self.activeTrajectory)

    def next_trajectory(self, trajectory: MPCTrajectory, elapsedTime):
        current_index = int(np.ceil((self.desiredVelocity * elapsedTime) / trajectory.resolution))
        delta_index = int(self.desiredVelocity * self.stepHorizon / trajectory.resolution)
        end_index = current_index + delta_index*self.N
        max_index = (len(trajectory.x) - 1)
        if end_index < max_index:
            x = trajectory.x[current_index : end_index : delta_index]
            y = trajectory.y[current_index : end_index : delta_index]
            z = trajectory.z[current_index : end_index : delta_index]
            psi = trajectory.psi[current_index : end_index : delta_index]
        else:
            x = trajectory.x[current_index : max_index : delta_index]
            y = trajectory.y[current_index : max_index : delta_index]
            z = trajectory.z[current_index : max_index : delta_index]
            psi = trajectory.psi[current_index : max_index : delta_index]

            # Pad with repeated last point
            missing = (self.N - len(x))
            x += [trajectory.x[-1]] * missing
            y += [trajectory.y[-1]] * missing
            z += [trajectory.z[-1]] * missing
            psi += [trajectory.psi[-1]] * missing

        self.trajectory = ca.reshape(ca.DM(np.vstack([x, y, z, psi])), -1, 1)

    def build_solver(self, activeTarget, activeTrajectory):
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

        if activeTarget:
            # Parameters: [current_state; previous_control; target_state]
            P = ca.SX.sym('P', self.n_states * 2 + self.n_controls)
        elif activeTrajectory:
            # Parameters: [current_state; previous_control; trajectory]
            P = ca.SX.sym('P', self.n_states + self.n_controls + self.n_states * self.N)

        # The kinematic model.
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
        v_actual = P[self.n_states:self.n_states + self.n_controls]

        for k in range(self.N):
            st_k = X[:, k]
            con_k = U[:, k]

            if activeTarget:
                cost_fn = cost_fn \
                          + (P[self.n_states + self.n_controls: 2 * self.n_states + self.n_controls] - st_k).T @ self.Q @ (P[self.n_states + self.n_controls: 2 * self.n_states + self.n_controls] - st_k) \
                          + con_k.T @ self.R @ con_k
            elif activeTrajectory:
                cost_fn = cost_fn \
                          + (P[self.n_states + self.n_controls + (self.n_states * k): self.n_states + self.n_controls + (self.n_states * (k + 1))] - st_k).T @ self.Q @ (P[self.n_states + self.n_controls + (self.n_states * k): self.n_states + self.n_controls + (self.n_states * (k + 1))] - st_k) \
                          + con_k.T @ self.R @ con_k

            st_next = X[:, k + 1]

            v_actual = v_actual + ( self.stepHorizon / self.tau ) * (con_k - v_actual)

            k1 = f(st_k, v_actual)
            k2 = f(st_k + self.stepHorizon / 2 * k1, v_actual)
            k3 = f(st_k + self.stepHorizon / 2 * k2, v_actual)
            k4 = f(st_k + self.stepHorizon * k3, v_actual)
            st_next_RK4 = st_k + (self.stepHorizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            g = ca.vertcat(g, st_next - st_next_RK4)

        g = ca.vertcat(g, U[:, 0] - P[self.n_states: self.n_states + self.n_controls])

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



def mpc_step(mpc: MPCController_, trajectory : MPCTrajectory) -> tuple:
    """
    Compute the optimal control using the controller solver.

    Args:
        mpc: An instance of MPCController.

    Returns:
        The computed control (DM) for the current step.
    """
    # Pack parameters: current state, target state, and previous control.
    if mpc.activeTrajectory:
        mpc.next_trajectory(trajectory, mpc.elapsed_time_trajectory)
        mpc.args['p'] = ca.vertcat(mpc.state, mpc.u_prev, mpc.trajectory)
    elif mpc.activeTarget:
        mpc.args['p'] = ca.vertcat(mpc.state, mpc.u_prev, mpc.target)

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
    #mpc.u_prev = u_apply
    mpc.X0 = ca.horzcat(X_opt[:, 1:], X_opt[:, -1])
    mpc.u0 = ca.horzcat(U_opt[:, 1:], U_opt[:, -1])

    # Scale each control value by 1000 and convert to integer.
    control_values = u_apply.full().flatten() # converts the DM to a NumPy array.
    scaled_control = np.asarray((np.round(control_values * 1000, 0)),dtype="int")
    scaled_control = np.asarray(control_values)
    u_ref, v_ref, w_ref, r_ref = scaled_control

    return u_ref, v_ref, w_ref, r_ref
