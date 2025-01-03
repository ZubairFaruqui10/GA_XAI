
import heyoka as hy
import pygmo as pg
import numpy as np
import scipy
import PIL
import time

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.cm as cm
from pymoo.core.problem import Problem
import numpy as np

class SPOC3PymooProblem(Problem):
    def __init__(self, n_sat, ic, T, mu, n_meas=3, scaling_factor=1e-4, inflation_factor=1.5, verbose=True):
        # Initialize problem dimensions
        self.n_sat = n_sat
        self.ic = ic
        self.T = T
        self.mu = mu
        self.n_meas = n_meas
        self.scaling_factor = scaling_factor
        self.inflation_factor = inflation_factor
        self.verbose = verbose

        # Create STMs and reference trajectory
        self.ref_state, self.stms = stm_factory(ic, T, mu, n_meas, verbose)

        # Number of variables = n_sat * 3 (positions) + n_sat * 3 (velocities)
        n_var = n_sat * 6
        n_obj = 1  # Assuming a single-objective problem
        n_constr = 0  # Update if there are constraints

        # Bounds from get_bounds()
        xl, xu = [-1.0] * (n_sat * 3) + [-10.0] * (n_sat * 3), [1.0] * (n_sat * 3) + [10.0] * (n_sat * 3)

        # Initialize Pymoo problem
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):

            """
            Fitness function without plotting.

            Args:
                x (`list` of length N): Chromosome containing initial relative positions and velocities of each satellite.

            Returns:
                float: Negative of the worst fill factor across all observations.
            """
            N = self.n_sat

            # Decode chromosome
            dx0 = np.array([(i, j, k, l, m, n) for (i, j, k, l, m, n) in zip(
                x[:N], x[N:2 * N], x[2 * N:3 * N], x[3 * N:4 * N], x[4 * N:5 * N], x[5 * N:]
            )])

            # Propagate formation and scale results
            rel_pos = []
            for stm in self.stms:
                d_ic = dx0 * self.scaling_factor
                fc = propagate_formation(d_ic, stm)
                rel_pos.append(fc / self.scaling_factor)
            rel_pos = np.array(rel_pos)

            # Compute fill factors
            fill_factors = []
            for k in range(self.n_meas):
                points_3D = rel_pos[k]

                # Scale positions for inflation factor, except at the first observation
                if k != 0:
                    points_3D = points_3D / self.inflation_factor

                # Crop points outside [-1, 1]
                points_3D = points_3D[np.max(points_3D, axis=1) < 1]
                points_3D = points_3D[np.min(points_3D, axis=1) > -1]

                # Map points to 3D grid
                pos3D = (points_3D * self.grid_size / 2).astype(int)
                pos3D = pos3D + int(self.grid_size / 2)

                # Compute projections onto XY, XZ, and YZ planes
                I = np.zeros((self.grid_size, self.grid_size, self.grid_size))
                for i, j, k_ in pos3D:
                    I[i, j, k_] = 1

                xy = np.max(I, axis=2)
                xz = np.max(I, axis=1)
                yz = np.max(I, axis=0)

                # Compute autocorrelations for each plane
                xyC = scipy.signal.correlate(xy, xy, mode="full")
                xzC = scipy.signal.correlate(xz, xz, mode="full")
                yzC = scipy.signal.correlate(yz, yz, mode="full")

                # Remove near-zero values (numerical roundoffs)
                xyC[abs(xyC) < 1e-8] = 0
                xzC[abs(xzC) < 1e-8] = 0
                yzC[abs(yzC) < 1e-8] = 0

                # Compute fill factors
                f1 = np.count_nonzero(xyC) / (xyC.shape[0] * xyC.shape[1])
                f2 = np.count_nonzero(xzC) / (xzC.shape[0] * xzC.shape[1])
                f3 = np.count_nonzero(yzC) / (yzC.shape[0] * yzC.shape[1])
                fill_factors.append(f1 + f2 + f3)

            # Return negative of the worst fill factor
            out["F"] -min(fill_factors)

def propagate_formation(dx0, stm):
    """From some initial (relative) position and velocities returns new (relative) positions at
    some future time (defined by the stm).
    Args:
        dx0 (`np.array` (N, 6)): initial relative positions and velocities.
        stm (`np.array` (6,6)): the state transition matrix at some future time.
    Returns:
        np.array (N,3): propagated positions
    """
    dxT = stm @ dx0.T
    # We return only the positions
    return dxT.T[:, :3]


def stm_factory(ic, T, mu, M, verbose=True):
    """Constructs all the STMS and reference trajectory in a CR3BP dynamics
    Args:
        ic (`np.array` (N, 6)): initial conditions (absolute).
        T (`float`): propagation time
        mu (`float`): gravity parameter
        M (`int`): number of grid points (observations)
        verbose (boolean): print time it took to build Taylor integrator and STMs
    Returns:
        (ref_state (M, 6), stms (M,6,6)): the propagated state and stms
    """
    # ----- We assemble the CR3BP equation of motion --------
    # The state
    x, y, z, vx, vy, vz = hy.make_vars("x", "y", "z", "vx", "vy", "vz")
    xarr = np.array([x, y, z, vx, vy, vz])
    # The dynamics
    r_1 = hy.sqrt((x + hy.par[0]) ** 2 + y**2 + z**2)
    r_2 = hy.sqrt((x - (1 - hy.par[0])) ** 2 + y**2 + z**2)
    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt = (
        2 * vy
        + x
        - (1 - hy.par[0]) * (x + hy.par[0]) / (r_1**3)
        - hy.par[0] * (x + hy.par[0] - 1) / (r_2**3)
    )
    dvydt = -2 * vx + y - (1 - hy.par[0]) * y / (r_1**3) - hy.par[0] * y / (r_2**3)
    dvzdt = -(1 - hy.par[0]) / (r_1**3) * z - hy.par[0] / (r_2**3) * z
    # This array contains the expressions (r.h.s.) of our dynamics
    farr = np.array([dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt])

    # We now compute the variational equations
    # 1 - Define the symbols
    symbols_phi = []
    for i in range(6):
        for j in range(6):
            # Here we define the symbol for the variations
            symbols_phi.append("phi_" + str(i) + str(j))
    phi = np.array(hy.make_vars(*symbols_phi)).reshape((6, 6))

    # 2 - Compute the gradient
    dfdx = []
    for i in range(6):
        for j in range(6):
            dfdx.append(hy.diff(farr[i], xarr[j]))
    dfdx = np.array(dfdx).reshape((6, 6))

    # 3 - Assemble the expressions for the r.h.s. of the variational equations
    dphidt = dfdx @ phi

    dyn = []
    for state, rhs in zip(xarr, farr):
        dyn.append((state, rhs))
    for state, rhs in zip(phi.reshape((36,)), dphidt.reshape((36,))):
        dyn.append((state, rhs))

    # These are the initial conditions on the variational equations (the identity matrix)
    ic_var = np.eye(6).reshape((36,)).tolist()

    start_time = time.time()
    ta = hy.taylor_adaptive(
        # The ODEs.
        dyn,
        # The initial conditions (do not matter, they will change)
        [0.1] * 6 + ic_var,
        # Operate below machine precision
        # and in high-accuracy mode.
        tol=1e-16,
    )
    if verbose:
        print(
            "--- %s seconds --- to build the Taylor integrator -- (do this only once)"
            % (time.time() - start_time)
        )
    # We set the Taylor integration param
    ta.pars[:] = [mu]
    # We set the ic
    ta.state[:6] = ic
    ta.state[6:] = ic_var
    ta.time = 0.0
    # The time grid
    t_grid = np.linspace(0, T, M)
    # We integrate
    start_time = time.time()
    sol = ta.propagate_grid(t_grid)
    if verbose:
        print("--- %s seconds --- to construct all stms" % (time.time() - start_time))

    ref_state = sol[4][:, :6]
    stms = sol[4][:, 6:].reshape(M, 6, 6)
    return (ref_state, stms)