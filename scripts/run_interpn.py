"""
Script to execute interpolation experiments with `interpn` from SciPy.
"""
import click
import minterpy as mp
import numpy as np

from functools import partial
from multiprocessing import Pool
from scipy.interpolate import interpn
from tqdm import tqdm

import utils

NUM_PROCESSES = 10


def runge_non_vec(*args, parameter: float) -> float:
    """Compute the M-dimensional Runge function (non-vectorized).

    Notes
    -----
    - This function is used to compute the training response
      for interpn; the function signature is specifically
      designed for the function to be passed to `create_train_yy()`.
    """
    yy = 0
    for arg in args:
        yy += arg**2

    return 1 / (1 + parameter**2 * yy)


def create_train_xx(spatial_dimensions: int, num_points_1d: int):
    """Create training data for `interpn`."""
    x = np.sort(mp.gen_points.chebychev_2nd_order(num_points_1d))

    points = spatial_dimensions * [x]

    return tuple(points)


def create_train_yy(fun, xx_train, *args, **kwargs):
    """Compute the response given training data for interpn."""
    return fun(*np.meshgrid(*xx_train, indexing='ij'), *args, **kwargs)


@click.command()
@click.option(
    "-m",
    "--spatial-dimension",
    required=True,
    type=int,
    help="The dimension of the function"
)
@click.option(
    "-mt",
    "--method",
    required=True,
    type=click.Choice(
        ["linear", "slinear", "cubic", "quintic", "pchip", "nearest"]
    ),
    help="Interpolation method",
)
@click.option(
    "-p",
    "--max-1d-points",
    required=True,
    type=int,
    help="Maximum number of one-dimensional points",
)
@click.option(
    "-rp",
    "--runge-parameter",
    required=True,
    type=float,
    help="Parameter of the Runge function"
)
def run_interpn(
  spatial_dimension: int,
  method: str,
  max_1d_points: int,
  runge_parameter: float,
):
    """Run the interpolation experiment on the Runge function w/ interpn."""
    # --- Input check
    method_min_points = {
        "linear": 2,
        "slinear": 2,
        "cubic": 4,
        "quintic": 6,
        "pchip": 4,
        "nearest": 1,
    }
    if method_min_points[method] > max_1d_points:
        raise ValueError(
            f"{method} interpolation requires at least "
            f"{methods_min_points[method]}, got instead "
            f"{max_1d_points}"
        )

    # --- Create testing data
    rng = np.random.default_rng(utils.RANDOM_SEED)
    xx_test = -1 + 2 * rng.random((utils.NUM_TEST, spatial_dimension))
    yy_test = utils.runge_vec(xx_test, runge_parameter)

    # --- Initialize output array
    min_points = method_min_points[method]
    num_iter = max_1d_points - (min_points - 1)
    linf_errors = np.empty((num_iter, 2))

    for i in tqdm(range(num_iter), bar_format=utils.BAR_FORMAT):
        # --- Create training data
        xx_train = create_train_xx(spatial_dimension, min_points + i)
        yy_train = create_train_yy(
            runge_non_vec,
            xx_train,
            parameter=runge_parameter,
        )
        # --- Create a spline interpolant
        interpol = partial(
            interpn,
            xx_train,
            yy_train,
            method=method,
        )

        # ---- Evaluate the interpolant
        num_batches = 1000
        yy_interpol = 1000 * [None]

        xx_split = np.array_split(xx_test, num_batches)

        pool = Pool(NUM_PROCESSES)
        for ind, res in enumerate(
                tqdm(
                    pool.imap(interpol, xx_split),
                    total=num_batches,
                    leave=False,
                    bar_format=utils.BAR_FORMAT,
                )
        ):
            yy_interpol[ind] = res
        yy_interpol = np.concatenate(yy_interpol)

        # --- Compute l-inf of the errors
        linf_errors[i, 0] = np.prod([len(x) for x in xx_train])
        linf_errors[i, 1] = utils.compute_linf_error(yy_test, yy_interpol)

    np.savetxt(
        "errors-interpn-"
        f"dim_{spatial_dimension}-"
        f"{method}-"
        f"max_1d_{max_1d_points}-"
        f"runge_param_{str(runge_parameter).replace('.', '_')}"
        f".csv",
        linf_errors,
        delimiter=",",
    )


if __name__ == "__main__":
    run_interpn()

