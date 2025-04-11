"""
Execute interpolation experiments with ndsplines.
"""
import click
import minterpy as mp
import ndsplines
import numpy as np

from tqdm import tqdm

import utils


def runge_non_vec(xx: np.ndarray, parameter: float) -> float:
    """Compute the M-dimensional Runge function (non-vectorized).

    Notes
    -----
    - This function is used to compute the training response
      for ndsplines; the function signature is specifically
      designed for the function to be passed to `create_train_yy()`.
    """
    return 1 / (1 + parameter**2 * np.sum(xx**2))


def create_train_xx(spatial_dimensions: int, num_points_1d: int):
    """Create a training data for NDSplines."""
    x = np.sort(mp.gen_points.chebychev_2nd_order(num_points_1d))
    xx = spatial_dimensions * [x]
    xx_mesh = np.meshgrid(*xx, indexing='ij')
    xx_grid = np.stack(tuple(xx_mesh), axis=-1)

    return xx_grid


def create_train_yy(fun, xx_train, *args, **kwargs):
    """Compute the response given training data for NDSplines."""
    yy = np.apply_along_axis(
        fun,
        axis=-1,
        arr=xx_train,
        *args,
        **kwargs,
    )

    return yy


@click.command()
@click.option(
    "-m",
    "--spatial-dimension",
    required=True,
    type=int,
    help="The dimension of the function"
)
@click.option(
    "-d",
    "--spline-degree",
    required=True,
    type=int,
    help="Degree of the splines",
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
def run_ndsplines(
  spatial_dimension: int,
  spline_degree: int,
  max_1d_points: int,
  runge_parameter: float,
):
    """Run the NDSplines experiment."""
    # --- Input check
    if spline_degree + 1 > max_1d_points:
        raise ValueError(
            f"Maximum number of 1D points ({max_1d_points}) must be equal to "
            f"or larger than spline-degree + 1 ({spline_degree + 1})"
        )
    if spline_degree < 0 or spline_degree > 5:
        raise ValueError(
            f"Can't run experiment for spline-degree ({spline_degree}) > 5"
        )

    # --- Create testing data
    rng = np.random.default_rng(23502)
    xx_test = -1 + 2 * rng.random((utils.NUM_TEST, spatial_dimension))
    yy_test = utils.runge_vec(xx_test, runge_parameter)

    # --- Initialize output array
    num_iter = max_1d_points - spline_degree
    linf_errors = np.empty((num_iter, 2))

    for i in tqdm(range(num_iter), bar_format=utils.BAR_FORMAT):
        # --- Create training data
        xx_train = create_train_xx(spatial_dimension, (spline_degree + 1) + i)
        yy_train = create_train_yy(
            runge_non_vec,
            xx_train,
            parameter=runge_parameter,
        )
        # --- Create a spline interpolant
        interpn = ndsplines.make_interp_spline(
            xx_train,
            yy_train,
            degrees=spline_degree,
        )
        # --- Predict the test points
        yy_interpolation = interpn(xx_test)

        # --- Update output
        linf_errors[i, 0] = int(np.prod(xx_train.shape[:-1]))
        linf_errors[i, 1] = utils.compute_linf_error(yy_test, yy_interpolation)

    degree_names = {
        1: "linear",
        2: "quadratic",
        3: "cubic",
        4: "quartic",
        5: "quintic",
    }

    np.savetxt(
        "errors-ndsplines-"
        f"dim_{spatial_dimension}-"
        f"{degree_names[spline_degree]}-"
        f"max_1d_{max_1d_points}-"
        f"runge_param_{str(runge_parameter).replace('.', '_')}"
        f".csv",
        linf_errors,
        delimiter=",",
    )


if __name__ == "__main__":
    run_ndsplines()

