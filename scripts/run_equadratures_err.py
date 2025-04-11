"""
Script to execute interpolation experiments with equadratures.
"""
import click
import equadratures as eq
import numpy as np
import pickle

from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

import utils


NUM_PROCESSES = 1


def wrap_linf_error(
    order: int,
    spatial_dimension: int,
    runge_parameter: float,
    xx_test: np.ndarray,
    yy_test: np.ndarray,
    target_directory: str,
):
    """Compute the l-inf error of a polynomial fit given test data."""
    # --- Load the polynomial
    fname = (
        f"{target_directory}/"
        f"fit-equadratures-"
        f"dim_{spatial_dimension}-"
        f"n_{order:03}-"
        f"runge_param_{str(runge_parameter).replace('.', '_')}"
        f".pkl"
    )

    with open(fname, "rb") as f:
        poly = pickle.load(f)

    yy_poly = poly.get_polyfit(xx_test).flatten()

    return len(poly.get_points()), utils.compute_linf_error(yy_test, yy_poly)


@click.command()
@click.option(
    "-m",
    "--spatial-dimension",
    required=True,
    type=int,
    help="The dimension of the function"
)
@click.option(
    "-nmin",
    "--min-poly-degree",
    required=True,
    type=int,
    help="Minimum polynomial degree",
)
@click.option(
    "-nmax",
    "--max-poly-degree",
    required=True,
    type=int,
    help="Maximum polynomial degree",
)
@click.option(
    "-rp",
    "--runge-parameter",
    required=True,
    type=float,
    help="Parameter of the Runge function"
)
@click.option(
    "-t",
    "--target-directory",
    default=".",
    show_default=True,
    type=str,
    help="Target directory where pkl files are located"
)
def run_equadratures_err(
    spatial_dimension: int,
    min_poly_degree: int,
    max_poly_degree: int,
    runge_parameter: float,
    target_directory: str,
):
    """Compute the l-inf error given a polynomial from EQ."""

    # --- Create a testing dataset
    rng = np.random.default_rng(utils.RANDOM_SEED)
    xx_test = -1 + 2 * rng.random((utils.NUM_TEST, spatial_dimension))
    yy_test = utils.runge_vec(xx_test, runge_parameter)

    # --- Wrap the function for parallel mapping
    wrap_fun = partial(
        wrap_linf_error,
        spatial_dimension=spatial_dimension,
        runge_parameter=runge_parameter,
        xx_test=xx_test,
        yy_test=yy_test,
        target_directory=target_directory,
    )

    # --- Iterate over polynomial degrees
    num_poly_degrees = max_poly_degree - min_poly_degree + 1
    linf_errors = np.empty((num_poly_degrees, 2))
    pool = Pool(NUM_PROCESSES)
    for ind, res in enumerate(
            tqdm(
                pool.imap(wrap_fun, range(min_poly_degree, max_poly_degree + 1)),
                total=num_poly_degrees,
                bar_format=utils.BAR_FORMAT,
            )
    ):
        # --- Compute l-inf of the errors
        linf_errors[ind, 0] = res[0]
        linf_errors[ind, 1] = res[1]

    # --- Dump the result
    fname = (
        f"errors-equadratures-"
        f"dim_{spatial_dimension}-"
        f"nmin_{min_poly_degree}-"
        f"nmax_{max_poly_degree}-"
        f"runge_param_{str(runge_parameter).replace('.', '_')}"
        f".csv"
    )

    np.savetxt(fname, linf_errors, delimiter=",")


if __name__ == "__main__":
    run_equadratures_err()

