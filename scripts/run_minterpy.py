"""
Script to execute interpolation experiments with Minterpy.
"""
import click
import minterpy as mp
import numpy as np

from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

import utils


NUM_PROCESSES = 10


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
    "-p",
    "--lp-degree",
    required=True,
    type=float,
    help="lp-degree of the polynomial",
)
@click.option(
    "-rp",
    "--runge-parameter",
    required=True,
    type=float,
    help="Parameter of the Runge function"
)
@click.option(
    "-nb",
    "--num-batches",
    required=True,
    type=int,
    help="Number of batches in the evaluation",
)
def run_minterpy(
    spatial_dimension: int,
    min_poly_degree: int,
    max_poly_degree: int,
    lp_degree: float,
    runge_parameter: float,
    num_batches: int,
):
    """Run the interpolation experiment on the Runge function with Minterpy.
    """
    # --- Create a testing dataset
    rng = np.random.default_rng(utils.RANDOM_SEED)
    xx_test = -1 + 2 * rng.random((utils.NUM_TEST, spatial_dimension))
    yy_test = utils.runge_vec(xx_test, runge_parameter)

    # --- Fix the runge parameter
    fun = partial(
        utils.runge_vec,
        parameter=runge_parameter,
    )

    # --- Iterate over polynomial degrees
    num_poly_degrees = max_poly_degree - min_poly_degree + 1
    linf_errors = np.empty((num_poly_degrees, 2))
    for i, poly_degree in tqdm(
            enumerate(range(min_poly_degree, max_poly_degree + 1)),
            total=num_poly_degrees,
            bar_format=utils.BAR_FORMAT,
    ):

        # --- Create an interpolant
        interpolant = mp.interpolate(
            fun,
            spatial_dimension,
            poly_degree,
            lp_degree,
        )
        nwt_poly = interpolant.to_newton()

        # ---- Evaluate the interpolant
        yy_interpol = num_batches * [None]

        xx_split = np.array_split(xx_test, num_batches)

        pool = Pool(NUM_PROCESSES)
        for ind, res in enumerate(
            tqdm(
                pool.imap(nwt_poly, xx_split),
                total=num_batches,
                leave=False,
                bar_format=utils.BAR_FORMAT,
            )
        ):
            yy_interpol[ind] = res
        yy_interpol = np.concatenate(yy_interpol)

        # --- Compute l-inf of the errors
        linf_errors[i, 0] = len(nwt_poly.multi_index)
        linf_errors[i, 1] = utils.compute_linf_error(yy_test, yy_interpol)

    if lp_degree == np.inf:
        lp_degree_str = "inf"
    else:
        lp_degree_str = str(lp_degree)

    np.savetxt(
        "errors-minterpy-"
        f"dim_{spatial_dimension}-"
        f"lp_{lp_degree_str}-"
        f"nmin_{min_poly_degree}-"
        f"nmax_{max_poly_degree}-"
        f"runge_param_{str(runge_parameter).replace('.', '_')}"
        f".csv",
        linf_errors,
        delimiter=",",
    )


if __name__ == "__main__":
    run_minterpy()

