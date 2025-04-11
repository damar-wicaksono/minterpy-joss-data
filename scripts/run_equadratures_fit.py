"""
Script to create the function approximation with equadratures.
"""
import click
import equadratures as eq
import pickle

from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

import utils


NUM_PROCESSES = 5


def create_parameter(spatial_dimension: int, order):
    """Create an input for the Runge function."""
    inputs = []
    for _ in range(spatial_dimension):
        inputs.append(eq.Parameter(lower=-1, upper=1, order=order))

    return inputs


def runge_non_vec(xx, parameter: float) -> float:
    """Compute the M-dimensional Runge function (non-vectorized).

    Notes
    -----
    - This function is used to compute the training response
      for interpn; the function signature is specifically
      designed for the function to be passed to `create_train_yy()`.
    """
    yy = 0
    for x in xx:
        yy += x**2

    return 1 / (1 + parameter**2 * yy)


def wrap_model(order: int, spatial_dimension: int, runge_parameter: float):
    # --- Create a set of parameters
    params = create_parameter(spatial_dimension, order)

    # --- Create a basis
    basis = eq.Basis("tensor-grid")

    # --- Create a polynomial
    my_runge = partial(
        runge_non_vec,
        parameter=runge_parameter,
    )
    poly = eq.Poly(
        params,
        basis,
        method="numerical-integration",
        override_cardinality=True,
    )
    poly.set_model(my_runge)

    return poly


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
def run_equadratures_fit(
    spatial_dimension: int,
    min_poly_degree: int,
    max_poly_degree: int,
    runge_parameter: float,
):
    """Construct a function approximation with equadratures.
    """
    # --- Partial function for mapping
    my_fun = partial(
        wrap_model,
        spatial_dimension=spatial_dimension,
        runge_parameter=runge_parameter,
    )

    # --- Iterate over polynomial degrees
    num_poly_degrees = max_poly_degree - min_poly_degree + 1

    with Pool(NUM_PROCESSES) as pool:
        for ind, res in enumerate(
                tqdm(
                    pool.imap(my_fun, range(min_poly_degree, max_poly_degree + 1)),
                    total=num_poly_degrees,
                    bar_format=utils.BAR_FORMAT,
                )
        ):
            # --- Dump the polynomial
            fname = (
                f"fit-equadratures-"
                f"dim_{spatial_dimension}-"
                f"n_{min_poly_degree+ind:03}-"
                f"runge_param_{str(runge_parameter).replace('.', '_')}"
                f".pkl"
            )

            with open(fname, "wb") as f:
                pickle.dump(res, f)


if __name__ == "__main__":
    run_equadratures_fit()

