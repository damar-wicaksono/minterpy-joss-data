"""
Script to execute interpolation experiments with Chaospy.
"""
import chaospy as cp
import click
import numpy as np

from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

import utils


NUM_PROCESSES = 10


def create_joint(spatial_dimension: int):
    """Create a joint input for the Runge function."""
    inputs = []
    for _ in range(spatial_dimension):
        inputs.append(cp.Uniform(-1, 1))

    return cp.J(*inputs)


def wrap_model(xx, model):
    """Wrap the fitted Chaospy model for parallel map."""
    return model(*xx.T)


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
    "-nb",
    "--num-batches",
    required=True,
    type=int,
    help="Number of batches in the evaluation",
)
def run_chaospy(
    spatial_dimension: int,
    min_poly_degree: int,
    max_poly_degree: int,
    runge_parameter: float,
    num_batches: int,
):
    """Run the interpolation experiment on the Runge function with ChaosPy.
    """
    # --- Create a testing dataset
    rng = np.random.default_rng(utils.RANDOM_SEED)
    xx_test = -1 + 2 * rng.random((utils.NUM_TEST, spatial_dimension))
    yy_test = utils.runge_vec(xx_test, runge_parameter)

    # --- Iterate over polynomial degrees
    num_poly_degrees = max_poly_degree - min_poly_degree + 1
    linf_errors = np.empty((num_poly_degrees, 2))
    for i, order in tqdm(
            enumerate(range(min_poly_degree, max_poly_degree + 1)),
            total=num_poly_degrees,
            bar_format=utils.BAR_FORMAT,
    ):

        # --- Create a joint input
        joint = create_joint(spatial_dimension)

        # --- Create quadrature points
        quads = cp.generate_quadrature(order, joint)
        nodes, _ = quads

        # --- Create evaluation
        evals = utils.runge_vec(nodes.T, runge_parameter)

        # --- Create expansion
        expansion = cp.generate_expansion(order, joint)

        # --- Fit the model
        model = cp.fit_quadrature(expansion, *quads, evals)
        my_model = partial(
                wrap_model,
                model=model,
                )

        # ---- Evaluate the interpolant
        yy_interpol = num_batches * [None]

        xx_split = np.array_split(xx_test, num_batches)

        pool = Pool(NUM_PROCESSES)
        for ind, res in enumerate(
            tqdm(
                pool.imap(my_model, xx_split),
                total=num_batches,
                leave=False,
                bar_format=utils.BAR_FORMAT,
            )
        ):
            yy_interpol[ind] = res
        yy_interpol = np.concatenate(yy_interpol)

        # --- Compute l-inf of the errors
        linf_errors[i, 0] = len(nodes.T)
        linf_errors[i, 1] = utils.compute_linf_error(yy_test, yy_interpol)

    np.savetxt(
        "errors-chaospy-"
        f"dim_{spatial_dimension}-"
        f"nmin_{min_poly_degree}-"
        f"nmax_{max_poly_degree}-"
        f"runge_param_{str(runge_parameter).replace('.', '_')}"
        f".csv",
        linf_errors,
        delimiter=",",
    )


if __name__ == "__main__":
    run_chaospy()

