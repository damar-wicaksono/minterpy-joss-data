# Data for Minterpy Paper Submitted to JOSS

This is a README file for the data used in the publication "Minterpy: Multivariate interpolation in Python" submitted to the Journal of Open Source Software (JOSS). The paper contains one figure (two subplots) of convergence comparison between Minterpy and several selected packages.

The organization of the archive directory is as follows:

```  
./
├── README.md       <- This file 
├── linf-errors     <- Directory with convergence data of interpolating 
|                      polynomials from different packages
├── plot            <- Directory with script to create the convergence plot
|                      that appears in the paper; the same plot is also
|                      stored here
├── requirements    <- Directory with package requirements for pip
└── scripts         <- Directory with scripts to reproduce the results stored
                       in "linf-errors"
```

## Numerical experiment details

The numerical experiment is based on the $m$-dimensional Runge function, defined as:

$$
f(\boldsymbol{x};r) = \frac{1}{1 + r^2 \lVert \boldsymbol{x} \rVert^2}, \boldsymbol{x} \in [-1, 1]^m.
$$

We consider two dimensions, $m = 3$ and $m = 4$, with a fixed Runge parameter $r = 1.0$.

The experiment measures the difference between the function and its polynomial approximation $Q_f$ using the infinity norm:

$$
\lVert f - Q_f \rVert_\infty = \max_{\boldsymbol{x} \in [-1, 1]} \lvert f(\boldsymbol{x}) - Q_f(\boldsymbol{x}) \rvert
$$

To evaluate this difference, we sample $1'000'000$ random points over the domain.

For each approximation, we evaluate two key performance metrics:

1. **Computational cost**: The number of function evaluations required to construct the approximation.
2. **Accuracy**: The error between the original function and its approximation, measured using the infinity norm.

We compare the performance of five packages that construct polynomial-based function approximations:

- [Minterpy](https://github.com/minterpy-project/minterpy)
- [Chaospy](https://github.com/jonathf/chaospy)
- [equadratures](https://github.com/equadratures/equadratures)
- [`interpn`](https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.interpolate.interpn.html) from [SciPy](https://github.com/scipy/scipy)
- [ndsplines](https://github.com/kb-press/ndsplines)

Each package has a distinct setup, which is described below.

## Package-specific setups, available results, and reproduction notes

This section provides a detailed description of the experimental setup for each package, explains the available convergence results in the archive, and offers notes on reproducing these results.

The available results are stored in the directory `linf-errors`:

```
./linf-errors/
├── chaospy         <- Convergence results from the Chaospy
├── equadratures    <- Convergence results from equadratures
├── interpn         <- Convergence results from interpn (SciPy)
├── minterpy        <- Convergence results from Minterpy
└── ndsplines       <- Convergence results from ndsplines
```

Each of these directories contains CSV files with two entries per row:

1. **Number of function evaluations**: The number of function evaluations required to construct the approximation.
2. **Infinity norm error**: The accuracy of the approximation, measured in terms of the infinity norm, evaluated at $1,000,000$ random points.

The scripts used to generate and reproduce these results can be found in the `scripts/` directory.

### Minterpy

Minterpy interpolating polynomials of increasing degrees $n$ with different $p$ are constructed, as detailed in the table below.

| No. | Dimension |    p     |    n    | Max. # Coeffs. |
|:---:|:---------:|:--------:|:-------:|:--------------:|
| 1.  |     3     |   1.0    | 0 - 100 |    176'851     | 
| 2.  |     3     |   2.0    | 0 - 70  |    185'366     |
| 3.  |     3     | $\infty$  | 0 - 55  |    175'616     |
| 4.  |     4     |   1.0    | 0 - 70  |   1'150'626    |
| 5.  |     4     |   2.0    | 0 - 40  |    858'463     |
| 6.  |     4     | $\infty$  | 0 - 35  |   1'679'616    |

Note that for Minterpy, the size of the multi-index set corresponds directly to the number of coefficients and the number of function evaluations.

The convergence results for Minterpy are stored in the `linf-errors/minterpy/` directory, relative to the archive source directory. This directory contains the following files, listed in the order corresponding to the setups in the table above:

1. `errors-minterpy-dim_3-lp_1_0-nmin_0-nmax_100-runge_param_1_0.csv`
2. `errors-minterpy-dim_3-lp_2_0-nmin_0-nmax_70-runge_param_1_0.csv`
3. `errors-minterpy-dim_3-lp_inf-nmin_0-nmax_55-runge_param_1_0.csv`
4. `errors-minterpy-dim_4-lp_1_0-nmin_0-nmax_75-runge_param_1_0.csv`
5. `errors-minterpy-dim_4-lp_2_0-nmin_0-nmax_45-runge_param_1_0.csv`
6. `errors-minterpy-dim_4-lp_inf-nmin_0-nmax_40-runge_param_1_0.csv`

These files can be reproduced using the `run_minterpy.py` script stored in the `scripts/` directory. The script can be used as follows:

```bash
$ python run_minterpy.py --help
Usage: run_minterpy.py [OPTIONS]

  Run the interpolation experiment on the Runge function with Minterpy.

Options:
  -m, --spatial-dimension INTEGER
                                  The dimension of the function  [required]
  -nmin, --min-poly-degree INTEGER
                                  Minimum polynomial degree  [required]
  -nmax, --max-poly-degree INTEGER
                                  Maximum polynomial degree  [required]
  -p, --lp-degree FLOAT           lp-degree of the polynomial  [required]
  -rp, --runge-parameter FLOAT    Parameter of the Runge function  [required]
  -nb, --num-batches INTEGER      Number of batches in the evaluation
                                  [required]
  --help                          Show this message and exit.

```

The commands used to produce each of the CSV files are:

1. `python run_minterpy.py -m 3 -nmin 0 -nmax 100 -p 1.0 -rp 1.0 -nb 1000`
2. `python run_minterpy.py -m 3 -nmin 0 -nmax 70 -p 2.0 -rp 1.0 -nb 1000`
3. `python run_minterpy.py -m 3 -nmin 0 -nmax 55 -p inf -rp 1.0 -nb 1000`
4. `python run_minterpy.py -m 4 -nmin 0 -nmax 75 -p 1.0 -rp 1.0 -nb 1000`
5. `python run_minterpy.py -m 4 -nmin 0 -nmax 45 -p 2.0 -rp 1.0 -nb 1000`
6. `python run_minterpy.py -m 4 -nmin 0 -nmax 40 -p inf -rp 1.0 -nb 1000`

### Chaospy

Function approximations via Legendre polynomial expansions in tensorial grid with increasing degree $n$ are constructed using Chaospy as detailed in the table below:

| No. | Dimension |   n    | Max. # Coeffs. |
|:---:|:---------:|:------:|:--------------:|
| 1.  |     3     | 0 - 20 |     9'261      |
| 2.  |     4     | 0 - 15 |     65'536     |

Note that for Chaospy, the number of coefficients corresponds directly to the number of function evaluations. However, it's worth noting that evaluating Chaospy polynomials can become unstable, particularly high degree as shown in the convergence plot.

The convergence results for equadratures are stored in the `linf-errors/chaospy` directory, relative to the archive source directory. This directory contains the following files, listed in the order corresponding to the setups in the table above:

1. `errors-chaospy-dim_3-nmin_0-nmax_20-runge_param_1_0.csv`
2. `errors-chaospy-dim_4-nmin_0-nmax_15-runge_param_1_0.csv`

These files can be reproduced using the `run_interpn.py` script stored in the `scripts/` directory. The script can be used as follows:

```bash
$ python run_chaospy.py --help
Usage: run_chaospy.py [OPTIONS]

  Run the interpolation experiment on the Runge function with ChaosPy.

Options:
  -m, --spatial-dimension INTEGER
                                  The dimension of the function  [required]
  -nmin, --min-poly-degree INTEGER
                                  Minimum polynomial degree  [required]
  -nmax, --max-poly-degree INTEGER
                                  Maximum polynomial degree  [required]
  -rp, --runge-parameter FLOAT    Parameter of the Runge function  [required]
  -nb, --num-batches INTEGER      Number of batches in the evaluation
                                  [required]
  --help                          Show this message and exit.
```

The commands used to produce each of the CSV files are:

1. `python run_chaospy.py -m 3 -nmin 0 -nmax 20 -rp 1 -nb 1000`
2. `python run_chaospy.py -m 4 -nmin 0 -nmax 15 -rp 1 -nb 1000`

### equadratures

Function approximations via Legendre polynomial expansions in tensorial grid with increasing degree $n$ are constructed using equadratures, as detailed in the table below.

| No. | Dimension |   n    | Max. # Coeffs. |
|:---:|:---------:|:------:|:--------------:|
| 1.  |     3     | 0 - 20 |     9'261      |
| 2.  |     4     | 0 - 10 |     14'641     |

Note that for equadratures, the number of coefficients corresponds directly to the number of function evaluations. However, it's worth noting that equadratures can become computationally expensive, particularly in terms of memory consumption, for degrees beyond those specified above.

The convergence results for equadratures are stored in the `linf-errors/equadratures` directory, relative to the archive source directory. This directory contains the following files, listed in the order corresponding to the setups in the table above:

1. `errors-equadratures-dim_3-nmin_0-nmax_20-runge_param_1_0.csv`
2. `errors-equadratures-dim_4-nmin_0-nmax_10-runge_param_1_0.csv`

These files can be reproduced in two steps using the scripts stored in the `scripts/` directory:

1. Constructing the equadrature approximation for the given dimension and degree using the `run_equadratures_fit.py` script.
2. Computing the error in infinity norm given the approximation using the `run_equadratures_err.py` script.

The `run_equadratures_fit.py` script can be used as follows:

```bash
$ python run_equadratures_fit.py --help
Usage: run_equadratures_fit.py [OPTIONS]

  Construct a function approximation with equadratures.

Options:
  -m, --spatial-dimension INTEGER
                                  The dimension of the function  [required]
  -nmin, --min-poly-degree INTEGER
                                  Minimum polynomial degree  [required]
  -nmax, --max-poly-degree INTEGER
                                  Maximum polynomial degree  [required]
  -rp, --runge-parameter FLOAT    Parameter of the Runge function  [required]
  --help                          Show this message and exit.
```

To produce all the approximations from equadratures stored in pickled files (`.pkl`), run the script with the following parameters:

1. `python run_equadratures_fit.py -m 3 -nmin 0 -nmax 20 -rp 1.0`
2. `python run_equadratures_fit.py -m 4 -nmin 0 -nmax 10 -rp 1.0`

The `run_equadratures_err.py` script can be used as follows:

```bash
$ python run_equadratures_err.py --help
Usage: run_equadratures_err.py [OPTIONS]

  Compute the l-inf error given a polynomial from EQ.

Options:
  -m, --spatial-dimension INTEGER
                                  The dimension of the function  [required]
  -nmin, --min-poly-degree INTEGER
                                  Minimum polynomial degree  [required]
  -nmax, --max-poly-degree INTEGER
                                  Maximum polynomial degree  [required]
  -rp, --runge-parameter FLOAT    Parameter of the Runge function  [required]
  -t, --target-directory TEXT     Target directory where pkl files are located
                                  [default: .]
  --help                          Show this message and exit.
```

Assuming that all pickled files are in the current working directory, run the script with the following parameters to produce the CSV files:

1. `python run_equadratures_err.py -m 3 -nmin 0 -nmax 20 -rp 1.0`
2. `python run_equadratures_err.py -m 4 -nmin 0 -nmax 10 -rp 1.0`

### interpn (Scipy)

Three different interpolation methods from the `interpn` module of SciPy are considered:

1. Linear interpolation
2. PCHIP (Piecewise cubic Hermite interpolating polynomial)
3. Nearest interpolation

Note that other interpolation methods, such as splines, are also available in SciPy, but they are mostly identical to the more performant implementations in ndsplines.

The interpolation methods from `interpn` require data given in a rectangular grid. To construct this data, a tensor product of Chebyshev-Lobatto points is used in one dimension, with an increasing number of points $N_{1D}$ considered. The total number of points is $(N_{1D})^m$, where $m$ is the number of dimensions, as shown in the following table.

| No. | Dimension | Method  | Max 1D points | Max. # Points. |
|:---:|:---------:|:-------:|:-------------:|:--------------:|
| 1.  |     3     | linear  |      100      |   1'000'000    |
| 2.  |     3     | nearest |      100      |   1'000'000    |
| 3.  |     3     |  pchip  |      100      |   1'000'000    |
| 4.  |     4     | linear  |      50       |   6'250'000    |
| 5.  |     4     | nearest |      50       |   6'250'000    |
| 6.  |     4     |  pchip  |      50       |   6'250'000    |

In contrast to Minterpy, Chaospy, or equadratures, the complexity of the interpolation method remains fixed, but the number of available interpolation points is increased.

The convergence results for Minterpy are stored in the `linf-errors/interpn/` directory, relative to the archive source directory. This directory contains the following files, listed in the order corresponding to the setups in the table above:

1. `errors-interpn-dim_3-linear-max_1d_100-runge_param_1_0.csv`
2. `errors-interpn-dim_3-nearest-max_1d_100-runge_param_1_0.csv`
3. `errors-interpn-dim_3-pchip-max_1d_100-runge_param_1_0.csv`
4. `errors-interpn-dim_4-linear-max_1d_50-runge_param_1_0.csv`
5. `errors-interpn-dim_4-nearest-max_1d_50-runge_param_1_0.csv`
6. `errors-interpn-dim_4-pchip-max_1d_50-runge_param_1_0.csv`

These files can be reproduced using the `run_interpn.py` script stored in the `scripts/` directory. The script can be used as follows:

```bash
$ python run_interpn.py  --help
Usage: run_interpn.py [OPTIONS]

  Run the interpolation experiment on the Runge function w/ interpn.

Options:
  -m, --spatial-dimension INTEGER
                                  The dimension of the function  [required]
  -mt, --method [linear|slinear|cubic|quintic|pchip|nearest]
                                  Interpolation method  [required]
  -p, --max-1d-points INTEGER     Maximum number of one-dimensional points
                                  [required]
  -rp, --runge-parameter FLOAT    Parameter of the Runge function  [required]
  --help                          Show this message and exit.
```

The commands used to produce each of the CSV files are:

1. `python run_interpn.py -m 3 -mt linear -p 100 -rp 1.0`
2. `python run_interpn.py -m 3 -mt nearest -p 100 -rp 1.0`
3. `python run_interpn.py -m 3 -mt pchip -p 100 -rp 1.0`
4. `python run_interpn.py -m 4 -mt linear -p 50 -rp 1.0`
5. `python run_interpn.py -m 4 -mt nearest -p 50 -rp 1.0`
6. `python run_interpn.py -m 4 -mt pchip -p 50 -rp 1.0`

## Reproducing the plot

The script `plot/create_plot.py` generates the convergence plot presented in the paper. It assumes that all result files are located in the `linf-errors/` directory, as described above.

The generated plot, `convergence.png`, which appears in the paper, is also stored in the same directory.

## Computing environment

The numerical experiments were conducted on a machine running Debian 12 Linux, equipped with a 32-core AMD EPYC processor and 256 GB of RAM. The experiments used Python v3.9.19 and relied on the following Python libraries and packages:

- `numpy==2.0.2`    
- `scipy==1.13.1`
- `matplotlib==3.9.2`
- `minterpy==0.3.0`
- `numba==0.6.0`
- `chaospy==4.3.18`
- `numpoly==1.3.6`
- `equadratures==10` _(incompatible with NumPy `v2.x`; revert to `v1.26.4`)_
- `ndsplines==0.2.0post0`
- `click==8.1.7`
- `tqdm==4.66.5`

The corresponding requirement files are located in the `requirements/` directory, relative to the root of the archive. Due to the incompatibility between equadratures `v10` (latest) and NumPy `v2.x`, alternative requirements are specified in `requirements-equadratures.txt` for experiments involving `equadratures`.