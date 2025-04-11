"""
Script to create the convergence plot in the paper.

The script assumes the necessary files are located
in "linf-errors/", relative to the root source
directory.
"""
import numpy as np
import matplotlib.pyplot as plt


def create_plot():
    """Create the convergence plot as appear in the paper."""

    # --- Read Minterpy data
    dir_name = "../linf-errors/minterpy"
    fnames_3d = [
        f"{dir_name}/errors-minterpy-dim_3-lp_1_0-nmin_0-nmax_100-runge_param_1_0.csv",
        f"{dir_name}/errors-minterpy-dim_3-lp_2_0-nmin_0-nmax_70-runge_param_1_0.csv",
        f"{dir_name}/errors-minterpy-dim_3-lp_inf-nmin_0-nmax_55-runge_param_1_0.csv",
    ]
    fnames_4d = [
        f"{dir_name}/errors-minterpy-dim_4-lp_1_0-nmin_0-nmax_75-runge_param_1_0.csv",
        f"{dir_name}/errors-minterpy-dim_4-lp_2_0-nmin_0-nmax_45-runge_param_1_0.csv",
        f"{dir_name}/errors-minterpy-dim_4-lp_inf-nmin_0-nmax_40-runge_param_1_0.csv",
    ]

    minterpy_dim_3 = []
    minterpy_dim_4 = []
    for fname_3d, fname_4d in zip(fnames_3d, fnames_4d):
        minterpy_dim_3.append(np.loadtxt(fname_3d, delimiter=","))
        minterpy_dim_4.append(np.loadtxt(fname_4d, delimiter=","))

    # --- Read NDSplines data
    dir_name = "../linf-errors/ndsplines"
    fnames_3d = [
        f"{dir_name}/errors-ndsplines-dim_3-linear-max_1d_100-runge_param_1_0.csv",
        f"{dir_name}/errors-ndsplines-dim_3-cubic-max_1d_100-runge_param_1_0.csv",
        f"{dir_name}/errors-ndsplines-dim_3-quintic-max_1d_100-runge_param_1_0.csv",
    ]
    fnames_4d = [
        f"{dir_name}/errors-ndsplines-dim_4-linear-max_1d_50-runge_param_1_0.csv",
        f"{dir_name}/errors-ndsplines-dim_4-cubic-max_1d_50-runge_param_1_0.csv",
        f"{dir_name}/errors-ndsplines-dim_4-quintic-max_1d_50-runge_param_1_0.csv",
    ]

    ndsplines_dim_3 = []
    ndsplines_dim_4 = []
    for fname_3d, fname_4d in zip(fnames_3d, fnames_4d):
        ndsplines_dim_3.append(np.loadtxt(fname_3d, delimiter=","))
        ndsplines_dim_4.append(np.loadtxt(fname_4d, delimiter=","))

    # --- Read SciPy data
    dir_name = "../linf-errors/interpn"
    fnames_3d = [
        f"{dir_name}/errors-interpn-dim_3-linear-max_1d_100-runge_param_1_0.csv",
        f"{dir_name}/errors-interpn-dim_3-pchip-max_1d_100-runge_param_1_0.csv",
        f"{dir_name}/errors-interpn-dim_3-nearest-max_1d_100-runge_param_1_0.csv",
    ]
    fnames_4d = [
        f"{dir_name}/errors-interpn-dim_4-linear-max_1d_50-runge_param_1_0.csv",
        f"{dir_name}/errors-interpn-dim_4-pchip-max_1d_50-runge_param_1_0.csv",
        f"{dir_name}/errors-interpn-dim_4-nearest-max_1d_50-runge_param_1_0.csv",
    ]

    interpn_dim_4 = []
    interpn_dim_3 = []
    for fname_3d, fname_4d in zip(fnames_3d, fnames_4d):
        interpn_dim_3.append(np.loadtxt(fname_3d, delimiter=","))
        interpn_dim_4.append(np.loadtxt(fname_4d, delimiter=","))

    # --- Read ChaosPy data
    dir_name = "../linf-errors/chaospy"
    chaospy_dim_3 = np.loadtxt(
        f"{dir_name}/errors-chaospy-dim_3-nmin_0-nmax_20-runge_param_1_0.csv",
        delimiter=",",
    )
    chaospy_dim_4 = np.loadtxt(
        f"{dir_name}/errors-chaospy-dim_4-nmin_0-nmax_15-runge_param_1_0.csv",
        delimiter=",",
    )

    # --- Read Equadratures data
    dir_name = "../linf-errors/equadratures"
    equadratures_dim_3 = np.loadtxt(
        f"{dir_name}/errors-equadratures-dim_3-nmin_0-nmax_20-runge_param_1_0.csv",
        delimiter=",",
    )
    equadratures_dim_4 = np.loadtxt(
        f"{dir_name}/errors-equadratures-dim_4-nmin_0-nmax_10-runge_param_1_0.csv",
        delimiter=",",
    )

    # --- Create the plot

    # Common settings
    fig, axs = plt.subplots(2, 1, figsize=(14, 9.5))
    linewidth = 0.5
    markersize = 9
    legend_fontsize = 16
    axis_fontsize = 18
    tick_fontsize = 16
    title_fontsize = 20
    colors = [
        "#4daf4a",
        "#377eb8",
        "#e41a1c",
        "#984ea3",
        "#ff7f00",
    ]
    markers = ["d", "o", "s", "x", "+", ".", "^", "p", "^", "v"]

    # Dimension 3
    axs[0].plot(
        minterpy_dim_3[0][:, 0],
        minterpy_dim_3[0][:, 1],
        linewidth=linewidth,
        marker=markers[0],
        markersize=markersize,
        label=r"Minterpy ($p = 1.0$)",
        color=colors[0],
        markerfacecolor='none',
    )
    axs[0].plot(
        minterpy_dim_3[1][:, 0],
        minterpy_dim_3[1][:, 1],
        linewidth=linewidth,
        marker=markers[1],
        markersize=markersize,
        label=r"Minterpy ($p = 2.0$)",
        color=colors[0],
        markerfacecolor='none',
    )
    axs[0].plot(
        minterpy_dim_3[2][:, 0],
        minterpy_dim_3[2][:, 1],
        linewidth=linewidth,
        marker=markers[2],
        markersize=markersize,
        label=r"Minterpy ($p = \infty$)",
        color=colors[0],
        markerfacecolor='none',
    )
    axs[0].plot(
        interpn_dim_3[1][:, 0],
        interpn_dim_3[1][:, 1],
        linewidth=linewidth,
        marker=markers[3],
        markersize=markersize,
        label=r"SciPy (pchip)",
        color=colors[2],
        markerfacecolor='none',
    )
    axs[0].plot(
        interpn_dim_3[2][:, 0],
        interpn_dim_3[2][:, 1],
        linewidth=linewidth,
        marker=markers[4],
        markersize=markersize,
        label=r"SciPy (nearest)",
        color=colors[2],
        markerfacecolor='none',
    )
    # Add Jitter
    jitter = np.random.normal(0, 1e-4, len(ndsplines_dim_3[0]))

    axs[0].plot(
        ndsplines_dim_3[0][:, 0],
        ndsplines_dim_3[0][:, 1],
        linewidth=linewidth,
        marker=markers[5],
        markersize=markersize,
        label=r"ndsplines (linear)",
        color=colors[1],
    )
    axs[0].plot(
        ndsplines_dim_3[1][:, 0],
        ndsplines_dim_3[1][:, 1],
        linewidth=linewidth,
        marker=markers[6],
        markersize=markersize,
        label=r"ndsplines (cubic)",
        color=colors[1],
        markerfacecolor='none',
    )
    axs[0].plot(
        ndsplines_dim_3[2][:, 0],
        ndsplines_dim_3[2][:, 1],
        linewidth=linewidth,
        marker=markers[7],
        markersize=markersize,
        label=r"ndsplines (quintic)",
        color=colors[1],
        markerfacecolor='none',
    )
    axs[0].plot(
        chaospy_dim_3[:, 0],
        chaospy_dim_3[:, 1],
        linewidth=linewidth,
        marker=markers[8],
        markersize=markersize,
        label=r"Chaospy (Legendre, tensor-grid)",
        color=colors[3],
    )
    axs[0].plot(
        equadratures_dim_3[:, 0],
        equadratures_dim_3[:, 1],
        linewidth=linewidth,
        marker=markers[9],
        markersize=markersize,
        label=r"equadratures (Legendre, tensor-grid)",
        color=colors[4],
    )
    axs[0].set_yscale("log")
    axs[0].set_xscale("log")
    axs[0].set_title("Dimension 3", fontsize=title_fontsize)
    axs[0].set_ylim([1e-15, 5])
    axs[0].set_xlim([0.5, 1e7])
    axs[0].set_ylabel(r"l-$\infty$ error", fontsize=axis_fontsize)
    axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axs[0].grid()

    # Dimension 4
    axs[1].plot(
        minterpy_dim_4[0][:, 0],
        minterpy_dim_4[0][:, 1],
        linewidth=linewidth,
        marker=markers[0],
        markersize=markersize,
        label=r"Minterpy ($p = 1.0$)",
        color=colors[0],
        markerfacecolor='none',
    )
    axs[1].plot(
        minterpy_dim_4[1][:, 0],
        minterpy_dim_4[1][:, 1],
        linewidth=linewidth,
        marker=markers[1],
        markersize=markersize,
        label=r"Minterpy ($p = 2.0$)",
        color=colors[0],
        markerfacecolor='none',
    )
    axs[1].plot(
        minterpy_dim_4[2][:, 0],
        minterpy_dim_4[2][:, 1],
        linewidth=linewidth,
        marker=markers[2],
        markersize=markersize,
        label=r"Minterpy ($p = \infty$)",
        color=colors[0],
        markerfacecolor='none',
    )
    axs[1].plot(
        interpn_dim_4[1][:, 0],
        interpn_dim_4[1][:, 1],
        linewidth=linewidth,
        marker=markers[3],
        markersize=markersize,
        label=r"SciPy (pchip)",
        color=colors[2],
        markerfacecolor='none',
    )
    axs[1].plot(
        interpn_dim_4[2][:, 0],
        interpn_dim_4[2][:, 1],
        linewidth=linewidth,
        marker=markers[4],
        markersize=markersize,
        label=r"SciPy (nearest)",
        color=colors[2],
        markerfacecolor='none',
    )
    axs[1].plot(
        ndsplines_dim_4[0][:, 0],
        ndsplines_dim_4[0][:, 1],
        linewidth=linewidth,
        marker=markers[5],
        markersize=markersize,
        label=r"ndsplines (linear)",
        color=colors[1],
    )
    axs[1].plot(
        ndsplines_dim_4[1][:, 0],
        ndsplines_dim_4[1][:, 1],
        linewidth=linewidth,
        marker=markers[6],
        markersize=markersize,
        label=r"ndsplines (cubic)",
        color=colors[1],
        markerfacecolor='none',
    )
    axs[1].plot(
        ndsplines_dim_4[2][:, 0],
        ndsplines_dim_4[2][:, 1],
        linewidth=linewidth,
        marker=markers[7],
        markersize=markersize,
        label=r"ndsplines (quintic)",
        color=colors[1],
        markerfacecolor='none',
    )
    axs[1].plot(
        chaospy_dim_4[:, 0],
        chaospy_dim_4[:, 1],
        linewidth=linewidth,
        marker=markers[8],
        markersize=markersize,
        label=r"Chaospy (Legendre, tensor-grid)",
        color=colors[3],
    )
    axs[1].plot(
        equadratures_dim_4[:, 0],
        equadratures_dim_4[:, 1],
        linewidth=linewidth,
        marker=markers[9],
        markersize=markersize,
        label=r"equadratures (Legendre, tensor-grid)",
        color=colors[4],
    )
    axs[1].set_yscale("log")
    axs[1].set_xscale("log")
    axs[1].set_title("Dimension 4", fontsize=title_fontsize)
    axs[1].set_ylim([1e-15, 5])
    axs[1].set_xlim([0.5, 1e7])
    axs[1].set_ylabel(r"l-$\infty$ error", fontsize=axis_fontsize)
    axs[1].set_xlabel(
        "Number of data points (or coefficients)",
        fontsize=axis_fontsize,
    )
    axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axs[1].legend(frameon=False, fontsize=legend_fontsize)
    axs[1].grid()

    fig.tight_layout()
    plt.savefig("convergence.png", dpi=600)


if __name__ == "__main__":
    create_plot()

