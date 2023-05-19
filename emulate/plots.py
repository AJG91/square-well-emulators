"""
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.typing import ArrayLike

sim_kwargs = dict(ls="solid", lw=1, zorder=1)
emu_kwargs = dict(dash_capstyle="round", ls=(0, (0.1, 2)), zorder=2, lw=9, 
                  path_effects=[mpe.withStroke(linewidth=4, foreground="k")])
basis_kwargs = dict(c="grey", lw=0.5, zorder=-0.5, alpha=0.3)

emu_color = "lightsalmon"
sim_color = "black"

emu_label = "Emulator"
sim_label = "Simulator"
basis_label = "Basis"

def setup_rc_params(presentation=False, constrained_layout=True, usetex=True, dpi=400):
    if presentation:
        fontsize = 11
    else:
        fontsize = 9
    black = "k"

    mpl.rcdefaults()  # Set to defaults
    x_minor_tick_size = y_minor_tick_size = 2.4
    x_major_tick_size = y_major_tick_size = 3.9

    # mpl.rc("text", usetex=True)
    mpl.rcParams["font.size"] = fontsize
    mpl.rcParams["text.usetex"] = usetex
    # mpl.rcParams["text.latex.preview"] = True
    mpl.rcParams["font.family"] = "serif"

    mpl.rcParams["axes.labelsize"] = fontsize
    mpl.rcParams["axes.edgecolor"] = black
    # mpl.rcParams['axes.xmargin'] = 0
    mpl.rcParams["axes.labelcolor"] = black
    mpl.rcParams["axes.titlesize"] = fontsize

    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["xtick.labelsize"] = fontsize
    mpl.rcParams["ytick.labelsize"] = fontsize
    mpl.rcParams["xtick.color"] = black
    mpl.rcParams["ytick.color"] = black
    # Make the ticks thin enough to not be visible at the limits of the plot (over the axes border)
    mpl.rcParams["xtick.major.width"] = mpl.rcParams["axes.linewidth"] * 0.95
    mpl.rcParams["ytick.major.width"] = mpl.rcParams["axes.linewidth"] * 0.95
    # The minor ticks are little too small, make them both bigger.
    mpl.rcParams["xtick.minor.size"] = x_minor_tick_size  # Default 2.0
    mpl.rcParams["ytick.minor.size"] = y_minor_tick_size
    mpl.rcParams["xtick.major.size"] = x_major_tick_size  # Default 3.5
    mpl.rcParams["ytick.major.size"] = y_major_tick_size
    plt.rcParams["xtick.minor.visible"] =  True
    plt.rcParams["ytick.minor.visible"] =  True

    ppi = 72  # points per inch
    mpl.rcParams["figure.titlesize"] = fontsize
    mpl.rcParams["figure.dpi"] = 150  # To show up reasonably in notebooks
    mpl.rcParams["figure.constrained_layout.use"] = constrained_layout
    # 0.02 and 3 points are the defaults:
    # can be changed on a plot-by-plot basis using fig.set_constrained_layout_pads()
    mpl.rcParams["figure.constrained_layout.wspace"] = 0.02
    mpl.rcParams["figure.constrained_layout.hspace"] = 0.0
    mpl.rcParams["figure.constrained_layout.h_pad"] = 0#3.0 / ppi
    mpl.rcParams["figure.constrained_layout.w_pad"] = 0#3.0 / ppi

    mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsfonts}"

    mpl.rcParams["legend.title_fontsize"] = fontsize
    mpl.rcParams["legend.fontsize"] = fontsize
    mpl.rcParams[
        "legend.edgecolor"
    ] = "inherit"  # inherits from axes.edgecolor, to match
    mpl.rcParams["legend.facecolor"] = (
        1,
        1,
        1,
        0.6,
    )  # Set facecolor with its own alpha, so edgecolor is unaffected
    mpl.rcParams["legend.fancybox"] = True
    mpl.rcParams["legend.borderaxespad"] = 0.8
    mpl.rcParams[
        "legend.framealpha"
    ] = None  # Do not set overall alpha (affects edgecolor). Handled by facecolor above
    mpl.rcParams[
        "patch.linewidth"
    ] = 0.8  # This is for legend edgewidth, since it does not have its own option

    mpl.rcParams["hatch.linewidth"] = 0.5
    
    return None

def compute_error(
    A: ArrayLike, 
    B: ArrayLike, 
    error: str
) -> ArrayLike:
    """
    Used to calculate relative or absolute error.

    Parameters
    ----------
    A : array
        Expected values.
    B : array
        Actual values observed.
    error : str
        Used to choose between relative or absolute errors.
        'Rel.' == relative errors
        'Abs.' == absolute errors

    Returns
    -------
    err : array
        Errors
    """
    if error == 'Rel.':
        return 2 * abs(A - B) / (abs(A) + abs(B))
    elif error == 'Abs.':
        return abs(A - B)
    else:
        raise ValueError('Check error input!')
        
        fig, ax = plt.subplots(1, 2, figsize=(16, 5))

def plot_scattering_wf(ps, psi_exact, psi_var, psi_b, V):
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    
    ax[0].plot(ps, V, color="black")
    ax[0].plot(ps, psi_var, c=emu_color, label=emu_label, **emu_kwargs)
    ax[0].plot(ps, psi_exact, c=sim_color, label=sim_label, **sim_kwargs)

    for wf_i in psi_b:
        ax[0].plot(ps, wf_i, **basis_kwargs)

    ax[0].plot([], [], label=basis_label, **basis_kwargs)

    ax[0].set_xlim(ps[0], ps[-1])
    props = dict(boxstyle="round", facecolor="white", alpha=0.0)
    ax[0].text(0.38, 0.25, r"$V_0$", fontsize=20, 
             transform=ax[0].transAxes, ha="right", bbox=props)    

    ax[0].set_xlabel(r"$r$ [fm]", fontsize=20)
    ax[0].set_ylabel(r"$u(r)$ [fm]", fontsize=20)
    ax[0].tick_params(axis='both', which='major', labelsize=20)
    ax[0].legend(bbox_to_anchor=(0.25, 1), loc="upper left", fontsize=20)

    ax[1].semilogy(ps, compute_error(psi_exact, psi_var, "Rel."), 
                   color='red', label=r'Std.', lw=2)
    ax[1].set_xlim(ps[0], ps[-1])
    ax[1].tick_params(axis='both', which='major', labelsize=20)
    ax[1].set_xlabel(r"$r$ [fm]", fontsize=20)
    ax[1].set_ylabel(r"Rel. Error", fontsize=20)

    fig.tight_layout()
    fig.savefig("plots/square_well_wfs_emulator_scattering.png", bbox_inches="tight")
    
    return None

def plot_bound_results(V0_pred, V0_b, E_b, E_sim, E_emu):
    for i, V0_i in enumerate(V0_b):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        ax1.plot(V0_pred, E_sim[i], c=sim_color, label=sim_label, **sim_kwargs)
        ax1.plot(V0_pred, E_emu[i], c=emu_color, label=emu_label, **emu_kwargs)

        for j in range(0, i + 1):
            ax1.plot(V0_b[j], E_b[j], marker='*', 
                     color='green', markersize=12, linestyle="none")

        ax1.plot([], [], label=basis_label, marker='*', 
                 color='green', markersize=12, linestyle="none")
        ax1.set_xlabel(r'$V_0$ [MeV]', fontsize=20)
        ax1.set_ylabel(r'$E_0$ [MeV]', fontsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax1.legend(loc='upper right', fontsize=20)

        ax2.semilogy(V0_pred, compute_error(E_sim[i], E_emu[i], "Rel."), 
                     color='red', label=f'Relative')
        ax2.set_ylim(1e-15, 1e2)
        ax2.tick_params(axis='both', which='major', labelsize=20)
        ax2.set_xlabel(r'$V_0$ [MeV]', fontsize=20)
        ax2.set_ylabel(r'Error', fontsize=20)
        ax2.legend(loc='upper right', fontsize=20)

        fig.tight_layout()
        fig.savefig("plots/square_well_wfs_emulator_bound_" + str(i) + ".png", bbox_inches="tight")
        
    return None

