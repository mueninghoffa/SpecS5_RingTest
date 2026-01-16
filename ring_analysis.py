"""
Fit an image of a ring with a 2D Gaussian ring model.
"""

import inspect
from typing import TypeAlias, cast

import lmfit
import numpy as np
import numpy.typing as npt
from astropy.io import fits
from matplotlib import collections as mc
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from logging_config import get_logger, plot_logger, set_up_logging

if __name__ != "__main__":
    logger = get_logger(__name__)

RealNDArray: TypeAlias = npt.NDArray[np.floating | np.integer]


def gaussian_ring(
    xy: RealNDArray,
    A: float,
    X: float,
    Y: float,
    R: float,
    sigma: float,
    C: float,
) -> np.ndarray:
    r"""
    Calculate the value of a Gaussian ring given the parameters.

    xy : 3D numpy array containing real numbers
        A numpy array containing the x and y positions at which to
        calculate. The outermost dimension of `xy` should contain two 2D
        arrays, one for the x positions and one for the y positions.
    A : float
        Amplitude
    X, Y : float
        Position of the center
    R : float
        Radius
    sigma : float
        Standard deviation of the Gaussian component
    C : float
        Constant background value

    Returns
    -------
    2D numpy array
        A 2D array containing the value at the positions in `xy`. This will
        have the same shape as the inner two dimensions of `xy`.

    Notes
    -----
    A Gaussian ring has a circle at which the maximum amplitude is reached,
    with the value falling off with radial distance from the circle.

    $$
    G( x, y; A, X, Y, R, \sigma, C ) =
    A \exp{ \left( - \frac{
        \left( R - \sqrt{ \left( X - x \right)^2 + \left( Y - y \right)^2 } )^2 }
                  { 2 \sigma^2 } \right) } + C
    $$
    """
    assert np.shape(xy)[0] == 2
    x = xy[0]
    y = xy[1]
    r = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
    return A * np.exp(-1 * (R - r) ** 2 / (2 * sigma**2)) + C


# Order of fit parameters in gaussian_ring
gaussian_ring_signature = inspect.signature(gaussian_ring)
PARAMETERS_ORDERED = list(gaussian_ring_signature.parameters.keys())[1:]


def guess_plot(
    img_data: RealNDArray,
    xygrid: RealNDArray,
    guess_params: dict[str, float],
) -> Figure:
    """
    Visually represent the parameter guessed values on the image.

    Parameters
    ----------
    img_data : 2D numpy array of real numbers
        Original image data to be plotted.
    xy_grid : 3D numpy array of position for `img_data`
        A numpy array containing the x and y positions of the `img_data`
        values. The outer dimension should have length two, containing the
        x and y positions as separate 2D arrays. The x and y position
        arrays must be the same shape as `img_data`.
    guess_params : dict(str : float)
        Dictionary of parameter guessed values.

    Returns
    -------
    matplotlib.figure.Figure
    """

    img_shape = np.shape(img_data)
    aspect_ratio = img_shape[0] / img_shape[1]
    fig, ax = plt.subplots(figsize=(5 / aspect_ratio + 2, 5))
    mesh = plt.pcolormesh(*xygrid, img_data, shading="nearest")
    # shading='nearest' centers squares on img_data points
    cbar = plt.colorbar(mesh)
    cbar.set_label("Pixel & Best Fit values", fontsize=14)

    # location of brightest pixel (A guess)
    max_idx = np.unravel_index(np.argmax(img_data), img_data.shape)
    xmax = int(xygrid[0][max_idx])
    ymax = int(xygrid[1][max_idx])
    plt.scatter(
        [xmax],
        [ymax],
        color="blue",
        marker="o",
        s=15,
        zorder=10,
        label=f"brightest pixel: ({xmax:d}, {ymax:d})",
    )

    # C-guess colored square
    C_guess_color = mesh.cmap(mesh.norm(guess_params["C"]))
    rect_w, rect_h = 0.15, 0.15
    rect_x, rect_y = 1 - rect_w, 0
    rect = patches.Rectangle(
        (rect_x, rect_y),
        rect_w,
        rect_h,
        linewidth=2,
        edgecolor="white",
        facecolor=C_guess_color,
        transform=ax.transAxes,  # use axes coords, not data coords
        zorder=10,
    )
    ax.add_patch(rect)

    rect_cx = rect_x + rect_w / 2
    rect_cy = rect_y + rect_h / 2
    ax.text(
        rect_cx,
        rect_cy,
        f"C guess: {guess_params["C"]:.1f}",
        color="white",
        fontsize=8,
        ha="center",  # horizontal/vertical alignment
        va="center",
        transform=ax.transAxes,  # use axes coords
        zorder=11,
    )

    # from photocenter to brightest pixel
    R_guess_line = mc.LineCollection(
        [[(guess_params["X"], guess_params["Y"]), (xmax, ymax)]],
        colors=(1, 0, 0, 1),
        linewidths=2,
        label=f"R guess: {guess_params["R"]:.1f}",
    )
    # line of length of sigma guess across ring
    sigma_guess_line = mc.LineCollection(
        [
            [
                (
                    guess_params["X"],
                    guess_params["Y"] + guess_params["R"] - guess_params["sigma"] / 2,
                ),
                (
                    guess_params["X"],
                    guess_params["Y"] + guess_params["R"] + guess_params["sigma"] / 2,
                ),
            ]
        ],
        colors=(1, 0.4, 0.7, 1),
        linewidths=3,
        label=f"sigma guess: {guess_params["sigma"]:.1f}",
    )
    ax.add_collection(R_guess_line)
    ax.add_collection(sigma_guess_line)

    # photocenter position
    plt.scatter(
        [guess_params["X"]],
        [guess_params["Y"]],
        color="white",
        marker="+",
        zorder=10,
        label=f"photocenter: ({guess_params["X"]:.1f}, {guess_params["Y"]:.1f})",
    )

    ax.legend(ncol=2, loc="upper center", fontsize=8)
    ax.set_xlabel("x (pix)", fontsize=16)
    ax.set_ylabel("y (pix)", fontsize=16)

    return fig


def guess_ring_params(
    img_data: RealNDArray,
    xygrid: RealNDArray,
    show_plot: bool = False,
    log_plot: bool = False,
) -> dict[str, float]:
    """
    Generate reasonable parameter initial values with heuristics.

    Parameter heuristics:

        * ``A``: value of the brightest pixel
        * ``C``: median pixel value
        * ``X``, ``Y``: photocenter (center of mass)
        * ``R``: distance from photocenter to brightest pixel
        * ``sigma``: determine FWHM_(full ring) of 1D-collapsed profile,
            then FWHM_(ring profile) ~ FWHM_(full ring)/2 - ``R``. Convert
            FWHM to ``sigma``.

    Several of these heuristics do not work if hot pixels are not removed
    from `img_data`.

    Parameters
    ----------
    img_data : 2D numpy array of real numbers
        The data to be fit.
    xy_grid : 3D numpy array of position for `img_data`
        A numpy array containing the x and y positions of the `img_data`
        values. The outer dimension should have length two, containing the
        x and y positions as separate 2D arrays. The x and y position
        arrays must be the same shape as `img_data`.
    show_plot, log_plot : bool, default=False
        Whether to show and/or log a plot visually showing the guessed values.

    Returns
    -------
    dict(str : float)
        Dictionary of parameter guesses.

    See Also
    --------
    guess_plot : generate plot of image with guess values shown
    """

    guess_params = dict()

    # Guess A to be max pixel
    # This requires that hot pixels be absent from image data
    guess_params["A"] = img_data.max()

    # Guess C to be median pixel value
    guess_params["C"] = np.median(img_data.flatten())

    # Guess X, Y from photocenter
    norm_img_data = img_data - guess_params["C"]
    guess_params["X"] = np.sum(xygrid[0] * norm_img_data) / np.sum(norm_img_data)
    guess_params["Y"] = np.sum(xygrid[1] * norm_img_data) / np.sum(norm_img_data)

    # Guess R from distance between photocenter and brightest pixel
    # This requires that hot pixels be absent from image data
    max_idx = np.unravel_index(np.argmax(img_data), img_data.shape)
    xmax = xygrid[0][max_idx]
    ymax = xygrid[1][max_idx]
    guess_params["R"] = np.sqrt(
        (guess_params["X"] - xmax) ** 2 + (guess_params["Y"] - ymax) ** 2
    )

    # Guess sigma from FWHM of ring minus R
    img_data1d = np.sum(img_data, axis=0)
    halfmax = (img_data1d.max() - img_data1d.min()) / 2 + img_data1d.min()
    img_datahalfmax = np.abs(img_data1d - halfmax)
    pixFWHM = abs(np.argmin(img_datahalfmax) - np.argmin(img_datahalfmax[::-1]))
    guess_params["sigma"] = abs((pixFWHM - 2 * guess_params["R"]) / 2.355)
    # 2 * sqrt( 2 * ln(2) ) =~ 2.355

    if show_plot or log_plot:
        fig = guess_plot(img_data, xygrid, guess_params)
        logger.debug("Guess params plot generated")

        if log_plot:
            plot_logger(fig)
        if show_plot:
            plt.show(block=True)
        else:
            plt.close()

    return guess_params


def plot_fit_strs(
    val_unc_params: dict[str, str],
) -> list[str]:
    """
    String together parameter values and uncertainties in the correct order.

    Parameters
    ----------
    val_unc_params : dict(str : str)
        Dictionary of parameter values and uncertainties. Dictionary values
        are a string of the form "<value>+/-<uncertainty>" rounded to the
        first significant figure of the uncertainty.

    Returns
    -------
    list of strings
        List of strings of the form "<name> = <value>+/-<uncertainty>".
    """
    result_str_list = [
        f"{name} = {val_unc_params[name]}" for name in PARAMETERS_ORDERED
    ]
    return result_str_list


def plot_fit_result(
    best_fit_params: dict[str, float],
    unc_params: dict[str, str],
    img_data: RealNDArray,
    xygrid: RealNDArray,
) -> Figure:
    """
    Plot the image, the best fit, and the residuals side by side.

    The image and the best fit share the same color map. Absolute residuals
    are shown with an independent color map. The values of the best fit
    parameters are shown in the upper righthand corner of the best fit plot.

    Parameters
    ----------
    best_fit_params : dict(str : float)
        Dictionary of initial parameter values.
    unc_params : dict(str, str)
        Parameter-name keyed dictionary containing a string of the best-fit
        value and its uncertainty.
    img_data : 2D numpy array of real numbers
        Original image data to be plotted.
    xy_grid : 3D numpy array of position for `img_data`
        A numpy array containing the x and y positions of the `img_data`
        values. The outer dimension should have length two, containing the
        x and y positions as separate 2D arrays. The x and y position
        arrays must be the same shape as `img_data`.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure of results.
    """
    best_fit_values = [best_fit_params[name] for name in PARAMETERS_ORDERED]
    fit_data = gaussian_ring(xygrid, *best_fit_values)
    residuals = img_data - fit_data

    img_shape = np.shape(img_data)
    aspect_ratio = img_shape[0] / img_shape[1]
    fig, axs = plt.subplots(
        figsize=(3 * 5 / aspect_ratio + 2, 5),
        nrows=1,
        ncols=5,
        layout="constrained",
        gridspec_kw=dict(width_ratios=(1, 0.03, 1, 1, 0.03)),
    )
    vmin = min(img_data.min(), fit_data.min())
    vmax = max(img_data.max(), fit_data.max())
    axs[0].pcolormesh(*xygrid, img_data, shading="nearest", vmin=vmin, vmax=vmax)
    fit_c = axs[2].pcolormesh(
        *xygrid, fit_data, shading="nearest", vmin=vmin, vmax=vmax
    )
    fig.colorbar(fit_c, cax=axs[1])
    resid_c = axs[3].pcolormesh(*xygrid, residuals, shading="nearest")
    fig.colorbar(resid_c, cax=axs[4])

    # best fit values and uncertainties
    results_str = ("\n").join(plot_fit_strs(unc_params))
    text_x, text_y = 0.99, 0.98
    axs[2].text(
        text_x,
        text_y,
        results_str,
        ha="right",
        va="top",
        fontsize=8,
        color="black",
        zorder=10,
        transform=axs[2].transAxes,  # use axes coords, not data coords
        bbox=dict(edgecolor="black", facecolor="white", alpha=0.6, boxstyle="round"),
    )

    axs[0].set_title("Data", fontsize=20)
    axs[2].set_title("Best Fit", fontsize=20)
    axs[3].set_title("Residuals", fontsize=20)

    axs[0].set_ylabel("y (pix)", fontsize=16)
    for ax in (axs[0], axs[2], axs[3]):
        ax.set_xlabel("x (pix)", fontsize=16)

    axs[1].set_ylabel("Pixel Values", fontsize=14, rotation=270, labelpad=15)
    axs[4].set_ylabel("Data - Fit", fontsize=14, rotation=270, labelpad=15)

    return fig


def calculate_weights(
    img_data: RealNDArray, guess_params: dict[str, float]
) -> RealNDArray:
    r"""
    Calculates weights for the pixel values based on Poissonian and pixel noise.

    Uses the initial guess values for the R, X, and Y parameters to define
    a box containing the ring. Pixels outside this box are used to
    calculate the pixel noise. Pixel weights are calculated as the square
    root of the sum of the squared Poisonnian noise and the square pixel noise.

    $$ \sigma_\text{tot} = \sqrt{N + \sigma_\text{pix}^2} $$

    Parameters
    ----------
    img_data : 2D numpy array of real numbers
        Image data to calculate weights for.
    guess_params : dict(str : float)
        Dictionary of initial parameter values.

    Returns
    -------
    2D numpy array of real numbers, same shape as `img_data`

    See Also
    --------
    guess_ring_params : Guesses parameter initial values.
    fit_gaussian_ring : Fit an image with a Gaussian ring.

    Notes
    -----
    The box to contain the ring is centered on the initial guesses for the
    ring's center (``X``, ``Y``), and has a side length of 3 times the
    initial gess for the radius (``R``). This is a hueristic that relies on
    the initial guesses being reasonable.
    """
    # Use pixels to calculate background if they lie outside the box
    # centered on the photocenter with sidelength 3 times guessed radius
    img_shape = img_data.shape
    exclusion_radius = 1.5 * guess_params["R"]
    x_bounds = (
        np.array(
            [guess_params["X"] - exclusion_radius, guess_params["X"] + exclusion_radius]
        )
        + img_shape[0]
    )
    y_bounds = (
        np.array(
            [guess_params["Y"] - exclusion_radius, guess_params["Y"] + exclusion_radius]
        )
        + img_shape[1]
    )
    x_bounds = np.round(x_bounds).astype(np.int32)
    y_bounds = np.round(y_bounds).astype(np.int32)

    background = []
    background += list(img_data[: x_bounds[0]].flatten())
    background += list(img_data[x_bounds[1] :].flatten())
    background += list(img_data[x_bounds[0] : x_bounds[1], : y_bounds[0]].flatten())
    background += list(img_data[x_bounds[0] : x_bounds[1], y_bounds[1] :].flatten())
    background = np.array(background).flatten()
    noise = np.std(background)

    clipped_poisson = np.max([img_data, np.zeros_like(img_data)], axis=0)

    weights = np.sqrt(clipped_poisson + noise**2)
    return weights


def fit_gaussian_ring(
    img_data: RealNDArray,
    show_plots: bool | tuple[bool, bool] = False,
    log_plots: bool | tuple[bool, bool] = True,
) -> lmfit.model.ModelResult:
    """
    Fits an image with a Gaussian ring using the lmfit package.

    Can generate plots for the initial guess and the results of the fit.

    Parameters
    ----------
    img_data : 2D numpy array of real numbers
        The data to be fit.
    show_plots, log_plots : bool or 2-tuple of bools, default=False
        Choose to show and/or log the initial guess plot and/or the best
        fit and residual results plot. Pass a pair of bools to control the
        plots independently or a single bool to control both. Shown plots
        must be closed before the function will continue.

    Returns
    -------
    lmfit.model.ModelResult
        The lmfit ``ModelResult`` object that results from the fit.

    See Also
    --------
    guess_ring_params : Guesses parameter initial values.
    calculate_weights : Calculates pixel weights from Poissonian and read noise.

    Notes
    -----
    While fitting with a Gaussian ring is conceptually the simplest way to
    analyze a ring, it is not the most efficient or robust way of doing so.
    For fitting, weights are calculated by summing the expected poisonnian
    noise and the pixel noise in quadrature (see `calculate_weights`). With
    this, the uncertanties returned are meaningful.
    """
    # duck typing
    if isinstance(show_plots, bool):
        show_plots = (show_plots, show_plots)
    assert len(show_plots) == 2 and all([isinstance(val, bool) for val in show_plots])
    if isinstance(log_plots, bool):
        log_plots = (log_plots, log_plots)
    assert len(log_plots) == 2 and all([isinstance(val, bool) for val in log_plots])

    model = lmfit.Model(gaussian_ring)
    params = model.make_params()

    x_num_pix, y_num_pix = np.shape(img_data)
    xlist = np.arange(x_num_pix)
    ylist = np.arange(y_num_pix)
    xygrid = cast(RealNDArray, np.meshgrid(ylist, xlist))
    guess_params = guess_ring_params(
        img_data, xygrid, show_plot=show_plots[0], log_plot=log_plots[0]
    )
    for name, value in guess_params.items():
        params[name].value = value

    img_data_weights = calculate_weights(img_data, guess_params)

    logger.debug("Fitting image with a Gaussian ring")
    result = model.fit(img_data, xy=xygrid, weights=img_data_weights, params=params)
    logger.info(result.fit_report())

    if show_plots[1] or log_plots[1]:
        fig = plot_fit_result(result.best_values, result.uvars, img_data, xygrid)
        logger.debug("Fit results plot generated")
        if log_plots[1]:
            plot_logger(fig)
        if show_plots[1]:
            plt.show(block=True)
        else:
            plt.close()

    return result


def run() -> None:
    """
    Simple script for development and testing.
    """
    set_up_logging(log_to_console=True, log_to_file=True)
    global logger
    logger = get_logger(__name__)
    img_path = "./ueye_fits_images/20260113_4103056678/ring_30s_Dsub/20260113_185250_ring_30s_Dsub_0016.fits"
    img_data = fits.open(img_path)[1].data  # pyright: ignore reportAttributeAccessIssue
    fit_gaussian_ring(img_data, show_plots=(False, False), log_plots=(True, True))


if __name__ == "__main__":
    run()
