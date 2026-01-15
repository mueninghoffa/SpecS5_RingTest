from typing import TypeAlias, cast

import lmfit
import numpy as np
import numpy.typing as npt
from astropy.io import fits
from matplotlib import collections as mc
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
    assert np.shape(xy)[0] == 2
    x = xy[0]
    y = xy[1]
    r = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
    return A * np.exp(-1 * (R - r) ** 2 / (2 * sigma**2)) + C


def guess_plot(
    img_data: RealNDArray,
    xygrid: RealNDArray,
    guess_params: dict[str, float],
) -> Figure:

    img_shape = np.shape(img_data)
    aspect_ratio = img_shape[0] / img_shape[1]
    fig, ax = plt.subplots(figsize=(5 / aspect_ratio + 2, 5))
    plt.pcolormesh(*xygrid, img_data, shading="nearest")
    # shading='nearest' centers squares on img_data points
    cbar = plt.colorbar()
    cbar.set_label("Pixel values", fontsize=14)

    # location of brightest pixel (A guess)
    max_idx = np.unravel_index(np.argmax(img_data), img_data.shape)
    xmax = xygrid[0][max_idx]
    ymax = xygrid[1][max_idx]

    # from photocenter to brightest pixel
    R_guess_line = mc.LineCollection(
        [[(guess_params["X"], guess_params["Y"]), (xmax, ymax)]],
        colors=(1, 0, 0, 1),
        linewidths=2,
        label="R guess",
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
        label="sigma guess",
    )
    ax.add_collection(R_guess_line)
    ax.add_collection(sigma_guess_line)

    plt.scatter(
        [xmax],
        [ymax],
        color="blue",
        marker="o",
        s=15,
        zorder=10,
        label="brightest pixel",
    )
    plt.scatter(
        [guess_params["X"]],
        [guess_params["Y"]],
        color="white",
        marker="+",
        zorder=10,
        label="photocenter",
    )

    ax.legend(ncol=4, loc="upper center", fontsize=8)
    ax.set_xlabel("x (mm)", fontsize=16)
    ax.set_ylabel("y (mm)", fontsize=16)

    return fig


def guess_ring_params(
    img_data: RealNDArray,
    xygrid: RealNDArray,
    show_plot: bool = False,
    log_plot: bool = False,
) -> dict[str, float]:

    guess_params = dict()

    # Guess A to be max pixel
    # This requires that hot pixels be absent from image data
    guess_params["A"] = img_data.max()

    # Note working very well
    # Guess C from histogram features
    # hist = np.histogram(img_data.flatten(), bins=50)
    # drop = hist[1][int(np.argmax(np.abs(np.diff(hist[0]))))]
    # noise_width = drop - img_data.min()
    # guess_params["C"] = img_data.min() + noise_width / 2
    guess_params["C"] = 0

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
    guess_params["sigma"] = (pixFWHM - 2 * guess_params["R"]) / 2.355
    # 2 * sqrt( 2 * ln(2) ) =~ 2.355

    if show_plot or log_plot:
        fig = guess_plot(img_data, xygrid, guess_params)
        logger.debug("Guess params plot generated")

        if log_plot:
            plot_logger(fig)
        if show_plot:
            plt.show(block=True)

    return guess_params


def plot_fit_result(
    fit_result: lmfit.model.ModelResult, img_data: RealNDArray, xygrid: RealNDArray
) -> Figure:
    # TODO: show best fit values in a legend
    fit_data = cast(RealNDArray, fit_result.eval(xy=xygrid))
    residuals = img_data - fit_data

    img_shape = np.shape(img_data)
    aspect_ratio = img_shape[0] / img_shape[1]
    fig, axs = plt.subplots(
        figsize=(3 * 5 / aspect_ratio + 8, 5),
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

    axs[0].set_title("Data", fontsize=20)
    axs[2].set_title("Best Fit", fontsize=20)
    axs[3].set_title("Residuals", fontsize=20)

    axs[0].set_ylabel("y (mm)", fontsize=16)
    for ax in (axs[0], axs[2], axs[3]):
        ax.set_xlabel("x (mm)", fontsize=16)

    axs[1].set_ylabel("Pixel Values", fontsize=14, rotation=270, labelpad=15)
    axs[4].set_ylabel("Data - Fit", fontsize=14, rotation=270, labelpad=15)

    return fig


def calculate_weights(
    img_data: RealNDArray, guess_params: dict[str, float]
) -> RealNDArray:
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
    xlist = np.arange(x_num_pix) - x_num_pix / 2
    ylist = np.arange(y_num_pix) - y_num_pix / 2
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
        fig = plot_fit_result(result, img_data, xygrid)
        logger.debug("Fit results plot generated")
        if log_plots[1]:
            plot_logger(fig)
        if show_plots[1]:
            plt.show(block=True)

    return result


def run() -> None:
    set_up_logging(log_to_console=True, log_to_file=True)
    global logger
    logger = get_logger(__name__)
    img_path = "./ueye_fits_images/20260113_4103056678/ring_30s_Dsub/20260113_185250_ring_30s_Dsub_0016.fits"
    img_data = fits.open(img_path)[1].data  # pyright: ignore reportAttributeAccessIssue
    fit_gaussian_ring(img_data, show_plots=(True, True), log_plots=True)


if __name__ == "__main__":
    run()
