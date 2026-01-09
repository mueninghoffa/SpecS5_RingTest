"""
Functions and a class for interacting with uEye cameras through pyueye.
Written  part by ChatGPT.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from astropy.io import fits
from pyueye import ueye

from ctypes_to_normal import ctypes_to_normal
from ueye_types import UEYE_CAMERA_LIST

from .ueye_commands import (
    alloc_image_mem,
    aoi_cmd,
    exit_camera,
    exposure_cmd,
    force_trigger,
    free_image_mem,
    gamma_cmd,
    get_camera_info,
    get_camera_list,
    get_number_of_cameras,
    get_sensor_info,
    init_camera,
    pixel_clock_cmd,
    set_binning,
    set_hardware_gain,
    set_image_mem,
    set_software_trigger,
)


def number_of_cameras() -> int:
    """
    Returns the number of connected cameras.
    """
    count = ueye.INT()
    get_number_of_cameras(count)
    return ctypes_to_normal(count)


def list_cameras(struct: bool = False) -> List[Dict[str, Any]] | UEYE_CAMERA_LIST:
    """
    Returns a list of camera info objects of connected cameras.

    Returns an empty list if no cameras are connected.

    Parameters
    ----------
    struct : bool, default=False
        Whether to return the pyueye struct as is (``True``) or convert it
        to a dictionary (``False``).
    """

    count = number_of_cameras()
    if count == 0:
        return []

    cam_list = ueye.UEYE_CAMERA_LIST()
    cam_list.dwCount = ueye.UINT(count)
    get_camera_list(cam_list)

    if not struct:
        cam_list = ctypes_to_normal(cam_list)["uci"]

    return cam_list


class UeyeCamera:
    """
    High-level, Pythonic interface to an IDS uEye camera.

    Supports:
    - Software-triggered single exposures
    - ROI, binning, gain, gamma, pixel clock
    - NumPy image retrieval
    - FITS HDU generation and saving
    """

    def __init__(self, camera_id: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.handle = ueye.HIDS(camera_id)

        self._mem_ptr = ueye.c_mem_p()
        self._mem_id = ueye.int()

        self._width = ueye.int()
        self._height = ueye.int()
        self._bpp = ueye.int(8)
        self._bytes_per_pixel = 1

        self._connected = False
        self._last_exposure_time: Optional[str] = None

    # ---- Lifecycle ----

    def connect(self) -> None:
        if self._connected:
            return

        init_camera(self.handle, None, camera_handle=self.handle)
        set_software_trigger(
            self.handle, ueye.IS_SET_TRIGGER_SOFTWARE, camera_handle=self.handle
        )

        self._allocate_image_memory()
        self._connected = True
        self.logger.info("Camera connected")

    def close(self) -> None:
        if not self._connected:
            return

        free_image_mem(
            self.handle, self._mem_ptr, self._mem_id, camera_handle=self.handle
        )
        exit_camera(self.handle, camera_handle=self.handle)

        self._connected = False
        self.logger.info("Camera disconnected")

    # ---- Configuration (properties) ----

    @property
    def exposure_time_ms(self) -> float:
        value = ueye.double()
        exposure_cmd(
            self.handle,
            ueye.IS_EXPOSURE_CMD_GET_EXPOSURE,
            value,
            ueye.sizeof(value),
            camera_handle=self.handle,
        )
        return float(value.value)

    @exposure_time_ms.setter
    def exposure_time_ms(self, ms: float) -> None:
        value = ueye.double(ms)
        exposure_cmd(
            self.handle,
            ueye.IS_EXPOSURE_CMD_SET_EXPOSURE,
            value,
            ueye.sizeof(value),
            camera_handle=self.handle,
        )

    @property
    def gain(self) -> int:
        return 0  # IDS does not provide a simple getter

    @gain.setter
    def gain(self, value: int) -> None:
        set_hardware_gain(
            self.handle,
            value,
            ueye.IS_IGNORE_PARAMETER,
            ueye.IS_IGNORE_PARAMETER,
            ueye.IS_IGNORE_PARAMETER,
            camera_handle=self.handle,
        )

    @property
    def gamma(self) -> float:
        return 1.0

    @gamma.setter
    def gamma(self, value: float) -> None:
        gamma_val = ueye.int(int(value * 100))
        gamma_cmd(
            self.handle,
            ueye.IS_GAMMA_CMD_SET,
            gamma_val,
            ueye.sizeof(gamma_val),
            camera_handle=self.handle,
        )

    @property
    def pixel_clock(self) -> int:
        return 0

    @pixel_clock.setter
    def pixel_clock(self, mhz: int) -> None:
        value = ueye.int(mhz)
        pixel_clock_cmd(
            self.handle,
            ueye.IS_PIXELCLOCK_CMD_SET,
            value,
            ueye.sizeof(value),
            camera_handle=self.handle,
        )

    # ---- Acquisition ----

    def expose(self) -> None:
        """
        Triggers a single software exposure.
        """
        force_trigger(self.handle, camera_handle=self.handle)

    def read_numpy(self, timeout_ms: int = 5000) -> np.ndarray:
        """
        Reads the most recently exposed image as a NumPy array.
        """
        freeze_video(self.handle, timeout_ms, camera_handle=self.handle)

        buffer = ueye.get_data(
            self._mem_ptr,
            self._width,
            self._height,
            self._bpp,
            self._width * self._bytes_per_pixel,
            copy=True,
        )

        return np.reshape(buffer, (self._height.value, self._width.value))

    # ---- FITS ----

    def _build_fits_header(self) -> fits.Header:
        """
        Constructs a FITS header containing IDS camera metadata.
        """
        cam_info = ueye.CAMINFO()
        get_camera_info(self.handle, cam_info, camera_handle=self.handle)

        sensor_info = ueye.SENSORINFO()
        get_sensor_info(self.handle, sensor_info, camera_handle=self.handle)

        header = fits.Header()
        header["CAMERA"] = "IDS uEye"
        header["MODEL"] = cam_info.Model.decode().strip()
        header["SERIAL"] = cam_info.SerNo.decode().strip()
        header["EXPTIME"] = self.exposure_time_ms / 1000.0
        header["WIDTH"] = self._width.value
        header["HEIGHT"] = self._height.value
        header["BITPIX"] = self._bpp.value

        return header

    def read_hdu(self) -> fits.ImageHDU:
        """
        Returns the most recent image as a FITS ImageHDU.
        """
        data = self.read_numpy()
        header = self._build_fits_header()
        return fits.ImageHDU(data=data, header=header)

    def save_fits(self, filename: str, overwrite: bool = False) -> None:
        """
        Saves the most recent image to a FITS file.
        """
        hdulist = fits.HDUList([fits.PrimaryHDU(), self.read_hdu()])
        hdulist.writeto(filename, overwrite=overwrite)
