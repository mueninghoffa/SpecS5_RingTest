"""
Functions and a class for interacting with uEye cameras through pyueye.
Written in part by ChatGPT.
"""

from __future__ import annotations

import ctypes
import time
from typing import Any, Optional

import numpy as np
from astropy.io import fits
from pyueye import ueye

from ctypes_to_normal import ctypes_to_normal
from logging_config import get_logger
from ueye_commands import (
    PyUeyeError,
    alloc_image_mem,
    device_info,
    event,
    exit_camera,
    exposure,
    free_image_mem,
    freeze_video,
    get_camera_info,
    get_camera_list,
    get_number_of_cameras,
    get_sensor_info,
    init_camera,
    pixel_clock,
    set_auto_parameter,
    set_color_mode,
    set_external_trigger,
    set_frame_rate,
    set_hardware_gain,
    set_hardware_gamma,
    set_image_mem,
)
from ueye_types import UEYE_CAMERA_LIST

logger = get_logger(__name__)


def number_of_cameras() -> int:
    """
    Get the number of connected cameras.

    Returns
    -------
    int
        Number of connected cameras.
    """
    count = ueye.INT()
    get_number_of_cameras(count)
    return ctypes_to_normal(count)


def list_cameras(struct: bool = False) -> list[dict[str, Any]] | UEYE_CAMERA_LIST:
    """
    Returns a list of camera info objects of connected cameras.

    Parameters
    ----------
    struct : bool, default=False
        Whether to return the pyueye struct as is (``True``) or convert it
        to a dictionary (``False``).

    Returns
    -------
    list of dicts or ``ueye.UEYE_CAMERA_LIST``
        Camera information list for connected cameras. The type is
        determined by `struct`. Returns an empty list if no cameras are
        connected.
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
    High-level, Pythonic interface to an IDS uEye camera via pyueye.

    Parameters
    ----------
    camera_id : int
        ID of the camera to be initialized, as stored in the camera's
        non-volatile memory. This is passed to ``ueye.is_InitCamera()``.
        If 0 (default), the first available camera will be initialized.

    Attributes
    ----------
    handle : `ueye.c_uint`
        Handle of the camera.
    device_info : dict
        Information about the device (Read-only).
    camera_info : dict
        Information about the camera (Read-only).
    sensor_info : dict
        Information about the sensor (Read-only).
    exposure_time_ms : float
        Exposure time in milliseconds (Read/Write).
    gain : float
        Global hardware gain (Read/Write).
    pixel_clock : int
        Pixel clock mode (Read/Write).
    gamma : bool
        Whether hardware gamma correction is enabled (Read/Write).

    Methods
    -------
    connect()
        Initialize the camera.
    close()
        Disconnect from camera and free allocated memory.
    """

    def __init__(self, camera_id: int) -> None:
        self.handle = ueye.HIDS(camera_id)

        self._mem_ptr = ueye.c_mem_p()
        self._mem_id = ueye.int()

        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._bits_per_pixel: Optional[int] = None

        self._connected = False
        self._last_exposure_time: Optional[str] = None
        self._image_available: Optional[ueye.c_uint] = None

    # ---- Lifecycle ----

    def _connect(self) -> None:
        if self._connected:
            logger.debug("Attempted to connect to camera while already connected")
            return

        init_camera(self.handle, None)
        set_external_trigger(self.handle, ueye.IS_SET_TRIGGER_SOFTWARE)
        self._width = self.sensor_info["nMaxWidth"]
        self._height = self.sensor_info["nMaxHeight"]

        self._connected = True
        logger.debug("Camera connected")

    def _enable_trigger(self):
        initwait = ueye.IS_INIT_EVENT()
        initwait.nEvent = ueye.UINT(ueye.IS_SET_EVENT_FRAME)
        initwait.bManualReset = ueye.BOOL(False)
        initwait.bInitialState = ueye.BOOL(False)
        event(self.handle, ueye.IS_EVENT_CMD_INIT, initwait, ueye.sizeof(initwait))
        self._image_available = ueye.UINT(ueye.IS_SET_EVENT_FRAME)
        event(
            self.handle,
            ueye.IS_EVENT_CMD_ENABLE,
            self._image_available,
            ueye.sizeof(self._image_available),
        )
        logger.debug("Software trigger event enabled")

    def reserve_memory(self) -> None:
        assert self._mem_ptr.value is None, "Memory pointer already assigned"
        assert self._mem_id.value == 0, "Memory ID already assigned"
        alloc_image_mem(
            self.handle,
            self._width,
            self._height,
            self._bits_per_pixel,
            self._mem_ptr,
            self._mem_id,
        )
        set_image_mem(self.handle, self._mem_ptr, self._mem_id)
        logger.debug("Camera memory reserved")

    def release_memory(self) -> None:
        if self._mem_ptr.value is None and self._mem_id.value == 0:
            logger.info("There is no memory to release")

        free_image_mem(self.handle, self._mem_ptr, self._mem_id)
        self._mem_ptr = ueye.c_mem_p()
        self._mem_id = ueye.c_int()
        logger.debug("Camera memory released")

    def _close(self) -> None:
        if not self._connected:
            logger.warning("Cannot disconnect from an unconnected camera")
            return

        exit_camera(self.handle)

        self._connected = False
        logger.debug("Camera disconnected")
        return

    def __enter__(self):
        self._connect()
        self._enable_trigger()
        return self

    def __exit__(self, _, __, ___) -> bool:
        if self._connected:
            self.release_memory()
            self._close()
        return False

    # ---- Hardware Information ----

    @property
    def device_info(self) -> dict:
        info = ueye.IS_DEVICE_INFO()
        device_info(
            self.handle,
            ueye.IS_DEVICE_INFO_CMD_GET_DEVICE_INFO,
            info,
            ueye.sizeof(info),
        )
        return ctypes_to_normal(info)

    @property
    def camera_info(self) -> dict:
        info = ueye.CAMINFO()
        get_camera_info(self.handle, info)
        return ctypes_to_normal(info)

    @property
    def sensor_info(self) -> dict:
        info = ueye.SENSORINFO()
        get_sensor_info(self.handle, info)
        return ctypes_to_normal(info)

    @property
    def pixel_clock_possible_values(self) -> list:
        num_clocks = ueye.uint()
        pixel_clock(
            self.handle,
            ueye.IS_PIXELCLOCK_CMD_GET_NUMBER,
            num_clocks,
            ueye.sizeof(num_clocks),
        )
        clocks = (ueye.UINT * num_clocks.value)()
        pixel_clock(
            self.handle, ueye.IS_PIXELCLOCK_CMD_GET_LIST, clocks, ueye.sizeof(clocks)
        )
        return ctypes_to_normal(clocks)

    # ---- Configuration (properties) ----

    @property
    def exposure_time_ms(self) -> float:
        value = ueye.double()
        exposure(
            self.handle,
            ueye.IS_EXPOSURE_CMD_GET_EXPOSURE,
            value,
            ueye.sizeof(value),
        )
        return ctypes_to_normal(value)

    @exposure_time_ms.setter
    def exposure_time_ms(self, ms: float):
        if ms >= 1e3:
            enable = ueye.c_uint(1)
            exposure(
                self.handle,
                ueye.IS_EXPOSURE_CMD_SET_LONG_EXPOSURE_ENABLE,
                enable,
                ueye.sizeof(enable),
            )
        value = ueye.double(ms)
        exposure(
            self.handle,
            ueye.IS_EXPOSURE_CMD_SET_EXPOSURE,
            value,
            ueye.sizeof(value),
        )

        # verify framerate is reasonable
        fps = ueye.IS_GET_FRAMERATE
        newfps = ueye.double()
        set_frame_rate(self.handle, fps, newfps)
        assert 1e-2 > abs(
            (newfps * self.exposure_time_ms / 1e3) - 1
        ), "Frame rate and exposure time do not match"

    @property
    def gain(self) -> int:
        mode = ueye.IS_GET_MASTER_GAIN
        ret = set_hardware_gain(
            self.handle,
            mode,
            ueye.IS_IGNORE_PARAMETER,
            ueye.IS_IGNORE_PARAMETER,
            ueye.IS_IGNORE_PARAMETER,
        )
        return ret

    @gain.setter
    def gain(self, value: int):
        set_hardware_gain(
            self.handle,
            value,
            ueye.IS_IGNORE_PARAMETER,
            ueye.IS_IGNORE_PARAMETER,
            ueye.IS_IGNORE_PARAMETER,
        )

    @property
    def pixel_clock(self) -> int:
        return 0

    @pixel_clock.setter
    def pixel_clock(self, mhz: int):
        value = ueye.int(mhz)
        pixel_clock(
            self.handle,
            ueye.IS_PIXELCLOCK_CMD_SET,
            value,
            ueye.sizeof(value),
            camera_handle=self.handle,
        )

    @property
    def gamma(self) -> bool:
        mode = ueye.IS_GET_HW_GAMMA
        ret = set_hardware_gamma(self.handle, mode)
        return bool(ret)

    @gamma.setter
    def gamma(self, active: bool):
        if active:
            mode = ueye.IS_SET_HW_GAMMA_ON
        else:
            mode = ueye.IS_SET_HW_GAMMA_OFF
        set_hardware_gamma(self.handle, mode)

    @property
    def color_mode(self) -> int:
        return set_color_mode(self.handle, ueye.IS_GET_COLOR_MODE)

    @color_mode.setter
    def color_mode(self, mode: int):
        set_color_mode(self.handle, mode)
        self._bits_per_pixel = set_color_mode(self.handle, ueye.IS_GET_BITS_PER_PIXEL)

    # ---- Configuration (non-properties) ----

    def _autoparam(
        self, setting: int, enable: bool, ignore_not_implemented: bool = False
    ):
        try:
            pval = ueye.c_double(int(enable))
            set_auto_parameter(self.handle, setting, pval, pval)
        except PyUeyeError as e:
            if e.error_code == 155 and ignore_not_implemented:
                logger.debug("Attempted to adjust unsupported auto-parameter setting")
                return
            raise

    def auto_parameters(self, params: dict[str, bool]):
        valid_auto_params = {"gain", "shutter", "white_balance", "frame_rate"}
        assert set(params.keys()).issubset(
            valid_auto_params
        ), "Invalid option in params"

        # gain
        if "gain" in params.keys():
            enable = params["gain"]
            self._autoparam(ueye.IS_SET_ENABLE_AUTO_GAIN, enable)
            self._autoparam(
                ueye.IS_SET_ENABLE_AUTO_SENSOR_GAIN, enable, ignore_not_implemented=True
            )
            self._autoparam(
                ueye.IS_SET_ENABLE_AUTO_SENSOR_GAIN_SHUTTER,
                enable,
                ignore_not_implemented=True,
            )

        # shutter (exposure time)
        if "shutter" in params.keys():
            enable = params["shutter"]
            self._autoparam(ueye.IS_SET_ENABLE_AUTO_SHUTTER, enable)
            self._autoparam(
                ueye.IS_SET_ENABLE_AUTO_SENSOR_SHUTTER,
                enable,
                ignore_not_implemented=True,
            )
            self._autoparam(
                ueye.IS_SET_ENABLE_AUTO_SENSOR_GAIN_SHUTTER,
                enable,
                ignore_not_implemented=True,
            )

        if "framerate" in params.keys():
            enable = params["framerate"]
            self._autoparam(ueye.IS_SET_ENABLE_AUTO_FRAMERATE, enable)
            self._autoparam(
                ueye.IS_SET_ENABLE_AUTO_SENSOR_FRAMERATE,
                enable,
                ignore_not_implemented=True,
            )

        if "whitebalance" in params.keys():
            enable = params["whitebalance"]
            self._autoparam(ueye.IS_SET_ENABLE_AUTO_WHITEBALANCE, enable)
            self._autoparam(
                ueye.IS_SET_ENABLE_AUTO_SENSOR_WHITEBALANCE,
                enable,
                ignore_not_implemented=True,
            )

    # ---- Acquisition ----

    def expose(self) -> None:
        """
        Triggers a single software exposure.
        """
        assert self._image_available is not None

        start = time.time()
        freeze_video(self.handle, ueye.IS_DONT_WAIT)

        wait_for_img = ueye.IS_WAIT_EVENT()
        wait_for_img.nEvent = self._image_available
        wait_for_img.nTimeoutMilliseconds = ueye.UINT(
            round(self.exposure_time_ms * 1.1)
        )
        event(
            self.handle,
            ueye.IS_EVENT_CMD_WAIT,
            wait_for_img,
            ueye.sizeof(wait_for_img),
        )
        end = time.time()
        elapsed = end - start
        logger.info(f"Sensor exposed for {elapsed:.2f} seconds")

    def read_numpy(self) -> np.ndarray:
        """
        Reads the most recently exposed image as a NumPy array.
        """
        assert isinstance(self._width, int)
        assert isinstance(self._height, int)
        assert isinstance(self._bits_per_pixel, int)

        total_bytes = self._width * self._height * ((self._bits_per_pixel + 7) // 8)
        buffer = ctypes.string_at(self._mem_ptr, total_bytes)
        arr = np.frombuffer(buffer, dtype=np.int16)
        img = arr.reshape((self._height, self._width))
        return img

    # ---- FITS ----

    def _build_fits_header(self) -> fits.Header:
        """
        Constructs a FITS header containing IDS camera metadata.
        """
        cam_info = ueye.CAMINFO()
        get_camera_info(self.handle, cam_info)

        sensor_info = ueye.SENSORINFO()
        get_sensor_info(self.handle, sensor_info)

        header = fits.Header()
        header["CAMERA"] = "IDS uEye"
        header["MODEL"] = cam_info.Model.decode().strip()
        header["SERIAL"] = cam_info.SerNo.decode().strip()
        header["EXPTIME"] = self.exposure_time_ms / 1000.0
        header["WIDTH"] = self._width.value
        header["HEIGHT"] = self._height.value
        header["BITPIX"] = self._bits_per_pixel.value

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
