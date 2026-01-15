"""
Functions and a class for interacting with uEye cameras through pyueye.
"""

from __future__ import annotations

import ctypes
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TypeAlias

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
    set_hardware_gain,
    set_hardware_gamma,
    set_image_mem,
)

Ueye_Camera_List: TypeAlias = ueye.UEYE_CAMERA_LIST  # type: ignore
Ueye_Color_Mode: TypeAlias = int
Ueye_Auto_Param: TypeAlias = int


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


def list_cameras(struct: bool = False) -> list[dict[str, Any]] | Ueye_Camera_List:
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
    base_dir : str, default="./ueye_fits_images"
        The parent directory where fits images will be saved.
    dark_library : str, default="./ueye_dark_library"
        The parent directory of the dark library.

    Attributes
    ----------
    handle : `ueye.c_uint`
        Handle of the camera.
    save_dir : `pathlib.Path`
        ``<base_dir>/YYMMDD_<camera serial number>``
        The directory within which saved fits images will be placed.
    dark_library : `pathlib.Path`
        The directory containing the dark library.
    device_info : dict
        Information about the device (Read-only).
    camera_info : dict
        Information about the camera (Read-only).
    sensor_info : dict
        Information about the sensor (Read-only).
    pixel_clock_possible_values : list of ints
        List of available pixel clock settings in MegaHertz (MHz) (Read-only).
    temperature : float
        Sensor temperature in Celsius (Read-only).
    exposure_time_ms : float
        Exposure time in milliseconds (Read/Write).
    gain : int
        Global hardware gain (Read/Write).
    pixel_clock : int
        Pixel clock mode (Read/Write).
    gamma : bool
        Whether hardware gamma correction is enabled (Read/Write).
    color_mode : int
        Corresponds to a ueye constant for a color mode (Read/Write).

    Methods
    -------
    auto_parameters(params)
        Enable/Disable automatic adjustment of camera settings.
    expose()
        Trigger an exposure. Blocks until exposure is complete.
    generate_dark(n=11)
        Generate and save a master dark by taking the median of `n` darks.
    generate_standard_darks(long=False)
        Generate darks for a number of common exposure times.
    read_numpy(subtract=False)
        Returns the most recent image as a numpy array.
    read_hdu(subtract=False)
        Returns the most recent image as n `astropy.io.fits` HDU.
    save_fits(suffix=None, subtract=False)
        Saves the most recent image as a fits file.

    See Also
    --------
    ueye_commands : Module for simplified use of pyueye functions

    Notes
    -----
    This class makes extensive use of the pyueye package.
    See the `uEye <https://www.1stvision.com/cameras/IDS/IDS-manuals/uEye_Manual>`
    documentation.
    """

    def __init__(
        self,
        camera_id: int,
        base_dir: str = "./ueye_fits_images",
        dark_library: str = "./ueye_dark_library",
    ) -> None:
        self.handle = ueye.HIDS(camera_id)
        self.save_dir = Path(base_dir)
        self.dark_library = Path(dark_library)
        self._darks = dict()

        self._mem_ptr = ueye.c_mem_p()
        self._mem_id = ueye.int()

        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._bits_per_pixel: Optional[int] = None
        self._serial_no: Optional[int] = None

        self._connected = False
        self._last_exposure_time: Optional[datetime] = None
        self._image_available: Optional[ueye.c_uint] = None
        self._image_counter = 0

    # ---- Lifecycle ----

    def _connect(self) -> None:
        """
        Initialize camera and connect to camera.

        Sets the camera to trigger an exposure from a software trigger.
        """
        if self._connected:
            logger.warning("Attempted to connect to camera while already connected")
            return

        init_camera(self.handle, None)
        set_external_trigger(self.handle, ueye.IS_SET_TRIGGER_SOFTWARE)
        self._width = self.sensor_info["nMaxWidth"]
        self._height = self.sensor_info["nMaxHeight"]
        self._serial_no = int(self.camera_info["SerNo"])

        now = datetime.now()
        date_str = now.strftime("%Y%m%d")

        self.save_dir = self.save_dir / f"{date_str}_{self._serial_no}"
        self.dark_library = self.dark_library / f"{self._serial_no}"

        self._connected = True
        logger.info(
            f"Initialized camera model {self.sensor_info["strSensorName"]}"
            + f"with camera ID {self.camera_info["Select"]}"
        )

    def _enable_frame_event(self) -> None:
        """
        Initialize and enable the wait event for a new image being available.
        """
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
        """
        Reserve memory for captured images.
        """
        assert self._connected, "Camera not initialized"
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
        """
        Release memory allocated to camera.
        """
        if self._mem_ptr.value is None and self._mem_id.value == 0:
            logger.info("There is no memory to release")

        free_image_mem(self.handle, self._mem_ptr, self._mem_id)
        self._mem_ptr = ueye.c_mem_p()
        self._mem_id = ueye.c_int()
        logger.debug("Camera memory released")

    def _close(self) -> None:
        """
        Close camera connection, if possible.
        """
        if not self._connected:
            logger.warning("Cannot disconnect from an unconnected camera")
            return

        exit_camera(self.handle)

        self._connected = False
        logger.debug("Camera disconnected")
        return

    def __enter__(self) -> UeyeCamera:
        """
        Connect to camera and enable image-available trigger.

        Returns
        -------
        UeyeCamera=self
        """
        logger.debug("Entered context")
        self._connect()
        self._enable_frame_event()
        return self

    def __exit__(self, _, __, ___) -> bool:
        """
        If camera is connected, release allocated memory and close camera.

        Returns
        -------
        bool=False
            Indicates that no exceptions were handled.
        """
        _ = __, ___  # shut up pyright

        logger.debug("Exited context")
        if self._connected:
            self.release_memory()
            self._close()
        return False

    # ---- Hardware Information ----

    @property
    def device_info(self) -> dict[str, Any]:
        """
        Get device info using `ueye.is_DeviceInfo`.

        Returns
        -------
        dict
            `ueye.IS_DEVICE_INFO` struct converted to a dict.
        """
        info = ueye.IS_DEVICE_INFO()
        device_info(
            self.handle,
            ueye.IS_DEVICE_INFO_CMD_GET_DEVICE_INFO,
            info,
            ueye.sizeof(info),
        )
        return ctypes_to_normal(info)

    @property
    def camera_info(self) -> dict[str, Any]:
        """
        Get camera info using `ueye.is_GetCameraInfo`.

        Returns
        -------
        dict
            `ueye.CAMINFO` struct converted to dict.
        """
        info = ueye.CAMINFO()
        get_camera_info(self.handle, info)
        return ctypes_to_normal(info)

    @property
    def sensor_info(self) -> dict[str, Any]:
        """
        Get sensor info using `ueye.is_GetSensorInfo`.

        Returns
        -------
        dict
            `ueye.SENSORINFO` struct converted to dict.
        """
        info = ueye.SENSORINFO()
        get_sensor_info(self.handle, info)
        return ctypes_to_normal(info)

    @property
    def pixel_clock_possible_values(self) -> list[int]:
        """
        Get camera's possible pixel clock values.

        Returns
        -------
        list of ints
            List containing the valid pixel clock values in MegaHertz (MHz).
        """
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

    @property
    def temperature(self) -> float:
        """
        Get the temperature (Celsius) of the sensor.

        The temperature is given as a 16-bit unsigned float (``c_ushort``)
        with the 15th bit encoding sign, bits 4 to 10 encoding the
        temperature before the decimal point, and bits 0 to 3 encoding the
        temperature after the decimal point.

        Returns
        -------
        float
            Sensor temperature in Celsius.
        """
        binary_temp = self.device_info["infoDevHeartbeat"]["wTemperature"]

        # Bits 4 to 10
        integer_part = (binary_temp & 0x07F0) >> 4

        # Bits 0 to 3
        decimal_part = binary_temp & 0x000F

        # Bit 15 encodes sign
        negative = (binary_temp >> 15) & 0x01

        temp = integer_part + decimal_part / 10

        if negative:
            temp = -temp

        return temp

    # ---- Configuration (properties) ----

    @property
    def exposure_time_ms(self) -> float:
        """
        Get the exposure time in milliseconds (ms).

        Returns
        -------
        float
            Current exposure time in ms.
        """
        value = ueye.double()
        exposure(
            self.handle,
            ueye.IS_EXPOSURE_CMD_GET_EXPOSURE,
            value,
            ueye.sizeof(value),
        )
        return ctypes_to_normal(value)

    @exposure_time_ms.setter
    def exposure_time_ms(self, ms: float) -> None:
        """
        Set the exposure time in milliseconds (ms).

        Warns if framerate is not adjusted to be consistent. Enables
        long exposures as necessary (>1s).

        Parameters
        ----------
        ms : float
            New exposure time in ms.

        Raises
        ------
        RuntimeError
            Raised if the framerate and exposure time are not compatible.

        Notes
        -----
        There are only a discrete number of available exposure times. The
        exposure time will be set to the closest one available. (The
        spacing of available exposure times might be influenced by the pixel clock.)

        If setting the exposure time results in long exposure being enabled
        or disabled, this may temporarily disable the event frame loop (see
        `self.exposure`).
        """
        if ms > 1e3:
            enable = ueye.c_uint(1)
        else:
            enable = ueye.c_uint(0)
        exposure(
            self.handle,
            ueye.IS_EXPOSURE_CMD_SET_LONG_EXPOSURE_ENABLE,
            enable,
            ueye.sizeof(enable),
        )
        # Have to re-enable event after going in/out of long exposure mode
        self._enable_frame_event()

        value = ueye.double(ms)
        exposure(
            self.handle,
            ueye.IS_EXPOSURE_CMD_SET_EXPOSURE,
            value,
            ueye.sizeof(value),
        )

        # warn if actual exposure time is far from requested
        if 0.1 < abs((ms - self.exposure_time_ms) / ms):
            logger.warning(
                f"Exposure time set to {self.exposure_time_ms:.3f},"
                + f" more than 10% different from requested value of {ms:.3f}"
            )

        logger.debug(f"Exposure time set to {self.exposure_time_ms:.3f} ms")

    @property
    def gain(self) -> int:
        """
        Get the current master gain setting.

        Does not get the individual color (rgb) gain settings.

        Returns
        -------
        int
            Current gain setting.
        """
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
    def gain(self, value: int) -> None:
        """
        Set the master gain.

        Cannot be used to set the individual color (rgb) gain settings.

        Parameters
        ----------
        value : int
            New master gain value.
        """
        val = ueye.uint(value)
        set_hardware_gain(
            self.handle,
            val,
            ueye.IS_IGNORE_PARAMETER,
            ueye.IS_IGNORE_PARAMETER,
            ueye.IS_IGNORE_PARAMETER,
        )
        logger.debug(f"Gain set to {value}")

    @property
    def pixel_clock(self) -> int:
        """
        Get the current pixel clock value.

        Returns
        -------
        int
            Current pixel clock in MegaHertz (MHz).

        See Also
        --------
        UeyeCamera.pixel_clock_possible_values : Get available values

        Notes
        -----
        There are only a discrete number of pixel clock values available.
        """
        value = ueye.uint()
        pixel_clock(self.handle, ueye.IS_PIXELCLOCK_CMD_GET, value, ueye.sizeof(value))
        return ctypes_to_normal(value)

    @pixel_clock.setter
    def pixel_clock(self, mhz: int) -> None:
        """
        Set the pixel clock value.

        Parameters
        ----------
        mhz : int
            New pixel clock setting in MegaHertz (MHz).

        See Also
        --------
        UeyeCamera.pixel_clock_possible_values : Get available values

        Notes
        -----
        There are only a discrete number of pixel clock values available.
        The camera is set to which valid value is closest to `mhz`.
        """
        value = ueye.int(mhz)
        pixel_clock(
            self.handle,
            ueye.IS_PIXELCLOCK_CMD_SET,
            value,
            ueye.sizeof(value),
        )

        actual_value = ueye.INT()
        pixel_clock(
            self.handle,
            ueye.IS_PIXELCLOCK_CMD_GET,
            actual_value,
            ueye.sizeof(actual_value),
        )
        logger.debug(f"Pixel clock set to {actual_value} MHz")
        if actual_value != value:
            logger.warning(
                f"Pixel clock was set to {actual_value} instead of requested value {value}"
            )

    @property
    def gamma(self) -> bool:
        """
        Whether gamma correction (white balance) is enabled.

        Returns
        -------
        bool
            Hardware gamma setting.
        """
        mode = ueye.IS_GET_HW_GAMMA
        ret = set_hardware_gamma(self.handle, mode)
        return bool(ret)

    @gamma.setter
    def gamma(self, active: bool) -> None:
        """
        Enable or disable gamma correction (white balance).

        Parameters
        ----------
        active : bool
            New hardware gamma setting.
        """
        if active:
            mode = ueye.IS_SET_HW_GAMMA_ON
        else:
            mode = ueye.IS_SET_HW_GAMMA_OFF
        set_hardware_gamma(self.handle, mode)
        logger.debug(f"Gamma set to {active}")

    @property
    def color_mode(self) -> int:
        """
        Get the current color mode.

        Returns
        -------
        int
            Corresponds to a color mode.

        Notes
        -----
        Should be updated to return string of color mode.
        """
        return set_color_mode(self.handle, ueye.IS_GET_COLOR_MODE)

    @color_mode.setter
    def color_mode(self, mode: Ueye_Color_Mode) -> None:
        """
        Set the color mode.

        Parameters
        ----------
        mode : valid ueye color mode
            A `ueye` constant corresponding to the desired color mode.
            Some common color modes are :

            *``ueye.IS_CM_MONO8``
            *``ueye.IS_BGR8_PACKED``
            *``ueye.IS_CM_SENSOR_RAW12``

            See the `uEye <https://www.1stvision.com/cameras/IDS/IDS-manuals/uEye_Manual/is_setcolormode.html>`
            documentation for the full list.
        """
        set_color_mode(self.handle, mode)
        self._bits_per_pixel = set_color_mode(self.handle, ueye.IS_GET_BITS_PER_PIXEL)
        logger.debug(
            f"Color mode set to {mode}"
        )  # should be updated to log literal name, not int

    # ---- Configuration (non-properties) ----

    def _autoparam(
        self,
        setting: Ueye_Auto_Param,
        enable: bool,
        ignore_not_implemented: bool = False,
    ) -> None:
        """
        Enable or disable an auto parameter.

        Parameters
        ----------
        setting : `ueye.uint`
            Ueye constant corresponding to the auto parameter being set.
            Some common parameters are:

            *``ueye.IS_SET_ENABLE_AUTO_GAIN``
            *``ueye.IS_SET_ENABLE_AUTO_SHUTTER``
            *``ueye.IS_SET_ENABLE_AUTO_FRAMERATE``

            See the `uEye <https://www.1stvision.com/cameras/IDS/IDS-manuals/uEye_Manual/is_setautoparameter.html>`
            documentation for the full list.
        enable : bool
            When to enable (``True``) or disable (``False``) the auro parameter.
        ignore_not_implemented : bool, default=True
            If ``True`` (default), then a `PyUeyeError` will not be raised
            if `setting` is not available on the camera model.
        """
        try:
            pval = ueye.c_double(int(enable))
            set_auto_parameter(self.handle, setting, pval, pval)
            logger.debug(f"Set auto parameter {setting} to {enable}")
        except PyUeyeError as e:
            if e.error_code == ueye.IS_NOT_SUPPORTED and ignore_not_implemented:
                logger.debug(
                    f"Attempted to adjust unsupported auto-parameter setting {setting}"
                )
                return
            raise

    def auto_parameters(self, params: dict[str, bool]) -> None:
        """
        Enable or disable the auto parameter settings for gain, shutter
        (exposure time), white balance (gamma), and frame rate.

        Parameters
        ----------
        params : dict of string-keyed bools
            Dict of key-value pairs for the auto parameters. Values must be
            bools. Valid keys:
                *"gain"
                *"shutter"
                *"white_balance"
                *"frame_rate"
        """
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

        Uses a ueye wait event to wait for the image to be available in
        memory before continuing.
        """

        assert self._image_available is not None, "Image available event is not enabled"

        logger.debug("Starting exposure")
        self._last_exposure_time = datetime.now(timezone.utc)
        start = time.time()
        freeze_video(self.handle, ueye.IS_DONT_WAIT)

        wait_for_img = ueye.IS_WAIT_EVENT()
        wait_for_img.nEvent = self._image_available
        wait_for_img.nTimeoutMilliseconds = ueye.UINT(
            round(max(self.exposure_time_ms * 1.1, self.exposure_time_ms + 300))
            # typical USB overhead around 150ms
        )
        event(
            self.handle,
            ueye.IS_EVENT_CMD_WAIT,
            wait_for_img,
            ueye.sizeof(wait_for_img),
        )
        end = time.time()
        elapsed = end - start
        logger.debug(f"Took image in {elapsed:.2f} seconds")
        # This log message is not the exposure time

    def _get_dark(self) -> np.ndarray | None:
        """
        Get the most recent master dark for the current exposure time.

        Returns
        -------
        np.ndarray or None
            Master dark image data in a numpy array, if it exists.
            Otherwise, returns ``None``.
        """
        if self.exposure_time_ms in self._darks.keys():
            return self._darks[self.exposure_time_ms]

        logger.debug(
            f"Looking for {self.exposure_time_ms}ms darks in {self.dark_library}"
        )
        pattern = f"*_dark_{int(self.exposure_time_ms):05d}ms.fits"
        candidates = list(self.dark_library.glob(pattern))

        if not candidates:
            logger.warning(f"No darks found for {self.exposure_time_ms}ms")
            return None

        candidates.sort()
        # fmt: off
        dark = fits.open(candidates[-1])[0].data.astype(np.int16)  # pyright: ignore reportAttributeAccessIssue
        # fmt: on

        self._darks[self.exposure_time_ms] = dark
        logger.debug(f"Loaded dark {dark}")
        return dark

    def generate_dark(self, n: int = 11) -> Path:
        """
        Generate a master dark by taking the median over `n` images.

        The dark will be saved as
        ``<self.dark_library>/<camera_serial_number>/YYMMDD_HHMMSS_dark_EEEEEms.fits``,
        where ``EEEEE`` is the zero-padded exposure time in milliseconds.

        Parameters
        ----------
        n : int, default=11
            How many dark images to take and combine into a master dark.
            Default to odd number of images to prevent non-integer values.

        Returns
        -------
        `pathlib.Path`
            Path to the master dark in the dark library.
        """
        logger.info(
            f"Taking {n} {int(self.exposure_time_ms)}ms dark exposures. Keep sensor covered!"
        )

        darks = []
        for _ in range(n):
            self.expose()
            darks.append(self.read_numpy(subtract=False))
        logger.info(f"Done taking {int(self.exposure_time_ms)}ms darks")
        darks = np.array(darks)
        master_dark = np.median(darks, axis=0)

        self.dark_library.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        date_time_str = now.astimezone().strftime("%Y%m%d_%H%M%S")
        filename = f"{date_time_str}_dark_{int(self.exposure_time_ms):05d}ms.fits"
        full_path = self.dark_library / filename
        header = self._generate_fits_header(include_start=False)
        header["CREATED"] = now.isoformat(), "UTC"
        hdu = fits.PrimaryHDU(master_dark, header)
        hdu.writeto(full_path, overwrite=True)

        logger.info(f"Saved {full_path}")
        return full_path

    def generate_standard_darks(self, long: bool = False) -> None:
        """
        Generate darks for a set of standard exposure times.

        Always generates darks for the following exposure times (ms):
            1, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000
        If `long` is set to ``True``, the darks are also generated for
        these exposure times (s):
            2, 3, 5, 10, 20, 30

        Parameters
        ----------
        long : bool, default=False
            Whether to generate darks for exposure times longer than 1 seconds.

        Notes
        -----
        The set of short exposures has a total exposure time of 22 seconds.
        The set of long exposure has a total exposure time of about 12 minutes.
        """
        original_exposure_time = self.exposure_time_ms
        exposure_times = (1, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000)  # ms
        long_exposure_times = (2e3, 3e3, 5e3, 1e4, 2e4, 3e4)
        total_length = sum(exposure_times) * 10
        if long:
            total_length += sum(long_exposure_times) * 10
        total_length *= 1e-3  # ms -> s
        logger.info(
            "Generating dark library"
            + (", including long exposures" if long else "")
            + f". This will take {total_length:.1f} seconds"
        )

        for exp_time in exposure_times:
            self.exposure_time_ms = exp_time
            self.generate_dark()

        if long:
            for exp_time in long_exposure_times:
                self.exposure_time_ms = exp_time
                self.generate_dark()

        self.exposure_time_ms = original_exposure_time
        logger.info("Standard dark library completed")

    def read_numpy(self, subtract: bool = False) -> np.ndarray:
        """
        Reads the most recently exposed image as a NumPy array.

        Subtracts a master dark from the image if one is available for the
        exposure time in the dark library.

        Parameters
        ----------
        subtract : bool, default=False
            Whether to subtract a dark, if available.

        Returns
        -------
        np.ndarray
            Numpy array of pixel data in the same shape as the sensor.
        """
        assert self._mem_ptr.value is not None, "No memory allocated"
        assert self._last_exposure_time is not None, "No image available"
        assert isinstance(self._width, int)
        assert isinstance(self._height, int)
        assert isinstance(self._bits_per_pixel, int)

        total_bytes = self._width * self._height * ((self._bits_per_pixel + 7) // 8)

        # accessing _mem_ptr without allocation will cause a hard crash
        buffer = ctypes.string_at(self._mem_ptr, total_bytes)
        arr = np.copy(np.frombuffer(buffer, dtype=np.int16))
        img = arr.reshape((self._height, self._width))

        if subtract:
            dark = self._get_dark()
            if dark is not None:
                img -= dark

        return img

    # ---- FITS ----

    def _generate_fits_header(self, include_start: bool = True) -> fits.Header:
        """
        Constructs a FITS header containing IDS camera metadata.

        Parameters
        ----------
        include_start : bool, default=True
            Whether to include the exposure's start time in the header.

        Returns
        -------
        astropy.io.fits.Header
            FITS header containing camera metadata.
        """
        camera_info = self.camera_info
        sensor_info = self.sensor_info
        device_info = self.device_info
        heartbeat = device_info["infoDevHeartbeat"]
        ueye_control = device_info["infoDevControl"]

        header = fits.Header()
        header["MANUF"] = camera_info["ID"]
        header["MODEL"] = sensor_info["strSensorName"]
        header["SERIAL"] = int(camera_info["SerNo"])
        header["CAMID"] = camera_info["Select"], "can be changed"
        header["DEVICEID"] = ueye_control["dwDeviceId"]
        header["SENSTYPE"] = sensor_info["SensorID"]
        header["FIRMVERS"] = heartbeat["dwRuntimeFirmwareVersion"], "firmware version"

        # header["WIDTH"] = self._width
        # header["HEIGHT"] = self._height
        header["SBITPIX"] = self._bits_per_pixel, "significant bits per pixel"
        header["PIXSIZE"] = sensor_info["wPixelSize"] / 1e2, "microns"
        header["GLOBSHUT"] = bool(sensor_info["bGlobShutter"]), "global shutter"

        header["EXPTIME"] = self.exposure_time_ms, "milliseconds"
        header["GAIN"] = self.gain
        header["PIXCLOCK"] = self.pixel_clock, "MegaHertz"
        header["TEMP"] = self.temperature, "Celsius"

        if include_start:
            assert self._last_exposure_time is not None, "No image available"
            header["DATE-OBS"] = (
                self._last_exposure_time.isoformat(),
                "UTC",
            )
            header["DATE-LOC"] = (
                self._last_exposure_time.astimezone().isoformat(),
                "local time",
            )

        return header

    def read_hdu(self, subtract: bool = False) -> fits.PrimaryHDU:
        """
        Returns the most recent image as a FITS ImageHDU.

        Parameters
        ----------
        subtract : bool, default=False
            Whether to subtract a dark, if available.

        Returns
        -------
        astropy.io.fits.ImageHDU
            HDU with the image data and a header with camera information.
        """
        data = self.read_numpy(subtract=subtract)
        header = self._generate_fits_header()
        return fits.PrimaryHDU(data=data, header=header)

    def save_fits(self, suffix: Optional[str] = None, subtract: bool = False) -> Path:
        """
        Saves the most recent image to a FITS file.

        If a `suffix` is not supplied, then the image is saved as
        ``<self.save_dir>/YYYYMMDD_HHMMSS_XXXXX.fits``, where ``XXXXX`` is
        a five digit index, starting from 1. If a `suffix` is provided, the
        image is saved as
        ``<self.save_dir>/suffix/YYYYMMDD_HHMMSS_<suffis>_XXXXX.fits``.

        Parameters
        ----------
        suffix : str, optional
            An optional name to be added to the file path and name.
        subtract : bool, default=False
            Whether to subtract a dark, if available.

        Returns
        -------
        pathlib.Path
            Full path of the saved image.
        """
        self._image_counter += 1

        assert self._last_exposure_time is not None, "No image available"
        date_time_str = self._last_exposure_time.astimezone().strftime("%Y%m%d_%H%M%S")

        if suffix:
            save_folder = self.save_dir / suffix
            filename = f"{date_time_str}_{suffix}_{self._image_counter:05d}.fits"
        else:
            save_folder = self.save_dir
            filename = f"{date_time_str}_{self._image_counter:05d}.fits"

        save_folder.mkdir(
            parents=True, exist_ok=True
        )  # creates parent folders if needed
        full_path = save_folder / filename
        hdu = self.read_hdu(subtract=subtract)
        hdu.writeto(full_path)

        logger.info(f"Saved {full_path}")
        return full_path
