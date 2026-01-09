"""
Wrapper for pyueye functions. Written in part by ChatGPT.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from pyueye import ueye


class PyUeyeError(RuntimeError):
    """
    Raised when a uEye command reports failure.
    """

    def __init__(
        self, command_name: str, error_code: int, error_text: Optional[str] = None
    ):
        message = f"{command_name} failed with error code {error_code}"
        if error_text:
            message += f": {error_text}"
        super().__init__(message)

        self.command_name = command_name
        self.error_code = error_code
        self.error_text = error_text


class UeyeCommand:
    """
    Callable wrapper around a pyueye function with robust error handling.

    Success is determined by a predicate function applied to the return
    value. `ueye.IS_SUCCESS` is always a valid return value.

    Parameters
    ----------
    func : `ueye` function
        The function to be wrapped.
    name : str, optional
        Name of the function, to be stored in `self.name`.

    Attributes
    ----------
    func : `ueye` function
        The wrapped function.
    name : str
        Name of the function. Defaults to `self.func.__name__` if not
        initialized with a `name` supplied.
        return value from `func` and false otherwise.

    Raises
    ------
    `PyUeyeError`
        Raised if the return value is not `ueye.IS_SUCCESS`.

    Notes
    -----
    This is to simplify and reduce the verbosity of calling pyueye
    functions and verifying their success. Commonly used functions should
    be wrapped in this class. If a return value other than
    `ueye.IS_SUCCESS` would be acceptable, it should be handled explicitly.
    For example, one might handle `ueye.is_FreezeVideo` returning
    `ueye.IS_TIMED_OUT` more gracefully than the raising of a
    `PyUeyeError`.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: Optional[str] = None,
    ):
        self.func = func
        self.name = name or func.__name__

    def __call__(self, *args, camera_handle: Optional[ueye.HIDS] = None) -> bool:
        result = self.func(*args)

        if result == ueye.IS_SUCCESS:
            return True

        error_text = None
        if camera_handle is not None:
            error_code = ueye.int()
            error_message = ueye.char_p()
            if (
                ueye.is_GetError(camera_handle, error_code, error_message)
                == ueye.IS_SUCCESS
            ):
                assert error_message.value is not None
                error_text = error_message.value.decode(errors="ignore")

        raise PyUeyeError(self.name, int(result), error_text)


# ---- Common commands used by the camera and driver ----

init_camera = UeyeCommand(ueye.is_InitCamera)
exit_camera = UeyeCommand(ueye.is_ExitCamera)

set_software_trigger = UeyeCommand(ueye.is_SetExternalTrigger)
force_trigger = UeyeCommand(ueye.is_ForceTrigger)

alloc_image_mem = UeyeCommand(ueye.is_AllocImageMem)
free_image_mem = UeyeCommand(ueye.is_FreeImageMem)
set_image_mem = UeyeCommand(ueye.is_SetImageMem)

get_camera_info = UeyeCommand(ueye.is_GetCameraInfo)
get_sensor_info = UeyeCommand(ueye.is_GetSensorInfo)
get_device_info = UeyeCommand(ueye.is_DeviceInfo)

exposure_cmd = UeyeCommand(ueye.is_Exposure)
gamma_cmd = UeyeCommand(ueye.is_Gamma)
pixel_clock_cmd = UeyeCommand(ueye.is_PixelClock)
set_hardware_gain = UeyeCommand(ueye.is_SetHardwareGain)
aoi_cmd = UeyeCommand(ueye.is_AOI)
set_binning = UeyeCommand(ueye.is_SetBinning)

get_number_of_cameras = UeyeCommand(ueye.is_GetNumberOfCameras)
get_camera_list = UeyeCommand(ueye.is_GetCameraList)
set_camera_id = UeyeCommand(ueye.is_SetCameraID)
# Using set_camera_id to get the current camera ID will result in a PyUeyeError
#   use get_device_info or get_camera_list instead
