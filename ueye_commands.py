"""
Wrapper for pyueye functions. Written in part by ChatGPT.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, cast

from pyueye import ueye

from logging_config import get_logger

logger = get_logger(__name__)

Predicate = Callable[[int, tuple[int, ...]], bool]


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
        Name of the function. Defaults to ``self.func.__name__`` if not
        initialized with a `name` supplied.

    Raises
    ------
    `PyUeyeError`
        Raised if the return value is not ``ueye.IS_SUCCESS`.`

    Notes
    -----
    This is to simplify and reduce the verbosity of calling pyueye
    functions and verifying their success. Commonly used functions should
    be wrapped in this class. Most `ueye` functions return only a status
    code, such as ``ueye.IS_SUCCESS``, but a few functions may also return
    a value, such as ``ueye.is_SetHardwareGain()`` when the value of arg
    ``nMaster`` is one of ``ueye.IS_GET_...``. Supplying a `success`
    predicate allows for the handling of these functions.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: Optional[str] = None,
        accept: Optional[Predicate] = None,
    ):
        self.func = func
        self.name = name or func.__name__
        if accept is None:
            accept = lambda _, __: False  # noqa: E731
        self.accept = cast(Predicate, accept)

    def __call__(self, *args, camera_handle: Optional[ueye.HIDS] = None) -> bool | int:
        """
        Evaluates the success of calling the wrapped function.

        Parameters
        ----------
        *args
            Arguments to be passed to `self.func`.
        camera_handle : `ueye.HIDS`, optional
            Camera handle for gettings errors if they occur. If the first
            arg is a camera handle, it does not need to be supplied
            separately.

        Returns
        -------
        result : bool, int
            Returns the result of the function call, if it succeeded.

        Raises
        ------
        PyUeyeError
            Raised if the function call did not succeed (`self.accept`
            returned ``False``).
        """
        result = self.func(*args)

        if result == ueye.IS_SUCCESS or self.accept(result, args):
            return result

        error_text = None
        if camera_handle is None and isinstance(args[0], ueye.c_uint):
            camera_handle = args[0]
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


# ---- Predicates ----


def gain_accept(result: int, args: tuple[int, ...]) -> bool:
    getters = {
        ueye.IS_GET_MASTER_GAIN,
        ueye.IS_GET_RED_GAIN,
        ueye.IS_GET_GREEN_GAIN,
        ueye.IS_GET_BLUE_GAIN,
        ueye.IS_GET_DEFAULT_MASTER,
        ueye.IS_GET_DEFAULT_RED,
        ueye.IS_GET_DEFAULT_GREEN,
        ueye.IS_GET_DEFAULT_BLUE,
    }
    if args[1] in getters:
        return 0 <= result <= 100
    return False


def gamma_accept(result: int, args: tuple[int, ...]) -> bool:
    if args[1] == ueye.IS_GET_HW_GAMMA:
        return isinstance(result, int) and result >= 0  # I think this is correct?
    if args[1] == ueye.IS_GET_HW_SUPPORTED_GAMMA:
        return result in (ueye.IS_SET_HW_GAMMA_ON, ueye.IS_SET_HW_GAMMA_OFF)
    return False


# ---- Common commands that only return status ----

init_camera = UeyeCommand(ueye.is_InitCamera)
exit_camera = UeyeCommand(ueye.is_ExitCamera)

set_software_trigger = UeyeCommand(ueye.is_SetExternalTrigger)
force_trigger = UeyeCommand(ueye.is_ForceTrigger)

alloc_image_mem = UeyeCommand(ueye.is_AllocImageMem)
free_image_mem = UeyeCommand(ueye.is_FreeImageMem)
set_image_mem = UeyeCommand(ueye.is_SetImageMem)

get_camera_info = UeyeCommand(ueye.is_GetCameraInfo)
get_sensor_info = UeyeCommand(ueye.is_GetSensorInfo)
device_info = UeyeCommand(ueye.is_DeviceInfo)

exposure_cmd = UeyeCommand(ueye.is_Exposure)
pixel_clock_cmd = UeyeCommand(ueye.is_PixelClock)
aoi_cmd = UeyeCommand(ueye.is_AOI)
set_binning = UeyeCommand(ueye.is_SetBinning)

get_number_of_cameras = UeyeCommand(ueye.is_GetNumberOfCameras)
get_camera_list = UeyeCommand(ueye.is_GetCameraList)
set_camera_id = UeyeCommand(ueye.is_SetCameraID)
# Using set_camera_id to get the current camera ID will result in a PyUeyeError
#   use get_device_info or get_camera_list instead

# ---- Common commands that might return a value ----

set_hardware_gain = UeyeCommand(ueye.is_SetHardwareGain, accept=gain_accept)
set_hardware_gamma = UeyeCommand(ueye.is_SetHardwareGamma, accept=gamma_accept)


logger.info("ueye command wrappers instantiated")
