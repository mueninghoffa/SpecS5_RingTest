"""
Find camera, and connect to it. Adjust settings to have all auto parameters off, and the hardware gain and exposure time set as specified by the user.

If there is a dark/background library, load it. Otherwise, make one for the specified settings.

Find positioner control and connect to it. Ensure full range of motion.

Connect to LED and turn it on. Take an image to verify there is a ring.

Have user confirm that the positioner is at the correct angle for the test. Take an image, then instruct user to add a (specific) gauge block to the camera focus stage, then take another image (maybe more than just two distances?). Analyze the images to determine fiber-beam angle.

Determine pseudorandom pattern of fiber positions throughout its range of motion (user specifies number of positions <1000).

Loop over the list of fiber positions, at each position taking an image and analyzing it.

Generate two maps over patrol area, one for ring size (tilt) and one for ring thickness (FRD). Display with interpolation and/or fit with some polynomial?

Generate pdf report containing table of all image fit values, histograms of ring size and thickness, and colormaps of ring size and thickness over patrol area (along with some mathematical info on it?).
"""

import matplotlib.pyplot as plt
import yaml
from pyueye import ueye

from logging_config import get_logger, set_up_logging
from ueye_camera import UeyeCamera, list_cameras, number_of_cameras

set_up_logging(log_to_console=True, log_to_file=True)
logger = get_logger(__name__)

config_filepath = "./camera_settings.yaml"
with open(config_filepath, "r") as stream:
    camera_info = yaml.safe_load(stream)
    camera_settings = camera_info["settings"]


def find_camera() -> int:
    num_cams = number_of_cameras()
    assert num_cams > 0, "No cameras connected"

    valid_cams = []
    camlist = list_cameras()
    for cam in camlist:
        if cam["FullModelName"] == camera_info["camera_model"]:
            valid_cams.append(cam)
    if len(valid_cams) > 0:
        logger.debug("One or more cameras found of specified model")
        valid_model = True
    else:
        logger.warning("No camera found of specified model.")
        valid_model = False
        valid_cams = camlist
    if camera_info["camera_ID"] == 0:
        selected_camera = valid_cams[0]
        camera_ID = selected_camera["dwCameraID"]
        logger.info(f"No camera ID specified, camera ID {camera_ID} chosen")
    else:
        for cam in camlist:
            if camera_info["camera_ID"] == cam["dwCameraID"]:
                selected_camera = cam
                camera_ID = selected_camera["dwCameraID"]
                logger.debug(f"Camera with ID {camera_ID} found")
                break
        else:
            logger.warning(f"No camera found with camera ID {camera_info["camera_ID"]}")
            selected_camera = camlist[0]
            camera_ID = selected_camera["dwCameraID"]
        if not valid_model:
            logger.warning(
                f"Camera matches specified camera ID {camera_ID} but is model {selected_camera["FullModelName"]}, not {camera_info["camera_model"]}"
            )

    return camera_ID


def configure_camera(camera: UeyeCamera) -> None:
    camera.auto_parameters(camera_settings["auto_params"])
    color_mode = getattr(ueye, camera_settings["color_mode"])
    camera.color_mode = color_mode
    camera.gain = camera_settings["gain"]
    if camera_settings["pixel_clock"].lower() == "min":
        pixel_clock = int(min(camera.pixel_clock_possible_values))
    else:
        pixel_clock = camera_settings["pixel_clock"]
    camera.pixel_clock = pixel_clock
    camera.exposure_time_ms = camera_settings["exposure_time"]

    logger.info(f"Camera configured according to {config_filepath}")

    camera.reserve_memory()


def run() -> None:
    camera_ID = find_camera()

    with UeyeCamera(camera_ID) as camera:
        configure_camera(camera)

        explist = (1, 2, 3, 5, 10, 15, 20, 30)
        for exp in explist:
            camera.exposure_time_ms = exp * 1e3
            camera.expose()
            # img = camera.read_numpy(subtract=True)
            # plt.imshow(img, vmin=-100, vmax=100)
            # plt.colorbar()
            # plt.show(block=True)
            camera.save_fits(suffix=f"ring_{exp}s", subtract=False)
            camera.save_fits(suffix=f"ring_{exp}s_Dsub", subtract=True)

        # camera.generate_standard_darks(long=True)


if __name__ == "__main__":
    run()
