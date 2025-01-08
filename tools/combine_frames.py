import os
import ffmpeg
from pathlib import Path
from argparse import ArgumentParser
from PIL import Image


def combine():

    pass


def split_frames(cams: int):
    pass


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="FrameCombiner", description="Combine Multi-Cam datasets into a video"
    )
    parser.add_argument("-s", "--scene", help="Scene folder containing images")
    parser.add_argument("-c", "--cameras", help="Number of cameras involved")
    args = parser.parse_args()
    cams = int(args.cameras)
    scene_path = Path(args.scene)
    if cams == 1:
        combine()

    elif cams == 3 or cams == 6:
        split_frames(scene, cams)

    else:
        raise AssertionError("Only 1, 3, 6 cameras are supported for now")
