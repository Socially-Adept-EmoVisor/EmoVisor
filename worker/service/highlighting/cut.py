from ..models import Highlight
from shutil import copyfile
import os
from typing import List

result_dir = "."


def cut_video(video_path: str, highlights: List[Highlight]) -> str:
    filename = os.path.basename(video_path)
    target = os.path.join(result_dir, filename)
    copyfile(video_path, target)

    return target
