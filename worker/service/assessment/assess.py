from typing import Tuple
from ..models import VideoResult, AudioResult
from .video.assess import assess_emotions as video_assess
from .audio.assess import assess_emotions as audio_assess


def assess_emotions(video_path: str) ->Tuple[VideoResult, AudioResult]:
    return (video_assess(video_path, 1), audio_assess(video_path))
