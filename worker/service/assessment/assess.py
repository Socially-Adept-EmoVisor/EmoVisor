from worker.service.models import AudioResult


from ..models import VideoResult, AudioResult
import video.assess
import audio.assess


def assess_emotions(video_path: str) -> tuple[VideoResult, AudioResult]:
    return (video.assess.assess_emotions(video_path, 1), audio.assess.assess_emotions(video_path))
