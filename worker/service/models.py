from dataclasses import dataclass
from typing import Optional


@dataclass
class TaskInfo:
    id: str
    video_url: str


@dataclass
class VideoEmotion:
    '''
    изменится в зависимости от модели оценивающий звук
    '''
    emotion: str


@dataclass
class AudioEmotion:
    '''
    изменится в зависимости от модели оценивающий звук
    '''
    emotion: str


@dataclass
class AudioDetection:
    time_start: float
    time_end: float
    emotion: AudioEmotion


@dataclass
class AudioResult:
    result: list[AudioDetection]


@dataclass
class VideoDetection:
    time_start: float
    time_end: float
    roi: tuple[int]
    emotion: VideoEmotion


@dataclass
class VideoResult:
    result: list[VideoDetection]


@dataclass
class Highlight:
    time_start: float
    time_end: float
    roi: Optional[tuple[int]]
