from dataclasses import dataclass
from typing import Optional,List,Tuple


@dataclass
class TaskInfo:
    id: str
    video_url: str


@dataclass
class VideoEmotion:
    arrousal:float
    valence:float
    expression:str


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
    result: List[AudioDetection]


@dataclass
class VideoDetection:
    time_start: float
    time_end: float
    roi: Tuple[int]
    emotion: VideoEmotion


@dataclass
class VideoResult:
    result: List[VideoDetection]


@dataclass
class Highlight:
    time_start: float
    time_end: float
    roi: Optional[Tuple[int]]
