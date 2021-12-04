from ..models import VideoResult, AudioResult, Highlight
from typing import List,Tuple


def select_best(moments: Tuple[VideoResult, AudioResult]) -> List[Highlight]:
    return [Highlight(v.time_start, v.time_end, v.roi)
            for v in VideoResult.result]
