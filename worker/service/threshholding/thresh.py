from ..models import VideoResult, AudioResult, Highlight


def select_best(moments: tuple[VideoResult, AudioResult]) -> list[Highlight]:
    return [Highlight(v.time_start, v.time_end, v.roi)
            for v in VideoResult.result]
