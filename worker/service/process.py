from .assessment.assess import assess_emotions
from .threshholding.thresh import select_best
from .highlighting.cut import cut_video


def process_video(full_video_path: str) -> str:
    print('started')
    moments_with_emotions = assess_emotions(full_video_path)
    highlights = select_best(moments_with_emotions)
    res_video_path = cut_video(full_video_path, highlights)
    return res_video_path

if __name__=="__main__":
    path = 'D:\Papka\work\EmoVisor\Boomers_Get_Mad_At_Tyler1.mp4'
    res = process_video(path)
    print(res)