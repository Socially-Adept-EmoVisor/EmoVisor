from .assessment.assess import assess_emotions
from .threshholding.thresh import select_best
from .highlighting.cut import cut_video


def process_video(full_video_path: str) -> str:
    print('started')
    moments_with_emotions = assess_emotions(full_video_path)
    highlights = select_best(moments_with_emotions)
    res_video_path = cut_video(full_video_path, highlights)
    return res_video_path


if __name__ == "__main__":
    path = r'C:\Users\hexpisos\Downloads\cut_xqc.mp4'
    filenames = [f'{i}.png' for i in range(832)]
    with imageio.get_writer('valence+arousal.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    with imageio.get_writer('polar.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread('p' + filename)
            writer.append_data(image)

    res = process_video(path)
