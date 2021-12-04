import cv2
from loguru import logger

from ...models import VideoDetection, VideoEmotions, VideoResult


def detect_faces(image) -> list[tuple[int]]:
    '''
    координаты лиц как в opencv: левый верхний угол, нижний правый угол
    '''
    return [(0, 0, 20, 20), (15, 40, 60, 60)]


def assess_face(face) -> VideoEmotions:
    return VideoEmotions("happy")


def assess_emotions(video_path: str, skipfames: int) -> VideoResult:
    try:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        res = VideoResult(result=[])
        cur = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.info("unexpeted EOF")
                break
            if cur % skipfames != 0:
                continue
            cur += 1

            rois = detect_faces(frame)
            for (x0, y0, x1, y1) in rois:
                face = frame[y0: y1, x0:x1]
                emotion = assess_face(face)
                if emotion is None:
                    continue
                res.result.append(VideoDetection(
                    time_start=min(0, (cur-skipfames) / fps), time_end=cur / fps, roi=(x0, y0, x1, y1), emotion=emotion))
    finally:
        cap.release()
