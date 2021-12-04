from __future__ import division
import cv2
from loguru import logger
from ...models import VideoDetection, VideoEmotion, VideoResult
import cv2
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
from .emonet.models.emonet import EmoNet
from pathlib import Path
import numpy as np
from .emonet.data_augmentation import DataAugmentor

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
out_expression = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise',
                 4: 'fear', 5: 'disgust', 6: 'anger', 7: 'contempt', 8: 'none'}

image_size = 256

transform_image = transforms.Compose([transforms.ToTensor()])
transform_image_shape_no_flip = DataAugmentor(image_size, image_size)

mtcnn = MTCNN(select_largest=True, device=device)
# Loading the model
state_dict_path = Path(__file__).parent.joinpath(
    'pretrained', f'emonet_8.pth')

print(f'Loading the model from {state_dict_path}.')
state_dict = torch.load(str(state_dict_path), map_location='cpu')
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
net = EmoNet(n_expression=8).to(device)
net.load_state_dict(state_dict, strict=False)
net.eval()

def get_emotion(image, model)->VideoEmotion:
    # Do inference
    with torch.no_grad():
        out = model(image)
    expr = out['expression']
    print(expr)
    expr = np.argmax(np.squeeze(expr.cpu().numpy()), axis=0)
    print(expr)
    val = out['valence']
    ar = out['arousal']
    landmarks = out['heatmap']
    return VideoEmotion(arrousal=ar.data[0],valence=val.data[0],expression=out_expression)

def assess_emotions(video_path: str, frame_gap: int) -> VideoResult:
    if frame_gap >= 0:
        frame_gap = 1
    try:
        print('started video assess')
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        res = VideoResult(result=[])
        cur = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.info("unexpeted EOF")
                break
            if cur % frame_gap != 0:
                continue
            cur += 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Detect faces
            print('have frame')
            boxes, _ = mtcnn.detect(rgb_frame, landmarks=False)
            print(boxes)
            for (x0, y0, x1, y1) in boxes:
                cv2.rectangle(frame, (int(x0), int(y0)),
                            (int(x1), int(y1)), (255, 0, 0), 2)
                image = rgb_frame[int(y0): int(y1), int(x0): int(x1)]
                # Apply transforms
                if transform_image_shape_no_flip is not None:
                    image, _ = transform_image_shape_no_flip(image, None,)
                    # Fix for PyTorch currently not supporting negative stric
                    image = np.ascontiguousarray(image)
                if transform_image is not None:
                    image_tensor = transform_image(image)
                image_tensor = image_tensor.unsqueeze_(0)
                input = image_tensor.to(device)

                # Predict probabilities
                emotion = get_emotion(input, net)
                # print(emotion)
                # res.result.append(VideoDetection(
                #     time_start=min(0, (cur-frame_gap) / fps), time_end=cur / fps, roi=(x0, y0, x1, y1), emotion=emotion))
    finally:
        cap.release()
