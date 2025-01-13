import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from timm.models import create_model
import argparse
import time
from models import *
from collections import Counter
import cv2


def load_trained_model(model_name, checkpoint_path, num_classes, device, args):
    model = create_model(
        model_name,
        extra_attention_block=args.extra_attention_block,
        args=args
    )
    model.reset_classifier(num_classes=num_classes)
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model


def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((360, 360)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(frame).unsqueeze(0)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(args.model, args.checkpoint_path, args.nb_classes, device, args)

    if args.video_source.isdigit():
        video_source = int(args.video_source)
    else:
        video_source = args.video_source
    cap = cv2.VideoCapture(video_source)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps if fps > 0 else 0.033
    frame_count = 0
    frame_skip = 5  # 每5帧处理一次
    skip_counter = 0
    second_predictions = []
    class_labels = ["asphalt", "concrete", "gravel", "mud", "snow"]
    current_second = 0
    last_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            skip_counter += 1
            if skip_counter >= frame_skip:
                skip_counter = 0
                processed_frame = preprocess_frame(frame)
                processed_frame = processed_frame.to(device)

                with torch.no_grad():
                    output = model(processed_frame)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, predicted_class = torch.max(probabilities, 1)

                second_predictions.append(predicted_class.item())
                frame_count += 1

            if frame_count >= fps // frame_skip:
                current_time = time.time()
                processing_time = current_time - last_time

                prediction_counts = Counter(second_predictions)
                most_common_prediction = prediction_counts.most_common(1)[0][0]
                most_common_label = class_labels[most_common_prediction]
                prediction_ratio = prediction_counts[most_common_prediction] / len(second_predictions) * 100

                current_second += 1
                print(
                    f"Second {current_second}: {most_common_label} ({prediction_ratio:.2f}%) - Processing time: {processing_time:.3f}s")

                frame_count = 0
                second_predictions = []
                last_time = current_time

    finally:
        cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video Stream Road Surface Classification')
    parser.add_argument('--model', type=str, default='mobilenetv4_conv_aa_large', help='模型名称')
    parser.add_argument('--checkpoint_path', type=str, default='./output/mobilenetv4_conv_aa_large_best_checkpoint.pth',
                        help='训练好的模型权重文件路径')
    parser.add_argument('--video_source', type=str, default='./test_video.mp4',
                        help='视频源（0为默认摄像头，或输入视频文件路径）')
    parser.add_argument('--nb_classes', type=int, default=5, help='数据集分类数量')
    parser.add_argument('--extra_attention_block', action='store_true', default=True, help='是否使用额外的注意力模块')
    parser.add_argument('--finetune', type=str, default='./models/model.safetensors', help='微调模型的路径')
    parser.add_argument('--freeze_layers', action='store_true', default=True, help='是否冻结层')
    parser.add_argument('--set_bn_eval', action='store_true', default=True, help='是否设置BN层为eval模式')

    args = parser.parse_args()
    main(args)