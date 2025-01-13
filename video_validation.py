import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from timm.models import create_model
import argparse
import time
from models import *
from collections import Counter


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
    frame_skip = 3  # 每3帧处理一次
    skip_counter = 0
    second_predictions = []
    class_labels = ["asphalt", "concrete", "gravel", "mud", "snow"]
    current_prediction = {"label": "", "ratio": 0, "frames": 0}

    try:
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            skip_counter += 1
            # 只在每第5帧时进行处理
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

            if frame_count >= fps // frame_skip:  # 调整计数阈值以适应跳帧
                prediction_counts = Counter(second_predictions)
                most_common_prediction = prediction_counts.most_common(1)[0][0]
                most_common_label = class_labels[most_common_prediction]
                prediction_ratio = prediction_counts[most_common_prediction] / len(second_predictions) * 100

                current_prediction = {
                    "label": most_common_label,
                    "ratio": prediction_ratio,
                    "frames": len(second_predictions)
                }

                frame_count = 0
                second_predictions = []

            if current_prediction["label"]:
                text = f"{current_prediction['label']}: {current_prediction['ratio']:.2f}% ({current_prediction['frames']} frames)"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Road Surface Classification', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            process_time = time.time() - start_time
            wait_time = max(0, frame_time - process_time)
            if wait_time > 0:
                time.sleep(wait_time)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video Stream Road Surface Classification')
    # 基本参数
    parser.add_argument('--model', type=str, default='mobilenetv4_conv_aa_large', help='模型名称')
    parser.add_argument('--checkpoint_path', type=str, default='./output/mobilenetv4_conv_aa_large_best_checkpoint.pth',
                        help='训练好的模型权重文件路径')
    parser.add_argument('--video_source', type=str, default='./test_video.mp4',
                        help='视频源（0为默认摄像头，或输入视频文件路径）')
    parser.add_argument('--nb_classes', type=int, default=5, help='数据集分类数量')

    # 必要的模型参数
    parser.add_argument('--extra_attention_block', action='store_true', default=True, help='是否使用额外的注意力模块')
    parser.add_argument('--finetune', type=str, default='./models/model.safetensors', help='微调模型的路径')
    parser.add_argument('--freeze_layers', action='store_true', default=True, help='是否冻结层')
    parser.add_argument('--set_bn_eval', action='store_true', default=True, help='是否设置BN层为eval模式')

    args = parser.parse_args()
    main(args)