import torch
import torchvision.transforms as transforms
from PIL import Image
from timm.models import create_model
from util.utils import load_model
import argparse
from models import *
import os


def load_trained_model(model_name, checkpoint_path, num_classes, device, args):
    # 创建模型
    model = create_model(
        model_name,
        extra_attention_block=args.extra_attention_block,
        args=args
    )
    model.reset_classifier(num_classes=num_classes)
    model.to(device)

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((360, 360)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = load_trained_model(
        args.model,
        args.checkpoint_path,
        args.nb_classes,
        device,
        args
    )

    # 读取并预处理图片
    image = preprocess_image(args.image_path)
    image = image.to(device)

    # 进行预测
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    class_labels = ["asphalt", "concrete", "gravel", "mud", "snow"]
    predicted_label = class_labels[predicted_class.item()]
    confidence_value = confidence.item() * 100

    print(f"预测的路面类别是: {predicted_label}")
    print(f"预测的置信度为: {confidence_value:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Classification Validation')
    # 基本参数
    parser.add_argument('--model', type=str, default='mobilenetv4_conv_aa_large', help='模型名称')
    parser.add_argument('--checkpoint-path', type=str, default='./output/mobilenetv4_conv_aa_large_best_checkpoint.pth',
                        help='训练好的模型权重文件路径')
    parser.add_argument('--image-path', type=str, default='./datasets/validate/test_image.png', help='要验证的图片路径')
    parser.add_argument('--nb-classes', type=int, default=5, help='数据集分类数量')

    # 从训练代码中添加的必要参数
    parser.add_argument('--extra-attention-block', action='store_true', default=True, help='是否使用额外的注意力模块')
    parser.add_argument('--finetune', type=str, default='./models/model.safetensors', help='微调模型的路径')
    parser.add_argument('--freeze-layers', action='store_true', default=True, help='是否冻结层')
    parser.add_argument('--set-bn-eval', action='store_true', default=True, help='是否设置BN层为eval模式')

    args = parser.parse_args()
    main(args)