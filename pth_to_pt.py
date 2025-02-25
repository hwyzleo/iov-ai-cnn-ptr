import torch
import os
from models import *
from models.build_mobilenet_v4 import custom_mobilenetv4_conv_aa_large as MyCustomModel


def main(args):
    # 设置设备
    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    # elif torch.backends.mps.is_available():
    #     device = torch.device('mps')
    # else:
    #     device = torch.device('cpu')
    # 设置成 cpu 保障通用性
    device = torch.device('cpu')

    # 实例化自定义模型
    model = MyCustomModel(num_classes=5)
    # print("Model structure before loading state_dict:")
    # print(model)

    # 加载模型检查点文件
    checkpoint_path = "output/mobilenetv4_conv_aa_large_best_checkpoint.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
    except Exception as e:
        raise Exception(f"Error loading checkpoint: {str(e)}")

    # 设置为评估模式并移到指定设备
    model = model.to(device)
    model.eval()

    # 确保所有子模块也转移到 CPU
    for module in model.modules():
        module.to(device)

    # 确保所有参数和缓冲区都在 CPU 上
    for param in model.parameters():
        param.data = param.data.to(device)
    for buffer in model.buffers():
        buffer.data = buffer.data.to(device)

    # 转换为TorchScript
    try:
        scripted_model = torch.jit.script(model)
    except Exception as e:
        raise Exception(f"Error during TorchScript conversion: {str(e)}")

    # 确保保存目录存在
    save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "mobilenetv4_conv_aa_large_best_checkpoint.pt")

    # 保存模型
    try:
        scripted_model.save(save_path)
        print(f"Model successfully saved to {save_path}")
    except Exception as e:
        raise Exception(f"Error saving model: {str(e)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert PyTorch model to TorchScript")
    args = parser.parse_args()
    main(args)
