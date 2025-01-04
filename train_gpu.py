""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, get_state_dict, ModelEma
from datasets import build_dataset
from datasets.threeaugment import new_data_aug_generator
from models import *
from estimate_model import Predictor, Plot_ROC, OptAUC
from util import utils as utils
from util.engine import train_one_epoch, evaluate
from util.losses import DistillationLoss
from util.samplers import RASampler


# 获取参数
def get_args_parser():
    arg_parser = argparse.ArgumentParser('MobileNetV4训练', add_help=False)
    arg_parser.add_argument('--batch-size', default=16, type=int,
                            help='每次训练时同时处理的图片数量，较大可以提高训练速度，更好地利用GPU并行计算能力，较小会引入更多随机性，可能有助于模型泛化')
    arg_parser.add_argument('--epochs', default=5, type=int,
                            help='训练轮数，较少容易欠拟合，较多容易过拟合')
    arg_parser.add_argument('--predict', default=True, type=bool, help='是否绘制ROC曲线和混淆矩阵')
    arg_parser.add_argument('--opt_auc', default=False, type=bool, help='是否优化AUC指标')

    # Model parameters
    arg_parser.add_argument('--model', default='mobilenetv4_conv_large', type=str, metavar='MODEL',
                            choices=['mobilenetv4_hybrid_large', 'mobilenetv4_hybrid_medium',
                                     'mobilenetv4_hybrid_large_075',
                                     'mobilenetv4_conv_large', 'mobilenetv4_conv_aa_large', 'mobilenetv4_conv_medium',
                                     'mobilenetv4_conv_aa_medium', 'mobilenetv4_conv_small',
                                     'mobilenetv4_hybrid_medium_075',
                                     'mobilenetv4_conv_small_035', 'mobilenetv4_conv_small_050',
                                     'mobilenetv4_conv_blur_medium'],
                            help='模型名称')
    arg_parser.add_argument('--extra_attention_block', default=False, type=bool,
                            help='添加额外的注意力机制模块，提升准确率，增加资源开销')
    arg_parser.add_argument('--input-size', default=384, type=int, help='图片尺寸')
    arg_parser.add_argument('--model-ema', action='store_true',
                            help='Model Exponential Moving Average (模型指数移动平均)，通过维护训练模型参数的移动平均值来提高模型性能和稳定性')
    arg_parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    arg_parser.set_defaults(model_ema=True)
    arg_parser.add_argument('--model-ema-decay', type=float, default=0.99996,
                            help='EMA的衰减率，较大的衰减率（接近1）意味着EMA模型会更多地保留历史参数值，变化更缓慢和平滑')
    arg_parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                            help='是否强制将EMA模型放在CPU上以节省GPU内存')

    # 优化参数
    arg_parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                            choices=['adamw', 'adam'],
                            help='指定优化器')
    arg_parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                            help='优化器的epsilon参数，这个参数对训练的影响通常很小，但在某些情况下（比如梯度值特别小或模型训练不稳定时）可能会变得重要')
    arg_parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                            help='Adam/AdamW 优化器中的 beta 参数，在大多数情况下，默认值就能很好地工作')
    arg_parser.add_argument('--clip-grad', type=float, default=0.02, metavar='NORM',
                            help='设置了梯度的最大范数(norm)阈值，避免过大的梯度更新对模型造成不稳定影响')
    arg_parser.add_argument('--clip-mode', type=str, default='agc',
                            choices=['norm', 'value', 'agc'],
                            help='梯度裁剪的具体方式')
    arg_parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                            help='用于 SGD（随机梯度下降）优化器，通常在 0.9-0.99 之间，默认值通常是 0.9。较大的动量值会使得模型训练更加稳定，但也可能会使得模型收敛速度变慢')
    arg_parser.add_argument('--weight-decay', type=float, default=0.025,
                            help='防止模型过拟合，范围在0.01到0.0001之间，值越大，正则化效果越强，但也可能导致模型欠拟合')

    # 学习率调度参数
    arg_parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                            choices=['cosine', 'step', 'linear'],
                            help='学习率调度器(Learning Rate Scheduler)的类型')
    arg_parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                            help='初始学习率，常用范围是0.001到0.1之间，较大可能导致训练不稳定或难以收敛，较小可能导致收敛速度过慢')
    arg_parser.add_argument('--adamw_lr', type=float, default=3e-3, metavar='AdamWLR',
                            help='AdamW优化器的学习率，范围在1e-4到1e-3之间')
    arg_parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                            help='向学习率中添加随机噪声')
    arg_parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                            help='学习率噪声的百分比大小')
    arg_parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                            help='控制学习率噪声的标准差，值越大意味着噪声波动范围越大，通常建议将标准差设置在较小的范围内（如0.1-0.3）')
    arg_parser.add_argument('--warmup-lr', type=float, default=1e-4, metavar='LR',
                            help='预热阶段的初始学习率，通常设置得比正常学习率小一个数量级或更多，例如如果目标学习率是0.1，那么预热初始学习率可能设为0.01或0.001')
    arg_parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                            help='设置学习率的最小值，防止学习率降得过低，确保模型持续学习的能力，通常设置为初始学习率的1/100到1/1000之间')
    arg_parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                            help='设置学习率衰减的epoch节点')
    arg_parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                            help='设置预热阶段的epochs占比，通常设置为总训练epochs的5%-10%')
    arg_parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                            help='设置学习率调度器的冷却阶段长度，一般为总训练epochs的5%左右')
    arg_parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                            help='指定在验证指标没有改善的情况下继续训练的最大epoch数，通常设置为10-20个epochs')
    arg_parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                            help='设置学习率衰减的比例，较大的衰减率（如0.1）会导致学习率急剧下降，可快速收敛；较小的衰减率（如0.5）则提供更平缓的学习过程，但可能需要更长的训练时间。')

    # 增强参数
    arg_parser.add_argument('--ThreeAugment', action='store_true',
                            help='当设置为True时，训练数据加载器会使用这三种数据增强方法来处理训练图像，从而增加数据的多样性，提高模型的泛化能力。')
    arg_parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                            help='用于数据增强中的颜色抖动，随机调整图像的亮度(brightness)、对比度(contrast)、饱和度(saturation)和色调(hue)')
    arg_parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                            help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    arg_parser.add_argument('--smoothing', type=float, default=0.1,
                            help='通过将原始的one-hot标签转换为软标签来防止模型过度自信，提高泛化能力')
    arg_parser.add_argument('--train-interpolation', type=str, default='bicubic',
                            choices=['bilinear', 'bicubic', 'nearest'],
                            help='图像插值方法')
    arg_parser.add_argument('--repeated-aug', action='store_true',
                            help='在每个epoch中对同一张图像进行多次数据增强，提高模型对不同图像变化的鲁棒性')
    arg_parser.add_argument('--no-repeated-aug',
                            action='store_false', dest='repeated_aug')
    arg_parser.set_defaults(repeated_aug=True)

    # 随机擦除参数
    arg_parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                            help='随机地将输入图像的某个矩形区域"擦除"，控制随机擦除数据增强的概率，值越大表示越多的训练图像会被应用随机擦除')
    arg_parser.add_argument('--remode', type=str, default='pixel',
                            choices=['pixel', 'rand', 'const'],
                            help='指定随机擦除时填充擦除区域的方式')
    arg_parser.add_argument('--recount', type=int, default=1,
                            help='指定在一张图像上进行随机擦除的次数，设置为1-3之间比较合适')
    arg_parser.add_argument('--resplit', action='store_true', default=False,
                            help='控制随机擦除区域的分割方式，true时允许将一个大的擦除区域分割成多个小区域，false时保持每个擦除区域为单个连续区域')

    # 混合参数
    arg_parser.add_argument('--mixup', type=float, default=0.8,
                            help='随机选择两张训练图片按照一定比例进行线性混合，提高模型的泛化能力和鲁棒性')
    arg_parser.add_argument('--cutmix', type=float, default=1.0,
                            help='从一张图片中随机裁剪出一个矩形区域，将这个区域替换为另一张图片中相同位置的内容，按照裁剪区域的面积比例进行混合')
    arg_parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                            help='裁剪区域大小的范围')
    arg_parser.add_argument('--mixup-prob', type=float, default=1.0,
                            help='数据增强的概率，0 表示完全不使用 Mixup，1 表示对所有批次都使用 Mixup')
    arg_parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                            help='控制在训练过程中切换不同数据增强方法（从 Mixup 切换到 CutMix）的概率')
    arg_parser.add_argument('--mixup-mode', type=str, default='batch',
                            choices=['batch', 'pair', 'elem'],
                            help='混合增强的具体实现方式')

    # 蒸馏参数
    arg_parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                            help='一个较大的预训练模型(教师模型)被用来指导一个较小模型(学生模型)的训练')
    arg_parser.add_argument('--teacher-path', type=str,
                            default='https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth',
                            help='教师模型地址')
    arg_parser.add_argument('--distillation-type', default='none', type=str,
                            choices=['none', 'soft', 'hard'],
                            help='知识蒸馏的类型')
    arg_parser.add_argument('--distillation-alpha', default=0.5, type=float,
                            help='控制原始损失和蒸馏损失的权重比例')
    arg_parser.add_argument('--distillation-tau', default=1.0, type=float,
                            help='温度参数，用于调节软标签的"软度"')

    # 微调参数
    arg_parser.add_argument('--finetune', default='./models/model.safetensors',
                            help='微调预训练模型，代码会从指定路径（可以是本地文件或 URL）加载预训练模型的权重')
    arg_parser.add_argument('--freeze_layers', type=bool, default=True,
                            help='为True时，预训练模型的特征提取部分会保持不变，只有分类器层的参数会被更新和优化')
    arg_parser.add_argument('--set_bn_eval', action='store_true', default=False,
                            help='控制批量归一化层的行为，当在小批量数据上微调大型预训练模型时，或当想保持预训练模型的特征分布特性时特别有用')

    # 数据集参数
    arg_parser.add_argument('--data_root', default='./datasets/dataset', type=str,
                            help='数据集路径')
    arg_parser.add_argument('--nb_classes', default=12, type=int,
                            help='数据集分类数量')
    arg_parser.add_argument('--data-set', default='IMNET', type=str,
                            choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                            help='数据集类型')
    arg_parser.add_argument('--inat-category', default='name', type=str,
                            choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                            help='semantic granularity')
    arg_parser.add_argument('--output_dir', default='./output',
                            help='指定输出目录的路径')
    arg_parser.add_argument('--writer_output', default='./',
                            help='TensorBoard 日志的输出目录')
    arg_parser.add_argument('--device', type=str, default='cuda',
                            choices=['cuda', 'mps', 'cpu'],
                            help='指定模型训练和推理时使用的硬件设备')
    arg_parser.add_argument('--seed', default=0, type=int,
                            help='设置随机数种子，确保实验的可重复性')
    arg_parser.add_argument('--resume', default='',
                            help='从之前保存的检查点恢复训练')
    arg_parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                            help='指定训练开始的轮次')
    arg_parser.add_argument('--eval', action='store_true',
                            help='用于将模型切换到评估模式')
    arg_parser.add_argument('--dist-eval', action='store_true', default=False,
                            help='启用分布式评估模式')
    arg_parser.add_argument('--num_workers', default=0, type=int,
                            help='配置数据加载时的并行工作进程数，0 表示仅使用主进程加载数据，通常建议设置为 CPU 核心数的 2-4 倍')
    arg_parser.add_argument('--pin-mem', action='store_true',
                            help='启用内存页锁定，提高 CPU 到 GPU 的数据传输速度，减少数据加载的延迟，提升整体训练效率')
    arg_parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                            help='不启用内存页锁定')
    arg_parser.set_defaults(pin_mem=True)

    # 训练参数
    arg_parser.add_argument('--world_size', default=1, type=int,
                            help='指定分布式训练中的进程总数')
    arg_parser.add_argument('--local_rank', default=0, type=int,
                            help='指定当前进程在本地机器上的进程序号')
    arg_parser.add_argument('--dist_url', default='env://',
                            help='指定分布式训练的URL地址')
    arg_parser.add_argument('--save_freq', default=1, type=int,
                            help='控制模型检查点的保存频率，指定每隔多少个epoch保存一次模型检查点')
    return arg_parser


# 获取设备类型
def get_device(args):
    if args.device == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def main(args):
    print(args)
    utils.init_distributed_mode(args)

    if args.local_rank == 0:
        writer = SummaryWriter(os.path.join(args.writer_output, 'runs'))

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError(
            "Finetuning with distillation not yet supported")

    device = get_device(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, dataset_val = build_dataset(args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval datasets not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    num_workers = 0 if device.type == 'mps' else args.num_workers  # MPS 时建议为0
    pin_memory = device.type != 'cpu'  # CPU 时不需要 pin_memory

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")

    model = create_model(
        args.model,
        extra_attention_block=args.extra_attention_block,
        args=args
    )
    model.reset_classifier(num_classes=args.nb_classes)

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = utils.load_model(args.finetune, model)

        checkpoint_model = checkpoint
        # state_dict = model.state_dict()
        # new_state_dict = utils.map_safetensors(checkpoint_model, state_dict)

        for k in list(checkpoint_model.keys()):
            if 'classifier' in k:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.freeze_layers:
            for name, para in model.named_parameters():
                if 'classifier' not in name:
                    para.requires_grad_(False)
                # else:
                #     print('training {}'.format(name))
            if args.extra_attention_block:
                for name, para in model.extra_attention_block.named_parameters():
                    para.requires_grad_(True)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but
        # before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    # args.lr = linear_scaled_lr
    #
    # print('*****************')
    # print('Initial LR is ', linear_scaled_lr)
    # print('*****************')

    # optimizer = create_optimizer(args, model_without_ddp)
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=2e-4,
                                  weight_decay=args.weight_decay) if args.finetune else create_optimizer(args,
                                                                                                         model_without_ddp)

    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is
    # 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    max_accuracy = 0.0

    output_dir = Path(args.output_dir)
    if args.output_dir and utils.is_main_process():
        with (output_dir / "model.txt").open("a") as f:
            f.write(str(model))
    if args.output_dir and utils.is_main_process():
        with (output_dir / "args.txt").open("a") as f:
            f.write(json.dumps(args.__dict__, indent=2) + "\n")
    if args.resume or os.path.exists(f'{args.output_dir}/{args.model}_best_checkpoint.pth'):
        args.resume = f'{args.output_dir}/{args.model}_best_checkpoint.pth'
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            print("Loading local checkpoint at {}".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
        msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:

            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():  # load parameters to cuda
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        if device.type == 'cuda':
                            state[k] = v.cuda()
                        elif device.type == 'mps':
                            state[k] = v.to(device)
                        else:
                            state[k] = v

            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            max_accuracy = checkpoint['best_score']
            print(f'Now max accuracy is {max_accuracy}')
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(
                    model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
    if args.eval:
        # util.replace_batchnorm(model) # Users may choose whether to merge Conv-BN layers during eval
        print(f"Evaluating model: {args.model}")
        print(f'No Visualization')
        test_stats = evaluate(data_loader_val, model, device, None, None, args, visualization=False)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
    # print(model)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, args.clip_mode, model_ema, mixup_fn,
            # set_training_mode=args.finetune == '',  # keep in eval mode during finetuning
            set_training_mode=True,
            set_bn_eval=args.set_bn_eval,  # set bn to eval if finetune
            writer=writer,
            args=args
        )

        lr_scheduler.step(epoch)

        test_stats = evaluate(data_loader_val, model, device, epoch, writer, args, visualization=True)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.output_dir:
                ckpt_path = os.path.join(output_dir, f'{args.model}_best_checkpoint.pth')
                checkpoint_paths = [ckpt_path]
                print("Saving checkpoint to {}".format(ckpt_path))
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'best_score': max_accuracy,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)

        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # plot ROC curve and confusion matrix
    if args.predict and utils.is_main_process():
        model_predict = create_model(
            args.model,
            extra_attention_block=args.extra_attention_block,
            args=args
        )

        model_predict.reset_classifier(num_classes=args.nb_classes)
        model_predict.to(device)
        print('*******************STARTING PREDICT*******************')
        Predictor(model_predict, data_loader_val, f'{args.output_dir}/{args.model}_best_checkpoint.pth', device)
        Plot_ROC(model_predict, data_loader_val, f'{args.output_dir}/{args.model}_best_checkpoint.pth', device)

        if args.opt_auc:
            OptAUC(model_predict, data_loader_val, f'{args.output_dir}/{args.model}_best_checkpoint.pth', device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'MobileNetV4 training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
