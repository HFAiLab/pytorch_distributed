import hf_env
hf_env.set_env('202111')

import os
import time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, models

import hfai
import hfai.nccl.distributed as dist
from hfai.nn.parallel import DistributedDataParallel


def train(dataloader, model, criterion, optimizer, scheduler, loss_scaler, epoch, local_rank, start_step, best_acc, save_path):
    model.train()
    for step, batch in enumerate(dataloader):
        if step < start_step:
            continue

        samples, labels = [x.cuda(non_blocking=True) for x in batch]

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, labels)
        loss_scaler.scale(loss).backward()
        
        loss_scaler.step(optimizer)
        loss_scaler.update()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # 保存
        if dist.get_rank() == 0 and local_rank == 0 and hfai.client.receive_suspend_command():
            state = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'acc': best_acc,
                'epoch': epoch,
                'step': step + 1
            }
            torch.save(state, save_path / 'latest.pt')
            time.sleep(5)
            hfai.client.go_suspend()


def validate(dataloader, model, criterion, epoch, local_rank):
    loss, correct1, correct5, total = torch.zeros(4).cuda()
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            samples, labels = [x.cuda(non_blocking=True) for x in batch]
            outputs = model(samples)
            loss += criterion(outputs, labels)
            _, preds = outputs.topk(5, -1, True, True)
            correct1 += torch.eq(preds[:, :1], labels.unsqueeze(1)).sum()
            correct5 += torch.eq(preds, labels.unsqueeze(1)).sum()
            total += samples.size(0)

    for x in [loss, correct1, correct5, total]:
        dist.reduce(x, 0)

    if local_rank == 0:
        loss_val = loss.item() / dist.get_world_size() / len(dataloader)
        acc1 = 100 * correct1.item() / total.item()
        acc5 = 100 * correct5.item() / total.item()
        print(f'Epoch: {epoch}, Loss: {loss_val}, Acc1: {acc1:.2f}%, Acc5: {acc5:.2f}%', flush=True)

    return correct1.item() / total.item()


def main(local_rank):
    # 超参数设置
    epochs = 100
    batch_size = 400
    num_workers = 4
    lr = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    save_path = Path('output/resnet_ddp_amp')
    save_path.mkdir(exist_ok=True, parents=True)

    # 多机通信
    ip = os.environ['MASTER_IP']
    port = os.environ['MASTER_PORT']
    hosts = int(os.environ['WORLD_SIZE'])  # 机器个数
    rank = int(os.environ['RANK'])  # 当前机器编号
    gpus = torch.cuda.device_count()  # 每台机器的GPU个数

    # world_size是全局GPU个数，rank是当前GPU全局编号
    dist.init_process_group(backend='nccl', init_method=f'tcp://{ip}:{port}', world_size=hosts * gpus, rank=rank * gpus + local_rank)
    torch.cuda.set_device(local_rank)

    # 模型、数据、优化器
    model = models.resnet50().cuda()
    model = DistributedDataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr,  momentum=momentum, weight_decay=weight_decay)
    loss_scaler = torch.cuda.amp.GradScaler(enabled=True)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])  # 定义训练集变换
    train_dataset = hfai.datasets.ImageNet(split='train', transform=train_transform)
    train_datasampler = DistributedSampler(train_dataset)
    train_dataloader = train_dataset.loader(batch_size=batch_size, sampler=train_datasampler, num_workers=num_workers, pin_memory=True)

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])  # 定义测试集变换
    val_dataset = hfai.datasets.ImageNet(split='val', transform=val_transform)
    val_datasampler = DistributedSampler(val_dataset)
    val_dataloader = val_dataset.loader(batch_size=batch_size, sampler=val_datasampler, num_workers=num_workers, pin_memory=True)

    # 加载
    best_acc, start_epoch, start_step = 0, 0, 0
    if (save_path / 'latest.pt').exists():
        ckpt = torch.load(save_path / 'latest.pt', map_location='cpu')
        model.module.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        best_acc, start_epoch, start_step = ckpt['acc'], ckpt['epoch'], ckpt['step']

    # 训练、验证
    for epoch in range(start_epoch, epochs):
        t1 = time.time()
        train_datasampler.set_epoch(epoch)
        train(train_dataloader, model, criterion, optimizer, scheduler, loss_scaler, epoch, local_rank, start_step, best_acc, save_path)
        start_step = 0 
        scheduler.step()
        acc = validate(val_dataloader, model, criterion, epoch, local_rank)
        t2 = time.time()

        # 保存
        if rank == 0 and local_rank == 0:
            print("cost time per epoch: {:.4f} s".format(t2-t1))
            if acc > best_acc:
                best_acc = acc
                print(f'New Best Acc: {100*acc:.2f}%!')
                torch.save(model.module.state_dict(), save_path / 'best.pt')


if __name__ == '__main__':
    ngpus = torch.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=True)