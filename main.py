'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
# from utils import progress_bar
from tqdm import tqdm
from utils import MYCIFAR10

from sklearn import metrics


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--num_class', default=4, type=int, help='num_class')
parser.add_argument('--data_name', default='cifar10_20000_triobject', type=str, help='data_name')
parser.add_argument('--job_id', default='local', type=str, help='job_id')
parser.add_argument('--load_model', '-r', action='store_true', help='load_model from checkpoint')
parser.add_argument('--load_model_path', default='', type=str, help='load_model_path')
parser.add_argument('--test_dbindex', action='store_true', default=False)
parser.add_argument('--just_test', action='store_true', default=False)
parser.add_argument('--no_save', action='store_true', default=False)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = MYCIFAR10(
    root='./data', train=True, download=False, transform=transform_train, data_name = args.data_name)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = MYCIFAR10(
    root='./data', train=False, download=False, transform=transform_test, data_name = args.data_name)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18(args.num_class)
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.load_model:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}.pth'.format(args.load_model_path))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_count = 0
    train_bar = tqdm(trainloader)
    for inputs, targets in train_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        feature, outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        batch_count += 1
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    #  % (train_loss/(batch_count), 100.*correct/total, correct, total))
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} | Acc: {:.3f}'.format(epoch, 200, train_loss/(batch_count), 100.*correct/total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_count = 0
    test_bar = tqdm(testloader)
    with torch.no_grad():
        for inputs, targets in test_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            feature, outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            batch_count += 1
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            test_bar.set_description('Test Epoch: [{}/{}] Loss: {:.4f} | Acc: {:.3f}'.format(epoch, 200, test_loss/(batch_count), 100.*correct/total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if not args.no_save and not args.just_test:
            torch.save(state, './checkpoint/{}_{}_ckpt.pth'.format(args.data_name, args.job_id))
        best_acc = acc
    return best_acc

def test_cluster():
    net.eval()
    test_bar = tqdm(testloader)
    feature_bank = []
    label_bank = []
    with torch.no_grad():
        for inputs, targets in test_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            feature, outputs = net(inputs)
            feature_bank.append(feature)
            label_bank.append(targets)
            
            test_bar.set_description('Feature extracting.')
    
    feature_bank = torch.cat(feature_bank, dim=0).cpu().numpy()
    label_bank = torch.cat(label_bank, dim=0).cpu().numpy()
    print(metrics.davies_bouldin_score(feature_bank, label_bank))

if not args.test_dbindex:
    if args.just_test:
        best_acc = test(start_epoch)
        test_cluster()
        print(best_acc)
    else:
        for epoch in range(start_epoch, start_epoch+200):
            train(epoch)
            test(epoch)
            scheduler.step()
else:
    test_cluster()
