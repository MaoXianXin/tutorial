import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
import time
import numpy as np
from autoaugment import ImageNetPolicy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

num_classes = 101
data_dir = './data/food-101'
# Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       ImageNetPolicy(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)

"""当num_workers>0时,它会自动预读取最多2个batch的数据,只要CPU,硬盘处理得足够快,就会形成1个送入model进行处理,1个在队列中等待,1个新的正在预处理的状态,这种情况下加prefetch是没用的"""
"""DataLoader早就加入了多进程读取机制,就是那个num_worker选项,所以完全没必要prefetch了"""
trainloader = torch.utils.data.DataLoader(train_data, batch_size=192, shuffle=True, num_workers=4, pin_memory=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=192, num_workers=4, pin_memory=True)
print(len(trainloader))

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = nn.DataParallel(model)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=[0.9, 0.999])


def train(n_epochs, trainloader, testloader, resnet, optimizer, criterion, save_path, scheduler):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    for epoch in range(n_epochs):
        running_loss = 0
        start = time.time()

        # 开始一个epoch的训练
        for inputs, labels in trainloader:
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = resnet(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        # end = time.time()
        # print(f"Device = cuda; Time per epoch: {(end - start):.3f} seconds")

        # 开始一个epoch的推断
        resnet.eval()
        valid_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = resnet(inputs)
                batch_loss = criterion(logits, labels)
                valid_loss += batch_loss.item()

                # Calculate accuracy
                top_p, top_class = logits.topk(1, dim=1)
                equals = top_class == labels.view(top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            if valid_loss <= valid_loss_min:
                print("Validation loss decreased Saving model")
                torch.save(resnet.state_dict(), save_path)
                valid_loss_min = valid_loss

        end = time.time()
        print(f"Device = cuda; Time per epoch: {(end - start):.3f} seconds")
        print(f"Epoch {epoch+1}/{n_epochs}.. "
              f"Train loss: {running_loss / len(trainloader):.3f}.. "
              f"Test loss: {valid_loss / len(testloader):.3f}.. "
              f"Test accuracy: {accuracy / len(testloader):.3f}")
        resnet.train()

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=13, gamma=0.1)
train(30, trainloader, testloader, model, optimizer, criterion, 'food_classifier_resnet50_noise.pth', exp_lr_scheduler)