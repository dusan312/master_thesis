
import numpy as np
from torchvision import transforms
from time import time, gmtime, strftime
import os
import torch
import torch.nn as nn
from imageio import imread
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import Normalize, Resize, ToTensor
import matplotlib.pyplot as plt
from nets import AlexNet, AlexNetSelf, RotClassifier
from Datasets import RotationDataset_STL10
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from rot_dataloader import RotDataLoader
import random
import torchvision
import math


def SSTrainRotation_STL10(folder_name, dataset_folder, save_name, model_load_name, save, load, lr, batch_size,
                          n_epochs, load_sec, save_sec, writerSt, gpu_n, print_every=100):
    writer = SummaryWriter(writerSt)
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_h%H_m%M")
    model_save_name = save_name+dt_string

    train_dataset = RotationDataset_STL10(dataset_folder+"unlab_img")

    train_loader = RotDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    device = torch.device(gpu_n if torch.cuda.is_available() else 'cpu')
    print('Device: ' + (gpu_n if torch.cuda.is_available() else 'cpu'))

    model = AlexNetSelf()
    model = model.to(device)

    rot_model = RotClassifier()
    rot_model = rot_model.to(device)

    optimizer = torch.optim.Adam([{'params': model.parameters()},
                                  {'params': rot_model.parameters()}],lr=lr, weight_decay=0.0005)
    #optimizer = torch.optim.Adam([{'params': model.parameters()},
    #                             {'params': rot_model.parameters()}], lr=lr, weight_decay=0.0005)
    loss_function = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=30, gamma=0.2)

    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])

    ##### training procedure
    saving_path = model_save_name
    loading_path = model_load_name
    if load:
        model.load_state_dict(torch.load(loading_path+'.pth'), strict=False)
        model.to(device)
        print("model weights are loaded on", device)
    if load_sec:
        rot_model.load_state_dict(torch.load(loading_path+'_BIN.pth'))

    model = model.to(device)
    model_rot = rot_model.to(device)
    model.train()
    model_rot.train()
    train_losses = []

    t1 = time()
    n_it = int(len(train_loader.dataset)/train_loader.batch_size)

    for epoch in range(n_epochs):
        t5 = time()
        losses = []
        # Training
        n_correct = 0
        for iteration, (images, labels) in enumerate(train_loader(epoch)):
            for img in range(images.size()[0]):
                images[img]=normalize(images[img])
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            output = model_rot(output)
            optimizer.zero_grad()
            loss = loss_function(output, labels.long())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            n_correct += torch.sum(output.argmax(1) == labels).item()
            # Print remaining time until end of epoch
            if iteration % print_every == 0:
                t2 = time()
                est_ep = (t2-t5)*(n_it-iteration)/(iteration+1)
                est_end = (t2-t1)*(n_it*n_epochs-iteration-epoch*n_it)/(iteration+1+epoch*n_it)
                print("Iteration: " + str(iteration) + " of " + str(n_it) + "   time until end of epoch: " +
                      strftime("%H:%M:%S", gmtime(est_ep)) + '   end of traning in ' +
                      strftime("%H:%M:%S", gmtime(est_end)))
        curr_loss = np.mean(np.array(losses))
        print('Loss after epoch '+str(epoch+1)+'/'+str(n_epochs)+' is:  '+str(curr_loss))
        accuracy_1 = 100.0 * n_correct / (4*len(train_loader.dataset))
        print('Accuracy after epoch '+str(epoch+1)+'/'+str(n_epochs)+' is:  '+str(accuracy_1))
        writer.add_scalar('Training accuracy', accuracy_1, epoch)
        writer.add_scalar('Training loss', curr_loss,epoch)
        train_losses.append(curr_loss)
        scheduler.step()
        if epoch % 5 == 0 and save:
            print("Model weights are saved from", device)
            torch.save(model.state_dict(),saving_path+'.pth')
        if epoch % 5 == 0 and save_sec:
            torch.save(rot_model.state_dict(), saving_path+'_BIN'+'.pth')

    if save:
        print("Model weights are saved from", device)
        torch.save(model.state_dict(), saving_path+'.pth')
    if save_sec:
        torch.save(rot_model.state_dict(), saving_path+'_ROT'+'.pth')

    writer.close()
    results = np.zeros((n_epochs,1))
    results[:, 0] = train_losses
    np.savetxt(save_name+'_lr='+str(lr)+'_nepochs='+str(n_epochs)+'_STL10.txt', results, fmt='%s')
    return dt_string

