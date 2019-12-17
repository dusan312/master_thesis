
import numpy as np
from time import time, gmtime, strftime
import torch
import math
import torchvision
import torch.nn as nn
from imageio import imread
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import RandomCrop, Resize, ToTensor, Normalize
import matplotlib.pyplot as plt
from nets import AlexNetSelf, BinClassifier
from Datasets import SSDataset_STL10
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from my_dataloader_v2 import MyDataLoader
import warnings


def SSTrain_STL10(folder_name, dataset_folder, save_name, model_load_name, save, load, lr,
                  batch_size, n_epochs, load_sec, save_sec, writerSt, gpu_n, print_every=100):
    writer = SummaryWriter(writerSt)
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_h%H_m%M")
    model_save_name = save_name+dt_string
    first_batch=True

    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    train_dataset = SSDataset_STL10(dataset_folder+"unlab_img")
    print(len(train_dataset))
    #exit(1)
    train_loader = MyDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    device = torch.device(gpu_n if torch.cuda.is_available() else 'cpu')
    print('Device: ' + (gpu_n if torch.cuda.is_available() else 'cpu'))

    model = AlexNetSelf()
    model.to(device)

    bin_model = BinClassifier()
    bin_model.to(device)

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': bin_model.parameters()}
    ], lr=lr, weight_decay=0.005)
    loss_function = nn.CrossEntropyLoss()
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    final_lr = 0.00001
    teta = final_lr/lr
    gam = math.exp(math.log(teta,math.e)/n_epochs)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.2)
    just_changed = True
    # training procedure
    saving_path = model_save_name
    loading_path = model_load_name
    if load:
        model.load_state_dict(torch.load(loading_path+'.pth'), strict=False)
        model.to(device)
        print("model weights are loaded on ", device)
    if load_sec:
        bin_model.load_state_dict(torch.load(loading_path+'_BIN.pth'))

    model.train()
    bin_model.train()
    train_losses = []
    t1 = time()
    n_it = int(len(train_loader.dataset)/train_loader.batch_size)
    for epoch in range(n_epochs):
        t5 = time()
        losses = []
        model.train()
        bin_model.train()
        n_correct = 0
        num_ones = 0
        num_zeros = 0
        # Training
        for iteration, (images, labels) in enumerate(train_loader(epoch)):
            org_img, new_img = images
            if first_batch:
                img_grid = torchvision.utils.make_grid(org_img)
                writer.add_image('first_batch_org', img_grid)
                img_grid = torchvision.utils.make_grid(new_img)
                writer.add_image('first_batch_new', img_grid)
                first_batch = False
            for img in range(org_img.size()[0]):
                org_img[img] = normalize(org_img[img])
            for img in range(new_img.size()[0]):
                new_img[img] = normalize(new_img[img])
            org_img = org_img.to(device)
            new_img = new_img.to(device)
            labels = labels.to(device)
            output1 = model(org_img)
            output2 = model(new_img)
            output12 = torch.cat((output1,output2), 1)
            output3 = bin_model(output12)
            optimizer.zero_grad()
            loss = loss_function(output3, labels.long())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            n_correct += torch.sum(output3.argmax(1) == labels).item()
            num_ones += torch.sum(labels == 1).item()
            num_zeros += torch.sum(labels == 0).item()
            # Print remaining time until end of epoch
            if iteration % print_every == 0:
                t2 = time()
                est_ep = (t2-t5)*(n_it-iteration)/(iteration+1)
                est_end = (t2-t1)*(n_it*n_epochs-iteration-epoch*n_it)/(iteration+1+epoch*n_it)
                print("Iteration: " + str(iteration) + " of " + str(n_it) + "   time until end of epoch: " + strftime("%d  %H:%M:%S", gmtime(est_ep)) + '   end of traning in ' + strftime("%d  %H:%M:%S", gmtime(est_end)))
        curr_loss = np.mean(np.array(losses))
        print('Loss after epoch '+str(epoch+1)+'/'+str(n_epochs)+' is:  '+str(curr_loss))
        accuracy_1 = 100.0 * n_correct / (len(train_loader.dataset)*2)
        print('Accuracy after epoch '+str(epoch+1)+'/'+str(n_epochs)+' is:  '+str(accuracy_1))
        writer.add_scalar('Training accuracy', accuracy_1, epoch)
        writer.add_scalar('Training loss', curr_loss,epoch)
        print('Number of zeros {}, number of ones {}'.format(num_zeros,num_ones))
        train_losses.append(curr_loss)
        scheduler.step()

        if save:
            print("Model weights are saved from ", device)
            torch.save(model.state_dict(),saving_path+'.pth')
        if save_sec:
            torch.save(bin_model.state_dict(), saving_path+'_BIN'+'.pth')

    if save:
        print("Model weights are saved from ", device)
        torch.save(model.state_dict(), saving_path+'.pth')
    if save_sec:
        torch.save(bin_model.state_dict(), saving_path+'_BIN'+'.pth')
    writer.close()
    results = np.zeros((n_epochs,1))
    results[:, 0] = train_losses
    np.savetxt(save_name+'_lr='+str(lr)+'_nepochs='+str(n_epochs)+'_STL10.txt', results, fmt='%s')
    return dt_string
