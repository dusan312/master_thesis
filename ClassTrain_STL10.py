import numpy as np
from time import time, gmtime, strftime
import os
import torch
import torch.nn as nn
from imageio import imread
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize
from nets import AlexNetSelf, Normal_test
from Datasets import ClassDataset_STL10
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import random
import torchvision
import math


def ClassTrain_STL10(freez, folder_name, dataset_folder, save_name, model_load_name, save, load, lr, batch_size, n_epochs,
                     load_sec, save_sec,  writerSt, gpu_n):
    writer = SummaryWriter(writerSt)
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_h%H_m%M")
    model_save_name = save_name+dt_string
    first_batch=True
    acc_fold=[]

    for fold in range(10):

        train_dataset = ClassDataset_STL10(dataset_folder+"img",fold_n=fold)
        val_dataset = ClassDataset_STL10(dataset_folder+"img_test")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

        device = torch.device(gpu_n if torch.cuda.is_available() else 'cpu')
        print('Device: ' + (gpu_n if torch.cuda.is_available() else 'cpu'))

        model = AlexNetSelf()
        class_model = Normal_test(num_classes=10)

        model = model.to(device)
        class_model = class_model.to(device)
        if freez:
            optimizer = torch.optim.Adam([
                #{'params': model.parameters()},
                {'params': class_model.parameters()}
            ], lr=lr, weight_decay=0.0005)
        else:
            optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': class_model.parameters()}
            ], lr=lr, weight_decay=0.0005)
        loss_function = nn.CrossEntropyLoss()
        final_lr = 0.000003
        teta = final_lr/lr
        gam = math.exp(math.log(teta,math.e)/n_epochs)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

        # trainig procedure
        saving_path = model_save_name
        loading_path = model_load_name
        if load:
            model.load_state_dict(torch.load(loading_path+'.pth'), strict=False)
            model.to(device)
            print("model weights are loaded on ", device)
        if load_sec:
            class_model.load_state_dict(torch.load(folder_name+load+'_cl.pth'), strict=False)
            model.to(device)
            print("Classfication model weights are loaded from ", device)

        best_loss = 999999999999999999999
        model.train()
        train_losses = []
        val_accuracies = []
        val_losses = []

        for epoch in range(n_epochs):
            t5 = time()
            losses = []
            # Training
            if freez:
                model.eval()
            else:
                model.train()
            class_model.train()
            n_correct_1 = 0
            for iteration, (images, labels) in enumerate(train_loader):
                if first_batch:
                    img_grid = torchvision.utils.make_grid(images)
                    writer.add_image('first_batch', img_grid)
                    first_batch = False
                for img in range(images.size()[0]):
                    images[img]=normalize(images[img])
                images = images.to(device)
                labels = labels.to(device)
                if freez:
                    with torch.no_grad():
                        output = model(images)
                else:
                    output = model(images)
                output = class_model(output)
                optimizer.zero_grad()
                loss1 = loss_function(output, labels.long())
                loss1.backward()
                optimizer.step()
                losses.append(loss1.item())
                n_correct_1 += torch.sum(output.argmax(1) == labels).item()
            curr_loss = np.mean(np.array(losses))
            print('Loss after epoch '+str(epoch+1)+'/'+str(n_epochs)+' is:  '+str(curr_loss))
            accuracy_1 = 100.0 * n_correct_1 / len(train_loader.dataset)
            print('Accuracy after epoch '+str(epoch+1)+'/'+str(n_epochs)+' is:  '+str(accuracy_1))
            train_losses.append(curr_loss)
            writer.add_scalar('Training loss, fold {} '.format(fold), curr_loss,epoch)
            writer.add_scalar('Training accuracy, fold {}'.format(fold), accuracy_1, epoch)

            model.eval()
            class_model.eval()
            #Test
            total = 0
            losses = []
            n_correct_1 = 0
            with torch.no_grad():
                for iteration, (images, labels) in enumerate(val_loader):
                    images = images.to(device)
                    labels = labels.to(device)
                    output = model(images)
                    output = class_model(output)
                    loss = loss_function(output, labels.long())
                    losses.append(loss.item())
                    _, predicted = torch.max(output.data, 1)
                    total += labels.size(0)
                    n_correct_1 += (predicted == labels).sum().item()
            curr_loss = np.mean(np.array(losses))
            val_losses.append(curr_loss)
            if val_losses[-1]<best_loss and save:
                best_loss=val_losses[-1]
                print("Models weights are saved on (autosave)", device)
                torch.save(model.state_dict(), saving_path+'_autosave_'+'.pth')
                torch.save(class_model.state_dict(), saving_path+'_autosave_CL'+'.pth')

            accuracy_1 = 100.0 * n_correct_1 / total
            print('VAL Loss after epoch {}/{} is: {}'.format(epoch+1,n_epochs,curr_loss))
            val_accuracies.append(accuracy_1)
            writer.add_scalar('Test accuracy, fold {}'.format(fold), accuracy_1,epoch)
            writer.add_scalar('Test loss, fold {}'.format(fold), curr_loss,epoch)
            print('Accuracy at test:  {}, fold {}'.format(str(accuracy_1),fold) + '\n')
            scheduler.step()

        if save:
            print("Model weights are saved from ", device)
            torch.save(model.state_dict(), saving_path+'.pth')
        if save_sec:
            torch.save(class_model.state_dict(), saving_path+'_CL'+'.pth')

        acc_fold.append(accuracy_1)
        writer.add_scalar('Fold_acc', accuracy_1, fold)

    accuracy_fold = np.mean(np.array(acc_fold))
    print('Final accuracy after 10 folds is {}\n'.format(accuracy_fold))
    writer.add_scalar('Fold_acc', accuracy_fold,11)
    writer.close()

    #####
    results = np.zeros((n_epochs,3))
    results[:, 0] = train_losses
    results[:, 1] = val_losses
    results[:, 2] = val_accuracies
    np.savetxt(str(model_save_name)+'_lr='+str(lr)+'_nepochs='+str(n_epochs)+'_STL10.txt', results, fmt='%s')
    return 'Training successfully completed!'
