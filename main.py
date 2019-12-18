from SSTrain_STL10 import SSTrain_STL10
from time import strftime
from datetime import datetime
from ClassTrain_STL10 import ClassTrain_STL10
from SSTrainRotation import SSTrainRotation_STL10
import os
import sys
import zipfile

'''
Hyperparameters
'''
to_home = '../../../'
type = 'rotSS' # 'mySS' or 'rotSS' or 'class'
###### if type is 'class'
cluster=True
######
dataset = 'stl10'
learning_rate = 0.0003
lr_class = 0.0003
experiment_name = 'Ex_38r'
folder = '../experiments_2/'
n_epochs = 100
n_epochs_class = 200
save = True
load = False
load_name = 'data/cvg/dusan/experiments/experiments_2/models/Ex_37m_model_mySS_17_12_2019_h10_m31'#Ex_12_model_rotSS_13_11_2019_h11_m03' # from home
load_second_part = False
save_second_part = True
batch_size = 128
batch_class = 128
test_class = True
frz = True


'''
Until here
'''

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))



if __name__ == '__main__':
    
    if sys.argv[1] == '--gpu1':
        gpu_n = 'cuda:0'
    else:
        gpu_n = 'cuda:1'
    
    if cluster:
        folder = '../../../data/cvg/dusan/experiments'+folder.strip('.')

    if type=='class' and test_class:
        print('Impossible to do classification after classification!!!')
        test_class=False

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_h%H_m%M")
    zipf = zipfile.ZipFile(folder+'Curr_ver_'+dt_string+'.zip', 'w', zipfile.ZIP_DEFLATED)
    zipdir('../Master_new', zipf)
    zipf.close()
    try:
        os.mkdir(to_home+'data/cvg/dusan/runs/'+experiment_name)
    except:
        print('There is already folder named {} in data/cvg/dusan/runs/'.format(experiment_name))
    model_load_name = to_home+load_name
    model_save_name = folder+'models/'+experiment_name+'_model_'+type+'_'
    class_exp_name='class_'+experiment_name
    writerSS = to_home+'data/cvg/dusan/runs/'+'/'+experiment_name+'/'+experiment_name+'_'+type+'_'+dt_string
    writerCL = to_home+'data/cvg/dusan/runs/'+'/'+experiment_name+'/'+experiment_name+'_'+'class'+'_'+dt_string

    if dataset == 'stl10':
        dataset_folder = (to_home+'data/cvg/dusan/stl10/' if cluster else '../../../stl10/')
    elif dataset == 'imgNet':
        dataset_folder = (to_home+'data/cvg/imagenet/ILSVRC2012/' if cluster else '../../../ILSVRC2012')
    else:
        print('Invalid dataset')
    print('Path to dataset folder: '+dataset_folder)

    if type == 'mySS':
        dt_string = SSTrain_STL10(folder, dataset_folder, model_save_name, model_load_name, save, load, learning_rate,
                                        batch_size, n_epochs, load_second_part, save_second_part, writerSS, gpu_n)
    elif type == 'class':
        dt_string = ''
        ClassTrain_STL10(frz, folder, dataset_folder, model_save_name, model_load_name, save, load,
                                    learning_rate, batch_size, n_epochs, load_second_part, save_second_part, writerSS, gpu_n)
    elif type == 'rotSS':
        dt_string = SSTrainRotation_STL10(folder, dataset_folder, model_save_name, model_load_name, save, load,
                                    learning_rate, batch_size, n_epochs, load_second_part, save_second_part, writerSS, gpu_n)
    print()
    print('Self-supervised training successfully completed!')
    print()
    if test_class:
        load = True
        now = datetime.now()
        model_load_name = model_save_name+dt_string + ''
        model_save_name = model_save_name+dt_string.strip('.pth') + '_classTest.pth'
        print(ClassTrain_STL10(frz, folder, dataset_folder, model_save_name, model_load_name, save, load,
                               lr_class, batch_class, n_epochs_class, load_second_part, save_second_part, writerCL))

