# %% [markdown]
# ---
# * Code for the paper published in the XIX Workshop on Computer Vision 2024 - WVC'2024.
# ---

# %% [markdown]
# ## Importing the libraries
# ---

# %%
import os
import random
import time
import platform
import sys
import argparse
import shutil
import datetime
import glob
import pickle   

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import PIL
import sklearn
from sklearn import metrics, preprocessing, model_selection
from PIL import Image

import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.optim import lr_scheduler 
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms, models, datasets, utils

import timm

# Explainable AI
#import pytorch_grad_cam
### from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
#from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
#from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
#from pytorch_grad_cam.utils.image import show_cam_on_image

# Local imports
from early_stopping import EarlyStopping
from models import create_model

# %% [markdown]
# ## Verificando de está rodando no Colab
# ---

# %%
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

# DEBUG
print(f'Running in Colab: {IN_COLAB}')

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')

# %% [markdown]
# ## Configuração da GPU
# ---

# %%
print('Configurando GPU...')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {DEVICE}')

# %% [markdown]
# ## Argument parsing
# ---

# %%
parser = argparse.ArgumentParser()

# Dataset name. ['BreakHis', 'USK-Coffee', 'deep-coffee', 'smallopticalsorter']
parser.add_argument('--ds', help='Dataset name.', type=str, default='CropPestAndDiseaseDetection')
# Model architecture [alexnet, vgg, resnet, densenet, squeezenet, (inception), ...]
parser.add_argument('--arch', help='CNN architecture', type=str, default='vit', )
parser.add_argument('--optim', help="Hyperparameter optmization: ['none', 'grid', 'random'].", type=str, default='none', )
parser.add_argument('--sm', help='Save the model?', default=True, action='store_true')

parser.add_argument('--seed', help='Seed for random number generator.', type=int, default=42)
parser.add_argument('--num_workers', help='Number of available cores.', type=int, default=2)
parser.add_argument('--debug', help="Is running in debug mode?", required=False, default=False, action='store_true')

# Hyperparameters 
# ---------------
parser.add_argument('--bs', help='Bach size.', type=int, default=64)
parser.add_argument('--lr', help='Learning rate.', type=float, default=0.0001)
parser.add_argument('--mm', help='Momentum.', type=float, default=0.9)
parser.add_argument('--ss', help='Step size.', type=int, default=5)
### parser.add_argument('--wd', help='Weight decay.', type=float, default=0.1)
parser.add_argument('--ep', help='Number of epochs', type=int, default=400) # 400

parser.add_argument('--optimizer', help="Optimizer. ['SGD', 'Adam'].", type=str, default='Adam')
parser.add_argument('--scheduler', help="Scheduler. ['steplr', 'cossine', 'plateau'].", type=str, default='plateau')

# Fine-tunning
parser.add_argument('--ft', help='Treinamento com fine-tuning.', default=True, action='store_true')
# Data augmentation stretegy. Ignorado quando otimização de hiperparametros
parser.add_argument('--da', help='Data augmentation stretegy. 0 = no data augmentation.',  type=int, default=0)
# Usa BCELoss em problemas com duas classes. Se False, usa CrossEntropyLoss para qualquer número de classes
parser.add_argument('--bce', help='Usa Binary Cross Entropy em problemas com duas classes.', default=True, action='store_true')
# Explainable AI
parser.add_argument('--xai', help='Perform eXplainable AI analysis.', default=False, action='store_true')

# Early stopping
parser.add_argument('--es', help='Use early stopping.', default=True, action='store_true')
parser.add_argument('--patience', help='Patience for early stopping.', type=int, default=21) # Use 21, if plateau
parser.add_argument('--delta', help='Delta for early stopping', type=float, default=0.0001)

### parser.add_argument('--wandb', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Use wandb.')

parser.add_argument('--ec', help='Experiment counter. Used for hp optimization.', type=int, default=0)

# Apenas para o BreakHis
parser.add_argument('--fold', help='Fold. [1, 2, 3, 4, 5]', type=int, default=1)
parser.add_argument('--magnification', help="Magnification. ['40X', '100X', '200X', '400X', '']", type=str, default='')

# ***** IMPORTANTE!!! *****
# Comentar esta linha após gerar o arquivo .py!
# *************************
### sys.argv = ['-f']

# Processa os argumentos informados na linha de comando
args = parser.parse_args()

# ***** IMPORTANTE!!! *****
# Set DEBUG mode:
# *************************
args.debug = False

if args.debug:
    args.ep = 1

# Performing Training...
TRAIN = True

# %%
if str(DEVICE) != 'cuda':
    # Caso não tenha uma GPU compatível disponível, executar apenas para prototipação.
    ### args.ep = 2

    print('CUDA not availavble. Finishing the program...')
    print('\nDone!\n\n')
    sys.exit()

if args.optim != 'none':
    args.sm = False
    ### args.da = 0
    # if hp optimization, always ignore XAI.
    args.xai = False

# %%
args_str = ''
for arg in vars(args):
    args_str += f'\n{arg}: {getattr(args, arg)}'
    print(f'{arg}: {getattr(args, arg)}')

# %%
def get_versions():
    str = ''
    str += f'\nNumPy: {np.__version__}'
    str += f'\nMatplotlib: {matplotlib.__version__}'
    str += f'\nPandas: {pd.__version__}'
    str += f'\nPIL: {PIL.__version__}'
    str += f'\nScikit-learn: {sklearn.__version__}'
    str += f'\nPyTorch: {torch.__version__}'
    str += f'\nTorchvision: {torchvision.__version__}'

    return str

# %% [markdown]
# ## Customized Dataset
# ---

# %%
class CropPestAndDeseaseDetectionDataset(Dataset):

    def __init__(self, path_list, label_list, transforms=None):
        self.path_list = path_list
        self.label_list = label_list
        self.transforms = transforms

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path = self.path_list[idx]
        ### image = io.imread(self.path_list[idx])
        image = Image.open(self.path_list[idx]) 

        # O dataset "Agricultural Pests Image Dataset" possui algumas imagens em níveis de cinza e algumas com canal de transparência.
        # É necessário trata-las, convertendo para imagens RGB (3 canais).
        if np.array(image).ndim != 3:
            # Trata imagens em níveis de cinza (com apenas 1 canal)
            ### print(path)
            ### print(np.array(image).ndim, np.array(image).shape)
            image = image.convert(mode='RGB')
            ### print(np.array(image).ndim, np.array(image).shape)
        elif np.array(image).shape[2] > 3:
            # Trata imagens com canal de transparência (com 4 canais)
            ### print(path)
            ### print(np.array(image).ndim, np.array(image).shape)
            image = image.convert(mode='RGB')
            ### print(np.array(image).ndim, np.array(image).shape)

        label = self.label_list[idx]

        if self.transforms:
            image = self.transforms(image)

        return (image, label, path)   

# %% [markdown]
# ## Configurating datasets
# ---

# %%
DS_PATH_MAIN = '/home/pedrocosta/Dev/Datasets/'

if args.ds == 'AgriculturalPestsDataset':
    DS_PATH = os.path.join(DS_PATH_MAIN, args.ds, 'images')

if args.ds == 'CropPestAndDiseaseDetection':
    DS_PATH = os.path.join(DS_PATH_MAIN, args.ds, args.ds)

# DEBUG
print(f'Dataset: {args.ds}')
print(f'Dataset Path: {DS_PATH}')

# %% [markdown]
# ## Gravação dos experimentos
# ---

# %%
# Pasta principal para armazenar os experimentos
EXP_PATH_MAIN = f'exp_{args.ds}'
if args.optim != 'none':
    EXP_PATH_MAIN = f'exp_hp_{args.ds}'

# Cria uma pasta para armazenar os experimentos, caso ainda não exista
if not os.path.isdir(EXP_PATH_MAIN):
    os.mkdir(EXP_PATH_MAIN)

mm_str = f'-mm_{args.mm}' if args.optimizer == 'SGD' else ''

ss_str = f'-ss_{args.ss}' if args.scheduler == 'steplr' else ''

# String contendo os valores dos hiperparametros deste experimento.
hp_str = f'-bs_{args.bs}-lr_{args.lr}-op_{args.optimizer}{mm_str}-sh_{args.scheduler}{ss_str}-epochs_{args.ep}'

# Ajusta para o nome da pasta
hp_optim = '' if args.optim == 'none' else f'-{args.optim}'

str_aux1 = '' if args.ds != 'BreakHis' else f'-mag_{args.magnification}-fold_{str(args.fold)}'

# Pasta que ira armazenar os resultados deste treinamento
EXP_PATH = os.path.join(EXP_PATH_MAIN, f'({args.ds})-{args.arch}{hp_optim}-da_{args.da}{hp_str}{str_aux1}')
print(f'Exp path: {EXP_PATH}')

# Check if EXP_PATH exists. If not, create it.
### if not glob.glob(EXP_PATH):
if not os.path.exists(EXP_PATH):
    os.mkdir(EXP_PATH)

else:
    # If the folder already exists, it is possible the experiment should (or shouldn't) be complete.
    # Nós verificamos, observando se o arquivo 'done.txt' está na pasta.
    # O arquivo 'done.txt' só é criado quando o experimento terminou por completo.
    ### if os.path.exists(os.path.join(EXP_PATH, 'done.txt')):
    if os.path.exists(os.path.join(EXP_PATH, 'done.txt')):
        # # The folder exists and the experiment is done.
        # print('Experiment already done. Finishing the program...')
        # print('\nDone!\n\n')
        # sys.exit()

        # The folder exists and the experiment is done.
        print('Model already trained! Performing prediction...')

        TRAIN = False

# %%
if TRAIN:
    with open(os.path.join(EXP_PATH, 'general_report.txt'), 'w') as model_file:
        model_file.write('\nArguments:')
        ### model_file.write(str(args.__str__()))
        model_file.write(args_str)
        model_file.write('\n\nPackage versions:')
        model_file.write(str(get_versions()))

# %%
EXP_PATH_PRED = os.path.join(EXP_PATH, f'prediction')

if not os.path.exists(EXP_PATH_PRED):
    os.mkdir(EXP_PATH_PRED)

if os.path.exists(os.path.join(EXP_PATH_PRED, 'done.txt')):
    # The folder exists and the experiment is done.
    print('Prediction is already done! Exiting...')

    sys.exit()

# %% [markdown]
# ## Reprodutibility configurations
# ---

# %%
random.seed(args.seed)
np.random.seed(args.seed)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

os.environ["PYTHONHASHSEED"] = str(args.seed)

# %% [markdown]
# ## Preparando o conjunto de dados
# ---

# %% [markdown]
# ### Definindo transformações para os dados (aumento de dados)

# %%
# Média e desvio padrão do ImageNet.
DS_MEAN = [0.485, 0.456, 0.406]
DS_STD =  [0.229, 0.224, 0.225]

# Data transforms 
# ---------------
if args.da == 0:  # Resize(224)
    # Treinamento
    data_transforms_train = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD)
    ])

    # Validacao
    data_transforms_val = transforms.Compose([
            transforms.Resize(size=(224, 224)), 
            transforms.ToTensor(), 
            transforms.Normalize(DS_MEAN, DS_STD)
    ])

    # Test
    data_transform_test = transforms.Compose([
            transforms.Resize(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize(DS_MEAN, DS_STD)
    ])

elif args.da == 1: # Resise + CentreCrop (Train = val = test)
    # Treinamento
    data_transforms_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD)
    ])

    # Define transformations for validation and test sets
    data_transforms_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

    data_transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

elif args.da == 2: # RandomResizedCrop (224)
    # Treinamento
    data_transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD)
    ])

    # Define transformations for validation and test sets
    data_transforms_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

    data_transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])
    
elif args.da == 3: # Data augmentation baseda
    # Training
    data_transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.25)),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

    # Validation
    data_transforms_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

    # Test
    data_transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

elif args.da == 4: # Data augmentation base, No HUE.
    # Training
    data_transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        ### transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

    # Validation
    data_transforms_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

    # Test
    data_transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

elif args.da == 5: # Data augmentation base. No HUE. No RandomErasing.
    # Training
    data_transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        ### transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        ### transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

    # Validation
    data_transforms_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

    # Test
    data_transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN, DS_STD),
    ])

# %% [markdown]
# ## Datasets and dataloaders
# ---

# %%
# Treino
# image_dataset_train = datasets.ImageFolder(os.path.join(DS_PATH, 'train'), data_transforms_train)

# Nomes e número de classes
class_names = os.listdir(DS_PATH)
num_classes = len(class_names)


X_fullDataset_ = []
y_fullDataset_ = []

for class_ in class_names:
    #lista ordenada dos arquivos (imagens) em cada pasta
    path_list_ = os.listdir(os.path.join(DS_PATH, class_))
    path_list_.sort()

    for path_image in path_list_:
        file_path = os.path.join(DS_PATH, class_, path_image)
        X_fullDataset_.append(file_path)
        y_fullDataset_.append(class_)
        
path_list = []


# Divisão do conjunto de treino para validação
# VAL_SIZE  = 0.2
# TRAIN_SIZE = 1. - VAL_SIZE

le = preprocessing.LabelEncoder()
le.fit(class_names)
y_fullDataset_idx = le.transform(y_fullDataset_)

# %%
print(f'DEBUG - Number of Classes: {num_classes} - Class names: ')
class_names_ = le.classes_
print(class_names_)
print(class_names)

class_names = class_names_

# %%
X_train, X_temp, y_train, y_temp= model_selection.train_test_split(X_fullDataset_, y_fullDataset_idx, test_size=0.4, stratify=y_fullDataset_idx, random_state=42)
X_val, X_test, y_val, y_test = model_selection.train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

train_dataset = CropPestAndDeseaseDetectionDataset(X_train, y_train, transforms=data_transforms_train)
val_dataset = CropPestAndDeseaseDetectionDataset(X_val, y_val, transforms=data_transforms_val)
test_dataset = CropPestAndDeseaseDetectionDataset(X_test, y_test, transforms=data_transform_test)

# Tamanho dos conjuntos de treino e de validação (número de imagens).
train_size = len(train_dataset)
val_size = len(val_dataset)
test_size = len(test_dataset)

print(train_size)
print(val_size)
print(test_size)

# %%
# Construindo os Dataloaders
dataloader_train = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=args.bs, 
                                               num_workers=args.num_workers,
                                               shuffle=True,
                                              )
dataloader_val = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.bs, 
                                             num_workers=args.num_workers,
                                             shuffle=True,
                                            )

# %% [markdown]
# ## Visualizando um lote de imagens
# ---

# %%
def save_batch(images_batch):
    """ Save one batch of images in a grid.

    References
    ----------
    * https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
    * https://pub.towardsai.net/image-classification-using-deep-learning-pytorch-a-case-study-with-flower-image-data-80a18554df63
    """

    # Unnormalize all channels (ImageNet weights)
    for t, m, s in zip(images_batch, DS_MEAN, DS_STD):
        t.mul_(s).add_(m)
        # The normalize code -> t.sub_(m).div_(s)

    images_batch_np = images_batch.numpy()
    fig_obj = plt.imshow(np.transpose(images_batch_np, (1, 2, 0)))
    
    # Grava a figura em disco
    plt.savefig(os.path.join(EXP_PATH, 'sample_batch.png')) 
    plt.savefig(os.path.join(EXP_PATH, 'sample_batch.pdf')) 


if TRAIN:
    items = iter(dataloader_train)
    image, label, paths = next(items)

    save_batch(utils.make_grid(image))

# %% [markdown]
# ## Inicializando o modelo
# ---

# %%
if TRAIN:
    print('\n>> Inicializando o modelo...')

    model, input_size, _ = create_model(args.arch, args.ft, num_classes, args.bce)

    # Envia o modelo para a GPU
    model = model.cuda() # Cuda
        
    # Imprime o modelo
    print(str(model))

    # Grava a modelo da rede em um arquivo .txt
    with open(os.path.join(EXP_PATH, 'model.txt'), 'w') as model_file:
        model_file.write(str(model))


print('chegou ate aqui')
# %% [markdown]
# ## Loss function and optimizer
# ---

# %%

print('chegou ate aqui 2')
if TRAIN:
    # Função de perda
    if num_classes > 2 or args.bce == False:
        # Classificação com mais de duas classes.
        criterion = nn.CrossEntropyLoss()
        print('criterion = nn.CrossEntropyLoss()')
        
    else:
        # Binary classification:
        # ----------------------
        # Do not use BCELoss. Instead use BCEWithLogitsLoss.
        # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        # https://discuss.pytorch.org/t/bceloss-vs-bcewithlogitsloss/33586/21
        ### criterion = nn.BCELoss()

        criterion = nn.BCEWithLogitsLoss()
        print('criterion = nn.BCEWithLogitsLoss()')

    # Otimizador
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mm)

    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

# %% [markdown]
# ### Scheduler

# %%
if TRAIN:
    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, )
        print(scheduler)

    elif args.scheduler == 'cossine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                            T_max=len(dataloader_train), 
                                                            eta_min=0,
                                                            last_epoch=-1)
        print(scheduler)

    elif args.scheduler ==  'steplr':

        # Step size of the learning rate
        if args.ss != 0:
            # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#StepLR
            scheduler = lr_scheduler.StepLR(optimizer, step_size=args.ss)
            print(scheduler)

    print(criterion)
    print(optimizer)

# %% [markdown]
# ## Training the model
# ---

# %%
if TRAIN:
    print('\n>> Training the model...')

    # Tempo total do treinamento (treinamento e validação)
    time_total_start = time.time()

    # Lista das perdas (loss) e acurácias (accuracy) de trino para cada época.
    train_loss_list = []
    train_acc_list = []

    # Lista das perdas (loss) e acurácias (accuracy) de validação para cada época.
    val_loss_list = []
    val_acc_list = []

    lr_list = []

    if args.es:
        early_stopping = EarlyStopping(patience=args.patience, delta=args.delta)

    for epoch in range(args.ep):
        # =========================================================================
        # TRAINING
        # =========================================================================
        # Inicia contagem de tempo deste época
        time_epoch_start = time.time()

        # Perdas (loss) nesta época
        train_loss_epoch = 0.
        # Número de amostras classificadas corretamente nesta época
        train_num_hits_epoch = 0  

        # Habilita o modelo para o modo de treino 
        model.train() 

        # Iterate along the batches of the TRAINING SET
        # ---------------------------------------------
        for i, (inputs, labels, paths) in enumerate(dataloader_train):

            inputs = inputs.to(DEVICE) 
            labels = labels.to(DEVICE) 

            # Zera os parametros do gradiente
            optimizer.zero_grad() 

            # FORWARD
            # ------>
            # Habilita o cálculo do gradiente
            torch.set_grad_enabled(True) 

            # Gera a saida a partir da entrada
            outputs = model(inputs) 

            if num_classes == 2 and args.bce:
                # Calculate probabilities
                # https://discuss.pytorch.org/t/bceloss-vs-bcewithlogitsloss/33586/27
                outputs_prob = torch.sigmoid(outputs) 
                preds = (outputs_prob > 0.5).float().squeeze()

                # Calcula a perda (loss)
                loss = criterion(outputs.squeeze(), labels.float())

            else:
                # 'outputs' estão em porcentagens. Tomar os maximos como a respostas.
                # Ex: batch=3 com 2 classes, entao preds = [1, 0, 1]
                preds = torch.argmax(outputs, dim=1).float() 

                # Calcula a perda (loss)
                loss = criterion(outputs, labels)

            # BACKWARD
            # <-------
            loss.backward() 

            # Atualiza o gradiente 
            optimizer.step()

            # Atualiza o loss da época
            train_loss_epoch += float(loss.item()) * inputs.size(0) 

            # Atualiza o número de amostras classificadas corretamente nessa época.
            train_num_hits_epoch += torch.sum(preds == labels.data) 

        train_loss_epoch /= len(train_dataset)
        train_acc_epoch = float(train_num_hits_epoch.double() / len(train_dataset))

        # Store loss and accuracy in lists
        train_loss_list.append(train_loss_epoch)
        train_acc_list.append(train_acc_epoch)

        # =========================================================================
        # VALIDATION
        # =========================================================================
        model.eval() 

        # Pego o numero de perda e o numero de acertos
        val_loss_epoch = 0. # Atual perda
        val_num_hits_epoch = 0 # Numero de itens corretos
        
        # Iterate along the batches of the VALIDATION SET
        # -----------------------------------------------
        for i, (inputs, labels, paths) in enumerate(dataloader_val):

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # Zero o gradiente antes do calculo do loss
            optimizer.zero_grad() 

            # Desabilita o gradiente, pois os parametros nao podem mudar durante etapa de validacao
            torch.set_grad_enabled(False) 

            # Gero um tensor cujas linhas representam o tamanho do "batch" do input
            outputs = model(inputs) 

            if num_classes == 2 and args.bce:
                # Calculate probabilities
                # https://discuss.pytorch.org/t/bceloss-vs-bcewithlogitsloss/33586/27
                outputs_prob = torch.sigmoid(outputs) 
                preds = ((outputs_prob > 0.5).float()).squeeze()

                # Calcula a perda (loss)
                loss = criterion(outputs.squeeze(), labels.float())

            else:
                # Retorna a maior predicao.
                preds = torch.argmax(outputs, dim=1).float()

                # Calcula a perda (loss)
                loss = criterion(outputs, labels) 

            # Loss acumulado na época
            val_loss_epoch += float(loss.item()) * inputs.size(0)

            # Acertos acumulados na época
            val_num_hits_epoch += torch.sum(preds == labels.data)

        # Ajusta o learning rate
        if args.scheduler == 'steplr' and args.ss != 0:
            scheduler.step() 
        
        elif args.scheduler == 'cossine' and epoch >= 10:
            scheduler.step()

        elif args.scheduler == 'plateau':
            scheduler.step(val_loss_epoch)

        lr_epoch = optimizer.param_groups[0]['lr']
        lr_list.append(lr_epoch)
            
        # Calculo a perda e acuracia de todo o conjunto de validacao
        val_loss_epoch /= len(val_dataset)
        val_acc_epoch = float(val_num_hits_epoch.double() / len(val_dataset))

        # Inserindo a perda e acuracia para os arrays 
        val_loss_list.append(val_loss_epoch)
        val_acc_list.append(val_acc_epoch)

        if args.es:
            early_stopping(val_loss_epoch, model, epoch)
            
            if early_stopping.early_stop:
                print(f'Early stopping in epoch {early_stopping.best_epoch}!')
                break

        # Calculo de tempo total da época
        time_epoch = time.time() - time_epoch_start
        
        # PRINTING
        # --------
        print(f'Epoch {epoch}/{args.ep - 1} - TRAIN Loss: {train_loss_epoch:.4f} VAL. Loss: {val_loss_epoch:.4f} - TRAIN Acc: {train_acc_epoch:.4f} VAL. Acc: {val_acc_epoch:.4f} ({time_epoch:.4f} seconds)')
    
    # Calcula do tempo total do treinamento (treinamento e validação)
    time_total_train = time.time() - time_total_start

    # PRINTING
    print('Treinamento finalizado. ({0}m {1}s)'.format(int(time_total_train // 60), int(time_total_train % 60)))

# %%
# if args.es:
#     # load the last checkpoint with the best model
#     model.load_state_dict(torch.load('checkpoint.pt'))

# %%
# # Saving the model
# if args.sm:
#     model_file = os.path.join(EXP_PATH, 'model.pth')
#     torch.save(model, model_file)

# %%
if TRAIN:
    if args.es:
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load('checkpoint.pt'))

        file_obj = open(os.path.join(EXP_PATH, 'es.obj'), 'wb')
        pickle.dump(early_stopping, file_obj)
        file_obj.close()

    # Saving the model
    if args.sm:
        model_file = os.path.join(EXP_PATH, 'model.pth')
        torch.save(model, model_file)

else:
    model = torch.load(os.path.join(EXP_PATH, 'model.pth'))
    model.eval()

    if os.path.exists(os.path.join(EXP_PATH, 'es.obj')):
        file_obj = open(os.path.join(EXP_PATH, 'es.obj'), 'rb')
        early_stopping = pickle.load(file_obj)
        file_obj.close()
    else:
        early_stopping = None

    train_acc_list = [0]
    val_acc_list = [0]
    time_total_train = 0

    device = next(model.parameters()).device
    print(f"The model is on: {device}")

# %% [markdown]
# ## Analisando o treinamento
# ---

# %%
if TRAIN:
    # Contar os parâmetros treináveis
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with open(os.path.join(EXP_PATH, f'general_report.txt'), 'w') as model_file:
        model_file.write(f'\n\nTrainable parameters: {trainable_params}')

# %%
if TRAIN:
    # Lista com os indices das épocas. [0, 1, ... num_epochs - 1]
    epochs_list = []
    for i in range(len(train_loss_list)):
        epochs_list.append(i)

    # Plot - Loss 
    # -----------
    fig_obj = plt.figure()

    plt.title('Loss')
    plt.plot(epochs_list, train_loss_list, c='magenta', label='Train loss', fillstyle='none')
    plt.plot(epochs_list, val_loss_list, c='green', label='Val. loss', fillstyle='none')
    if args.es:
        plt.axvline(x=early_stopping.best_epoch, color='r', label='Early stopping')
        ### plt.text(early_stopping.best_epoch + 0.1, (-early_stopping.best_score) + .05, str(f'{-early_stopping.best_score:.4f}'), color = 'blue', fontweight = 'bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')

    # Grava a figura em disco
    plt.savefig(os.path.join(EXP_PATH, 'chart_loss.png'))
    plt.savefig(os.path.join(EXP_PATH, 'chart_loss.pdf')) 


    # Plot - Accuracy
    # ---------------
    fig_obj = plt.figure()

    plt.title('Accuracy')
    plt.plot(epochs_list, train_acc_list, c='magenta', label='Train accuracy', fillstyle='none')
    plt.plot(epochs_list, val_acc_list, c='green', label='Val. accuracy', fillstyle='none')
    if args.es:
        plt.axvline(x=early_stopping.best_epoch, color='r', label='Early stopping')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')

    # Grava a figura em disco
    plt.savefig(os.path.join(EXP_PATH, 'chart_acc.png')) 
    plt.savefig(os.path.join(EXP_PATH, 'chart_acc.pdf')) 


    # Plot LR
    # ---------------
    fig_obj = plt.figure()

    plt.title('LR')
    plt.plot(epochs_list, lr_list, c='magenta', label='LR', fillstyle='none')
    if args.es:
        plt.axvline(x=early_stopping.best_epoch, color='r', label='Early stopping')
    plt.xlabel('Epochs')
    plt.ylabel('LR')
    plt.legend(loc='best')

    # Grava a figura em disco
    plt.savefig(os.path.join(EXP_PATH, 'chart_lr.png')) 
    plt.savefig(os.path.join(EXP_PATH, 'chart_lr.pdf')) 


# %% [markdown]
# ### Saving trainning report

# %%
if TRAIN:
    # Arquivo CSV que irá armazenar todos os losses e acuracias.
    report_filename = os.path.join(EXP_PATH, 'training_report' + '.csv')

    # Cria o arquivo CSV
    report_file = open(report_filename, 'w')

    header = 'Epoch\tTrain Loss\tVal. Loss\tTrain Acc.\tVal. Acc.\n'
    report_file.write(header)

    # Putting values from each epoch inside of archive
    for i in range(0, len(train_loss_list)):
        text = str(i) + '\t' + str(train_loss_list[i]) + '\t' + str(val_loss_list[i]) + '\t' + str(train_acc_list[i]) + '\t' + str(val_acc_list[i]) + '\t' + str(lr_list[i]) 
        if args.es and i == early_stopping.best_epoch:
            text += '\t *\n'
        else:
            text += '\n'

        report_file.write(text)

    if args.es:
        report_file.write(f'Early stopping: \t {early_stopping.best_epoch}')

    # Closing
    report_file.close()

# %% [markdown]
# ## Avaliando o modelo
# ---

# %%
# DataLoaders
# -----------
dataloader_test = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=args.bs,  
                                              num_workers=args.num_workers,
                                              shuffle=False,
                                             )

dataloader_val  = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=args.bs,
                                              num_workers=args.num_workers,
                                              shuffle=False, 
                                             )

# %% [markdown]
# ### Conjunto de testes

# %%
# Habilita o modelo para avaliação
model.eval()

# Listas contendo as classes reais (true_test), as classes preditas pelo modelo ('pred_test') e
# os caminhos para cada imagem. Apenas para o conjunto de testes.
true_test_list = []
pred_test_list = []
path_test_list = []

prob_test_list = []

# Inicia a contagem do tempo apenas para teste
time_test_start = time.time()

# Itera sobre o dataloader_test
# -----------------------------
for i, (img_list, true_list, path_list) in enumerate(dataloader_test):

    # Cuda extension
    img_list = img_list.to(DEVICE)
    true_list = true_list.to(DEVICE)

    # Para que o gradiente nao se atualize!
    torch.set_grad_enabled(False)

    outputs = model(img_list)

    if num_classes == 2 and args.bce:
        # Calculate probabilities
        # https://discuss.pytorch.org/t/bceloss-vs-bcewithlogitsloss/33586/27
        outputs_prob = torch.sigmoid(outputs) 
        preds = (outputs_prob > 0.5).float().squeeze()

        prob_test_batch = np.asarray(outputs_prob.cpu())

        # Temos a probabilidade só da classe 1. 
        # Probability of class 0 is (1 - prob(c_1))
        prob_test_batch = np.c_[1. - prob_test_batch, prob_test_batch]

    else:
        ### _, preds = torch.max(output, dim=1)
        preds = torch.argmax(outputs, dim=1)

        # https://discuss.pytorch.org/t/obtain-probabilities-from-cross-entropy-loss/157259
        outputs_prob = nn.functional.softmax(outputs, dim=1)

        prob_test_batch = np.asarray(outputs_prob.cpu())

    # Lista de labels com a resposta (batch)
    true_test_batch = np.asarray(true_list.cpu())
    # Lista de labels com a predicao (batch)
    pred_test_batch = np.asarray(preds.cpu())

    ###  print(prob_test_batch)

    # Lista com os caminhos das imagens (batch)
    path_test_batch = list(path_list)

    # Consolidate das listas de predicao e resposta
    # for i in range(0, len(pred_test_batch)):
    #     true_test_list.append(true_test_batch[i])
    #     pred_test_list.append(pred_test_batch[i])
    #     path_test_list.append(path_test_batch[i])
    #     prob_test_list.append(prob_test_batch[i])

    for i in range(len(pred_test_batch)):
        true_test_list.append(true_test_batch[i])
        pred_test_list.append(pred_test_batch[i])
        path_test_list.append(path_test_batch[i])
        prob_test_list.append(prob_test_batch[i])

    # true_test_list.append(true_test_batch)
    # pred_test_list.append(pred_test_batch)
    # path_test_list.append(path_test_batch)
    # prob_test_list.append(prob_test_batch)


# print(type(prob_test_list))
# print(type(prob_test_list[0]))
# print(prob_test_list[0])


print(len(true_test_list))
print(len(pred_test_list))
print(len(path_test_list))
print(len(prob_test_list))
# print('----')
# print(len(prob_test_list[0]))
# print(len(true_test_list[1]))
# print(len(pred_test_list[2]))
# print(len(path_test_list[3]))
# print(len(prob_test_list[4]))

# Calculo o tempo final 
finish_test = time.time()

# Calculo do tempo total de teste
time_total_test = finish_test - time_test_start

# %% [markdown]
# ### Conjunto de validação

# %%
# resposta_val
true_val_list = []
pred_val_list = []
path_val_list = []

prob_val_list = []

# Itera sob o dataloader_val
# --------------------------
for i, (img_list, labelList, path_list) in enumerate(dataloader_val):

    if DEVICE.type == 'cuda':
        # Cuda extension
        img_list = img_list.to(DEVICE)
        labelList = labelList.to(DEVICE)

    # Nao atualizar o gradiente
    torch.set_grad_enabled(False) 

    # >>>>> FORWARD
    outputs = model(img_list)

    if num_classes == 2 and args.bce:
        # Calculate probabilities
        # https://discuss.pytorch.org/t/bceloss-vs-bcewithlogitsloss/33586/27
        outputs_prob = torch.sigmoid(outputs) 
        preds = (outputs_prob > 0.5).float().squeeze()

        prob_val_batch = np.asarray(outputs_prob.cpu())

        prob_val_batch = np.c_[1. - prob_val_batch, prob_val_batch]

    else:
        ### _, preds = torch.max(output, 1)
        preds = torch.argmax(outputs, dim=1)

        # https://discuss.pytorch.org/t/obtain-probabilities-from-cross-entropy-loss/157259
        outputs_prob = nn.functional.softmax(outputs, dim=1)

        prob_val_batch = np.asarray(outputs_prob.cpu())

    # Obtém as classes reais (True) e classes preditas (pred) deste lote.
    true_val_batch = np.asarray(labelList.cpu())
    pred_val_batch = np.asarray(preds.cpu())
        
    # Obtém os caminhos das imagens deste lote
    path_val_batch = list(path_list)

    # Itera sob cada item predito. (Esse FOR tem tamanho do batch_size)
    for i in range(0, len(pred_val_batch)):
        true_val_list.append(true_val_batch[i])
        pred_val_list.append(pred_val_batch[i])
        path_val_list.append(path_val_batch[i])

        prob_val_list.append(prob_val_batch[i])

# %% [markdown]
# ### Matriz de confusão e relatórios de classificação (Scikit-learn)

# %%
# TEST SET
# -------------------------------------------------------------------------

# Confusion matrix
conf_mat_test = metrics.confusion_matrix(true_test_list, pred_test_list)
# Classification report - Scikit-learn
class_rep_test = metrics.classification_report(true_test_list, pred_test_list, 
                                               target_names=class_names, digits=4)
# Accuracy
acc_test = metrics.accuracy_score(true_test_list, pred_test_list)

class_rep_path = os.path.join(EXP_PATH_PRED, 'classification_report_test.txt')
file_rep = open(class_rep_path, 'w')

file_rep.write('\n\nTEST SET:')
file_rep.write('\n---------------')
file_rep.write('\nConfusion matrix:\n')
file_rep.write(str(conf_mat_test))
file_rep.write('\n')
file_rep.write('\nClassification report:\n')
file_rep.write(class_rep_test)
file_rep.write('\n')
file_rep.write('\nAccuracy:\t' + str(acc_test))

file_rep.close()

# Ploting the confusion matrix
plt.figure(figsize=(10, 8))
metrics.ConfusionMatrixDisplay(conf_mat_test, display_labels=class_names).plot()
plt.xticks(rotation=45)
# Save figure in disk
plt.savefig(os.path.join(EXP_PATH_PRED, 'conf_mat_test.png'), bbox_inches='tight')
plt.savefig(os.path.join(EXP_PATH_PRED, 'conf_mat_test.pdf'), bbox_inches='tight')

print('\nTEST SET:')
print('\n---------------')
print('\nConfusion matrix:\n')
print(str(conf_mat_test))
print('\nClassification report:\n')
print(class_rep_test)
print('\nAccuracy:\t' + str(acc_test))

# %%
# VALIDATION SET
# -------------------------------------------------------------------------
# Confusion matrix
conf_mat_val = metrics.confusion_matrix(true_val_list, pred_val_list)
# Classification report - Scikit-learn
class_rep_val = metrics.classification_report(true_val_list, pred_val_list, 
                                              target_names=class_names, digits=4)
# Accuracy
acc_val = metrics.accuracy_score(true_val_list, pred_val_list)

class_rep_path = os.path.join(EXP_PATH_PRED, 'classification_report_val.txt')
file_rep = open(class_rep_path, 'w')

file_rep.write('VALIDATION SET:')
file_rep.write('\n---------------')
file_rep.write('\nConfusion matrix:\n')
file_rep.write(str(conf_mat_val))
file_rep.write('\n')
file_rep.write('\nClassification report:\n')
file_rep.write(class_rep_val)
file_rep.write('\n')
file_rep.write('\nAccuracy:\t' + str(acc_val))

file_rep.close()

# Ploting the confusion matrix
plt.figure(figsize=(10, 8))
metrics.ConfusionMatrixDisplay(conf_mat_val, display_labels=class_names).plot()
plt.xticks(rotation=45)
# Save figure in disk
plt.savefig(os.path.join(EXP_PATH_PRED, 'conf_mat_val.png'), bbox_inches='tight')
plt.savefig(os.path.join(EXP_PATH_PRED, 'conf_mat_val.pdf'), bbox_inches='tight')

print('VALIDATION SET:')
print('---------------')
print('\nConfusion matrix:\n')
print(str(conf_mat_val))
print('\nClassification report:\n')
print(class_rep_val)
print('\nAccuracy:\t' + str(acc_val))

# %% [markdown]
# ### Precision-Recal curve

# %%
# Concatenate results (if they are already NumPy arrays)
all_probs_test = np.array(prob_test_list)
all_probs_val = np.array(prob_val_list)

# %%
# Binarize the labels for multiclass precision-recall
n_classes = all_probs_test.shape[1]
binary_targets = preprocessing.label_binarize(true_test_list, classes=np.arange(n_classes))

# Define a list of 12 distinct colors (manually chosen)
custom_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", '#ff9896', '#c5b0d5', '#c49c94','#f7b6d2','#c7c7c7','#7f7f7f',
    '#bcbd22','#17becf','#9edae5','#f5b7b1',
]

# Plot Precision-Recall curve for each class
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    precision, recall, thresholds = metrics.precision_recall_curve(binary_targets[:, i], all_probs_test[:, i])
    auc_pr = metrics.auc(recall, precision)
    plt.plot(recall, precision, label=f"{class_names[i]} ({auc_pr:.2f})", color=custom_colors[i])

plt.xlabel("Recall", fontsize=18)
plt.ylabel("Precision", fontsize=18)
### plt.title("Precision-Recall Curve")
plt.legend(loc="best", fontsize=16)
plt.tick_params(axis='both', labelsize=16)
plt.grid()

# Save figure in disk
plt.savefig(os.path.join(EXP_PATH_PRED, 'prec_rec_test.png')) 
plt.savefig(os.path.join(EXP_PATH_PRED, 'prec_rec_test.pdf')) 

plt.show()

# %%
# Binarize the labels for multiclass precision-recall
n_classes = all_probs_val.shape[1]
binary_targets = preprocessing.label_binarize(true_val_list, classes=np.arange(n_classes))

# Plot Precision-Recall curve for each class
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    precision, recall, thresholds = metrics.precision_recall_curve(binary_targets[:, i], all_probs_val[:, i])
    auc_pr = metrics.auc(recall, precision)
    plt.plot(recall, precision, label=f"{class_names[i]} ({auc_pr:.2f})", color=custom_colors[i])

plt.xlabel("Recall", fontsize=18)
plt.ylabel("Precision", fontsize=18)
### plt.title("Precision-Recall Curve")
plt.legend(loc="best", fontsize=16)
plt.tick_params(axis='both', labelsize=16)
plt.grid()

# Save figure in disk
plt.savefig(os.path.join(EXP_PATH_PRED, 'prec_rec_val.png')) 
plt.savefig(os.path.join(EXP_PATH_PRED, 'prec_rec_val.pdf')) 

plt.show()

# %% [markdown]
# ### Classification report

# %%
# Usa o método __get_item__ da classe ... extendida da classe Dataset

# Conjunto de validação
file_details_path = os.path.join(EXP_PATH_PRED, 'classification_details_val.csv')
file_details = open(file_details_path, 'w')

file_details.write('VALIDATION SET')
file_details.write('\n#\tFile path\tTarget\tPrediction')

for class_name in class_names:
    file_details.write('\t' + str(class_name))

for i, (target, pred, probs) in enumerate(zip(true_val_list, pred_val_list, prob_val_list)):
    image_name = str(path_val_list[i])
    file_details.write('\n' + str(i) + '\t' + image_name + '\t' + str(target) + '\t' + str(pred))

    for prob in probs:
        file_details.write('\t' + str(prob))

file_details.close()

# %%
# Conjunto de testes
file_details_path = os.path.join(EXP_PATH_PRED, 'classification_details_test.csv')
file_details = open(file_details_path, 'w')

file_details.write('TEST SET')
file_details.write('\n#\tFile path\tTarget\tPrediction')

for class_name in class_names:
    file_details.write('\t' + str(class_name))

for i, (target, pred, probs) in enumerate(zip(true_test_list, pred_test_list, prob_test_list)):
    image_name = str(path_test_list[i])
    file_details.write('\n' + str(i) + '\t' + image_name + '\t' + str(target) + '\t' + str(pred))

    for prob in probs:
        file_details.write('\t' + str(prob))

file_details.close()

# %% [markdown]
# ### Hyperparameter optimization report

# %%
if args.optim != 'none':
    print('\n>> Relatório da otimização de hiperparametros...')
    # O nome de um arquivo CSV. Irá armazenar todos os losses e acuracias.
    hp_filename = os.path.join(EXP_PATH_MAIN, '(' + args.ds + ')-' + args.arch + '-' + args.optim + '.csv')
else:
    print('\n>> Relatório do conjunto de experimentos...')
    # O nome de um arquivo CSV. Irá armazenar todos os losses e acuracias.
    hp_filename = os.path.join(EXP_PATH_MAIN, '(' + args.ds + ')-' + args.optim + '.csv')


if args.ec == 0:
    # Cria o arquivo CSV
    hp_file = open(hp_filename, 'w')

    # Criar cabeçalho
    header = '#\tDS\tARCH\tHP\tFT\tBS\tLR\tMM\tSS\tEP\tES\tACC_VAL\tACC_TEST\tACC_TRAIN(*)\tACC_VAL(*)\tTIME\n'
    hp_file.write(header)

else:
    # Cria o arquivo CSV
    hp_file = open(hp_filename, 'a')

# if args.es:
#     info = f'{args.ec}\t{args.ds}\t{args.arch}\t{args.optim}\t{args.ft}\t{args.bs}\t{args.lr}\t{args.mm}\t{args.ss}\t{args.ep}\t{early_stopping.best_epoch}\t{acc_val}\t{acc_test}\t{train_acc_list[-1]}\t{val_acc_list[-1]}\t{str(datetime.timedelta(seconds=time_total_train))}\n'
# else:
#     info = f'{args.ec}\t{args.ds}\t{args.arch}\t{args.optim}\t{args.ft}\t{args.bs}\t{args.lr}\t{args.mm}\t{args.ss}\t{args.ep}\t{args.ep - 1}\t{acc_val}\t{acc_test}\t{train_acc_list[-1]}\t{val_acc_list[-1]}\t{str(datetime.timedelta(seconds=time_total_train))}\n'

# Momentum. Only if optmizer is SGD
mm_str = f'{args.mm}' if args.optimizer == 'SGD' else ''
# Step size. Only if scheduler is steplr
ss_str = f'{args.ss}' if args.scheduler == 'steplr' else ''
# Early stopping
if args.es:
    if early_stopping == None:
        es_ = 0
    else:
        es_ = early_stopping.best_epoch
else:
    es_ = args.ep - 1

if TRAIN == False:
    train_acc_ = ''
    val_acc_ = ''
else:
    # train_acc_ = train_acc_list[early_stopping.best_epoch]
    # val_acc_ = val_acc_list[early_stopping.best_epoch]
    train_acc_ = train_acc_list[es_]
    val_acc_ = val_acc_list[es_]

info = f'{args.ec}\t{args.ds}\t{args.arch}\t{args.optim}\t{args.ft}\t{args.bs}\t{args.lr}\t{mm_str}\t{ss_str}\t{args.ep}\t{es_}\t{acc_val}\t{acc_test}\t{train_acc_}\t{val_acc_}\t{str(datetime.timedelta(seconds=time_total_train))}\n'

hp_file.write(info)

hp_file.close()

# %% [markdown]
# ## Done!
# ---

# %%
# Se o arquivo "done.txt" estiver na pasta, o experimento foi finalizado com sucesso!
done_file = open(os.path.join(EXP_PATH, 'done.txt'), 'w')
done_file.close()

done_file = open(os.path.join(EXP_PATH_PRED, 'done.txt'), 'w')
done_file.close()

print('\nDone!\n\n')

# %% [markdown]
# ## References 
# ---
# 
# * https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# * Finetuning Torchvision Models
#     * https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/finetuning_torchvision_models_tutorial.ipynb 
# * https://github.com/Spandan-Madan/Pytorch_fine_tuning_Tutorial
# * https://huggingface.co/docs/transformers/training
# * torchvision models
#     * https://pytorch.org/vision/stable/models.html
# * TIMM Models
#     * https://paperswithcode.com/lib/timm


