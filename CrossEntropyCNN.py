#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 19:11:43 2024

@author: victor
"""

# 1. Load all the libraries:
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from datetime import datetime
from sklearn.utils import class_weight as cw 
import brevitas.nn as qnn
from brevitas.quant import Int8Bias
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from sklearn.metrics import f1_score
#from brevitas.export import FINNManager
#from ray import tune  
#from ray.tune import CLIReporter
#from ray.tune.schedulers import ASHAScheduler
 
# ----------------------------------------------------------------------------------------------------------------------
# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
dim = 64
# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 200

PATIENCE = 5
IMG_SIZE = 64
N_CLASSES = 14

EpocasAMostrar = 1
percent_test = 0.1 # 22 de 222 muestras
percent_val = 0.2 # 44 de 222 muestras

# PATH = "/home/miguel/finn/notebooks/CD_50MHz/QCNN_Laura_8b_3cnv.pth"
# Creating the image data and the labels from the images in the folder
def create_dataset(folder):
    img_data_array = []
    class_name = []

    for dir1 in os.listdir(folder):
        for data in os.listdir(os.path.join(folder, dir1)):
            for file in os.listdir(os.path.join(folder, dir1, data)):
                # Obtain the path:
                path = os.path.join(folder, dir1, data, file)
                # Load the images and sequences:
                if data == 'Img':
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    # Expand the dimensions for the network functionality:
                    img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_AREA)
                    img = np.array(img)
                    # Keras use the format float32:
                    img = img.astype('float32')
                    # Normalization:
                    img /= 255
                    img_data_array.append(img)
                    class_name.append(dir1)

    return np.array(img_data_array), class_name  # extract the image array and class name
# ----------------------------------------------------------------------------------------------------------------------

class CustomTensorDataset(Dataset):
  def __init__(self, features, labels, transform=None):
    features_tensor, label_tensor = torch.from_numpy(features), torch.from_numpy(labels)
    self.features_tensor = features_tensor
    #self.label_tensor = label_tensor.type(torch.FloatTensor)
    self.label_tensor = label_tensor.type(torch.LongTensor)

  def __getitem__(self, index):
    features = self.features_tensor[index]
    labels = self.label_tensor[index]

    return features, labels

  def __len__(self):
    return self.features_tensor.size(0)

# 2. Load and preprocess the data and definition of variables:

def train(train_loader, model, criterion, optimizer, device):
    
    model.train() # esta funcion no entrena al modelo si no que "avisa" al modelo de que va a ser entrenado, pues hay algunos layers como BatchNorm que actuan diferente si estan en entrenamiento
    running_loss = 0
    
    for X, y_true in train_loader:

        optimizer.zero_grad()
        
        X = X.to(device)
        y_true = y_true.to(device)
        #y_true = y_true.type(torch.float)
        # Forward pass
        y_hat = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss

def validate(valid_loader, model, criterion, device):
    model.eval()
    running_loss = 0

    for X, y_true in valid_loader:
    
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
        
    return model, epoch_loss

def get_accuracy(model, data_loader, device):
    correct_pred = 0 
    n = 0
    m = nn.Softmax(dim = 1)
    
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device)

            y_prob = model(X)
            y_prob = m(y_prob)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n

def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    # set objects for storing metrics
    train_losses = []
    valid_losses = []
    train_accs   = []
    valid_accs   = []
    pred_valid_loss = 0
    val_wait = 0

    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            
            train_acc = get_accuracy(model, train_loader, device=device)
            train_accs.append(train_acc)
            valid_acc = get_accuracy(model, valid_loader, device=device)
            valid_accs.append(valid_acc)

            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')
        if valid_loss > pred_valid_loss:
            val_wait += 1
        else:
            val_wait = 0
        if val_wait == PATIENCE:
            break
        pred_valid_loss = valid_loss
        

    return model, optimizer, train_losses, valid_losses, train_accs, valid_accs

# # 2.1. Creating the image data and the labels from the images in the folder:
# folder = r'/media/victor/HHDLinux/Victor/Bases_de_datos/Sets_datos_para_red/Alvaro_detector/Set/Train/'
# img_train, class_train = create_dataset(folder)
# folder = r'/media/victor/HHDLinux/Victor/Bases_de_datos/Sets_datos_para_red/Alvaro_detector/Set/Test/'
# img_test, class_test = create_dataset(folder)
# folder = r'/media/victor/HHDLinux/Victor/Bases_de_datos/Sets_datos_para_red/Alvaro_detector/Set/Val/'
# img_val, class_val = create_dataset(folder)

# classes = np.unique(class_train)
# target_dict = {k: v for v, k in enumerate(np.unique(class_train))}
# numClasses = len(target_dict)
# labels = np.arange(0, numClasses, dtype=int)

# # 2.2. Prepare input signals:
#     # TRAINING:
# x_train = np.array(img_train, np.float32)
# x_train = np.expand_dims(x_train, axis=1)
# train_dict = {k: v for v, k in enumerate(np.unique(class_train))}
# y_train = [train_dict[class_train[i]] for i in range(len(class_train))]
# y_train = np.array(y_train)
# class_weights = cw.compute_class_weight(class_weight='balanced',
#                                         classes=np.unique(y_train),
#                                         y=y_train[:]) # Obtain the weights
# class_weights = torch.tensor(class_weights, dtype = torch.float)
# train_dataset = CustomTensorDataset(features = x_train, labels = y_train)
# train_loader = DataLoader(dataset=train_dataset, 
#                           batch_size=BATCH_SIZE, 
#                           shuffle=True)
#     # TEST:
# x_test = np.array(img_test, np.float32)
# x_test = np.expand_dims(x_test, axis=1)
# test_dict = {k: v for v, k in enumerate(np.unique(class_test))}
# y_test = [test_dict[class_test[i]] for i in range(len(class_test))]
# y_test = np.array(y_test)
# test_dataset = CustomTensorDataset(features = x_test, labels = y_test)
# test_loader = DataLoader(dataset=test_dataset, 
#                           batch_size=BATCH_SIZE, 
#                           shuffle=False)
#     # VALIDATION:
# x_val  = np.array(img_val, np.float32)
# x_val  = np.expand_dims(x_val, axis=1)
# val_dict = {k: v for v, k in enumerate(np.unique(class_val))}
# y_val = [val_dict[class_val[i]] for i in range(len(class_val))]
# y_val = np.array(y_val)
# val_dataset = CustomTensorDataset(features = x_val, labels = y_val)
# val_loader = DataLoader(dataset=val_dataset, 
#                           batch_size=BATCH_SIZE, 
#                           shuffle=True)

# ----------------------------------------------------------------------------------------------------------------------
# 3. Creation and compilation of the model:
# 3.1. Define the variables:
im_shape = (dim, dim, 1)

class BobEsponja(nn.Module):
    def __init__(self, classes, b_witdh, in_width):
        super(BobEsponja, self).__init__()
        self.QuantInput = qnn.QuantIdentity(return_quant_tensor=True, bit_width=in_width)
        self.conv1 = qnn.QuantConv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), weight_bit_width = b_witdh, return_quant_tensor=True, bias = False, bias_quant=Int8Bias, padding = 1)
        self.relu1 = qnn.QuantReLU(bit_width=b_witdh, return_quant_tensor=True)
        self.maxpool1 = qnn.QuantMaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # self.dropout1 = nn.Dropout(p=0.6)
        self.conv2 = qnn.QuantConv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), weight_bit_width = b_witdh, return_quant_tensor=True, bias = True, bias_quant=Int8Bias, padding = 1)
        self.relu2 = qnn.QuantReLU(bit_width=b_witdh, return_quant_tensor=True)
        self.maxpool2 = qnn.QuantMaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # self.dropout2 = nn.Dropout(p=0.6)
        self.conv3 = qnn.QuantConv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), weight_bit_width = b_witdh, return_quant_tensor=True, bias = True, bias_quant=Int8Bias, padding = 1)
        self.relu3 = qnn.QuantReLU(bit_width=b_witdh, return_quant_tensor=True)
        self.maxpool3 = qnn.QuantMaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # self.dropout3 = nn.Dropout(p=0.6)

        self.fc1 = qnn.QuantLinear(in_features=4096, out_features=128, weight_bit_width=4, return_quant_tensor=True, bias = True, bias_quant=Int8Bias)
        self.relu4 = qnn.QuantReLU(bit_width=b_witdh, return_quant_tensor=True)
        self.fc2 = qnn.QuantLinear(in_features=128, out_features=classes, return_quant_tensor=True, weight_bit_width=b_witdh, bias = False)

    def forward(self, x):
        x = self.QuantInput(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = x.reshape(x.size(0),-1)
        x = self.fc1(x) 
        x = self.relu4(x)
        out = self.fc2(x)
        return out 

# model = BobEsponja(10,4,8)

# print(model.parameters())
# params = list(model.parameters())
# print(len(params))
# print(params[0].size())
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas = (0.9,0.999), eps = 1e-9)
# criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

# model, optimizer, train_loss, valid_loss, train_acc, valid_acc = training_loop(model, criterion, optimizer, train_loader, val_loader, N_EPOCHS, DEVICE)

# train_loss = np.array(train_loss) 
# valid_loss = np.array(valid_loss)
# train_acc = np.array(train_acc) 
# valid_acc = np.array(valid_acc)

# PATH = "/media/victor/HHDLinux/Victor/Bases_de_datos/Sets_datos_para_red/Alvaro_detector/Modelo_Generado/QCNN_Laura_8b_3cnv.pth"
# torch.save(model.state_dict(), PATH)
# print('Trained model written to',PATH)



def test(model, data_loader, device):
    m = nn.Softmax(dim = 1)
    # epsilon = 1e-5

    with torch.no_grad():
        model.eval()
        m = nn.Softmax(dim=1)
        true_y = np.array(0)
        predicted_labels = np.array(0)
        for X, y_true in data_loader:

            X = X.to(device)
            
            # #----------------------------------------------------------------------------------
            # #Normalizacion de batch
            # # Calcula la media y la varianza a lo largo de el batch
            # mean = torch.mean(X, dim=(0, 2, 3), keepdim=True)  # Media por canal
            # var = torch.var(X, dim=(0, 2, 3), keepdim=True)  # Varianza por canal
            # # Normalización de batch
            # X = (X - mean) / torch.sqrt(var + epsilon)
            # #----------------------------------------------------------------------------------
            
            y_true = y_true.to(device)

            y_pred = model(X)
            y_pred = m(y_pred)
            #_, predicted_labels = torch.max(y_prob, 1)
            predicted_labels = np.append(predicted_labels, torch.argmax(y_pred.cpu(), dim=1).numpy())
            true_y = np.append(true_y, y_true.cpu().numpy())
            
            X = None
            y_pred = None
            torch.cuda.empty_cache()
        
        true_y = np.delete(true_y, 0)
        predicted_labels = np.delete(predicted_labels, 0)
            
        return true_y, predicted_labels

# 2.1. Creating the image data and the labels from the images in the folder:
folder = r'C:\Users\rubni\OneDrive - Universidad Rey Juan Carlos\LF_UAH_Train\dataSET_14\\'
data, label = create_dataset(folder)

last_index = 0

classes = np.unique(label)
target_dict = {k: v for v, k in enumerate(np.unique(label))}
numClasses = len(target_dict)
labels = np.arange(0, numClasses, dtype=int)
# print(labels)

indices_total = np.arange(len(label)) # Array con todos los indices
archivos_por_carpeta = int(len(label)/numClasses)

true_y_acum = np.array(0)
predicted_labels_acum = np.array(0)


resultados_por_etapa = []

for pasos in range(int(1/percent_test)):
    
    print('///////////////////////////////////////////////////////////////////////////////////////////////////////')
    print(f'ETAPA {pasos}:')
    print(' ')
    
    clase_act = 0
    indices_test = np.array(0)
    indices_train = np.array(0)
    indices_val = np.array(0)
    
    while clase_act < numClasses:
        indices_test_temp = indices_total[(last_index+archivos_por_carpeta*clase_act):(last_index+(archivos_por_carpeta*clase_act)+(round(archivos_por_carpeta*percent_test)))]
        indices_interm = np.delete(indices_total[archivos_por_carpeta*clase_act:archivos_por_carpeta*(clase_act+1)], indices_test_temp-archivos_por_carpeta*clase_act)
        indices_val_temp = indices_interm[0:round(archivos_por_carpeta*percent_val)]
        indices_train_temp = indices_interm[round(archivos_por_carpeta*percent_val):len(indices_interm)]
        clase_act = clase_act + 1
        indices_test = np.append(indices_test, indices_test_temp)
        indices_train = np.append(indices_train, indices_train_temp)
        indices_val = np.append(indices_val, indices_val_temp)
                                 
    indices_test = np.delete(indices_test, 0)
    indices_train = np.delete(indices_train, 0)
    indices_val = np.delete(indices_val, 0)
    # 2.2. Prepare input signals:
    
        # TRAINING:
    label_train = np.array(0)
    for inter in range (len(indices_train)):
        label_train = np.append(label_train, label[indices_train[inter]])
    label_train = np.delete(label_train, 0)
    x_train = data[indices_train]
    x_train = np.expand_dims(x_train, axis=1)
    train_dict = {k: v for v, k in enumerate(np.unique(label_train))}
    y_train = [train_dict[label_train[i]] for i in range(len(label_train))]
    y_train = np.array(y_train)
    #print("y_train es: ", y_train)
    class_weights = cw.compute_class_weight(class_weight='balanced',
                                            classes=np.unique(y_train),
                                            y=y_train[:]) # Obtain the weights
    class_weights = torch.tensor(class_weights, dtype = torch.float)
    
    train_dataset = CustomTensorDataset(features = x_train, labels = y_train)
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=BATCH_SIZE, 
                              shuffle=True)
    
        # TEST:
    label_test = np.array(0)
    for inter in range (len(indices_test)):
        label_test = np.append(label_test, label[indices_test[inter]])
    label_test = np.delete(label_test, 0)
    x_test = data[indices_test]
    x_test = np.expand_dims(x_test, axis=1)
    test_dict = {k: v for v, k in enumerate(np.unique(label_test))}
    y_test = [test_dict[label_test[i]] for i in range(len(label_test))]
    y_test = np.array(y_test)
    test_dataset = CustomTensorDataset(features = x_test, labels = y_test)
    test_loader = DataLoader(dataset=test_dataset, 
                              batch_size=BATCH_SIZE, 
                              shuffle=False)
    print("len(x_test): ", len(x_test))
    
        # VALIDATION:
    label_val = np.array(0)
    for inter in range (len(indices_val)):
        label_val = np.append(label_val, label[indices_val[inter]])
    label_val = np.delete(label_val, 0)
    x_val  = data[indices_val]
    x_val = np.expand_dims(x_val, axis=1)
    val_dict = {k: v for v, k in enumerate(np.unique(label_val))}
    y_val = [val_dict[label_val[i]] for i in range(len(label_val))]
    y_val = np.array(y_val)
    val_dataset = CustomTensorDataset(features = x_val, labels = y_val)
    val_loader = DataLoader(dataset=val_dataset, 
                              batch_size=BATCH_SIZE, 
                              shuffle=True)
    
    y_val = None
    torch.cuda.empty_cache()
    
    #------------------------------------------------------------------------------------
    model = BobEsponja(N_CLASSES,4,8).to(DEVICE) # ENTRADA-1 es sin wavelet de aproximacion
    
    #model.load_state_dict(torch.load(PATH)) # Cargar un modelo preentrenado
    
    
    #print("model ocupa: ", sys.getsizeof(model), " bytes")
    print(model.parameters())
    params = list(model.parameters())
    print("len(params): ", len(params))
    print("params[0] size: ", params[0].size())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas = (0.9,0.999), eps = 1e-9)
    print("class_weights: ", class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean').to(DEVICE)
    #print("criterion ocupa: ", sys.getsizeof(criterion), " bytes")
    
    model, optimizer, train_loss, valid_loss, train_acc, valid_acc = training_loop(model, criterion, optimizer, train_loader, val_loader, N_EPOCHS, DEVICE, EpocasAMostrar)
    
    
    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)
    
    train_acc1 = np.array([item.cpu().item() for item in train_acc])
    valid_acc1 = np.array([item.cpu().item() for item in valid_acc])
    
    # Gráficas
    # plt.style.use("ggplot")
    # plt.figure(dpi=90)
    # eje_x = np.arange(0, len(train_loss))
    # plt.plot(train_loss, label="Training Loss")
    # plt.plot(valid_loss, label="Validation Loss")
    # plt.plot(train_acc1, label="Training Precision")
    # plt.plot(valid_acc1, label="Validation Precision")
    # plt.ylim(0, 2)
    # plt.title("Bob Esponja Results")
    # plt.xlabel('Epochs #')
    # plt.ylabel("Loss/Accuracy")
    # plt.legend()
    # plt.show()
    
    last_index = last_index + round(archivos_por_carpeta*percent_test)


    #------------------------------------------------------------------------------------------
    # Evaluación del modelo final
    print("[INFO]: Evaluating the model")
    
    # Efectuamos la predicción
    
    true_y, predicted_labels = test(model, test_loader, DEVICE)
    
    true_y_acum = np.append(true_y_acum, true_y)
    predicted_labels_acum = np.append(predicted_labels_acum, predicted_labels)
    
    # Guardar resultados de esta etapa
    resultados_por_etapa.append({
        'etapa': pasos,
        'true': true_y,
        'pred': predicted_labels
    })

    PATH = f'C:\\Users\\rubni\\OneDrive - Universidad Rey Juan Carlos\\LF_UAH_Train\\revista_etapa_{pasos}.pth'
    torch.save(model.state_dict(), PATH)
    print(f'Trained model for etapa {pasos} written to {PATH}')

# Sacamos la matriz de confusión y el report para el conjunto de datos test acumulados

true_y_acum = np.delete(true_y_acum, 0)
predicted_labels_acum = np.delete(predicted_labels_acum, 0)

# Calcular la matriz de confusión
cm = confusion_matrix(true_y_acum, predicted_labels_acum)
# Convertir a DataFrame para etiquetar filas y columnas
cm_df = pd.DataFrame(cm, index=classes, columns=classes)

print("\n [INFO]: Matriz de confusión (test)")
print(confusion_matrix(true_y_acum, predicted_labels_acum))
print("\n[INFO]: Report (test)")
print(classification_report(true_y_acum, predicted_labels_acum, target_names=classes, digits=4))

print(" [INFO]: Matriz de confusión con nombres de clases:")
print(cm_df)


f1_scores_por_etapa = []

for resultado in resultados_por_etapa:
    f1 = f1_score(resultado['true'], resultado['pred'], average='weighted')
    f1_scores_por_etapa.append({'etapa': resultado['etapa'], 'f1_score': f1})

# Mostrar todos los F1-Scores
print("\n[INFO]: F1-Scores por etapa:")
for item in f1_scores_por_etapa:
    print(f"Etapa {item['etapa']}: F1-Score = {item['f1_score']:.4f}")

# Mejor etapa
mejor_etapa = max(f1_scores_por_etapa, key=lambda x: x['f1_score'])
print(f"\n[INFO]: Mejor etapa: {mejor_etapa['etapa']} con F1-Score = {mejor_etapa['f1_score']:.4f}")



#----------------------------------------------------------------------------------------------
# CHECK VARIABLES ALLOCATED ON THE GPU

# variables_on_GPU()

#-----------------------------------------------------------------------------------------

# PATH = r'C:\Users\rubni\OneDrive - Universidad Rey Juan Carlos\LF_UAH_Train\dataSET_14\revista.pth'
# torch.save(model.state_dict(), PATH)
# print('Trained model written to',PATH)