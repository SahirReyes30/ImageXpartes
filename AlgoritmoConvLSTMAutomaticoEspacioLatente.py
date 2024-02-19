import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import zipfile
import sys

def create_shifted_frames_2(data):
    x = data[:, 0 : data.shape[1] - 1, :, :]
    y = data[:, data.shape[1]-1, :, :]
    return x, y

#Toma todos los colores existentes en la imagen
def get_colors(image):
  aux = []
  band = True
  for i in image:
    for j in i:

      for k in aux:
        if j.tolist() == k:
          band = False
          break
      if band:
        aux.append(j.tolist())
      band = True
  return np.array(aux)

def balance_img_categories(img, palette, balancer):
  #palette = np.sort(palette)
  rows = len(img)
  cols = len(img[0])
  for i in range(rows):
    for j in range(cols):
      pos = np.where(palette == img[i,j])[0][0]
      img[i,j] = balancer[pos]
  return img

#Función para dada una paleta solo tomar los colores de esa paleta en la imagen
def quantizetopalette(silf, palette, dither=False, mode="P"):
  """Convert an RGB or L mode image to use a given P image's palette."""
  silf.load()
  palette.load()
  im = silf.im.convert(mode, 0, palette.im)
  # the 0 above means turn OFF dithering making solid colors
  return silf._new(im)

#Realiza las operaciones necesarias para obtener una imagen RGB por una paleta de colores
def rgb_quantized(img, palette):
  rows, cols = len(img), len(img[0])
  total_vals = 1
  for i in palette.shape:
    total_vals *= i
  palettedata = palette.reshape(total_vals).tolist()
  palImage = Image.new('P', (rows, cols))
  palImage.putpalette(palettedata*32)
  oldImage = Image.fromarray(img).convert("RGB")
  newImage = quantizetopalette(oldImage,palImage)
  res_image = np.asarray(newImage.convert("RGB"))
  return res_image

def gray_quantized(img, palette):
  rows, cols = len(img), len(img[0])
  total_vals = 1
  for i in palette.shape:
    total_vals *= i
  palettedata = palette.reshape(total_vals).tolist()
  palImage = Image.new('L', (rows, cols))
  palImage.putpalette(palettedata*32)
  oldImage = Image.fromarray(img, 'L')
  newImage = quantizetopalette(oldImage,palImage, mode="L")
  res_image = np.asarray(newImage)
  return res_image

def recolor_greys_image(data, palette):
    rows, cols = len(data), len(data[0])
    aux = np.zeros((rows, cols), dtype=np.uint64)
    for i in range(rows):
        for j in range(cols):
            aux[i,j] = min(palette, key= lambda x:abs(x-data[i,j]))
    return aux

def agroup_window(data, window):
    new_data = [data[i:window+i] for i in range(len(data)-window+1)]
    return np.array(new_data)

def add_last(data, new_vals):
    print(f"data: {data.shape} y new_val: {new_vals.shape}")
    x_test_new = data[:,1:]
    print(f"x_test_new: {x_test_new.shape}")

    l = []
    for i in range(len(x_test_new)):
        l.append(np.append(x_test_new[i], new_vals[i]))
    x_test_new = np.array(l).reshape(data.shape[:])
    print("CX", x_test_new.shape)
    return x_test_new

def add_lastNew(data, new_val):
    print(f"data: {data.shape} y new_val: {new_val.shape}")
    x_test_new = data[:,1:,...]  # Omite el primer paso de tiempo
    print(f"x_test_new: {x_test_new.shape}")

    # Asumiendo que new_val es una única predicción que se debe añadir a cada paso de tiempo en x_test_new
    new_val = new_val.squeeze(axis=0)  # Elimina la dimensión del batch, si es necesario

    print(new_val.shape)
    # Añadir new_val a cada elemento en x_test_new
    x_test_new = np.concatenate((x_test_new, np.expand_dims(new_val, axis=1)), axis=1)

    print("CX", x_test_new.shape)
    return x_test_new

#Crea cubos con su propia información de tamaño h
def get_cubes(data, h):
    new_data = []
    for i in range(0, len(data)-h):
        new_data.append(data[i:i+h])
    new_data = np.array(new_data)
    print(new_data.shape)
    return new_data


channels = 1
window = 10
categories = [0, 35, 70, 119, 177, 220, 255] 
horizon = 4


parte = "EspacioLatente"

carpeta = ""

#leer una entrada de usuario por consola para variable de carpeta
carpeta = input("Ingrese el nombre de la carpeta: ")
print(carpeta)

#crear carpeta si no existe
if not os.path.exists("DroughtDatasetMask/ResultadosEspacioLatente/"+carpeta):
    os.makedirs("DroughtDatasetMask/ResultadosEspacioLatente/"+carpeta)

imagenInicial = 300

x = np.load("/media/mccdual2080/Almacenamiengto/SahirProjects/SahirReyes/dataSetAutoencoder/DatasetAutoencoder/DataSetLatentSpace/Npy/Dataset120x360Encoded.npy")

strategy = tf.distribute.MirroredStrategy()
#print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():

    rows = x.shape[1]
    cols = x.shape[2]
    print("rows",rows)
    print("cols",cols)    
    
    print("Parte", parte)
    print("x", x.shape)
    print("x", x.dtype)
    print("x", x.min())
    print("x", x.max())

    x_2 = agroup_window(x, window)
    print(x_2.shape)
    x_train = x_2[:int(len(x_2)*.7)]
    x_test = x_2[int(len(x_2)*.7):]
    x_validation = x_train[int(len(x_train)*.8):]
    x_train = x_train[:int(len(x_train)*.8)]

    x_train = x_train.reshape(len(x_train), window, rows, cols, channels)
    x_validation = x_validation.reshape(len(x_validation), window, rows, cols, channels)
    x_test = x_test.reshape(len(x_test), window, rows, cols, channels)

    print("Forma de datos de entrenamiento: {}".format(x_train.shape))
    print("Forma de datos de validación: {}".format(x_validation.shape))
    print("Forma de datos de pruebas: {}".format(x_test.shape))

    x_train, y_train = create_shifted_frames_2(x_train)
    x_validation, y_validation = create_shifted_frames_2(x_validation)
    x_test, y_test = create_shifted_frames_2(x_test)

    print("Training dataset shapes: {}, {}".format(x_train.shape, y_train.shape))
    print("Validation dataset shapes: {}, {}".format(x_validation.shape, y_validation.shape))
    print("Test dataset shapes: {}, {}".format(x_test.shape, y_test.shape))


    np.save("DroughtDatasetMask/ResultadosEspacioLatente/"+carpeta+"/x_test_mask.npy", x_test)
    np.save("DroughtDatasetMask/ResultadosEspacioLatente/"+carpeta+"/y_test_mask.npy", y_test)
    np.save("DroughtDatasetMask/ResultadosEspacioLatente/"+carpeta+"/x_train_mask.npy", x_train)
    np.save("DroughtDatasetMask/ResultadosEspacioLatente/"+carpeta+"/y_train_mask.npy", y_train)
    np.save("DroughtDatasetMask/ResultadosEspacioLatente/"+carpeta+"/x_validation_mask.npy", x_validation)
    np.save("DroughtDatasetMask/ResultadosEspacioLatente/"+carpeta+"/y_validation_mask.npy", y_validation)

    # Define the path where you want to save the log file
    log_file_path = "DroughtDatasetMask/ResultadosEspacioLatente/"+carpeta+"/InfoConvLSTM2D_Mask"+str(rows)+"_"+str(cols)+".txt"

    # Save the original stdout so we can restore it later
    original_stdout = sys.stdout

    #Construction of Convolutional LSTM network
    inp = keras.layers.Input(shape=(None, *x_train.shape[2:]))
    #It will be constructed a 3 ConvLSTM2D layers with batch normalization,
    #Followed by a Conv3D layer for the spatiotemporal outputs.
    m = keras.layers.ConvLSTM2D(16, (5,5), padding= "same", return_sequences= True, activation= "relu")(inp)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(16, (5,5), padding= "same", return_sequences= True, activation= "relu")(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.ConvLSTM2D(16, (3,3), padding= "same", activation= "relu")(m)
    m = keras.layers.Conv2D(channels, (3,3), activation= "sigmoid", padding= "same")(m)
    model = keras.models.Model(inp, m)
    model.compile(loss= "binary_crossentropy", optimizer= "Adam")
    print(model.summary())
    #Callbacks
    early_stopping = keras.callbacks.EarlyStopping(monitor= "val_loss", patience= 6, restore_best_weights= True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor= "val_loss", patience= 6)
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath= "DroughtDatasetMask/ResultadosEspacioLatente/"+carpeta+"/ConvLSTM2D_Mask"+str(rows)+"_"+str(cols)+".h5",
        monitor= "val_loss",
        save_best_only= True,
        mode= "min"
    )
    # Model training with logs redirected to a file
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file  # Redirect stdout to the log file
        model.fit(
            x_train, y_train,
            batch_size=2,
            epochs=30,
            validation_data=(x_validation, y_validation),
            callbacks=[early_stopping, reduce_lr]
        )
        sys.stdout = original_stdout  # Restore stdout back to normal

    print(f"Training log was saved to {log_file_path}")

    #Guardar el modelo
    
    model.save("DroughtDatasetMask/ResultadosEspacioLatente/"+carpeta+"/ConvLSTM2D_Mask"+str(rows)+"_"+str(cols)+".h5")

    print(imagenInicial)

    example = x_test[imagenInicial]

    print(example.shape)

    err = model.evaluate(x_test, y_test, batch_size= 2)
    print("El error del modelo es: {}".format(err))
    preds = model.predict(x_test, batch_size= 2)
    print("preds",preds.shape)
    x_test_new = add_last(x_test, preds[:])
    preds2 = model.predict(x_test_new, batch_size= 2)
    print("preds2",preds2.shape)
    x_test_new = add_last(x_test_new, preds2[:])
    preds3 = model.predict(x_test_new, batch_size= 2)
    print ("preds3",preds3.shape)
    x_test_new = add_last(x_test_new, preds3[:])
    preds4 = model.predict(x_test_new, batch_size= 2)
    print ("preds4",preds4.shape)
    res_forecast = add_last(x_test_new, preds4[:])
    print("PREDSS",res_forecast.shape)

    np.save("DroughtDatasetMask/ResultadosEspacioLatente/"+carpeta+"/PredictionsConvolutionLSTM_forecast_"+str(rows)+"_"+str(cols)+"_"+parte+"_w"+str(window)+".npy", res_forecast)  #Guardar el vector de predicciones

    print("Res_forecast" , res_forecast.shape)

    print("x_test" , x_test.shape)
    print("x_test_new" , x_test_new.shape)
    print("y_test" , y_test.shape)