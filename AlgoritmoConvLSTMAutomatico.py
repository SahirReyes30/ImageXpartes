import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
#from google.colab import files
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
window = 5
categories = [0, 35, 70, 119, 177, 220, 255] 
horizon = 4
parte0_0 = "Part0_0"
parte0_1 = "Part0_1"
parte1_0 = "Part1_0"
parte1_1 = "Part1_1"
carpeta = ""

#leer una entrada de usuario por consola para variable de carpeta
carpeta = input("Ingrese el nombre de la carpeta: ")
print(carpeta)

#crear carpeta si no existe
if not os.path.exists("DroughtDatasetMask/dataset/BordesNuevos/"+carpeta):
    os.makedirs("DroughtDatasetMask/dataset/BordesNuevos/"+carpeta)


imagenInicial = 300

x00 = np.load("DroughtDatasetMask/NPY61_180Part0_0/DroughtDatasetMask_Part0_0.npy")
x01 = np.load("DroughtDatasetMask/dataset/BordesNuevos/61_180Part0_1/DroughtDatasetMaskBordesNuevos0_1v2.npy")
x10 = np.load("DroughtDatasetMask/dataset/BordesNuevos/61_180Part1_0/DroughtDatasetMaskBordesNuevos1_0v2.npy")
x11 = np.load("DroughtDatasetMask/dataset/BordesNuevos/61_180Part1_1/DroughtDatasetMaskBordesNuevos1_1v2.npy")
print ("00",x00.shape)
print ("01",x01.shape)
print ("10",x10.shape)
print ("11",x11.shape)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():

    for x in [x00 ,x01, x10, x11]:
        if x is x00:
            parte = parte0_0
            rows = x00.shape[1]
            cols = x00.shape[2]
            print("rows",rows)
            print("cols",cols)
        elif x is x01:
            parte = parte0_1
            rows = x01.shape[1]
            cols = x01.shape[2]
            print("rows",rows)
            print("cols",cols)
        elif x is x10:
            parte = parte1_0
            rows = x10.shape[1]
            cols = x10.shape[2]
            print("rows",rows)
            print("cols",cols)
        elif x is x11:
            parte = parte1_1
            rows = x11.shape[1]
            cols = x11.shape[2]
            print("rows",rows)
            print("cols",cols)
        
        print("Parte", parte)

        print("x", x.shape)
        print("x", x.dtype)
        print("x", x.min())
        print("x", x.max())
        
        #x = np.array([gray_quantized(i, np.array(categories)) for i in x])
        #colors_greys = get_colors(x[1168])
        #print(f"Colores {colors_greys}")
        #print(x.shape)
#
        ##inicio
        #x = np.array([gray_quantized(i, np.array(categories)) for i in x])
        #colors_greys = get_colors(x[1168])
        #print(f"Colores {colors_greys}")
        #print(x.shape)
#
        #x_greys = np.array([recolor_greys_image(img, categories) for img in x])
        #x = x_greys.astype('float32') / 255
        #print(get_colors(x[1168]))
        #print(x.shape)
#
        #x_2 = agroup_window(x, window)
        #print(x_2.shape)
        #x_train = x_2[:int(len(x_2)*.7)]
        #x_test = x_2[int(len(x_2)*.7):]
        #x_validation = x_train[int(len(x_train)*.8):]
        #x_train = x_train[:int(len(x_train)*.8)]
#
        #x_train = x_train.reshape(len(x_train), window, rows, cols, channels)
        #x_validation = x_validation.reshape(len(x_validation), window, rows, cols, channels)
        #x_test = x_test.reshape(len(x_test), window, rows, cols, channels)
#
        #print("Forma de datos de entrenamiento: {}".format(x_train.shape))
        #print("Forma de datos de validación: {}".format(x_validation.shape))
        #print("Forma de datos de pruebas: {}".format(x_test.shape))
#
        #x_train, y_train = create_shifted_frames_2(x_train)
        #x_validation, y_validation = create_shifted_frames_2(x_validation)
        #x_test, y_test = create_shifted_frames_2(x_test)
#
        #print("Training dataset shapes: {}, {}".format(x_train.shape, y_train.shape))
        #print("Validation dataset shapes: {}, {}".format(x_validation.shape, y_validation.shape))
        #print("Test dataset shapes: {}, {}".format(x_test.shape, y_test.shape))
#
        #crear carpeta
        if not os.path.exists("DroughtDatasetMask/dataset/BordesNuevos/"+carpeta+"/"+str(rows)+"_"+str(cols)+parte):
            os.makedirs("DroughtDatasetMask/dataset/BordesNuevos/"+carpeta+"/"+str(rows)+"_"+str(cols)+parte)
            
#
        ##DroughtDatasetMask/dataset/BordesNuevos/61_180Part0_1
        #np.save("DroughtDatasetMask/dataset/BordesNuevos/"+carpeta+"/61_180"+parte+"/x_test_mask.npy", x_test)
        #np.save("DroughtDatasetMask/dataset/BordesNuevos/"+carpeta+"/61_180"+parte+"/y_test_mask.npy", y_test)
        #np.save("DroughtDatasetMask/dataset/BordesNuevos/"+carpeta+"/61_180"+parte+"/x_train_mask.npy", x_train)
        #np.save("DroughtDatasetMask/dataset/BordesNuevos/"+carpeta+"/61_180"+parte+"/y_train_mask.npy", y_train)
        #np.save("DroughtDatasetMask/dataset/BordesNuevos/"+carpeta+"/61_180"+parte+"/x_validation_mask.npy", x_validation)
        #np.save("DroughtDatasetMask/dataset/BordesNuevos/"+carpeta+"/61_180"+parte+"/y_validation_mask.npy", y_validation)

        #cargar datos    
        x_test = np.load("DroughtDatasetMask/dataset/BordesNuevos/61_180"+parte+"/x_test_mask.npy")
        y_test = np.load("DroughtDatasetMask/dataset/BordesNuevos/61_180"+parte+"/y_test_mask.npy")
        x_train = np.load("DroughtDatasetMask/dataset/BordesNuevos/61_180"+parte+"/x_train_mask.npy")
        y_train = np.load("DroughtDatasetMask/dataset/BordesNuevos/61_180"+parte+"/y_train_mask.npy")
        x_validation = np.load("DroughtDatasetMask/dataset/BordesNuevos/61_180"+parte+"/x_validation_mask.npy")
        y_validation = np.load("DroughtDatasetMask/dataset/BordesNuevos/61_180"+parte+"/y_validation_mask.npy")


        # Define the path where you want to save the log file
        log_file_path = "DroughtDatasetMask/dataset/BordesNuevos/"+carpeta+"/"+ str(rows)+"_"+str(cols)+parte+"/InfoConvLSTM2D_Mask"+str(rows)+"_"+str(cols)+".txt"

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
            filepath= "DroughtDatasetMask/dataset/BordesNuevos/"+carpeta+"/"+str(rows)+"_"+str(cols)+parte+"/ConvLSTM2D_Mask"+str(rows)+"_"+str(cols)+".h5",
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
        
        model.save("DroughtDatasetMask/dataset/BordesNuevos/"+carpeta+"/"+str(rows)+"_"+str(cols)+parte+"/ConvLSTM2D_Mask"+str(rows)+"_"+str(cols)+".h5")
    

        print (f"lengeth x_test: {len(x_test)}")
#        imagenInicial = np.random.choice(range(len(x_test)), size= 1)[0]
        print(imagenInicial)

        example = x_test[imagenInicial]

        print(example.shape)

        err = model.evaluate(x_test, y_test, batch_size= 2)
        print("El error del modelo es: {}".format(err))
        preds = model.predict(x_test, batch_size= 2)
        print(preds.shape)
        x_test_new = add_last(x_test, preds[:])
        preds2 = model.predict(x_test_new, batch_size= 2)
        #print(preds2.shape)
        x_test_new = add_last(x_test_new, preds2[:])
        preds3 = model.predict(x_test_new, batch_size= 2)
        x_test_new = add_last(x_test_new, preds3[:])
        preds4 = model.predict(x_test_new, batch_size= 2)
        res_forecast = add_last(x_test_new, preds4[:])
        print("PREDSS",res_forecast.shape)

        np.save("DroughtDatasetMask/dataset/BordesNuevos/"+carpeta+"/"+str(rows)+"_"+str(cols)+parte+"/PredictionsConvolutionLSTM_forecast_"+str(rows)+"_"+str(cols)+"_"+parte+"_w"+str(window)+".npy", res_forecast)  #Guardar el vector de predicciones

        print("Res_forecast" , res_forecast.shape)

        print("x_test" , x_test.shape)
        print("x_test_new" , x_test_new.shape)
        print("y_test" , y_test.shape)

        # Selecciona la primera imagen y elimina la dimensión de canal singular con squeeze()
        #plt.imshow(preds[0].squeeze(), cmap='gray')
        #plt.title("First Predicted Image")
        #plt.axis('off')
        #plt.show()
    

    #matriz de confus DroughtDatasetMask/dataset/BordesNuevos/v5         /  71         _       190  Part1_1/PredictionsConvolutionLSTM_forecast_71_190_Part1_1_w5.npy
    data00 = "1"
    data01 = "2"
    data10 = "3"
    data11 = "4"
    for data in [data00 ,data01, data10, data11]:
        if data is data00:
            parte = parte0_0 #DroughtDatasetMask/dataset/BordesNuevos/61_180Part0_0/x_test_mask.npy
            x_test = np.load("DroughtDatasetMask/dataset/BordesNuevos/61_180Part0_0/x_test_mask.npy")
            y_test = np.load("DroughtDatasetMask/dataset/BordesNuevos/61_180Part0_0/y_test_mask.npy")
            rows = len(x_test[0,0])
            cols= len(x_test[0,0,0])
            data = np.load("DroughtDatasetMask/dataset/BordesNuevos/"+carpeta+"/"+str(rows)+"_"+str(cols)+parte+"/PredictionsConvolutionLSTM_forecast_61_180_Part0_0_w5.npy")
        elif data is data01:
            parte = parte0_1
            x_test = np.load("DroughtDatasetMask/dataset/BordesNuevos/61_180Part0_1/x_test_mask.npy")
            y_test = np.load("DroughtDatasetMask/dataset/BordesNuevos/61_180Part0_1/y_test_mask.npy")
            rows = len(x_test[0,0])
            cols= len(x_test[0,0,0])
            data = np.load("DroughtDatasetMask/dataset/BordesNuevos/"+carpeta+"/"+str(rows)+"_"+str(cols)+parte+"/PredictionsConvolutionLSTM_forecast_61_190_Part0_1_w5.npy")
        elif data is data10:
            parte = parte1_0
            x_test = np.load("DroughtDatasetMask/dataset/BordesNuevos/61_180Part1_0/x_test_mask.npy")
            y_test = np.load("DroughtDatasetMask/dataset/BordesNuevos/61_180Part1_0/y_test_mask.npy")
            rows = len(x_test[0,0])
            cols= len(x_test[0,0,0])
            data = np.load("DroughtDatasetMask/dataset/BordesNuevos/"+carpeta+"/"+str(rows)+"_"+str(cols)+parte+"/PredictionsConvolutionLSTM_forecast_71_180_Part1_0_w5.npy")
        elif data is data11:
            parte = parte1_1
            x_test = np.load("DroughtDatasetMask/dataset/BordesNuevos/61_180Part1_1/x_test_mask.npy")
            y_test = np.load("DroughtDatasetMask/dataset/BordesNuevos/61_180Part1_1/y_test_mask.npy")
            rows = len(x_test[0,0])
            cols= len(x_test[0,0,0])
            data = np.load("DroughtDatasetMask/dataset/BordesNuevos/"+carpeta+"/"+str(rows)+"_"+str(cols)+parte+"/PredictionsConvolutionLSTM_forecast_71_190_Part1_1_w5.npy")
            
        print ("data",data.shape)
        print ("BordesNuevos/"+carpeta+"/"+str(rows)+"_"+str(cols)+parte+"/PredictionsConvolutionLSTM_forecast_"+str(rows)+"_"+str(cols)+"_"+parte+"_w5.npy")
        classes = np.array([0, 255, 220, 177, 119, 70, 35]) # 255, 220, 177, 119, 70, 35  0
        classes_rgb = np.array([[0,0,0], [35,35,35], [70,70,70], [119,119,119], [177,177,177], [220,220,220], [255,255,255]])
        rows = len(x_test[0,0])
        cols= len(x_test[0,0,0])
        print(rows)
        print(cols)
        h = 4

        print(data.shape)
        print(x_test.shape)
        print(y_test.shape)

        y_test = get_cubes(y_test, h)

        colors = get_colors(x_test[-10,0])
        print("COLORSS", colors)
        print("COLORS", colors.shape)

        colorss = get_colors(data[-10,0])
        print("COLORSS", colorss)

        naive = x_test[:-4]
        data = data[1:-3]

        #y_real = y_test[:, -h:]*255
        new_data = data[:, -h:]
        n_real = naive[:, -h:]*255

        #y_test = y_test[:, -h:]
        naive = naive[:, -h:]

        print("XX")
        print(y_test.shape)
        print(new_data.shape)
        print(n_real.shape)

        print(min(new_data[0,0,60]))
        print(max(new_data[0,0,60]))

        new_data = new_data * 255
        new_data = new_data.astype(np.uint8)

        print("new_data", new_data.shape)
        print(colorss.shape)
        print(min(new_data[0,0,60]))
        print(max(new_data[0,0,60]))

        new_data = new_data.reshape(new_data.shape[:-1])
        print("HoY", new_data.shape)

        aux = []
        for i in new_data:
            aux2 = []
            for j in i:
                #res = cv2.cvtColor(j, cv2.COLOR_GRAY2RGB)
                #res = recolor_greys_image(j, classes)
                #rgb_quantized(res, classes_rgb)
                #res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
                res = gray_quantized(j, classes)
                res = recolor_greys_image(res, classes)
                aux2.append(res)
            aux.append(np.array(aux2))


        new_data = np.array(aux)
        print("SHAPEE", new_data.shape)
        color_data = get_colors(new_data[-10,0])
        print("DCOLORS", color_data)
        new_data = new_data.reshape(new_data.shape[0],new_data.shape[1],new_data.shape[2],new_data.shape[3],1)

        y_test = y_test * 255
        naive = naive * 255

        print("YCOLORS", get_colors(y_test[-10,0]))
        print("NCOLORS", get_colors(naive[-10,0]))
        print("DCOLORS", get_colors(new_data[-10,0]))

        print("XS")
        print(f"new data shape {new_data.shape}")
        print(f"y_test.shape {y_test.shape}")
        print(f"new data shape {naive.shape}")

        l_clas = len(classes)

        #print 
        print (f"lengeth x_test: {y_test.shape[0]}")
        print (f"h: {h}")
        print (f"rows: {rows}")
        print (f"cols: {cols}")

        cm_f = np.zeros((l_clas, l_clas), dtype=np.uint64)
        cm_n = np.zeros((l_clas, l_clas), dtype=np.uint64)
        print(cm_f)

        for e in range(y_test.shape[0]):
            for k in range(h):
                for i in range(rows):
                    for j in range(cols):
                        #print(f"e: {e}, k: {k}, i: {i}, j: {j}")
                        pos1 = np.where(classes == y_test[e, k, i, j])[0][0]
                        pos2 = np.where(classes == new_data[e, k, i, j])[0][0]
                        pos3 = np.where(classes == naive[e, k, i, j])[0][0]
                        cm_f[pos1, pos2] += 1
                        cm_n[pos1, pos3] += 1

        print("Matriz de confusión de pronóstico")
        print(cm_f)
        print("Matriz de confusión de naive")
        print(cm_n)

        import pandas as pd

        # Convert cm_f numpy array to pandas DataFrame
        df_cm_f = pd.DataFrame(cm_f)

        #print(df_cm_f)

        df_cm_n = pd.DataFrame(cm_n)

        #print(df_cm_n)

        # Crear el DataFrame de la primera matriz de confusión como antes
        df_cm_f = pd.DataFrame(cm_f, index=[f'True_{i}' for i in range(len(cm_f))],
                               columns=[f'Pred_{i}' for i in range(len(cm_f[0]))])

        # Crear el DataFrame de la segunda matriz de confusión como antes
        df_cm_n = pd.DataFrame(cm_n, index=[f'True_{i}' for i in range(len(cm_n))],
                               columns=[f'Pred_{i}' for i in range(len(cm_n[0]))])

        # Calcular el desplazamiento necesario para la segunda matriz (longitud de la primera matriz + 2 por la columna vacía)
        offset = df_cm_f.shape[1] + 2

        # Crear un escritor de Excel
        with pd.ExcelWriter("DroughtDatasetMask/dataset/BordesNuevos/"+carpeta+"/"+str(rows)+"_"+str(cols)+parte+"/combined_confusion_matrices.xlsx") as writer:
            # Escribir la primera matriz en la hoja de cálculo empezando en la primera columna
            df_cm_f.to_excel(writer, startcol=0, index=True)

            # Escribir la segunda matriz en la hoja de cálculo con un desplazamiento
            df_cm_n.to_excel(writer, startcol=offset, index=True)

        #with pd.ExcelWriter("DroughtDatasetMask/NPY61_180"+parte+"/cm_f_n.xlsx") as writer:
        #    df_cm_f.to_excel(writer, sheet_name='cm_f')
        #    df_cm_n.to_excel(writer, sheet_name='cm_n')


        

        #fig = plt.figure(figsize=(20,20))
        #r = 3
        #c = 4
        #ac = 1
        #pos = 100

        #for i in range(h):
        #    ax = fig.add_subplot(r, c, ac)
        #    ax.imshow(y_test[pos,i], cmap='gray')
        #    ax.axis('off')
        #    ax.set_title('Original_t+{}'.format(i+1))
        #    ac += 1
        #plt.tight_layout()
        #plt.show()
        #fig = plt.figure(figsize=(20,20))
        #for i in range(h):
        #    ax = fig.add_subplot(r, c, ac)
        #    ax.imshow(new_data[pos,i], cmap='gray')
        #    ax.axis('off')
        #    ax.set_title('Pronóstico_t+{}'.format(i+1))
        #    ac += 1
        #plt.tight_layout()
        #plt.show()
        #fig = plt.figure(figsize=(20,20))

        #for i in range(h):
        #    ax = fig.add_subplot(r, c, ac)
        #    ax.imshow(naive[pos,i], cmap='gray')
        #    ax.axis('off')
        #    ax.set_title('Naive_t+{}'.format(i+1))
        #    ac += 1

        ## Ajustar el espaciado entre subplots
        #plt.subplots_adjust(wspace=.1, hspace=0.05)  # Puedes disminuir estos valores si es necesario
        #plt.tight_layout()
        #plt.show()
