#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:07:25 2024

@author: alisonmatusbello

"""

# %% Importar bilbiotecas
from pandas.io.clipboard import paste
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import linregress
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.dates as mdates
import pywt
import seaborn as sns

# %% Definición del Modelo LSTM 
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, device, batch_size):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.device = device
        
        """ Definición de las capas de LSTM """
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size1, batch_first=True) # Capa 1 LSTM: salida de toda la secuencia
        self.lstm2 = nn.LSTM(input_size=hidden_size1, hidden_size=hidden_size2, batch_first=True) # Capa 2 LSTM: salida del último paso
        self.fc = nn.Linear(hidden_size2, output_size) # Capa totalmente conectada: asigna la salida final del LSTM al tamaño de salida deseado
        return
    
    """ Definición del método propagración hacia adelante """
    def forward(self, x):
        out, _ = self.lstm1(x) # LSTM capa 1: salida de toda la secuencia
        out, _ = self.lstm2(out) # LSTM capa 2: salida del último paso
        out = out[:, -1, :]  # Toma solo la salida del último paso
        out = self.fc(out)  # Asignar la salida final del LSTM al tamaño de salida deseado (transformación a la dimensión deseada)
        return out

#%% Clase principal para manejar el flujo de datos y entrenamiento
class course():
    def __init__(self, file, input_size, hidden_size1, hidden_size2, output_size, batch_size, past, future, clip, prediction):
        super(course, self).__init__()
        self.file = file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.batch_size = batch_size
        self.past = past
        self.future = future
        self.clip = clip  # gradient clipping threshold
        self.prediction = prediction
        self.s = 0 # Media del conjunto de datos
        self.m = 0 # Desviación estándar del conjunto de datos

    
    """ Procesamiento de datos: denoising, normalización, y creación de conjuntos de entrenamiento y validación. """
    def data_process(self):
        
        def sure_shrink(data):
            
            """ Aplicar la selección de umbrales de Stein's Unbiased Risk Estimate (SURE)."""
            # Cálculo de la desviación absoluta mediana (MAD) como estimación del ruido
            sigma = np.median(np.abs(data - np.median(data))) / 0.03835  # 0.03835
            return sigma * np.sqrt(2 * np.log(len(data)))

        def soft_threshold(data, threshold):
            
            """ Función (soft thresholding( function) """
            return np.sign(data) * np.maximum(np.abs(data) - threshold, 0)

# %%
        def wavelet_denoising(data, wavelet='sym4', level=3): # Tipo wavelet: sym4, db4, morl, etc.
      
            """
            Utilizar la función de eliminación de ruido wavelet
            parámetro：
                data: input
                wavelet: función base wavelet，por defecto es 'db4' o 'sym4'
                level: el número de niveles de descomposición wavelet
            """
            # Descomposición wavelet 
            coeffs = pywt.wavedec(data, wavelet, level=level)

            # Aplicar umbralización SURE a los coeficientes de detalle mediante umbralización suave
            threshold = sure_shrink(coeffs[-1])  # Selección del umbral basada en los coeficientes de mayor nivel de detalle
            coeffs[1:] = [soft_threshold(c, threshold) for c in coeffs[1:]]

            # Reconstruir la señal
            denoised_data = pywt.waverec(coeffs, wavelet)

            return denoised_data


        df = pd.read_csv('data_selected3h.csv', delimiter=';')
        print(df)
        # Establecer la columna 'time' como índice y convertirla a formato datetime
        df['time'] = pd.to_datetime(df['time'], unit='s') #+ pd.Timedelta(hours=8)
        df.set_index('time', inplace=True)

        # Filtrado wavelet
        columns_to_denoise = ['PWV APEX', 'PWV UCSC', 'Humedad', 'Temperatura']
        df[columns_to_denoise] = df[columns_to_denoise].apply(wavelet_denoising)

        # Normalización（Z-score）
        self.m = df[self.prediction].mean()
        self.s = df[self.prediction].std()  # Calcular la media y la desviación típica, utilizadas para la desnormalización
        df = (df - df.mean()) / df.std()


        # Construir conjunto de entrenamiento y conjunto de prueba。
        X = []
        y = []
        timestamps = []  # Usar para registrar las marcas de tiempo.
        X_test = []
        y_test = []

        d = {'PWV APEX': 0, 'PWV UCSC': 1, 'Humedad': 2, 'Temperatura': 3}
        a = d[self.prediction]
                      
        for i in range(self.past, len(df) - self.future + 1, 4):
            # Utilizar los últimos 24 pasos como entrada
            X.append(df.iloc[i - self.past: i].values)
            # Utilice los 4 pasos siguientes como etiquetas (labels)
            y.append(df.iloc[i: i + self.future, a].values)
            # Registrar las marcas de tiempo
            timestamps.append(df.index[i: i + self.future])

         # Validación de las dimensiones de los datos procesados
        if len(X) == 0 or len(y) == 0:
           raise ValueError("El conjunto de datos no tiene suficientes muestras para procesar.")
        if len(timestamps) < len(X):
           raise ValueError("Las marcas de tiempo no coinciden con el número de muestras.")
           
        if len(X) > 0 and len(y) > 0:  # Solo imprime si hay datos
           print("Primera muestra de entrada (X):")
           print(pd.DataFrame(X[0], columns=df.columns))  # Muestra las características
           print("\nPrimera muestra de salida (y):")
           print(y[0])  # Solo muestra la salida de PWV UCSC

        # Divida el conjunto de datos en conjuntos de entrenamiento y de prueba con una proporción de 8:2.
        X, y = np.array(X), np.array(y)
        train_val_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_val_size], X[train_val_size:]
        y_train, y_test = y[:train_val_size], y[train_val_size:]

        # Convertir las marcas de tiempo en un array numpy
        self.timestamps = np.array(timestamps[train_val_size:])  # Guardar las marcas de tiempo del conjunto de prueba
        
        print("Train timestamps range:", timestamps[:train_val_size][0], timestamps[:train_val_size][-1])
        print("Test timestamps range:", timestamps[train_val_size:][0], timestamps[train_val_size:][-1])


        print('train data'.center(64, '-'))
        print(X_train.shape)
        print(y_train.shape)
        print('test data'.center(64, '-'))
        print(X_test.shape)
        print(y_test.shape)

        # Convertir los datos en torch tensors.
        X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
        X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
        y_train_tensor = torch.from_numpy(y_train.astype(np.float32))
        y_test_tensor = torch.from_numpy(y_test.astype(np.float32))

        self.X_train_tensor = X_train_tensor
        self.X_test_tensor = X_test_tensor
        self.y_train_tensor = y_train_tensor
        self.y_test_tensor = y_test_tensor

        # Construct dataset loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            # Cuando el número de muestras no sea divisible por el tamaño del lote, elimine el último lote incompleto.
            shuffle=True
            # Al principio de cada época, baraja los datos para evitar que el modelo se ajuste en exceso a una disposición específica de los datos.
        )

    def train_and_test(self):
        # Crear el modelo
        model = LSTMModel(
            input_size=self.input_size,
            hidden_size1=self.hidden_size1,
            hidden_size2=self.hidden_size2,
            output_size=self.output_size,
            batch_size=self.batch_size,
            device=self.device
        ).to(self.device)

        # Entrenamiento y prueba del modelo
        # Entrenamiento
        #optimizer：Adam con tasa de aprendizaje inicial de 0.01
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)
        #optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        # Scheduler de tasa de aprendizaje dinámica: decae a 1/10 después de 100 épocas
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        # Criterio de pérdida (Loss)：MSE
        criterion = nn.MSELoss()

        train_loss_arr = []
        val_loss_arr = []

        for epoch in range(180):
            # training mode
            model.train()
            train_loss = []
            val_loss = []

            for i, (inputs, labels) in enumerate(self.train_loader):
                # Mover tensor a GPU
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Agregar los print para inspeccionar las entradas (solo la primera iteración del primer epoch)
                if epoch == 0 and i == 0:  # Solo primera iteración del primer epoch
                  print("Input tensor shape:", inputs.shape)
                  print("Mean of input sample (first batch):", inputs[0].mean(dim=0))
                  print("Std of input sample (first batch):", inputs[0].std(dim=0))

                # forward propagation
                outputs = model(inputs)
                # Borrar los gradientes de las épocas anteriores
                optimizer.zero_grad()
                # Calcular "Loss"
                loss = criterion(outputs, labels)
                # backpropagate the loss and update the weight matrices
                loss.backward()
                # gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), self.clip)
                # collect loss
                train_loss.append(loss.item())
                # update optimizer
                optimizer.step()

            # updata learnings
            scheduler.step()
            # calculate the loss of each epoch
            epoch_train_loss = sum(train_loss) / len(train_loss)
            train_loss_arr.append(epoch_train_loss)
            print(f'Epoch [{epoch + 1}/{180}], Train Loss: {epoch_train_loss:.4f}')

        # show the loss figure
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(train_loss_arr, label='tran_loss of' + self.prediction)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        ax.legend()
        plt.grid(False)
        plt.show()


        # Modo Evaluación
        model.eval()
        # use the model output to generate the predicted results
        with torch.no_grad():
            X_test_tensor = self.X_test_tensor.to(self.device)
            predictions = model(X_test_tensor).cpu().numpy()
            y_test_np = self.y_test_tensor.cpu().numpy()

        # Normalización
        predictions = np.ravel(predictions, order='C')
        y_test_np = np.ravel(y_test_np, order='C')
        predictions = predictions * self.s + self.m
        y_test_np = y_test_np * self.s + self.m
        
        # Realizar un ajuste lineal entre los valores reales y los valores previstos
        timestamps = np.array(self.timestamps)
        timestamps = timestamps.flatten()  # Convierte la matriz de marcas de tiempo en una matriz unidimensional
        plt.figure(figsize=(35, 20))
        plt.plot(timestamps, y_test_np, linewidth=1.2, label='Real', color='blue')  # valores actuales
        plt.plot(timestamps, predictions, linewidth=1.2, label='Predicted', color='red')  # valores predichos
        plt.title('Prediction vs Actual ' + self.prediction, fontsize=40)  # Título más grande
        plt.xlabel('Datetime', fontsize=27)  # Etiqueta del eje X más grande
        plt.ylabel('PWV UCSC (mm)', fontsize=27)  # Etiqueta del eje Y más grande
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # Seleccionar automáticamente el tiempo como eje x
        plt.gca().xaxis.set_major_formatter(
            mdates.DateFormatter('%Y-%m-%d %H:%M'))  # Convertir la hora a formato datetime
        plt.tick_params(axis='both', which='major', labelsize=16)  # Tamaño de las etiquetas de los ejes
        plt.legend(fontsize=35)  # Tamaño de la leyenda más grande
        plt.grid(False)  # Eliminar el grid
        plt.show()

        # Graficar las curvas de los resultados previstos y los resultados reales
        slope, intercept, r_value, p_value, std_err = linregress(y_test_np, predictions)


        # Métricas de Evaluación
        r_squared = r2_score(y_test_np, predictions) # Coeficiente de determinación (R^2)
        rmse = np.sqrt(np.mean((y_test_np - predictions) ** 2)) # Root Mean Squared Error (RMSE)
        mae = np.mean(np.abs(y_test_np - predictions))  # Mean Absolute Error (MAE)
        mape = np.mean(np.abs((y_test_np - predictions) / y_test_np)) * 100  # Mean Absolute Percentage Error (MAPE)

        # plot el diagrama de dispersión y la línea ajustada
        plt.figure(figsize=(10, 10))  # Ajustar el tamaño de la figura
        plt.scatter(y_test_np, predictions, alpha=0.5, s=50, label='Data points')  # Tamaño y transparencia de los puntos
        plt.plot(y_test_np, slope * y_test_np + intercept, color='red',
                 label=f'Fit line: y={slope:.2f}x+{intercept:.2f}', linewidth=2)  # Ajustar grosor de la línea
        plt.xlabel('Actual ' + self.prediction, fontsize=20)  # Tamaño de fuente de la etiqueta X
        plt.ylabel('Predicted ' + self.prediction, fontsize=20)  # Tamaño de fuente de la etiqueta Y
        plt.title(f'Scatter plot of Actual vs Predicted\nR²: {r_squared:.4f} | RMSE: {rmse:.2f} mm',
                 fontsize=20, pad=20)  # Tamaño y posición del título
        plt.legend(fontsize=20)  # Tamaño de la leyenda
        plt.tick_params(axis='both', which='major', labelsize=20)  # Tamaño de las etiquetas de los ejes
        plt.grid(False)  # Eliminar la rejilla
        plt.tight_layout()  # Ajustar el espaciado
        plt.show()

        
        # Mostrar las métricas como una tabla
        metrics_table = pd.DataFrame({
          "Metric": ["R²", "RMSE", "MAE", "MAPE"],
          "Value": [r_squared, rmse, mae, mape]
})
       # Configurar la visualización de Seaborn
        plt.figure(figsize=(5, 2))
        sns.set_theme(style="whitegrid")
        sns.heatmap(metrics_table.set_index("Metric"), annot=True, fmt=".4f", cmap="Blues", cbar=False, linewidths=0.5)
        plt.title("Evaluation Metrics")
        plt.show()
        
# %%
# Definición de parámetros
LSTM = course(
    file='data_selected3h.csv',
    input_size = 4,
    hidden_size1 = 64,
    hidden_size2 = 64,
    output_size =4,
    batch_size = 16,
    past = 48,  # input times-step
    future = 4,  # output times-step
    clip= 1.0,  # gradient clipping threshold
    prediction = 'PWV UCSC'
)
 

""" La 'prediction' representa el contenido predicho por el modelo, tiene cuatro opciones, que contienen:
'PWV APEX', 'PWV UCSC, 'Humedad', 'Temperatura' """


# Procesamiento de datos
LSTM.data_process()

# Entrenamiento y validación
LSTM.train_and_test()       

