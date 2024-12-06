# Modelo de Predicción de Vapor de Agua Precipitable (PWV)

Este repositorio contiene el código y la documentación del modelo de predicción de vapor de agua precipitable (PWV) desarrollado utilizando datos de dos radiómetros de 183 GHz y datos meteorológicos del Llano de Chajnantor, en Atacama, Chile. El proyecto combina técnicas avanzadas de aprendizaje profundo para mejorar la comprensión y predicción de las condiciones atmosféricas en una de las áreas más importantes para la observación astronómica.

## Descripción del Proyecto

El objetivo principal de este proyecto es desarrollar un modelo robusto para predecir el PWV en escalas temporales de hasta 1 día, integrando información de variables meteorológicas como temperatura, humedad, y datos históricos de PWV.

El modelo implementado utiliza **redes neuronales LSTM (Long Short-Term Memory)**, lo que permite capturar las dependencias temporales inherentes en los datos, mejorando la precisión en las predicciones.

## Estructura del Repositorio

- **`data_selected3h/`**: Contiene la base de datos utilizada en esta investigación.
- **`Código_modelo/`**: Contiene el código completo del modelo LSTM.
- **`README.md`**: Esta descripción y guía de uso del repositorio.

## Características Principales

- Predicción de PWV a partir de datos históricos y meteorológicos.
- Modelo ajustado y optimizado para los datos del Llano de Chajnantor.

## Requisitos

- **Python 3.8 o superior.**
- Librerías principales: `PyTorch`, `NumPy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `pywt`.

---
