# Proyecto de Mantenimiento de Sistemas Mecatrónicos

Este código está escrito en Python, requiere instalar algunas librerías. Usa una red neuronal recurrente de tipo BiLSTM para hacer la estimación de remaining useful life de un conjunto de motores
basados en el reto PHM08 de la NASA. Los procedimientos están basados en: https://doi.org/10.1016/j.ymssp.2019.05.005

Para correr el código, todos los archivos adjuntos tienen que estar en la misma carpeta:

1-Correr el código train_and_val_split.py. Este toma los datos del dataset de entrenamienmto del archivo train.txt

2-Correr normalization.py

3-Cambiar el nombre del modelo dentro del código Linear_regression_HI_df_creation.py, por el nombre del archivo .pt creado luego de correr normalization.py. Normalmente tiene un nombre
como bilstm_autoencoder_loss_0.XXXXX.pt

4-Correr Linear_regression_HI_df_creation.py

5-Correr Linear_regression_model_training

6-Correr HI_library

Adicionalmente, si se busca hacer validaciones o demostraciones:

-view_train_HI_curves.py para visualizar las curvas de Health Index entrenadas. Se puede cambiar la variable N_UNITS para modificar la cantidad de curvas del gráfico de salida

-validacion_del_modelo.py permite hacer la validación del modelo usando los datos separados del dataset inicial

-resultados_validacion.py muestra gráficamente un resultado aleatorio obtenido con el código de validacion_del_modelo.py. Para ver múltiples resultados, solo se debe correr el código varias veces

-test.py fue utilizado para la validación usando el dataset de prueba que fue dado por el profesor. Este dataset contiene curvas incompletas de 10 motores en el archivo test.xlsx
