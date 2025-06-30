**Busqueda hiperparametros CNN**

El mayor test accuracy fue 70.16% con train accuracy de 78.05%. La diferencia de 8 puntos entre train y test inidican overfitting. Este modelo tiene los siguientes hiper parametros:

- BatchSize: 16
- Dropout: 0.1
- HFlip: 0.5
- VFlip: 0
- RBContrast: 0

El segundo mejor modelo logro un test accuracy de 69.61% y train accuracy de 71.31%, la similitud entre estos valores indican que esta red generaliza mejor que la anterior y no se encuentra overfitteada a los datos de entrenamiento. Los hiperparametros de este modelo son: 

- BatchSize: 16
- Dropout: 0.1
- HFlip: 0.5
- VFlip: 0
- RBContrast: 0
