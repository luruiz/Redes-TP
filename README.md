# skin-dataset-clasification  
---
**Las respuestas estan en `Preguntas.md`**
---
Para generar el entorno
```
$ python -m venv .venv
$ . ./.venv/bin/activate
$ pip install -r requirements.txt
```

Para levantar el notebook, mlflow y tensorboard
```
$ python -m jupyter lab
$ mlflow ui --port 5000
$ tensorboard --logdir=/path/to/logdir
```

Tensorboard suele abrir en el 6006, pero si tenes algo corriendo ahi sube al 6007 y sucesivamente.
