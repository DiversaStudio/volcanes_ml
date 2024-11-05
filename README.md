## Introducción

Este repositorio está dedicado a la comprensión y análisis de imágenes térmicas de volcanes en Ecuador. Utilizamos la librería `flirpy` para procesar y visualizar datos térmicos capturados de diferentes volcanes. Este proyecto tiene como objetivo inicial ayudar a los investigadores y científicos a interpretar mejor los datos térmicos y mejorar la monitorización y análisis de la actividad volcánica en la región.



## Configuración del Entorno de Trabajo

Para un entorno limpio y organizado, se recomienda utilizar un entorno virtual. Sigue estos pasos para configurar el entorno y ejecutar el proyecto.



## Clonar el repositorio


Clona este repositorio en tu máquina local:

```bash
git clone https://github.com/DiversaStudio/volcanes_ml.git
cd volcanes_ml



 Preparar e Instalar la Librería flirpy

La librería flirpy es necesaria para manejar los datos térmicos en este proyecto. Sigue estos pasos para instalarla:

1. ### Clona el repositorio de `flirpy`:


git clone https://github.com/LJMU-Astroecology/flirpy.git
cd flirpy

2. ### Instala las dependencias de flirpy:
pip install -r requirements.txt


3. ### Instala flirpy localmente:
pip install .


## Crear y Activar un Entorno Virtual

Para aislar las dependencias, crea y activa un entorno virtual con los siguientes comandos:
En Windows
python -m venv env
env\Scripts\activate

En MacOS / Linux
python3 -m venv env
source env/bin/activate

4. ###  Instalar Dependencias del Proyecto

Con el entorno virtual activo, instala las dependencias necesarias ejecutando:
pip install -r requirements.txt