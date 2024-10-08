## Introducción

Este repositorio está dedicado a la comprensión y análisis de imágenes térmicas de volcanes en Ecuador. Utilizamos la librería `flirpy`, el modelo de deteccion de bordes`DexiNed` y `Marigold` para segmentar basado en profundidad. 

Se procesa y visualiza datos térmicos capturados de diferentes volcanes. Este proyecto tiene como objetivo inicial ayudar a los investigadores y científicos a interpretar mejor los datos térmicos y mejorar la monitorización y análisis de la actividad volcánica en la región.

## Breve explicacion de los archivos

+ __scripts/Segmentation-Model__ 
Esta carpeta contiene un clon de este repo: https://github.com/xavysp/DexiNed. Este modelo permite detectar bordes y puede ser ejecutado localmente. Para usarlo, este cuenta con un README o a su vez se puede referir a este vide: https://www.youtube.com/watch?v=Hz0uU04B3U8
+ __scripts/base_connect__
Este archivo nos permite crear una base de datos local usando Postgres para almarcenar data sobre las imágenes.
+ __scripts/data_preprocessing__
Este archivo nos permite normalizar los datos de temperatura y visualizar las imágenes.
+ __scripts/feature__
Este archivo nos permite segmentar las imagenes con un metodo diferente al de Marigold. 
+ __scripts/Marigold__
Este archivo nos permite detectar profundidad en las imagenes volcanicas, pero debe ser ejectuado en google cloud para hacer uso de GPUs


## Clonar el repositorio

Para clonar el repositorio, utiliza el siguiente comando en tu terminal:

```sh
!git clone https://github.com/LJMUAstroecology/flirpy.git
%cd flirpy
## Instalar las dependencias

Asegúrate de tener pip instalado. Luego, instala las dependencias requeridas utilizando:

sh

!pip install -r requirements.txt

!pip install .
