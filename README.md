# Sistema de Análisis de Imágenes Térmicas Volcánicas

## Descripción General
Sistema de aprendizaje profundo para el monitoreo automatizado de actividad volcánica mediante imágenes térmicas. El proyecto implementa una arquitectura de Red Neuronal Convolucional Multi-Rama para el procesamiento de imágenes térmicas FLIR, alcanzando una precisión del 98.86% en la detección de eventos de emisión.

## Características
- Procesamiento de imágenes térmicas brutas (.fff) mediante Flirpy
- Análisis de tensores multidimensionales para reconocimiento de patrones temporales
- Detección automatizada de bordes y análisis de umbrales térmicos
- Integración con PostgreSQL para gestión de metadatos y resultados
- Predicción de nuevas imágenes

## Arquitectura
El sistema emplea una arquitectura neural de tres ramas:
1. Rama térmica: Procesa datos de temperatura corregidos
2. Rama de detección de bordes: Analiza características morfológicas
3. Rama de umbrales: Evalúa anomalías térmicas

## Estructura del Proyecto
```plaintext
proyecto/
├── data/
│   ├── input/           # Imágenes térmicas .fff brutas
│   ├── processed/       # Tensores preprocesados
│   └── output/          # Predicciones del modelo
├── src/
│   ├── data/            # Módulos de procesamiento de datos
│   ├── features/        # Extracción de características
│   ├── models/          # Implementación de red neuronal
│   └── utils/           # Funciones auxiliares y visualización
└── notebooks/           # Cuadernos de desarrollo
    ├── 01_DataPreprocessing
    ├── 02_FeatureEngineering
    ├── 03_ModelTraining
    └── 04_Prediction
```

## Instalación
```bash
# Clonar repositorio
git clone https://github.com/username/analisis-termico-volcanico.git

# Instalar dependencias
pip install -r requirements.txt

# Configurar base de datos PostgreSQL
psql -U postgres -f setup/database.sql
```
## Métricas de Rendimiento

Detección de Emisiones: 98.86%
Detección de Cielo Despejado: 87.70%
Condiciones Nubladas: 81.25%

## Entrenamiento
El modelo fue entrenado con 7,024 imágenes térmicas bajo la siguiente configuración:

Tamaño de lote: 250
Tasa de aprendizaje: 0.001
Tasa de dropout: 0.6
Paciencia para parada temprana: 7

## Citación
Si utiliza este código en su investigación, por favor cite:
```bash
@article{
  title={Monitoreo de Actividad Volcánica Utilizando Arquitectura CNN Multi-Rama},
  affiliation={1:Diversa, 2:EPN},
  contributhors={Mosquera, Diana [1] & Gallegos, Francisco [1] & Vasconez, Juan [1] & Merino, Pedro [2]},  
  journal={Pending},
  year={2025}
}
```
## Agradecimientos

- Instituto Geofísico de la Escuela Politécnica Nacional de Ecuador
- Centro de Modelización Matemática (MODEMAT) de la Escuela Politécnica Nacional de Ecuador

## Contacto

Email: hello@diversa.studio
Diversa

