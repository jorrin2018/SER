# 🎭 SER (Speech Emotion Recognition)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Development-yellow)

Algoritmo para clasificar sentimientos a partir de la señal de voz utilizando técnicas de aprendizaje profundo.

## 👨‍🔬 Autor
- **Nombre:** Jorge Luis Jorrin Coz
- **Institución:** Instituto Politécnico Nacional (IPN)
- **Programa:** BEIFI (Beca de Estímulo Institucional de Formación de Investigadores)
- **Periodo:** Junio 2025
- **Email:** jjorrinc2100@alumno.ipn.mx, jljorrincoz@gmail.com

## 📝 Descripción
Este proyecto implementa un sistema de reconocimiento de emociones en el habla utilizando una arquitectura de aprendizaje profundo que combina:
- BiLSTM (Red neuronal recurrente bidireccional)
- Mecanismos de atención (self-attention, multi-head attention, temporal attention)
- Características espectrales (MFCC)
- Características prosódicas (RMS, pitch)
- X-vectors (opcional)

## Características
- Procesamiento modular de características acústicas
- Atención configurable (single vs. multi-head, self vs. temporal)
- Balanceo de clases
- Validación cruzada opcional
- Entrenamiento con early stopping
- Visualización de resultados (curvas de pérdida, matriz de confusión)

## 📊 Dataset

El proyecto utiliza el dataset IEMOCAP (Interactive Emotional Dyadic Motion Capture) para el entrenamiento y evaluación. IEMOCAP es uno de los datasets más completos y ampliamente utilizados en el campo del reconocimiento de emociones en el habla.

### Características del Dataset
- **Nombre completo:** Interactive Emotional Dyadic Motion Capture Database
- **Tamaño:** ~12 horas de datos audiovisuales
- **Participantes:** 10 actores en sesiones diádicas
- **Idioma:** Inglés
- **Tipos de diálogo:** 
  - Actuados
  - Improvisados
  - Interacciones naturales

### Emociones clasificadas
| Emoción | Etiqueta | Descripción |
|---------|----------|-------------|
| Neutral | neu | Estado emocional equilibrado |
| Frustración | fru | Sentimiento de impotencia o molestia |
| Sorpresa | sur | Asombro o perplejidad |
| Enojo | ang | Ira o molestia intensa |
| Felicidad | hap | Estado de alegría o júbilo |
| Tristeza | sad | Melancolía o pesar |
| Excitación | exc | Estado de alta activación emocional |
| Miedo | fea | Sensación de amenaza o peligro |
| Otros | xxx | Emociones no categorizadas |

### Referencia
> Busso, C., Bulut, M., Lee, C. C., Kazemzadeh, A., Mower, E., Kim, S., ... & Narayanan, S. S. (2008). IEMOCAP: Interactive emotional dyadic motion capture database. Language resources and evaluation, 42(4), 335-359.

## 🛠️ Requisitos

```bash
# Instalación de dependencias
pip install -r requirements.txt
```

Principales dependencias:
- 🔥 PyTorch >= 1.9.0
- 🎵 Librosa >= 0.8.1
- 🧮 NumPy >= 1.19.5
- 🐼 Pandas >= 1.3.0
- 🧪 Scikit-learn >= 0.24.2
- 📊 Matplotlib >= 3.4.2
- 🗣️ SpeechBrain >= 0.5.12

## 🚀 Uso

### Preparación del entorno
1. Clona el repositorio
```bash
git clone https://github.com/jjorrinc2100/SER.git
cd SER
```

2. Instala las dependencias
```bash
pip install -r requirements.txt
```

### Configuración del Dataset
1. Descarga el dataset IEMOCAP (requiere solicitud de acceso)
2. Coloca los archivos de audio en la siguiente estructura:
```
IEMOCAP_full_release/
├── Session1/
├── Session2/
├── Session3/
├── Session4/
└── Session5/
```

### Ejecución
```bash
python main.py
```

## 📈 Resultados

### Archivos generados
- `best_model.pt`: Mejor modelo guardado durante el entrenamiento
- `loss_curve.png`: Visualización de las curvas de pérdida

### Métricas y Visualizaciones
El modelo genera automáticamente:
- 📊 Métricas de precisión por validación cruzada (5-fold)
- 📉 Curvas de pérdida de entrenamiento y validación
- 🎯 Matriz de confusión
- 📋 Reporte detallado de clasificación con:
  - Precisión por clase
  - Recall por clase
  - F1-score
  - Accuracy global

## 📚 Cita

Si utilizas este trabajo en tu investigación, por favor cítalo como:

```bibtex
@misc{jorrin2025ser,
  author = {Jorrin-Coz, Jorge},
  title = {SER: Speech Emotion Recognition using Deep Learning},
  year = {2025},
  publisher = {GitHub},
  institution = {Instituto Politécnico Nacional},
  howpublished = {\url{https://github.com/jorrin2018/SER}}
}
```

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.
