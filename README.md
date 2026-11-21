# Clasificación de nube de puntos de una fábrica

Proyecto de procesamiento de una nube de puntos 3D de una fábrica. El flujo tiene dos fases:

1. **Clasificación geométrica**: analiza la nube y clasifica puntos según características geométricas, normales y color.
2. **Clasificación con semillas**: usa anotaciones humanas dispersas para entrenar un clasificador y refinar algunas clases de la primera fase.

La segunda fase utiliza como entrada el PLY generado por la primera.

## Estructura

```text
.
├── fabrica_puntos.ply                 # Nube de puntos de entrada
├── seed_annotations.csv               # Anotaciones para la fase 2
├── fase1_clasificacion.py             # Pipeline geométrico (fase 1)
├── fase2_clasificacion_semillas.py    # Clasificación supervisada (fase 2)
├── preview_renderer.py                # Formato común de las vistas previas
├── requirements.txt                   # Dependencias de Python
└── results/
    ├── fase1_clasificacion.ply
    ├── fase1_estadisticas.json
    ├── fase1_comparacion.png
    ├── fase2_clasificacion.ply
    ├── fase2_estadisticas.json
    └── fase2_comparacion.png               # Vistas oblicua y superior
```

Los archivos de `results/` se generan o actualizan al ejecutar las fases correspondientes. Cada fase produce una imagen con vistas oblicua y superior, leyenda y conteo por clase.

## Requisitos

Se necesita Python y las dependencias indicadas en `requirements.txt`. En Windows PowerShell, desde la carpeta del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Ejecución

Ejecuta los scripts desde la carpeta del proyecto y en orden:

```powershell
python fase1_clasificacion.py
python fase2_clasificacion_semillas.py
```

La fase 1 crea el PLY clasificado, sus estadísticas y una imagen de dos vistas. La fase 2 requiere ese PLY y `seed_annotations.csv`; produce la clasificación refinada, sus estadísticas y su propia imagen con el mismo formato.

El CSV de semillas debe tener las columnas `class,x,y,z`, con coordenadas XYZ en metros. Las clases admitidas son `piso`, `tuberia`, `viga`, `persona` y `maquina_vehiculo`.

La ruta de entrada de la fase 1 puede cambiarse con `PLY_PATH`, y el directorio de resultados con `OUT_DIR`. La configuración de la fase 1 también admite ajustes mediante `CFG_JSON`.
