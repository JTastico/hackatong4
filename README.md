# Jebi Hackathon 2026 - Grupo 04

Repositorio oficial de entrega para el grupo 04 del Jebi Hackathon 2026.

## El reto

**Tema: Mining Productivity 2.0**

Usando 15 minutos de datos de una pala Hitachi EX-5600 cargando camiones (video estereo + IMU), proponer una solucion que ayude al operador de mina a incrementar la productividad.

**Principio guia:** "No se puede gestionar lo que no se puede medir."

## Como entregar

Su entrega final es lo que esta en la branch `main` con el tag `submission` antes de las **17:00 del 18 de abril**.

```bash
git add .
git commit -m "Final submission"
git push origin main
git tag submission
git push --tags
```

## Contrato del entrypoint

Jebi va a clonar este repo y correr `bash run.sh` contra un dataset de testeo distinto al de desarrollo. Su solucion debe:

1. Leer los datos desde `./inputs/` (mismos nombres de archivo que el dev dataset: `shovel_left.mp4`, `shovel_right.mp4`, `imu_data.csv`)
2. Escribir resultados en `./outputs/` (formato libre: JSON, CSV, reporte HTML, video anotado)
3. Completar en menos de **10 minutos** en una laptop estandar
4. Ser reproducible: sin claves hardcoded, sin paths absolutos

**Si `run.sh` no corre contra el test dataset, su grupo queda fuera de los 8 finalistas.**

## Estructura del repo

```
.
├── README.md           # Este archivo (pueden reemplazarlo con el suyo)
├── run.sh              # Entrypoint que Jebi va a ejecutar (editar)
├── requirements.txt    # Dependencias Python (editar segun necesiten)
├── solution/           # Su codigo va aqui
├── inputs/             # Vacio. Jebi pone aqui el test dataset al evaluar
└── outputs/            # Vacio. Su run.sh escribe aqui
```

## Setup local

Para desarrollar, descarguen el dataset de desarrollo desde el link en su correo y colocando los 3 archivos en `./inputs/` localmente. **No commiteen los archivos de video al repo** (ya estan en `.gitignore`).

```bash
# Una vez con los datos en ./inputs/
bash run.sh
# Ver resultados en ./outputs/
```

## Tips

- Empiecen por explorar los datos antes de escribir codigo. Que ven en el video? Que muestra el IMU?
- Definan que significa "productividad" en su solucion antes de medirla
- Pregunten a los mentores. El reto es deliberadamente vago
- Sonnet 4.6 es el modelo default de Claude Code y alcanza para casi todo. No cambien a Opus sin necesidad real
- Los 4 entregables minimos al cierre del Block 2:
  1. Que midieron, y como?
  2. Que les dijeron los datos?
  3. Que le recomendarian al operador de mina?
  4. Si tuvieran 6h mas, que construirian despues?

Buena suerte.
