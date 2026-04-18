#!/bin/bash
# Jebi Hackathon 2026 - Entrypoint
#
# Este script lo va a ejecutar Jebi contra un dataset de testeo distinto
# al de desarrollo. Editen este archivo para que llame a su solucion.
#
# Inputs disponibles en ./inputs/:
#   - shovel_left.mp4
#   - shovel_right.mp4
#   - imu_data.csv
#
# Outputs deben escribirse en ./outputs/

set -e  # Salir al primer error

echo "Jebi Hackathon 2026 - Grupo XX"
echo "Inputs:"
ls -la inputs/

# TODO: Equipo, reemplacen esta linea con la llamada a su solucion
# Ejemplos:
#   python solution/main.py
#   node solution/index.js
#   python -m solution.run

echo "ERROR: run.sh no ha sido implementado todavia"
echo "Editen run.sh para llamar a su solucion"
exit 1

echo "Outputs generados:"
ls -la outputs/
echo "Done."
