from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import find_peaks


@dataclass
class ImuSwingEvent:
    start_time_seconds: float
    peak_time_seconds: float
    end_time_seconds: float
    peak_value: float


class ImuProcessor:
    """
    Adaptador de salida para señales IMU.

    Analiza la serie de Yaw para detectar swings con `scipy.signal.find_peaks`.
    Soporta insumo en JSON, CSV y NPY.

    TODO integración real:
    - Ajustar pre-procesado (filtro pasa-bajo/pasa-banda).
    - Calibrar umbrales y ventana mínima con dataset real.
    - Incorporar otras señales (pitch, roll, aceleraciones) para robustez.
    """

    def detect_swing_events(self, imu_path: Path) -> Dict[str, List[ImuSwingEvent]]:
        imu_data = self._load_imu(imu_path)
        times = self._normalize_time_seconds(imu_data["time_seconds"])
        yaw = np.asarray(imu_data["yaw"], dtype=float)

        if len(times) < 5:
            return {"swing_events": []}

        # Suavizado para eliminar vibraciones de alta frecuencia.
        smoothed_yaw = self._smooth_signal(yaw, window_size=9)
        sample_period = self._estimate_sample_period(times)
        min_distance_seconds = 10.0
        min_distance_samples = max(1, int(min_distance_seconds / sample_period))
        dynamic_prominence = max(0.5, float(np.std(smoothed_yaw)) * 0.7)

        peaks, peak_props = find_peaks(
            smoothed_yaw,
            prominence=dynamic_prominence,
            distance=min_distance_samples,
        )
        swings: List[ImuSwingEvent] = []
        for peak_idx in peaks:
            peak_time = float(times[peak_idx])
            peak_value = float(smoothed_yaw[peak_idx])
            half_window = max(1, int(2.0 / sample_period))
            start_idx = max(0, peak_idx - half_window)
            end_idx = min(len(times) - 1, peak_idx + half_window)

            swings.append(
                ImuSwingEvent(
                    start_time_seconds=float(times[start_idx]),
                    peak_time_seconds=peak_time,
                    end_time_seconds=float(times[end_idx]),
                    peak_value=peak_value,
                )
            )

        return {
            "swing_events": swings,
            "peak_prominences": peak_props.get("prominences", []).tolist(),
            "sample_period_seconds": sample_period,
            "applied_prominence": dynamic_prominence,
            "applied_distance_samples": min_distance_samples,
        }

    def _load_imu(self, imu_path: Path) -> Dict[str, np.ndarray]:
        suffix = imu_path.suffix.lower()
        if suffix == ".json":
            return self._load_json_imu(imu_path)
        if suffix == ".csv":
            return self._load_csv_imu(imu_path)
        if suffix == ".npy":
            return self._load_npy_imu(imu_path)
        raise ValueError(f"Formato IMU no soportado: {imu_path.name}")

    def _load_json_imu(self, imu_path: Path) -> Dict[str, np.ndarray]:
        payload = json.loads(imu_path.read_text(encoding="utf-8"))
        records = payload["samples"] if isinstance(payload, dict) and "samples" in payload else payload
        if not isinstance(records, list) or not records:
            raise ValueError(f"El JSON de IMU no contiene muestras válidas: {imu_path}")

        times: List[float] = []
        yaw: List[float] = []
        for idx, sample in enumerate(records):
            t_value = sample.get("timestamp_seconds", idx * 0.1)
            yaw_value = sample.get("yaw", sample.get("Yaw", 0.0))
            times.append(float(t_value))
            yaw.append(float(yaw_value))

        return {"time_seconds": np.array(times), "yaw": np.array(yaw)}

    def _load_csv_imu(self, imu_path: Path) -> Dict[str, np.ndarray]:
        data = np.genfromtxt(imu_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        names = set(data.dtype.names or [])
        if "timestamp_seconds" in names:
            times = np.array(data["timestamp_seconds"], dtype=float)
        elif "time" in names:
            times = np.array(data["time"], dtype=float)
        else:
            times = np.arange(len(data), dtype=float) * 0.1

        if "yaw" in names:
            yaw = np.array(data["yaw"], dtype=float)
        elif "Yaw" in names:
            yaw = np.array(data["Yaw"], dtype=float)
        else:
            raise ValueError(f"No se encontró columna yaw/Yaw en CSV IMU: {imu_path}")

        return {"time_seconds": times, "yaw": yaw}

    def _load_npy_imu(self, imu_path: Path) -> Dict[str, np.ndarray]:
        """
        Soporta formatos .npy comunes:
        - vector 1D: se interpreta como yaw.
        - matriz 2D: intenta usar columnas [time, yaw] o al menos yaw en última columna.
        - dict serializado (allow_pickle): claves esperadas time/timestamp_seconds y yaw/Yaw.
        """
        arr = np.load(imu_path, allow_pickle=True)

        # Caso: dict serializado con np.save.
        if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
            obj = arr.item()
            if isinstance(obj, dict):
                time_key = "timestamp_seconds" if "timestamp_seconds" in obj else ("time" if "time" in obj else None)
                yaw_key = "yaw" if "yaw" in obj else ("Yaw" if "Yaw" in obj else None)
                if yaw_key is None:
                    raise ValueError(f"NPY dict sin clave yaw/Yaw: {imu_path}")
                yaw = np.asarray(obj[yaw_key], dtype=float)
                if time_key is not None:
                    times = np.asarray(obj[time_key], dtype=float)
                else:
                    times = np.arange(len(yaw), dtype=float) * 0.1
                return {"time_seconds": times, "yaw": yaw}

        # Caso: señal 1D yaw.
        if isinstance(arr, np.ndarray) and arr.ndim == 1:
            yaw = np.asarray(arr, dtype=float)
            times = np.arange(len(yaw), dtype=float) * 0.1
            return {"time_seconds": times, "yaw": yaw}

        # Caso: matriz 2D.
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            if arr.shape[1] >= 2:
                times = np.asarray(arr[:, 0], dtype=float)
                yaw = np.asarray(arr[:, 1], dtype=float)
            elif arr.shape[0] >= 2:
                # Fallback para forma transpuesta (2, N)
                times = np.asarray(arr[0, :], dtype=float)
                yaw = np.asarray(arr[1, :], dtype=float)
            else:
                raise ValueError(f"NPY 2D inválido para IMU: {imu_path}")
            return {"time_seconds": times, "yaw": yaw}

        raise ValueError(f"No se pudo interpretar el archivo NPY de IMU: {imu_path}")

    def _smooth_signal(self, signal: np.ndarray, window_size: int = 9) -> np.ndarray:
        if window_size <= 1 or len(signal) < window_size:
            return signal
        kernel = np.ones(window_size, dtype=float) / float(window_size)
        return np.convolve(signal, kernel, mode="same")

    def _estimate_sample_period(self, times: np.ndarray) -> float:
        if len(times) < 2:
            return 0.1
        deltas = np.diff(times)
        deltas = deltas[np.isfinite(deltas)]
        deltas = deltas[deltas > 0]
        if len(deltas) == 0:
            return 0.1
        return float(np.median(deltas))

    def _normalize_time_seconds(self, raw_times: np.ndarray) -> np.ndarray:
        times = np.asarray(raw_times, dtype=float)
        if len(times) == 0:
            return times

        # Corrige telemetrías en epoch (ms/us/ns) y las lleva a segundos relativos.
        if float(np.nanmax(times)) > 1e12:
            times = times / 1e9  # ns -> s
        elif float(np.nanmax(times)) > 1e9:
            times = times / 1e3  # ms -> s

        start = float(times[0])
        times = times - start
        return times
