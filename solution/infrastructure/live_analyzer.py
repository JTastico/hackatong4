from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List

import cv2


@dataclass
class LiveStatus:
    running: bool = False
    frame_idx: int = 0
    total_frames: int = 0
    fps: float = 0.0
    timestamp_seconds: float = 0.0
    progress_percent: float = 0.0
    phase: str = "idle"
    video_name: str = ""
    updated_at: float = 0.0


@dataclass
class LiveAnalyzerStreamer:
    status: LiveStatus = field(default_factory=LiveStatus)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def available_videos(self, inputs_dir: Path) -> List[str]:
        return sorted([path.name for path in inputs_dir.glob("*.mp4")])

    def snapshot(self) -> Dict[str, float | int | str | bool]:
        with self._lock:
            return {
                "running": self.status.running,
                "frame_idx": self.status.frame_idx,
                "total_frames": self.status.total_frames,
                "fps": self.status.fps,
                "timestamp_seconds": round(self.status.timestamp_seconds, 3),
                "progress_percent": round(self.status.progress_percent, 2),
                "phase": self.status.phase,
                "video_name": self.status.video_name,
            }

    def stream(self, video_path: Path) -> Generator[bytes, None, None]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"No se pudo abrir el video para streaming: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 15.0)
        if fps <= 0:
            fps = 15.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_interval = 1.0 / fps

        with self._lock:
            self.status = LiveStatus(
                running=True,
                frame_idx=0,
                total_frames=total_frames,
                fps=fps,
                timestamp_seconds=0.0,
                progress_percent=0.0,
                phase="digging",
                video_name=video_path.name,
                updated_at=time.time(),
            )

        frame_idx = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                frame_idx += 1
                timestamp_seconds = frame_idx / fps
                progress = (frame_idx / total_frames * 100.0) if total_frames > 0 else 0.0
                phase = self._phase_for_progress(progress)

                self._draw_overlay(frame, frame_idx, total_frames, timestamp_seconds, phase, progress)

                with self._lock:
                    self.status.running = True
                    self.status.frame_idx = frame_idx
                    self.status.total_frames = total_frames
                    self.status.fps = fps
                    self.status.timestamp_seconds = timestamp_seconds
                    self.status.progress_percent = progress
                    self.status.phase = phase
                    self.status.video_name = video_path.name
                    self.status.updated_at = time.time()

                ok_jpg, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if not ok_jpg:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
                time.sleep(frame_interval)
        finally:
            capture.release()
            with self._lock:
                self.status.running = False
                self.status.phase = "completed"
                self.status.updated_at = time.time()

    def _phase_for_progress(self, progress_percent: float) -> str:
        # Simulación de fases para vista en vivo.
        cycle_pos = progress_percent % 100.0
        if cycle_pos < 25.0:
            return "digging"
        if cycle_pos < 55.0:
            return "swinging_loaded"
        if cycle_pos < 70.0:
            return "dumping"
        return "swinging_empty"

    def _draw_overlay(
        self,
        frame,
        frame_idx: int,
        total_frames: int,
        timestamp_seconds: float,
        phase: str,
        progress_percent: float,
    ) -> None:
        cv2.rectangle(frame, (10, 10), (600, 130), (0, 0, 0), thickness=-1)
        cv2.putText(
            frame,
            "Analisis sincronico (demo)",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Frame: {frame_idx}/{total_frames} | t={timestamp_seconds:.2f}s",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Fase estimada: {phase}",
            (20, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (120, 220, 120),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Progreso: {progress_percent:.1f}%",
            (20, 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 255),
            2,
            cv2.LINE_AA,
        )
