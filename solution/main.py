from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel

from solution.application.analyzer_service import AnalyzeLoadingCycleUseCase
from solution.domain.models import AnalysisReport
from solution.infrastructure.imu_processor import ImuProcessor
from solution.infrastructure.live_analyzer import LiveAnalyzerStreamer
from solution.infrastructure.video_processor import VideoProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

app = FastAPI(title="Mining Productivity 2.0 API", version="0.1.0")
live_streamer = LiveAnalyzerStreamer()


class BatchResponse(BaseModel):
    status: str
    processed_reports: int
    output_files: List[str]
    message: str


def _build_use_case() -> AnalyzeLoadingCycleUseCase:
    return AnalyzeLoadingCycleUseCase(video_processor=VideoProcessor(), imu_processor=ImuProcessor())


def discover_inputs(inputs_dir: Path) -> Tuple[List[Tuple[Path, Optional[Path]]], Optional[Path]]:
    mp4_files = sorted(inputs_dir.glob("*.mp4"))
    json_files = sorted(inputs_dir.glob("*.json"))
    csv_files = sorted(inputs_dir.glob("*.csv"))
    npy_files = sorted(inputs_dir.glob("*.npy"))

    json_imu_files = [p for p in json_files if "imu" in p.stem.lower()]
    csv_imu_files = [p for p in csv_files if "imu" in p.stem.lower()]
    npy_imu_files = [p for p in npy_files if "imu" in p.stem.lower()]

    # Prioridad: archivo explícito con "imu" en el nombre.
    if json_imu_files:
        imu_path: Optional[Path] = json_imu_files[0]
    elif csv_imu_files:
        imu_path = csv_imu_files[0]
    elif npy_imu_files:
        imu_path = npy_imu_files[0]
    else:
        # Fallback flexible: primer JSON/CSV/NPY disponible que no sea output/report.
        candidate_json = [p for p in json_files if "report" not in p.stem.lower() and "output" not in p.stem.lower()]
        candidate_csv = [p for p in csv_files if "report" not in p.stem.lower() and "output" not in p.stem.lower()]
        candidate_npy = [p for p in npy_files if "report" not in p.stem.lower() and "output" not in p.stem.lower()]
        imu_path = (
            candidate_json[0]
            if candidate_json
            else (candidate_csv[0] if candidate_csv else (candidate_npy[0] if candidate_npy else None))
        )

    left = [f for f in mp4_files if "left" in f.stem.lower()]
    right = [f for f in mp4_files if "right" in f.stem.lower()]
    pairs: List[Tuple[Path, Optional[Path]]] = []

    if left:
        for idx, left_video in enumerate(left):
            right_video = right[idx] if idx < len(right) else None
            pairs.append((left_video, right_video))
    elif mp4_files:
        # Fallback cuando no vienen etiquetados como left/right.
        pairs.append((mp4_files[0], mp4_files[1] if len(mp4_files) > 1 else None))

    return pairs, imu_path


def _safe_input_file(name: str, suffix: str) -> Path:
    candidate = (INPUTS_DIR / Path(name).name).resolve()
    if candidate.parent != INPUTS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Ruta no permitida.")
    if candidate.suffix.lower() != suffix:
        raise HTTPException(status_code=400, detail="Tipo de archivo no permitido.")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Archivo no encontrado.")
    return candidate


def _safe_output_json(name: str) -> Path:
    candidate = (OUTPUTS_DIR / Path(name).name).resolve()
    if candidate.parent != OUTPUTS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Ruta no permitida.")
    if candidate.suffix.lower() != ".json":
        raise HTTPException(status_code=400, detail="Solo archivos JSON.")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Reporte no encontrado.")
    return candidate


def run_batch(inputs_dir: Path = INPUTS_DIR, outputs_dir: Path = OUTPUTS_DIR) -> BatchResponse:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    pairs, imu_path = discover_inputs(inputs_dir)

    if not pairs:
        raise RuntimeError("No se encontraron videos .mp4 en /inputs.")
    if imu_path is None:
        raise RuntimeError("No se encontró telemetría IMU (JSON, CSV o NPY) en /inputs.")

    use_case = _build_use_case()
    generated_files: List[str] = []

    for left_video_path, right_video_path in pairs:
        report: AnalysisReport = use_case.execute(
            left_video_path=left_video_path,
            right_video_path=right_video_path,
            imu_path=imu_path,
        )
        output_file = outputs_dir / report.output_filename()
        output_file.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        generated_files.append(str(output_file))

    return BatchResponse(
        status="success",
        processed_reports=len(generated_files),
        output_files=generated_files,
        message=f"Batch completado en {datetime.utcnow().isoformat()}Z",
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/live/ui")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.post("/batch/run", response_model=BatchResponse)
def batch_run() -> BatchResponse:
    try:
        return run_batch()
    except Exception as exc:  # noqa: BLE001 - exponer error para diagnóstico en hackathon.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/batch/discover")
def batch_discover() -> dict:
    pairs, imu_path = discover_inputs(INPUTS_DIR)
    return {
        "video_pairs": [{"left": str(left), "right": str(right) if right else None} for left, right in pairs],
        "imu_path": str(imu_path) if imu_path else None,
    }


@app.get("/live/videos")
def live_videos() -> dict:
    return {"videos": live_streamer.available_videos(INPUTS_DIR)}


@app.get("/media/video/{filename}")
def serve_input_video(filename: str) -> FileResponse:
    path = _safe_input_file(filename, ".mp4")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/live/reports")
def list_productivity_reports() -> dict:
    reports = sorted(
        [p for p in OUTPUTS_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".json"],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return {"reports": [p.name for p in reports]}


@app.get("/live/report/{report_name}")
def get_productivity_report(report_name: str) -> dict:
    path = _safe_output_json(report_name)
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/live/status")
def live_status() -> dict:
    return live_streamer.snapshot()


@app.get("/live/stream")
def live_stream(video: Optional[str] = None) -> StreamingResponse:
    chosen_video = video
    if chosen_video is None:
        videos = live_streamer.available_videos(INPUTS_DIR)
        if not videos:
            raise HTTPException(status_code=404, detail="No hay videos .mp4 en /inputs.")
        chosen_video = videos[0]

    video_path = (INPUTS_DIR / chosen_video).resolve()
    if not video_path.exists() or video_path.suffix.lower() != ".mp4":
        raise HTTPException(status_code=404, detail="Video inválido.")
    if video_path.parent != INPUTS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Ruta de video no permitida.")

    try:
        return StreamingResponse(
            live_streamer.stream(video_path),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/live/ui", response_class=HTMLResponse)
def live_ui() -> HTMLResponse:
    html = r"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Dashboard — Mining Productivity</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }
    header { padding: 14px 20px; border-bottom: 1px solid #334155; background: #111827; }
    h1 { font-size: 1.15rem; margin: 0; font-weight: 600; }
    .hint { font-size: 0.8rem; color: #94a3b8; margin-top: 6px; }
    .wrap { display: flex; gap: 16px; padding: 16px; align-items: flex-start; flex-wrap: wrap; max-width: 1400px; margin: 0 auto; }
    .left { flex: 1 1 520px; min-width: 280px; }
    .right { flex: 0 1 420px; min-width: 300px; }
    .card { background: #111827; border: 1px solid #374151; border-radius: 12px; padding: 14px; margin-bottom: 12px; }
    video { width: 100%; border-radius: 8px; background: #000; border: 1px solid #475569; }
    label { display: block; font-size: 0.75rem; color: #94a3b8; margin-bottom: 4px; }
    select { width: 100%; padding: 8px 10px; border-radius: 8px; border: 1px solid #475569; background: #1f2937; color: #f8fafc; }
    .panel h2 { font-size: 0.95rem; margin: 0 0 10px 0; color: #38bdf8; }
    .big { font-size: 1.35rem; font-weight: 700; color: #f8fafc; }
    .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.85rem; }
    .muted { color: #94a3b8; font-size: 0.8rem; }
    .phase { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #1e293b; }
    .warn { color: #fbbf24; font-size: 0.8rem; }
    #timeLine { font-variant-numeric: tabular-nums; }
  </style>
</head>
<body>
  <header>
    <h1>Análisis asíncrono (video + métricas sincronizadas por tiempo)</h1>
    <div class="hint">Usa pausa, adelantar y retroceder en el reproductor. El panel derecho sigue el tiempo actual del video.</div>
  </header>
  <div class="wrap">
    <div class="left">
      <div class="card">
        <label>Reporte JSON (outputs/)</label>
        <select id="reportSelect"></select>
      </div>
      <div class="card">
        <label>Video a reproducir</label>
        <select id="videoSelect"></select>
      </div>
      <div class="card">
        <video id="v" controls playsinline preload="metadata"></video>
        <div class="muted" style="margin-top:8px;">Tiempo: <span id="timeLine">0.00</span> s</div>
      </div>
    </div>
    <div class="right panel">
      <div class="card">
        <h2>Camión y posición</h2>
        <div id="truckBlock" class="muted">Cargue un reporte.</div>
      </div>
      <div class="card">
        <h2>Ciclo actual (excavación → volcado)</h2>
        <div id="cycleBlock" class="muted">—</div>
      </div>
      <div class="card">
        <h2>Acumulado en este camión (hasta el instante del video)</h2>
        <div id="accBlock" class="muted">—</div>
        <div class="warn" style="margin-top:8px;">Toneladas estimadas = Σ (factor_llenado × toneladas_nominales_por_pase). Ajusta nominal en el reporte (metadata).</div>
      </div>
    </div>
  </div>
  <script>
    const reportSelect = document.getElementById("reportSelect");
    const videoSelect = document.getElementById("videoSelect");
    const video = document.getElementById("v");
    const timeLine = document.getElementById("timeLine");
    const truckBlock = document.getElementById("truckBlock");
    const cycleBlock = document.getElementById("cycleBlock");
    const accBlock = document.getElementById("accBlock");

    let report = null;
    let nominalTons = 95;

    function basename(p) {
      if (!p) return "";
      const s = String(p);
      const i = Math.max(s.lastIndexOf("/"), s.lastIndexOf("\\"));
      return i >= 0 ? s.slice(i + 1) : s;
    }

    function cycleAtTime(t) {
      if (!report || !report.cycles) return null;
      return report.cycles.find(function (c) { return t >= c.start_time_seconds && t < c.end_time_seconds; }) || null;
    }

    function completedPassesForTruck(truckId, t) {
      if (!report || !report.cycles || !truckId) return 0;
      return report.cycles.filter(function (c) {
        return c.truck_id === truckId && c.end_time_seconds <= t;
      }).length;
    }

    function tonsCompletedForTruck(truckId, t) {
      if (!report || !report.cycles || !truckId) return 0;
      return report.cycles
        .filter(function (c) { return c.truck_id === truckId && c.end_time_seconds <= t; })
        .reduce(function (sum, c) {
          return sum + (Number(c.estimated_fill_factor) || 0) * nominalTons;
        }, 0);
    }

    function fillVideoOptions() {
      videoSelect.innerHTML = "";
      if (!report || !report.video_info) return;
      const left = basename(report.video_info.left_video_path);
      const right = report.video_info.right_video_path ? basename(report.video_info.right_video_path) : null;
      if (left) {
        const o = document.createElement("option");
        o.value = left;
        o.textContent = "Izquierda — " + left;
        videoSelect.appendChild(o);
      }
      if (right) {
        const o = document.createElement("option");
        o.value = right;
        o.textContent = "Derecha — " + right;
        videoSelect.appendChild(o);
      }
    }

    function attachVideoSrc() {
      const name = videoSelect.value;
      if (!name) return;
      video.src = "/media/video/" + encodeURIComponent(name);
      video.load();
    }

    function renderPanel() {
      const t = video.currentTime || 0;
      timeLine.textContent = t.toFixed(2);

      if (!report) {
        truckBlock.textContent = "Sin reporte.";
        cycleBlock.textContent = "—";
        accBlock.textContent = "—";
        return;
      }

      const c = cycleAtTime(t);
      if (c) {
        const lane = c.loading_lane || "desconocido";
        truckBlock.innerHTML =
          "<div class='big'>" + (c.truck_id || "?") + "</div>" +
          "<div class='muted'>Lado de descarga (heurística): <strong>" + lane + "</strong></div>" +
          "<div class='muted'>Compara con la posición real en video para validar.</div>";
        const ph = c.phase_times || {};
        cycleBlock.innerHTML =
          "<div class='muted'>Ciclo #" + c.cycle_id + "</div>" +
          "<div class='big'>" + (Number(c.duration_seconds) || 0).toFixed(2) + " s</div>" +
          "<div class='muted'>Duración total del ciclo (modelo)</div>" +
          "<div style='margin-top:10px; font-size:0.85rem;'>Fases (s)</div>" +
          "<div class='phase'><span>Excavación</span><span>" + (ph.digging_seconds || 0) + "</span></div>" +
          "<div class='phase'><span>Giro cargado</span><span>" + (ph.swinging_loaded_seconds || 0) + "</span></div>" +
          "<div class='phase'><span>Volcado</span><span>" + (ph.dumping_seconds || 0) + "</span></div>" +
          "<div class='phase'><span>Giro vacío</span><span>" + (ph.swinging_empty_seconds || 0) + "</span></div>";

        const passes = completedPassesForTruck(c.truck_id, t);
        const tons = tonsCompletedForTruck(c.truck_id, t);
        const inProgress = t < c.end_time_seconds;
        accBlock.innerHTML =
          "<div><span class='muted'>Pases completados a este camión:</span> <strong>" + passes + "</strong></div>" +
          "<div style='margin-top:6px;'><span class='muted'>Toneladas estimadas (pases terminados):</span> <strong>" +
          tons.toFixed(1) + " t</strong></div>" +
          (inProgress ? "<div class='warn' style='margin-top:8px;'>Hay un pase en curso; al terminar el ciclo se sumará al acumulado.</div>" : "");
      } else {
        truckBlock.innerHTML = "<div class='muted'>Fuera de un ciclo detectado (entre ciclos o fuera de rango).</div>";
        cycleBlock.innerHTML = "<div class='muted'>—</div>";
        accBlock.innerHTML = "<div class='muted'>Avanza el video hasta un tramo con ciclo activo.</div>";
      }
    }

    async function loadReports() {
      const data = await fetch("/live/reports").then(function (r) { return r.json(); });
      reportSelect.innerHTML = "";
      (data.reports || []).forEach(function (name) {
        const o = document.createElement("option");
        o.value = name;
        o.textContent = name;
        reportSelect.appendChild(o);
      });
      if (reportSelect.options.length) await loadReport(reportSelect.value);
    }

    async function loadReport(name) {
      if (!name) return;
      report = await fetch("/live/report/" + encodeURIComponent(name)).then(function (r) { return r.json(); });
      nominalTons = (report.metadata && report.metadata.nominal_tons_per_bucket_fill) || 95;
      fillVideoOptions();
      if (videoSelect.options.length) {
        videoSelect.selectedIndex = 0;
        attachVideoSrc();
      }
      renderPanel();
    }

    reportSelect.addEventListener("change", function () { loadReport(reportSelect.value); });
    videoSelect.addEventListener("change", attachVideoSrc);
    video.addEventListener("timeupdate", renderPanel);
    video.addEventListener("seeked", renderPanel);
    video.addEventListener("loadedmetadata", renderPanel);

    loadReports();
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mining Productivity 2.0 - Batch + API")
    parser.add_argument("--batch", action="store_true", help="Ejecuta procesamiento batch y termina.")
    parser.add_argument("--host", default="0.0.0.0", help="Host FastAPI/uvicorn")
    parser.add_argument("--port", default=8000, type=int, help="Puerto FastAPI/uvicorn")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch:
        response = run_batch()
        print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
        return

    uvicorn.run("solution.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
