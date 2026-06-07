from __future__ import annotations

import argparse
import cgi
import html
import json
import mimetypes
import shutil
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import YOLO_CONF_THRESHOLD
from src.knn_ocr import KNNOCR
from src.pipeline import process_vehicle_image
from src.yolo_detector import YOLOPlateDetector

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local web app for license plate validity recognition")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--yolo-model", default="models/yolov8m_plate_detector.pt")
    parser.add_argument("--plate-ocr-model", default="models/knn_plate_ocr.joblib")
    parser.add_argument("--validity-ocr-model", default="models/knn_validity_ocr.joblib")
    parser.add_argument("--conf-threshold", type=float, default=YOLO_CONF_THRESHOLD)
    parser.add_argument("--output-dir", default="outputs/web_app")
    return parser.parse_args()


def page(title: str, body: str) -> bytes:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Arial, Helvetica, sans-serif;
      background: #f4f6f8;
      color: #1f2933;
    }}
    body {{
      margin: 0;
      padding: 24px;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
    }}
    h1, h2, h3 {{
      margin: 0 0 12px;
    }}
    .panel {{
      background: #ffffff;
      border: 1px solid #d8dee6;
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
      gap: 24px;
    }}
    .debug-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
    }}
    .stage {{
      border: 1px solid #d8dee6;
      border-radius: 8px;
      background: #fbfcfd;
      padding: 12px;
    }}
    .stage img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 6px;
      background: #ffffff;
      border: 1px solid #eef1f4;
    }}
    .chars {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .char-box {{
      display: flex;
      flex-direction: column;
      align-items: center;
      background: #f8fafc;
      border: 1px solid #d8dee6;
      border-radius: 8px;
      padding: 8px;
      width: 80px;
    }}
    .char-box img {{
      max-width: 100%;
      height: auto;
      border-radius: 4px;
      margin-bottom: 8px;
    }}
    .char-label {{
      font-weight: bold;
      font-size: 20px;
      color: #1f6feb;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
    }}
    td, th {{
      border-bottom: 1px solid #e3e8ef;
      padding: 12px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      width: 180px;
      color: #4b5563;
    }}
    code, pre {{
      background: #f1f4f7;
      border-radius: 6px;
    }}
    pre {{
      overflow-x: auto;
      padding: 16px;
      font-size: 14px;
    }}
    .button {{
      background: #1f6feb;
      color: white;
      border: 0;
      border-radius: 8px;
      padding: 12px 24px;
      cursor: pointer;
      font-weight: 600;
      font-size: 16px;
      transition: background 0.2s;
    }}
    .button:hover {{
      background: #1858c7;
    }}
    .button:disabled {{
      background: #d8dee6;
      cursor: not-allowed;
    }}
    
    /* Upload Area Styles */
    #upload-container {{
      position: relative;
    }}
    #dropzone {{
      border: 2px dashed #d8dee6;
      border-radius: 12px;
      padding: 40px;
      text-align: center;
      background: #fbfcfd;
      cursor: pointer;
      transition: all 0.2s;
    }}
    #dropzone:hover, #dropzone.dragover {{
      border-color: #1f6feb;
      background: #f0f7ff;
    }}
    #dropzone p {{
      margin: 10px 0;
      font-size: 16px;
      color: #4b5563;
    }}
    #dropzone .icon {{
      font-size: 48px;
      color: #94a3b8;
      margin-bottom: 12px;
    }}
    #file-input {{
      display: none;
    }}
    #preview-container {{
      display: none;
      position: relative;
      margin-top: 10px;
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid #d8dee6;
    }}
    #preview-image {{
      width: 100%;
      display: block;
      max-height: 400px;
      object-fit: contain;
      background: #000;
    }}
    .remove-btn {{
      position: absolute;
      top: 12px;
      right: 12px;
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid #d8dee6;
      border-radius: 50%;
      width: 32px;
      height: 32px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      font-weight: bold;
      color: #ef4444;
      font-size: 18px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .remove-btn:hover {{
      background: #ffffff;
      color: #dc2626;
    }}
    .actions {{
      margin-top: 20px;
      display: flex;
      justify-content: center;
    }}
    
    .final-preview {{
      margin-bottom: 20px;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid #d8dee6;
    }}
    .final-preview img {{
      width: 100%;
      display: block;
      max-height: 300px;
      object-fit: contain;
      background: #f8fafc;
    }}

    .copy-btn {{
      float: right;
      background: #f1f4f7;
      border: 1px solid #d8dee6;
      border-radius: 4px;
      padding: 4px 8px;
      font-size: 12px;
      cursor: pointer;
      font-weight: bold;
      color: #4b5563;
    }}
    .copy-btn:hover {{
      background: #e3e8ef;
    }}

    .muted {{
      color: #667085;
      font-size: 14px;
    }}
  </style>
</head>
<body>
<main>
{body}
</main>

<script>
  function copyToClipboard(button, elementId) {{
    const text = document.getElementById(elementId).innerText;
    navigator.clipboard.writeText(text).then(() => {{
      const originalText = button.innerText;
      button.innerText = 'Copied!';
      setTimeout(() => {{ button.innerText = originalText; }}, 2000);
    }});
  }}

  const fileInput = document.getElementById('file-input');
  const dropzone = document.getElementById('dropzone');
  const previewContainer = document.getElementById('preview-container');
  const previewImage = document.getElementById('preview-image');
  const removeBtn = document.getElementById('remove-btn');
  const submitBtn = document.getElementById('submit-btn');

  if (dropzone) {{
    dropzone.addEventListener('click', () => fileInput.click());
    
    dropzone.addEventListener('dragover', (e) => {{
      e.preventDefault();
      dropzone.classList.add('dragover');
    }});
    
    dropzone.addEventListener('dragleave', () => {{
      dropzone.classList.remove('dragover');
    }});
    
    dropzone.addEventListener('drop', (e) => {{
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {{
        fileInput.files = e.dataTransfer.files;
        handleFiles(e.dataTransfer.files[0]);
      }}
    }});

    fileInput.addEventListener('change', (e) => {{
      if (e.target.files.length > 0) {{
        handleFiles(e.target.files[0]);
      }}
    }});

    function handleFiles(file) {{
      if (file && file.type.startsWith('image/')) {{
        const reader = new FileReader();
        reader.onload = (e) => {{
          previewImage.src = e.target.result;
          previewContainer.style.display = 'block';
          dropzone.style.display = 'none';
          if (submitBtn) submitBtn.disabled = false;
        }};
        reader.readAsDataURL(file);
      }}
    }}

    if (removeBtn) {{
      removeBtn.addEventListener('click', (e) => {{
        e.stopPropagation();
        fileInput.value = '';
        previewContainer.style.display = 'none';
        dropzone.style.display = 'block';
        if (submitBtn) submitBtn.disabled = true;
      }});
    }}
  }}
</script>
</body>
</html>""".encode("utf-8")


def file_link(path: Path) -> str:
    relative = path.resolve().relative_to(ROOT.resolve())
    return f"/file?path={quote(relative.as_posix())}"


def image_stage(label: str, path: Path) -> str:
    if not path.exists():
        return ""
    return f"""
    <div class="stage">
      <h3>{html.escape(label)}</h3>
      <a href="{file_link(path)}" target="_blank"><img src="{file_link(path)}" alt="{html.escape(label)}"></a>
    </div>
    """


def char_strip(label: str, directory: Path, sequence: str = "") -> str:
    images = sorted(directory.glob("char_*.jpg"))
    if not images:
        return ""
    
    items = []
    for i, path in enumerate(images):
        char = sequence[i] if i < len(sequence) else "?"
        items.append(f"""
        <div class="char-box">
          <a href="{file_link(path)}" target="_blank"><img src="{file_link(path)}" alt="{html.escape(path.name)}"></a>
          <div class="char-label">{html.escape(char)}</div>
        </div>
        """)
        
    return f"""
    <div class="panel">
      <h2>{html.escape(label)}</h2>
      <div class="chars">{"".join(items)}</div>
    </div>
    """


def result_table(result: dict) -> str:
    rows = [
        ("Plate number", result.get("plate_number")),
        ("Raw plate OCR", result.get("raw_plate_number")),
        ("Validity text", result.get("validity_text")),
        ("Raw validity OCR", result.get("raw_validity_text")),
        ("Validity month", result.get("validity_month")),
        ("Validity year", result.get("validity_year")),
        ("Validity status", result.get("validity_status")),
        ("Plate bbox", result.get("plate_bbox")),
        ("Detection confidence", result.get("detection_confidence")),
        ("Error", result.get("error")),
    ]
    body = "\n".join(
        f"<tr><th>{html.escape(label)}</th><td>{html.escape('None' if value is None else str(value))}</td></tr>"
        for label, value in rows
    )
    return f"<table>{body}</table>"


class PlateRecognitionApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = (ROOT / args.output_dir).resolve()
        self.upload_dir = self.output_dir / "uploads"
        self.debug_dir = self.output_dir / "debug"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.detector = YOLOPlateDetector(args.yolo_model, conf_threshold=args.conf_threshold)
        self.plate_ocr = KNNOCR.load(args.plate_ocr_model)
        self.validity_ocr = KNNOCR.load(args.validity_ocr_model)

    def render_home(self, message: str = "") -> bytes:
        msg = f'<div class="panel" style="border-color: #ef4444; color: #ef4444;"><strong>{html.escape(message)}</strong></div>' if message else ""
        body = f"""
        <div style="text-align: center; margin-bottom: 40px;">
          <h1>Indonesian License Plate Recognition</h1>
          <p class="muted">Upload a vehicle image to detect license plate and check its validity period.</p>
        </div>
        {msg}
        <div class="panel">
          <form method="post" action="/upload" enctype="multipart/form-data">
            <div id="upload-container">
              <div id="dropzone">
                <p><strong>Click to upload</strong> or drag and drop</p>
                <p class="muted">Supported formats: JPG, PNG, WEBP, BMP</p>
              </div>
              <div id="preview-container">
                <img id="preview-image" src="" alt="Preview">
                <div id="remove-btn" class="remove-btn" title="Remove image">X</div>
              </div>
              <input type="file" name="image" id="file-input" accept="image/*" required>
            </div>
            <div class="actions">
              <button class="button" type="submit" id="submit-btn" disabled>Run Recognition</button>
            </div>
          </form>
        </div>
        <div class="panel">
          <h2>System Information</h2>
          <table>
            <tr><th>YOLO detector</th><td><code>{html.escape(self.args.yolo_model)}</code></td></tr>
            <tr><th>Plate OCR KNN</th><td><code>{html.escape(self.args.plate_ocr_model)}</code></td></tr>
            <tr><th>Validity OCR KNN</th><td><code>{html.escape(self.args.validity_ocr_model)}</code></td></tr>
          </table>
        </div>
        """
        return page("License Plate Recognition", body)

    def render_result(self, result: dict, debug_root: Path, uploaded_path: Path) -> bytes:
        stages = [
            ("01 Original Image", debug_root / "01_original.jpg"),
            ("02 YOLO Bounding Box", debug_root / "02_detected_plate_bbox.jpg"),
            ("03 Cropped Plate", debug_root / "03_cropped_plate.jpg"),
            ("04 Grayscale Plate", debug_root / "04_grayscale_plate.jpg"),
            ("05 Otsu Binary Plate", debug_root / "05_binary_plate.jpg"),
            ("06 Cleaned Binary Plate", debug_root / "06_cleaned_binary_plate.jpg"),
            ("07 Upper Number Region", debug_root / "07_upper_region.jpg"),
            ("08 Lower Validity Region", debug_root / "08_lower_validity_region.jpg"),
        ]
        stage_html = "\n".join(image_stage(label, path) for label, path in stages)
        
        orig_img_path = debug_root / "01_original.jpg"
        orig_img_html = f'<div class="final-preview"><img src="{file_link(orig_img_path)}" alt="Original Image"></div>' if orig_img_path.exists() else ""

        body = f"""
        <h1>Recognition Result</h1>
        <p><a href="/" style="text-decoration: none; color: #1f6feb; font-weight: bold;">Upload another image</a></p>
        <div class="grid">
          <div class="panel">
            <h2>Final Result</h2>
            {orig_img_html}
            {result_table(result)}
          </div>
          <div class="panel">
            <button class="copy-btn" onclick="copyToClipboard(this, 'json-output')">Copy</button>
            <h2>JSON Output</h2>
            <pre id="json-output">{html.escape(json.dumps(result, indent=2, ensure_ascii=False))}</pre>
          </div>
        </div>
        <div class="panel">
          <h2>Processing Stages</h2>
          <div class="debug-grid">{stage_html}</div>
        </div>
        {char_strip("Upper Plate Character Segmentation", debug_root / "09_upper_segmented_chars", result.get("raw_plate_number", ""))}
        {char_strip("Lower Validity Digit Segmentation", debug_root / "10_lower_segmented_digits", result.get("raw_validity_text", ""))}
        """
        return page("Recognition Result", body)

    def handle_upload(self, handler: BaseHTTPRequestHandler) -> bytes:
        form = cgi.FieldStorage(
            fp=handler.rfile,
            headers=handler.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": handler.headers.get("Content-Type", ""),
            },
        )
        field = form["image"] if "image" in form else None
        if field is None or not getattr(field, "filename", ""):
            return self.render_home("No image file was uploaded.")

        suffix = Path(field.filename).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            return self.render_home(f"Unsupported image extension: {suffix}")

        run_id = time.strftime("%Y%m%d_%H%M%S")
        safe_name = f"{run_id}_{Path(field.filename).name.replace('/', '_').replace(' ', '_')}"
        uploaded_path = self.upload_dir / safe_name
        with uploaded_path.open("wb") as file:
            shutil.copyfileobj(field.file, file)

        debug_root = self.debug_dir / Path(safe_name).stem
        result = process_vehicle_image(
            str(uploaded_path),
            self.detector,
            self.plate_ocr,
            validity_ocr=self.validity_ocr,
            debug_dir=str(debug_root),
        )
        return self.render_result(result, debug_root, uploaded_path)


def make_handler(app: PlateRecognitionApp):
    class Handler(BaseHTTPRequestHandler):
        def send_html(self, payload: bytes, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_html(app.render_home())
                return
            if parsed.path == "/file":
                query = parse_qs(parsed.query)
                requested = unquote(query.get("path", [""])[0])
                path = (ROOT / requested).resolve()
                try:
                    path.relative_to(ROOT.resolve())
                except ValueError:
                    self.send_error(403)
                    return
                if not path.exists() or not path.is_file():
                    self.send_error(404)
                    return
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/upload":
                try:
                    self.send_html(app.handle_upload(self))
                except Exception as exc:
                    self.send_html(app.render_home(f"Recognition failed: {exc}"), status=500)
                return
            self.send_error(404)

    return Handler


def main() -> int:
    args = parse_args()
    app = PlateRecognitionApp(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    url = f"http://{args.host}:{args.port}"
    print(f"Serving license plate recognition app at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
