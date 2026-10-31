# Egg segmentation and tracking

Reproducible solution to the CV Intern Task. It compares a classical egg segmentation method with a pretrained MobileSAM reference, logs development experiments, evaluates a held-out split, and tracks two eggs in the supplied video.

## Results

The supplied dataset is **Roboflow egg v1**, 110 images. It contains 108 images in the `egg` folder and 2 `Unlabeled` images. The linked v2/v3 pages list 160 images, but the supplied alternative ZIP is v1. The dataset has classification folders, **not hand-drawn segmentation masks**. Thus the reported scores measure agreement with reviewed MobileSAM pseudo-labels.

| Evaluation metric | Mean on 22 images |
|---|---:|
| IoU | 0.565 |
| Dice | 0.648 |
| Pixel precision | 0.712 |
| Pixel recall | 0.677 |
| Object precision at IoU 0.5 | 0.644 |
| Object recall at IoU 0.5 | 0.600 |

The selected graph-cut method improved development mean IoU from 0.361 for the strongest threshold baseline to 0.461. The image-bootstrap 95% interval for evaluation IoU is 0.416–0.705. See [`outputs/Egg_Segmentation_Tracking_Report.pdf`](outputs/Egg_Segmentation_Tracking_Report.pdf) for analysis and [`outputs/explorer/index.html`](outputs/explorer/index.html) for an interactive worst-case gallery.

## Setup

Use Python 3.12 on macOS/Linux. From the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
YOLO_CONFIG_DIR="$PWD/.cache" .venv/bin/python -c "from ultralytics import SAM; SAM('mobile_sam.pt')"
```

Place the provided `egg.v1i.folder` directory in the repository root. Its `README.dataset.txt` identifies the Roboflow version and CC BY 4.0 license. The MobileSAM checkpoint is deliberately excluded from Git; its SHA-256 is recorded in `outputs/reference_provenance.json` after inference.

To obtain the assignment video for local analysis:

```bash
mkdir -p data/video
.venv/bin/yt-dlp --remote-components ejs:github --extractor-args 'youtube:player_client=web,android' \
  -f 'best[ext=mp4][height<=720]/best[height<=720]' \
  -o 'data/video/source.%(ext)s' 'https://www.youtube.com/watch?v=Y4JyV94y2_A'
```

The source MP4 is excluded from Git. The tracker reads `data/video/source.mp4` and writes the annotated output.

The [annotated tracking video is publicly viewable on Google Drive](https://drive.google.com/file/d/1oLbEB3EDDzFkoCDZdwqK_lTCbqnHanMv/view?usp=drivesdk).

## Reproduce the experiments

```bash
YOLO_CONFIG_DIR="$PWD/.cache" .venv/bin/python scripts/generate_reference.py
YOLO_CONFIG_DIR="$PWD/.cache" .venv/bin/python scripts/repair_reference.py
.venv/bin/python scripts/run_experiments.py
.venv/bin/python scripts/track_video.py
MPLCONFIGDIR="$PWD/.cache/matplotlib" .venv/bin/python scripts/make_report.py
```

The first command generates automatic MobileSAM proposals and saves masks for all 110 images. The second applies recorded human-selected positive point and box prompts to visible missed eggs; it records every prompt in `outputs/reference_repairs.json` and creates a QC contact sheet. Two boiling-water frames have no visible egg mask. The experiment command uses a deterministic 88/22 split that keeps strong perceptual matches and ORB-verified scene matches together; it saves all trial decisions and per-image held-out metrics. The tracker writes a full 60-second MP4 and `outputs/tracks.csv`. To rebuild the report with the verified public video link, rerun the final command with `--video-url 'https://drive.google.com/file/d/1oLbEB3EDDzFkoCDZdwqK_lTCbqnHanMv/view?usp=drivesdk'`.

## Method and limitations

The final image method forms bright, low-saturation seeds, filters implausible components, and runs local GrabCut to grow the seeds to object boundaries. This uses only classical image processing. The video uses a stricter brightness threshold and a fixed table region because that scene contains hands, captions, and reflections. The tracker performs global minimum-distance association and maintains two IDs; uncertain reidentification after an occlusion is marked with `?` and logged in the CSV.

The reference masks are imperfect, especially where eggs touch similar-colored backgrounds or are partly obscured. The evaluation was exploratory: method design was informed by early inspection of errors, so independent human-labeled masks are needed for a definitive estimate. The fixed video region is tailored to this clip.

## Files

- `PLAN.md` — plan and evaluation guardrails written before implementation.
- `eggcv/core.py` — split, classical methods, metrics, and overlays.
- `scripts/generate_reference.py` — pretrained mask generation.
- `scripts/repair_reference.py` — recorded visual-QC prompts and final contact sheet.
- `scripts/run_experiments.py` — trial log, selection, evaluation, and error explorer.
- `scripts/track_video.py` — separate egg tracks and annotated video.
- `scripts/make_report.py` — PDF built from measured outputs.
- `outputs/` — masks, figures, logs, metrics, explorer, video, and report.

## Attribution and AI use

Dataset: [Fire, egg v1 on Roboflow Universe](https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1), CC BY 4.0. Pretrained model: [Ultralytics MobileSAM](https://docs.ultralytics.com/models/sam/). Source video: [Mocomi raw and boiled egg experiment](https://www.youtube.com/watch?v=Y4JyV94y2_A).

AI coding assistance was used for drafting code, debugging, and report wording. The methods, parameters, visual-QC prompts, trial log, and failure analysis are retained so the work can be explained and modified in an interview.
