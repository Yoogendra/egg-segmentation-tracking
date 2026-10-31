"""Build the assignment PDF from measured artifacts, never hard-coded scores."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from PIL import Image as PILImage, ImageDraw
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def chart(experiments: list[dict], path: Path) -> None:
    names = [x["name"].replace(" graph cut", " GC") for x in experiments]
    values = [x["dev"]["iou"] for x in experiments]
    fig, ax = plt.subplots(figsize=(8, 2.7))
    bars = ax.bar(names, values, color=["#17a398" if x["decision"].startswith("kept") else "#8293a8" for x in experiments])
    ax.set_ylim(0, .55)
    ax.set_ylabel("Mean dev IoU")
    ax.grid(axis="y", alpha=.2)
    ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value+.01, f"{value:.3f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def video_strip(path: Path) -> None:
    video = cv2.VideoCapture(str(OUT / "tracked_eggs.mp4"))
    canvas = PILImage.new("RGB", (900, 195), "white")
    draw = ImageDraw.Draw(canvas)
    for i, second in enumerate([15, 24, 36]):
        video.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        ok, frame = video.read()
        if ok:
            im = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            im.thumbnail((290, 164))
            canvas.paste(im, (i * 300 + 5, 0))
            draw.text((i * 300 + 8, 168), f"{second} s", fill="black")
    video.release()
    canvas.save(path)


def make(video_url: str) -> Path:
    exp = json.loads((OUT / "experiment_log.json").read_text())
    scores = json.loads((OUT / "eval_summary.json").read_text())
    split = json.loads((OUT / "split.json").read_text())
    manifest = json.loads((OUT / "reference_manifest.json").read_text())
    per_image = sorted(csv.DictReader((OUT / "eval_per_image.csv").open()), key=lambda x: float(x["iou"]))
    tracks = list(csv.DictReader((OUT / "tracks.csv").open()))
    chart(exp, OUT / "experiment_chart.png")
    video_strip(OUT / "video_examples.jpg")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleX", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=colors.HexColor("#102a43"), spaceAfter=12))
    styles.add(ParagraphStyle(name="SubX", parent=styles["Heading2"], fontSize=13, leading=16, textColor=colors.HexColor("#176b77"), spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodyX", parent=styles["BodyText"], fontSize=9, leading=13, spaceAfter=6))
    styles.add(ParagraphStyle(name="SmallX", parent=styles["BodyText"], fontSize=8, leading=11, spaceAfter=4))
    styles.add(ParagraphStyle(name="CaptionX", parent=styles["BodyText"], fontSize=8, leading=11, textColor=colors.HexColor("#526477"), alignment=TA_CENTER, spaceAfter=8))
    path = OUT / "Egg_Segmentation_Tracking_Report.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=(612,792), rightMargin=44, leftMargin=44, topMargin=48, bottomMargin=45)
    story = []
    P = lambda text, style="BodyX": Paragraph(text, styles[style])
    story += [P("Egg segmentation and tracking", "TitleX"), P("Computer Vision Intern Task | reproducible experimental report", "BodyX")]
    story += [P("Executive summary", "SubX"), P(f"A pretrained MobileSAM model produced reference egg masks for all 110 images. A classical local graph-cut method, selected on the development set, achieved mean IoU <b>{scores['iou']:.3f}</b> and Dice <b>{scores['dice']:.3f}</b> on {scores['eval_count']} evaluation images. A two-ID tracker processed the complete 60-second reference video and marks uncertain reidentification after occlusion.")]
    story += [P("Dataset and split", "SubX"), P(f"Roboflow egg v1 classification export: 110 images (108 in the egg folder and 2 Unlabeled), 640 x 640, CC BY 4.0. The supplied ZIP is v1; the online v2/v3 pages list 160 images but were not the provided local export. A deterministic 4:1 split gives {scores['dev_count']} development and {scores['eval_count']} evaluation images. pHash and geometrically verified ORB matches form {split['group_count']} scene groups; all members of a group remain on one side of the split.")]
    sample_paths = [ROOT / manifest[i]["image"] for i in [2,5,14,57,86]]
    story += [Table([[Image(str(p),width=93,height=93) for p in sample_paths]],colWidths=[102]*5), P("The provided images span hands, ash, cookware, fabric and patterned backgrounds.","CaptionX")]
    story += [P("Task 1 - pretrained reference segmentation", "SubX"), P(f"MobileSAM automatic proposals at 384 px were filtered by area, shape, border contact and overlap. Visual audit found missed eggs, so {sum('human_prompted' in x for x in manifest)} images received human-selected positive prompts; four difficult images used boxes. Prompts were chosen from the image, independent of classical predictions. Two boiling-water images have no visible egg mask. All masks, prompt coordinates and model checksum are saved. These are <b>pseudo-labels</b>, not human ground truth; agreement with them does not prove absolute accuracy.")]
    story += [P("Task 2 - classical method", "SubX"), P("The final method converts each frame to grayscale and HSV, selects bright low-saturation seed regions, removes tiny noise, filters implausible components by shape, and runs local GrabCut around each seed. GrabCut fits foreground/background color models and expands a seed toward the full egg boundary. The output is a union of candidate masks. The video uses a documented stricter brightness setting and table ROI; it does not use MobileSAM at runtime.")]
    story += [PageBreak(), P("Experiment log", "TitleX"), P("Each trial used the same 88-image development set. Parameters and reasons are in outputs/experiment_log.json; the table below gives the decision path. The selection rule was highest mean dev IoU.")]
    table_data = [["Trial", "Change / reason", "Dev IoU", "Decision"]]
    for x in exp:
        table_data.append([x["name"], x["reason"], f"{x['dev']['iou']:.3f}", "Keep" if x["decision"].startswith("kept") else "Drop"])
    table = Table([[P(str(v),"SmallX") for v in row] for row in table_data], colWidths=[112,255,65,62], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dbe9ed")),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#c3d0d8")),("VALIGN",(0,0),(-1,-1),"TOP"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f6f9fb")]),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    story += [table, Spacer(1,8), Image(str(OUT/"experiment_chart.png"),width=500,height=169)]
    story += [P("Task 3 - held-out evaluation", "SubX"), P(f"Mean IoU {scores['iou']:.3f} (image-bootstrap 95% interval {scores['iou_95_ci'][0]:.3f}-{scores['iou_95_ci'][1]:.3f}); Dice {scores['dice']:.3f}; pixel precision {scores['precision']:.3f}; pixel recall {scores['recall']:.3f}. Object matching at IoU &ge; 0.5 yields precision {scores['object_precision']:.3f} and recall {scores['object_recall']:.3f}; mean absolute count error {scores['count_error']:.2f}. Metrics are per-image macro means; the bootstrap interval reflects small-sample variation, not label uncertainty.")]
    story += [PageBreak(), P("Qualitative error analysis", "TitleX"), P("Green = overlap; red = MobileSAM only; blue = classical method only. Samples are the lowest-IoU evaluation cases, chosen by a fixed ranking. Open outputs/explorer/index.html for the full interactive gallery.")]
    pictures = []
    captions = []
    for row in per_image[:3]:
        pictures.append(Image(str(OUT / "explorer" / f"{row['key']}.jpg"),width=160,height=160))
        captions.append(P(f"{row['key']}<br/>IoU {float(row['iou']):.3f}","CaptionX"))
    story += [Table([pictures,captions],colWidths=[170]*3), Spacer(1,12)]
    story += [P("The first case selects bright tissue while missing the egg. The second is an Unlabeled boiling-water frame: a reflection is falsely predicted as an egg. The third misses a shadowed egg entirely. These cases expose different failure mechanisms rather than one generic threshold error.")]
    story += [P("Observed gaps", "SubX"), P("Dark or shadowed eggs can fail the bright-seed step. Bright cookware, tissue and reflections can become false positives. GrabCut can absorb a similar-colored background or split an egg occluded by a rail. The reference pipeline itself misses partially submerged eggs and sometimes selects non-egg proposals; the saved QC sheet and manual prompt log make this visible. Next improvements: human instance masks for a representative subset, illumination normalization, temporal consistency for video, and a dedicated egg detector to provide proposals.")]
    story += [PageBreak(), P("Video tracking and reproducibility", "TitleX"), P(f"Task 4 uses a table ROI and a stricter brightness seed to suppress hands, captions and reflections. Connected components are assigned to two persistent tracks with global minimum-distance matching; a frame gap over 24 frames or a jump over 100 px is marked uncertain. Across 1,440 video frames, the output contains {len(tracks)} egg detections and {sum(x['uncertain_reid']=='True' for x in tracks)} uncertain reidentification events. Identity during hand occlusion cannot be established confidently from two visually similar eggs; these events are explicitly signalled rather than silently relabelled.")]
    story += [Image(str(OUT/"video_examples.jpg"),width=510,height=110), P("Three frames from the annotated output. IDs remain distinct; an uncertain recovery is marked with ?.","CaptionX")]
    story += [P("Tracking video", "SubX")]
    if video_url:
        story += [P(f"Public Google Drive video: <link href='{video_url}' color='blue'>{video_url}</link>")]
    else:
        story += [P("Local review video: outputs/tracked_eggs.mp4. Public Drive publication is pending approval.")]
    story += [P("Reproduce", "SubX"), P("Create a Python 3.12 environment, install requirements.txt, keep the supplied egg.v1i.folder at the repository root, and run scripts/generate_reference.py, scripts/repair_reference.py, scripts/run_experiments.py, scripts/track_video.py, and scripts/make_report.py in that order. See README.md for exact commands. The MobileSAM checkpoint is downloaded separately and identified by checksum in outputs/reference_provenance.json.")]
    story += [P("Attribution and AI use", "SubX"), P("Dataset: Fire, egg v1, Roboflow Universe, CC BY 4.0, https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1 . Model: Ultralytics MobileSAM, https://docs.ultralytics.com/models/sam/ . Source video: Mocomi, https://www.youtube.com/watch?v=Y4JyV94y2_A . AI coding assistance was used for code drafting, debugging, and report wording; the workflow, visual QC, method comparison, and interpretation are documented for independent review.")]
    story += [P("Limitations", "SubX"), P("The dataset contains only classification folders, not segmentation annotations. The same images were inspected during development, so the reported agreement is exploratory and should be confirmed with independent human-labelled masks. Scene grouping reduces but cannot eliminate leakage from visually related images. The video-specific ROI is tailored to this clip and should not be assumed to transfer to other scenes.")]
    def footer(canvas, doc):
        canvas.saveState();canvas.setFont("Helvetica",8);canvas.setFillColor(colors.HexColor("#718096"));canvas.drawString(44,25,"Egg segmentation and tracking | reproducible assignment");canvas.drawRightString(568,25,str(doc.page));canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-url",default="")
    args=parser.parse_args()
    print(make(args.video_url))
