# Features and results review

## Assignment coverage

| Requirement | Implemented evidence |
|---|---|
| Pretrained segmentation for every image | MobileSAM masks for all 110 supplied images; model checksum, selection rules, visual QC sheet, and recorded corrective prompts in `outputs/`. Two boiling-water images have no visible egg mask. |
| Classical image processing | Nine controlled trials; final bright low-saturation seed selection plus local GrabCut, with a global-mask fallback only when graph cut finds nothing. No pretrained model is used by this method at inference time. |
| 4:1 development/evaluation split | Exactly 88 dev and 22 evaluation images. Source IDs, pHash, HSV color histograms and ORB geometry form 99 scene groups. An audit confirms no cross-split exact duplicate or pair meeting the grouping rules. |
| Quantitative evaluation | Per-image IoU, Dice, pixel precision/recall, pixel accuracy and balanced accuracy, object precision/recall at IoU 0.5, count error, and image-bootstrap confidence intervals. |
| Qualitative evaluation | A ranked error gallery and examples in the PDF showing missed dark eggs, reflections, and bright background false positives. |
| Video tracking | The entire 60-second source clip is processed with two IDs and an annotated MP4. Three uncertain reidentifications after occlusion are explicitly marked. |
| Reproducible submission | Setup commands, model/data attribution, saved parameters, experiment log, tests, PDF report, and AI-use disclosure. |

## Measured results

| Metric | Value |
|---|---:|
| Dataset images | 110 (108 `egg`, 2 `Unlabeled`) |
| Dev / evaluation images | 88 / 22 |
| Scene groups | 99 |
| Best global-threshold baseline, dev mean IoU | 0.356 |
| Best morphology/color threshold, dev mean IoU | 0.370 |
| Graph cut without fallback, dev mean IoU | 0.477 |
| Final graph cut with fallback, dev mean IoU | 0.494 |
| Graph cut without fallback, grouped evaluation mean IoU | 0.509 |
| Final method, grouped evaluation mean IoU | 0.532 |
| Evaluation mean Dice | 0.627 |
| Evaluation pixel precision | 0.684 |
| Evaluation pixel recall | 0.682 |
| Evaluation pixel accuracy | 0.910 |
| Evaluation balanced pixel accuracy | 0.813 |
| All-background pixel accuracy baseline | 0.872 |
| Evaluation object precision at IoU 0.5 | 0.598 |
| Evaluation object recall at IoU 0.5 | 0.554 |
| Evaluation mean absolute count error | 0.95 eggs/image |
| Evaluation mean IoU, bootstrap 95% interval | 0.394–0.663 |
| Video length / processed frames | 60 s / 1,440 |
| Recorded egg detections / tracked IDs | 1,351 / 2 |
| Flagged uncertain reidentifications | 3 |

The evaluation numbers compare classical masks with reviewed **MobileSAM pseudo-labels**, because the source dataset has classification folders rather than human segmentation masks. They quantify agreement with that reference, not absolute segmentation accuracy. Pixel accuracy is particularly misleading because most pixels are background. The stricter grouping replaced an earlier split after its results had been inspected, so the current evaluation is exploratory rather than a pristine holdout. The small set gives a wide interval. An independent capture-session test with human masks would be needed for a definitive claim.

[Watch the public tracking video](https://drive.google.com/file/d/1oLbEB3EDDzFkoCDZdwqK_lTCbqnHanMv/view?usp=drivesdk).

## What makes this stronger than a minimal submission

1. **The improvement is experimentally demonstrated.** The final method raises development mean IoU from 0.370 for the strongest threshold baseline to 0.494, and improves grouped evaluation IoU from 0.509 to 0.532 over graph cut alone. The nine-trial log records parameters, reasons, outcomes, and discarded ideas. OpenCV's graph-cut RNG is fixed so these scores reproduce.
2. **The reference labels are audited.** MobileSAM proposals are useful but imperfect. The project saves its checkpoint checksum and all masks, displays a QC contact sheet, records human-selected positive points and boxes for missed eggs, and states where the pseudo-labels remain uncertain. This avoids presenting automated masks as unquestioned ground truth.
3. **Leakage is addressed explicitly.** An audit caught related hand and pot captures crossing the first split. The revised exact 4:1 split groups matching names, adjacent source captures, perceptual hashes, HSV color histograms, and ORB geometry. The audit file verifies zero exact cross-split duplicates and zero pairs meeting the grouping rules, while naming the remaining limitations.
4. **Failure modes are easy to inspect.** The interactive explorer sorts the held-out images by IoU and overlays true overlap, missed reference area, and extra prediction area. An interviewer can open a specific failure and ask for an algorithm change on the spot.
5. **Tracking reports uncertainty.** The two egg IDs stay distinct in the rendered video, while uncertain recovery after hand occlusion is marked with `?` and recorded in `tracks.csv`. This is more defensible than claiming perfect identity continuity for visually identical objects.
6. **The results are reproducible.** The repository includes setup commands, explicit model/data provenance, a report generator that reads measured files, a complete annotated video, and tests for metrics and split integrity.

## Interview walkthrough

Start with the dataset: one class, 110 images, multiple eggs and cluttered backgrounds, no pixel labels. Explain MobileSAM proposal filtering and why visual audit was necessary. Show the 88/22 scene-grouped split and trial chart. Explain why thresholding misses shadowed eggs and why GrabCut grows bright cores toward full boundaries. Open a low-IoU case in the explorer, then show the video around a hand occlusion and the uncertainty marker. Conclude with limitations and the next validation step: independent human instance masks.
