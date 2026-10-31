# Features and results review

## Assignment coverage

| Requirement | Implemented evidence |
|---|---|
| Pretrained segmentation for every image | MobileSAM masks for all 110 supplied images; model checksum, selection rules, visual QC sheet, and recorded corrective prompts in `outputs/`. Two boiling-water images have no visible egg mask. |
| Classical image processing | Eight controlled trials; final bright low-saturation seed selection plus local GrabCut and component filtering. No pretrained model is used by this method at inference time. |
| 4:1 development/evaluation split | 88 dev and 22 evaluation images. Deterministic grouping keeps strong pHash and geometrically verified ORB scene matches on one side of the split. |
| Quantitative evaluation | Per-image IoU, Dice, pixel precision/recall, object precision/recall at IoU 0.5, count error, and image-bootstrap confidence intervals. |
| Qualitative evaluation | A ranked error gallery and examples in the PDF showing missed dark eggs, reflections, and bright background false positives. |
| Video tracking | The entire 60-second source clip is processed with two IDs and an annotated MP4. Three uncertain reidentifications after occlusion are explicitly marked. |
| Reproducible submission | Setup commands, model/data attribution, saved parameters, experiment log, tests, PDF report, and AI-use disclosure. |

## Measured results

| Metric | Value |
|---|---:|
| Dataset images | 110 (108 `egg`, 2 `Unlabeled`) |
| Dev / evaluation images | 88 / 22 |
| Scene groups | 103 |
| Best global-threshold baseline, dev mean IoU | 0.359 |
| Best morphology/color threshold, dev mean IoU | 0.361 |
| Final graph-cut method, dev mean IoU | 0.461 |
| Final graph-cut method, evaluation mean IoU | 0.565 |
| Evaluation mean Dice | 0.648 |
| Evaluation pixel precision | 0.712 |
| Evaluation pixel recall | 0.677 |
| Evaluation object precision at IoU 0.5 | 0.644 |
| Evaluation object recall at IoU 0.5 | 0.600 |
| Evaluation mean absolute count error | 1.00 eggs/image |
| Evaluation mean IoU, bootstrap 95% interval | 0.416–0.705 |
| Video length / processed frames | 60 s / 1,440 |
| Recorded egg detections / tracked IDs | 1,351 / 2 |
| Flagged uncertain reidentifications | 3 |

The evaluation numbers compare classical masks with reviewed **MobileSAM pseudo-labels**, because the source dataset has classification folders rather than human segmentation masks. They quantify agreement with that reference, not absolute segmentation accuracy. The small evaluation set gives a wide interval. Early error inspection influenced method design, so an independent human-labeled set would be needed for a definitive claim.

[Watch the public tracking video](https://drive.google.com/file/d/1oLbEB3EDDzFkoCDZdwqK_lTCbqnHanMv/view?usp=drivesdk).

## What makes this stronger than a minimal submission

1. **The improvement is experimentally demonstrated.** The final method raises development mean IoU from 0.361 for the strongest threshold baseline to 0.461. The eight-trial log records parameters, reasons, outcomes, and discarded ideas, so the choice can be defended rather than described as a single lucky threshold.
2. **The reference labels are audited.** MobileSAM proposals are useful but imperfect. The project saves its checkpoint checksum and all masks, displays a QC contact sheet, records human-selected positive points and boxes for missed eggs, and states where the pseudo-labels remain uncertain. This avoids presenting automated masks as unquestioned ground truth.
3. **Leakage is addressed explicitly.** Similar images are grouped using pHash and ORB with geometric verification before the 4:1 split. The split file makes the partition auditable and repeatable.
4. **Failure modes are easy to inspect.** The interactive explorer sorts the held-out images by IoU and overlays true overlap, missed reference area, and extra prediction area. An interviewer can open a specific failure and ask for an algorithm change on the spot.
5. **Tracking reports uncertainty.** The two egg IDs stay distinct in the rendered video, while uncertain recovery after hand occlusion is marked with `?` and recorded in `tracks.csv`. This is more defensible than claiming perfect identity continuity for visually identical objects.
6. **The results are reproducible.** The repository includes setup commands, explicit model/data provenance, a report generator that reads measured files, a complete annotated video, and tests for metrics and split integrity.

## Interview walkthrough

Start with the dataset: one class, 110 images, multiple eggs and cluttered backgrounds, no pixel labels. Explain MobileSAM proposal filtering and why visual audit was necessary. Show the 88/22 scene-grouped split and trial chart. Explain why thresholding misses shadowed eggs and why GrabCut grows bright cores toward full boundaries. Open a low-IoU case in the explorer, then show the video around a hand occlusion and the uncertainty marker. Conclude with limitations and the next validation step: independent human instance masks.
