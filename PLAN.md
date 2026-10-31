# Egg segmentation and tracking: implementation plan

## Input audit

The supplied local export is Roboflow `egg` v1: 110 classification images (77 train, 22 valid, 11 test), licensed CC BY 4.0. The assignment's alternative ZIP contains this version. The Roboflow v2 and v3 pages advertise 160 images, but the local dataset is v1. We use the supplied version and record that choice rather than silently claiming 160 images.

1. Audit image sizes, exact/near duplicates, scene families, and images with multiple eggs.
2. Make a deterministic 4:1 dev/eval split grouped by scene or near-duplicate clusters. Keep eval sealed during tuning.
3. Produce pretrained masks for every image. Persist model provenance and mask files. Review a sample and mark weak pseudo-labels.
4. Run classical methods as a controlled experiment: brightness threshold, color/contrast threshold, and a refined method with component filtering and morphology. Log parameters, dev metrics, and discarded ideas.
5. Freeze the selected algorithm. Report pixel and object metrics on eval, uncertainty intervals, and representative failure cases. Make a small HTML error explorer as an extra, interview-friendly deliverable.
6. Apply the final algorithm to the linked video. Track separate eggs over time with stable IDs, export an annotated MP4 and tracking diagnostics.
7. Write a PDF report, reproducible commands, AI-use disclosure, and data attribution. Verify code, figures, report layout, and outputs. Publish only after artifacts are reviewable and access is available.

## Evaluation guardrails

- Task 1 masks are pseudo ground truth, not human labels. Report agreement with them, not absolute segmentation accuracy.
- Do not tune on eval or on video output; video-specific adjustments must be documented separately.
- If the model or video cannot be obtained, leave those results explicitly incomplete rather than fabricate numbers or links.
