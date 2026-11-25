# AI-Powered Smart Road Intelligence and Traffic Safety Monitoring System

An integrated computer-vision project that explores road-scene understanding, semantic segmentation, object detection, video tracking and traffic-safety analysis.

The project is being developed as a modular academic prototype. Its objective is to demonstrate how visual information from road images and video can be processed into useful scene, object and motion information. It is not a certified driver-assistance or traffic-control system.

## Project objectives

- Prepare and validate road-scene image, mask and video datasets.
- Build image-level classification and pixel-level semantic segmentation baselines.
- Investigate augmentation and transfer-learning approaches to improve model performance.
- Develop object detection, localisation and counting workflows.
- Process video to track road users and analyse their trajectories.
- Explore interpretable traffic-safety indicators using detections and motion information.
- Evaluate the implemented components and document their performance and limitations.

## Project overview

The intended workflow is:

1. Validate datasets and annotations.
2. Prepare images, masks, metadata and train/validation/test splits.
3. Train and evaluate image classification and segmentation models.
4. Extend the pipeline to object detection and video tracking.
5. Derive and evaluate traffic-safety indicators from the integrated outputs.

A concise description of the intended components and their relationships will be maintained separately in [architecture.md](architecture.md). This README focuses on understanding the project, preparing the environment, running the notebooks and interpreting their outputs.

## Repository structure

The following reflects the current repository layout on the `main` branch. The dataset folders contain the CamVid assets and generated artefacts used by the notebooks.

```text
AAI-521-Project/
├── LICENSE
├── README.md
├── datasets/
│   └── camvid/
│       ├── images/
│       ├── masks/
│       ├── bluechannel/
│       ├── label_colors.txt
│       ├── preprocessed/
│       │   ├── active_classes.csv
│       │   ├── all_source_classes.csv
│       │   ├── augmentation_config.json
│       │   ├── camvid_preprocessed_metadata.csv
│       │   ├── train_metadata.csv
│       │   ├── validation_metadata.csv
│       │   └── test_metadata.csv
│       ├── models/
│       └── outputs/
└── notebooks/
    ├── 01_dataset_exploration.ipynb
    ├── 02_data_preprocessing.ipynb
    ├── 03_data_augmentation.ipynb
    ├── 04_baseline_cnn.ipynb
    ├── 05_baseline_segmentation_cnn.ipynb
    ├── 06_improved_segmentation_unet_resnet18.ipynb
    └── 07_resnet18_transfer_learning.ipynb
```

The repository currently documents the CamVid image workflows. Additional datasets and notebooks for detection, tracking and safety analysis will be incorporated as those components are implemented. Do not assume that a folder or notebook not shown in the repository already exists.

## Datasets and data conventions

### CamVid

CamVid is the primary dataset for the implemented image-based workflows. It contains road-scene images and pixel-level semantic annotations. The source images inspected for this project are typically 960 × 720 pixels.

The main locations are:

| Path | Purpose |
|---|---|
| `datasets/camvid/images/` | Road-scene images |
| `datasets/camvid/masks/` | Original colour-coded semantic masks |
| `datasets/camvid/bluechannel/` | Converted class-index mask representation used by segmentation workflows |
| `datasets/camvid/label_colors.txt` | Source label colour information |
| `datasets/camvid/preprocessed/` | Metadata, active class list and augmentation configuration |
| `datasets/camvid/models/` | Saved model checkpoints |
| `datasets/camvid/outputs/` | Generated metrics, mappings and validation artefacts |

### Metadata and split files

The preprocessing workflow creates a master metadata file and three split-specific files:

- `camvid_preprocessed_metadata.csv`: master metadata for the CamVid records, including the split assignment.
- `train_metadata.csv`: training records.
- `validation_metadata.csv`: validation records.
- `test_metadata.csv`: held-out test records.
- `active_classes.csv`: the 29 active semantic classes used for image-level class-presence labels and downstream model configuration.
- `augmentation_config.json`: shared image size, batch size, seed and augmentation settings used by the CNN workflow.

The master metadata contains 701 records. The sequence-based classification split used in the baseline CNN is:

| Split | Images | Sequence allocation |
|---|---:|---|
| Training | 406 | Two sequences |
| Validation | 171 | One sequence |
| Testing | 124 | One sequence |
| **Total** | **701** | Four independent sequences |

Splitting at the sequence level avoids placing frames from the same sequence in more than one split, reducing temporal data leakage. It also means the test set represents one unseen sequence only. Class support can be highly imbalanced between splits; this limitation is important when interpreting per-class metrics.

### Original RGB masks, Bluechannel masks and ignored pixels

The original masks are colour-coded RGB annotations. The Bluechannel representation is a converted label representation consumed by segmentation workflows. They are not interchangeable: use the mask format and class-ID mapping expected by the specific notebook.

- The conversion maps recognised source colours to class IDs.
- `255` is reserved for void/ignored pixels in the segmentation target.
- Unknown source colours must not be silently assigned to an arbitrary class.
- Ignored pixels must be excluded from segmentation loss and evaluation metrics.

The mask validation and conversion process identified unmapped RGB colour values. The conversion workflow therefore does not produce a valid Bluechannel mask for every source record. In particular, `Seq05VD_f02610` is the known excluded record: it has an original image/mask record but was skipped during Bluechannel conversion because the mask contained colours outside the defined mapping. The resulting Bluechannel set has 700 valid pairs. This is a known data limitation, not a reason to fabricate a label mask.

### Image-level class presence

For the multi-label CNN, each image is represented by 29 binary class-presence targets derived from its annotations. The preprocessing workflow uses a minimum-pixel threshold when deciding whether a class is present in an image. The initial threshold discussed for this project is 20 pixels. Small objects below this threshold may not be represented as positive image-level labels. If this threshold or class mapping changes, regenerate the metadata and record the change before comparing experiments.

## Environment and execution

### Runtime

The notebooks were developed for Python 3 and a Google Colab workflow, with Google Drive used to access project data and save model outputs. A GPU runtime is recommended for training; dataset inspection and some preprocessing checks can run on CPU.

The notebooks use the PyTorch / torchvision stack together with common data and image libraries such as NumPy, pandas, Matplotlib, Pillow and OpenCV. Individual notebooks import the packages they require. There is currently no root-level `requirements.txt` in the repository, so do not assume that a pinned dependency file is available.

### Prepare the project in Google Drive

The current notebook configuration uses this project root:

```python
PROJECT_ROOT = Path("/content/drive/MyDrive/AAI-521/project")
```

To use the notebooks without changing their path configuration:

1. Clone or upload the repository into Google Drive.
2. Make the project available at `MyDrive/AAI-521/project` (the directory must contain the `datasets/` and `notebooks/` folders).
3. Open the required notebook in Colab and select a Python 3 runtime.
4. Enable a GPU runtime for model training where available.
5. Run the Drive mount cell when prompted.
6. Verify that the notebook's `PROJECT_ROOT` points to the directory containing `datasets/camvid/`.

If the repository is stored under a different folder name or location, update `PROJECT_ROOT` in the notebook before executing it. Do not change individual data paths independently unless necessary; they are derived from this root.

### Execution order

Run the notebooks in numerical order for the full CamVid image workflow:

| Order | Notebook | Purpose | Main dependency / output |
|---:|---|---|---|
| 1 | [01_dataset_exploration.ipynb](notebooks/01_dataset_exploration.ipynb) | Inspect images, masks, dimensions, class colours and data integrity | Raw CamVid images, masks and label definitions |
| 2 | [02_data_preprocessing.ipynb](notebooks/02_data_preprocessing.ipynb) | Validate image-mask records, create converted labels and metadata, and establish dataset splits | Generates metadata and preprocessing artefacts used downstream |
| 3 | [03_data_augmentation.ipynb](notebooks/03_data_augmentation.ipynb) | Define and inspect training-time augmentation and save shared configuration | Uses preprocessed metadata; produces/updates augmentation configuration |
| 4 | [04_baseline_cnn.ipynb](notebooks/04_baseline_cnn.ipynb) | Train and evaluate a multi-label CNN for image-level class presence | Uses split metadata, active classes and augmentation configuration |
| 5 | [05_baseline_segmentation_cnn.ipynb](notebooks/05_baseline_segmentation_cnn.ipynb) | Establish a baseline pixel-level semantic segmentation workflow | Uses image/Bluechannel mask pairs and class-ID conventions |
| 6 | [06_improved_segmentation_unet_resnet18.ipynb](notebooks/06_improved_segmentation_unet_resnet18.ipynb) | Explore an improved segmentation model using a U-Net decoder and ResNet18 encoder | Uses the sequence-based 700-pair segmentation split |
| 7 | [07_resnet18_transfer_learning.ipynb](notebooks/07_resnet18_transfer_learning.ipynb) | Investigate ResNet18-based transfer learning for the image-level classification task | Uses the prepared image-level metadata and class labels |

Notebooks 4 and 5 are separate modelling tasks: image-level multi-label classification and pixel-level semantic segmentation. Notebook 6 is an improved segmentation experiment, not a replacement for the baseline. Notebook 7 explores transfer learning for the classification workflow. Run each notebook from top to bottom in a fresh runtime unless its own instructions state otherwise.

**Important:** Notebooks 1–7 are present in the repository. Their presence does not guarantee that every cell has been executed successfully in a new environment. Check the notebook outputs and saved artefacts after each run.

## Implemented notebook guide

### 01 — Dataset exploration

**File:** `notebooks/01_dataset_exploration.ipynb`

This notebook establishes what is available in the raw CamVid data before preprocessing. It inspects representative images and masks, image dimensions, mask modes and unique RGB colours, and checks whether image/mask records can be paired consistently.

**How to read the results:** Review the sample visualisations together with the tabular integrity and colour summaries. Unknown colours or missing pairs require investigation before conversion or training. A mask that looks plausible visually can still contain colour values that are not represented in the mapping.

**Expected outcome:** A clear understanding of the source data, its annotation representation and any records requiring special handling.

### 02 — Data preprocessing

**File:** `notebooks/02_data_preprocessing.ipynb`

This notebook prepares the CamVid assets for modelling. It validates the source records, analyses annotation colours, maps recognised mask colours to class IDs, handles void/unknown pixels, derives image-level class-presence labels and writes the master and split-specific metadata.

The preprocessing outputs used by later notebooks are stored under `datasets/camvid/preprocessed/`. Related validation artefacts and class mappings are stored under `datasets/camvid/outputs/`.

**How to read the results:**

- Confirm that image and mask paths exist and that dimensions are compatible.
- Review unknown RGB colour reports and the number of ignored pixels.
- Confirm that the split counts reconcile with the master metadata.
- Distinguish the 701 source metadata records from the 700 valid Bluechannel segmentation pairs.
- Check that the active class list and class mapping are consistent with downstream model output dimensions.

Do not manually edit generated metadata to hide missing or invalid records. Correct the source or mapping decision in the preprocessing workflow and regenerate the outputs instead.

### 03 — Data augmentation

**File:** `notebooks/03_data_augmentation.ipynb`

This notebook defines and visualises transformations intended to increase training-data diversity. The shared configuration is saved in `datasets/camvid/preprocessed/augmentation_config.json`.

The committed configuration records:

| Setting | Value |
|---|---:|
| Random seed | 42 |
| Image size | 224 × 224 |
| Batch size | 32 |
| Horizontal flip probability | 0.50 |
| Brightness | 0.15 |
| Contrast | 0.15 |
| Saturation | 0.10 |
| Rotation | ±3 degrees |
| Translation | Up to 2% in each direction |
| Scale | 0.98–1.02 |
| Gaussian blur probability | 0.15 |
| Evaluation augmentation | Disabled |

The transformations are intended for training data. Evaluation/test data should not receive random augmentation. For segmentation, geometric transforms must be applied identically to the image and mask, and categorical masks must use nearest-neighbour interpolation so that new invalid class IDs are not introduced.

**How to read the results:** Compare original and transformed samples. Check that road-scene semantics remain plausible and that objects/labels remain aligned. Augmentation quality should ultimately be judged by validation performance, not by the number of transformations applied.

### 04 — Baseline CNN

**File:** `notebooks/04_baseline_cnn.ipynb`

This notebook builds a CNN for **multi-label image classification**. It predicts which of the 29 active classes are present in a road-scene image. It does not produce pixel-level segmentation masks.

The notebook loads the train, validation and test metadata, validates image paths and class columns, analyses class imbalance, defines and trains a CNN, saves the best checkpoint, and evaluates predictions on the held-out test split.

**Recorded configuration and preprocessing:**

- Random seed: 42.
- Device: CUDA when available, otherwise CPU.
- Input size: 224 × 224.
- Source image resize: 224 × 168, followed by 28-pixel padding at the top and bottom.
- Padding colour: `(124, 116, 104)`.
- Training augmentations: horizontal flip, colour jitter, small affine transformation and occasional Gaussian blur.
- Data-loader workers: 2 (reduce to 0 if the runtime has worker/process issues).
- Maximum epochs: 30.
- Early-stopping patience: 5.
- Best checkpoint: `datasets/camvid/models/baseline_cnn_best.pth`.

The notebook uses the augmentation configuration and active class list generated by the preceding notebooks. Confirm these files exist before starting training.

**Outputs and interpretation:**

The current repository includes:

- `datasets/camvid/outputs/baseline_cnn_overall_metrics.csv`
- `datasets/camvid/outputs/baseline_cnn_per_class_metrics.csv`
- `datasets/camvid/models/baseline_cnn_best.pth`

The recorded test metrics include micro precision, recall and F1, macro precision, recall and F1, test loss and exact-match accuracy. The committed reference run reports micro F1 of approximately 0.830 and macro F1 of approximately 0.484. Exact-match accuracy is 0.0, meaning none of the test images had every one of the 29 labels predicted exactly correctly. This is a strict metric in a multi-label task and should be read alongside per-class and micro/macro metrics.

Per-class results must be interpreted using positive and negative support. Some classes have no positive examples in the 124-image test split, so precision/recall or Average Precision may be undefined or uninformative for those classes. Other classes have very small support. In addition, the baseline run has classes for which recall is zero, showing that the model does not identify all classes effectively. The high micro score should therefore not be treated as uniformly strong performance across all 29 classes.

The sequence-based split is useful for avoiding sequence leakage, but the test set is only one unseen sequence. Results are an initial baseline, not a claim of broad real-world generalisation.

### 05 — Baseline segmentation CNN

**File:** `notebooks/05_baseline_segmentation_cnn.ipynb`

This notebook establishes the initial pixel-level semantic segmentation workflow. Unlike Notebook 04, its output is a class label for each pixel in the image.

The data pipeline uses image tensors with three channels and integer target masks. A validated batch example has image shape `[8, 3, 224, 224]`, logits shape `[8, 29, 224, 224]`, and target/prediction shapes `[8, 224, 224]`.

**Important interpretation points:**

- The 29 output channels represent the configured semantic classes.
- Target masks contain class IDs, not RGB colour triplets.
- The ignore label `255` must be excluded from loss and metric calculations.
- A successful forward pass verifies tensor compatibility; it does not prove that segmentation quality is good.
- Inspect input, ground-truth mask and predicted mask together.
- Use per-class IoU and mean IoU (mIoU), with the averaging and ignored-pixel handling stated explicitly. Small and infrequent classes can perform much worse than the aggregate score suggests.

The valid Bluechannel image-mask set has 700 pairs. The excluded `Seq05VD_f02610` record should not be counted as a valid converted-mask pair.

### 06 — Improved segmentation with U-Net and ResNet18

**File:** `notebooks/06_improved_segmentation_unet_resnet18.ipynb`

This notebook explores a stronger semantic segmentation architecture using a ResNet18 encoder with a U-Net-style decoder. It reuses the validated 700-image sequence-based segmentation split and the established class mapping.

The configuration in the notebook includes:

| Parameter | Value |
|---|---:|
| Seed | 42 |
| Image size | 320 × 240 (width × height) |
| Classes | 29 |
| Ignore index | 255 |
| Batch size | 8 |
| Data-loader workers | 2 |
| Maximum epochs | 40 |
| Early-stopping patience | 7 |
| Encoder learning rate | 0.0001 |
| Decoder learning rate | 0.001 |
| Weight decay | 0.0001 |
| Dice loss weight | 1.0 |
| Class-weight power | 0.5 |
| Class-weight bounds | 0.5–5.0 |

The configured best-checkpoint path is `datasets/camvid/models/improved_unet_resnet18/best_unet_resnet18.pth`.

**How to interpret results:** Compare this experiment with the baseline segmentation model using the same test split, class mapping, ignore-label treatment and metric definitions. Inspect per-class IoU and qualitative overlays, not only aggregate mIoU. The architecture and configuration alone do not establish that the improved model outperforms the baseline; use the actual completed training and evaluation outputs to make that conclusion.

### 07 — ResNet18 transfer learning

**File:** `notebooks/07_resnet18_transfer_learning.ipynb`

This notebook explores transfer learning using a ResNet18-based model for road-scene image classification. It should be interpreted as an image-level classification experiment and compared with Notebook 04 using the same sequence-based train/validation/test split and the same 29 class definitions.

When executing or reviewing this notebook, record the pretrained weight source, input normalisation, frozen/fine-tuned layers, optimiser and learning rates, stopping criterion, checkpoint path and evaluation metrics from its configuration and outputs. Compare micro and macro metrics and per-class support; do not claim improvement from training loss alone.

## Reading model outputs and evaluation results

Use metrics that match the task:

| Task | Useful metrics | Interpretation |
|---|---|---|
| Multi-label image classification | Micro/macro precision, recall and F1; per-class Average Precision; exact-match accuracy | Class-presence prediction quality across images |
| Semantic segmentation | Per-class IoU, mIoU, Dice | Pixel-level overlap between predicted and reference regions |
| Object detection (planned) | mAP, per-class precision and recall | Object localisation and classification |
| Object counting (planned) | MAE | Difference between predicted and reference counts |
| Multi-object tracking (planned) | IDF1, ID switches | Identity consistency across video frames |
| Runtime (planned) | FPS, latency | Processing throughput and responsiveness |

For classification, micro metrics aggregate decisions across labels and are influenced by frequent classes. Macro metrics average across classes and reveal whether rare classes are being neglected. Exact-match accuracy is especially strict for a 29-label prediction because every label for an image must be correct.

For segmentation, mIoU can conceal weak results on small classes. Always inspect per-class metrics and qualitative examples. State whether ignored pixels and absent classes are excluded from metric averaging.

Learning curves should be read together: decreasing training loss with increasing validation loss can indicate overfitting; high, flat losses can indicate underfitting or a data/label/optimisation problem. Use validation data for model selection and reserve the test split for final evaluation.

## Planned extensions

The following areas are part of the integrated project scope but are not yet represented by completed notebooks in the current repository:

### Object detection, segmentation and counting

Add the notebook link and implementation guide here when the detection workflow is committed. Document the dataset and annotation format, object classes, model and checkpoint, thresholds, evaluation metrics, output files and known limitations.

### Video processing, tracking and trajectory analysis

Add the notebook link and implementation guide here when the tracking workflow is committed. Document the video subset, frame rate, detection-to-tracker interface, tracker settings, tracking metrics and trajectory outputs.

### Traffic-safety analysis

Add the notebook link and implementation guide here when the safety-analysis workflow is committed. Define each indicator, coordinate system, calibration assumptions, temporal thresholds and validation approach. Pixel displacement must not be presented as physical distance or speed without suitable camera calibration and geometric assumptions.

When these components are added, update the execution sequence and describe the actual data passed between notebooks. Keep the project documentation consolidated rather than splitting it into separate workstream reports.

## Reproducibility checklist

Before running a notebook:

- [ ] Confirm the project root and dataset paths.
- [ ] Confirm the required metadata and mask files exist.
- [ ] Run the preceding data-preparation notebooks when their outputs are missing or stale.
- [ ] Check the active class list and mask encoding expected by the model.
- [ ] Review missing records, unknown colours and ignored-pixel counts.
- [ ] Confirm the runtime has sufficient memory and select GPU where appropriate.
- [ ] Record configuration changes and library versions.

After running a notebook:

- [ ] Confirm that training or evaluation completed without errors.
- [ ] Inspect learning curves and sample predictions.
- [ ] Confirm the checkpoint and metric files were written to the expected locations.
- [ ] Record test sample counts and metric definitions.
- [ ] Review per-class results and class support before drawing conclusions.

## Constraints and limitations

- CamVid is a finite dataset; results may not generalise to different roads, regions, cameras, weather or traffic conditions.
- The sequence-based split reduces temporal leakage but leaves only one sequence in the test set.
- Class imbalance and absent classes in a split can make some per-class metrics undefined or misleading.
- The Bluechannel conversion excludes one known record because of unmapped annotation colours.
- The 20-pixel image-level presence threshold can omit very small objects.
- Training time and feasible batch size depend on the available GPU and memory.
- Video tracking can be affected by occlusion, camera movement, frame gaps and missed detections.
- Distance and speed estimation require calibration and suitable geometric assumptions.
- This is an academic prototype; model outputs are not operational safety instructions.

## Contribution and documentation

- Keep notebook filenames and numbering consistent with the repository.
- Keep configuration values in clearly identified notebook cells.
- Preserve the established split and class mapping when comparing experiments.
- Do not silently change label encoding, ignore-index handling or metric definitions.
- Save important metrics and checkpoints under the existing `datasets/camvid/` structure or document any agreed change.
- Update this README when notebooks, data paths, configurations or outputs change.
- Maintain the system-level component description separately in [architecture.md](architecture.md).

