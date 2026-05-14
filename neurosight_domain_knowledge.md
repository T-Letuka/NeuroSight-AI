## Domain Knowledge

### Clinical Context

Radiology and Neuroradiology rely heavily on Magnetic Resonance Imaging (MRI) for detecting and characterizing brain tumors. Brain tumor diagnosis is not a simple image classification problem. It is a high-stakes clinical reasoning task involving anatomy, pathology, spatial localization, and uncertainty estimation simultaneously.

A radiologist reviewing a brain MRI must answer three critical questions:

1. What is the lesion?
Tumor classes such as Glioma, Meningioma, Schwannoma, and metastatic lesions have fundamentally different biological behavior, prognosis, and treatment pathways.

2. Where is the lesion located?
Tumor position directly affects neurosurgical planning, biopsy targeting, radiation therapy boundaries, and risk assessment. Spatial precision is clinically essential because even small localization errors can affect patient outcomes.

3. How reliable is the interpretation?
Tumors appear differently across MRI weightings and across patients. Radiologists synthesize information from multiple imaging modalities rather than relying on a single image representation.

NeuroSight AI is designed around this real-world diagnostic workflow rather than treating medical imaging as a generic computer vision task.

---

## MRI Weightings and Their Clinical Meaning

Brain MRI is not a single image. Different MRI sequences emphasize different tissue properties and reveal complementary diagnostic information.

### T1-Weighted MRI
T1-weighted imaging emphasizes anatomical structure.

- Fat and white matter appear relatively bright
- Water and cerebrospinal fluid appear dark
- Useful for identifying overall brain anatomy and structural boundaries

T1 imaging provides the anatomical baseline against which abnormalities are interpreted.

### T1C+ (Contrast-Enhanced T1)
T1C+ imaging is obtained after administering a Gadolinium-based contrast agent.

Tumors often disrupt the blood-brain barrier, allowing contrast material to accumulate in abnormal tissue. As a result:

- Active tumor regions become bright
- Lesion boundaries become more visible
- Vascularized tumor components are highlighted

This weighting is one of the most clinically important sequences for tumor detection and characterization.

### T2-Weighted MRI
T2-weighted imaging highlights water content and edema.

- Fluid appears bright
- Swelling surrounding tumors becomes highly visible
- Tumor infiltration beyond the core lesion can often be observed

T2 imaging is especially important for understanding tumor extent and surrounding tissue involvement.

---

## 2D Slice-Based Medical Imaging

A full MRI scan is inherently a 3D volumetric dataset composed of multiple axial, sagittal, or coronal slices.

NeuroSight AI operates on curated 2D slices rather than full volumetric MRI reconstruction. Each image in the dataset represents a diagnostically selected slice extracted from the larger MRI volume.

This simplifies the computational problem while preserving clinically relevant information for classification and localization tasks.

The approach reflects a realistic compromise frequently used in medical AI research:

- Lower computational cost than 3D CNNs
- Easier annotation pipelines
- Faster experimentation
- Reduced memory requirements
- Clinically meaningful diagnostic slices retained

---

## Tumor Localization and Point Annotations

Unlike many medical imaging datasets, this dataset includes radiologist-provided lesion center annotations.

Each abnormal MRI slice contains:

- A tumor class label
- An (x, y) coordinate marking the lesion center

The annotation represents the approximate spatial center of the visible lesion within a 512×512 image.

For normal scans, no lesion exists, so the coordinate defaults to the image center.

These point annotations provide weak but highly valuable spatial supervision.

---

## Multi-Task Learning in Medical Imaging

NeuroSight AI uses a multi-task convolutional neural network architecture that performs:

- Tumor classification
- Lesion localization

simultaneously.

This approach introduces an important inductive bias:

"The model cannot classify accurately without learning spatially meaningful tumor representations."

Traditional black-box classifiers may achieve high accuracy while focusing on irrelevant background artifacts or scanner-specific patterns. Multi-task learning reduces this risk by forcing the shared feature backbone to encode lesion-aware representations.

The localization branch acts as a spatial regularizer that improves interpretability and encourages clinically grounded feature learning.

---

## Explainability and Attention Validation

Most medical imaging AI systems rely on post-hoc visualization methods such as Grad-CAM to produce attention heatmaps.

However, heatmaps alone do not prove clinical validity.

NeuroSight AI introduces a quantitative explainability evaluation pipeline:

1. Generate Grad-CAM++ attention maps
2. Extract the attention centroid
3. Compare the centroid to the radiologist-provided lesion point
4. Compute Euclidean distance between the two

This transforms explainability from subjective visualization into a measurable spatial alignment problem.

The result is a clinically interpretable metric that evaluates whether the model is attending to the actual lesion rather than irrelevant image regions.

This addresses a major trust gap in current medical AI literature.

---

## Clinical Reporting and Deterministic AI

Instead of using a Large Language Model for report generation, NeuroSight AI uses a deterministic rule-based reporting engine.

This design choice is important in medical systems because it provides:

- Fully auditable outputs
- Zero hallucination risk
- Reproducibility
- Traceability from prediction to report
- Regulatory friendliness

Every generated finding is directly linked to model predictions and predefined templates.

This mirrors the structured reporting philosophy increasingly adopted in clinical radiology systems.

---

## Core Challenges in Brain Tumor Classification

Brain tumor MRI interpretation is inherently difficult because of:

### Inter-Patient Variability
The same tumor type can appear dramatically different across patients due to:

- Tumor size
- Anatomical location
- Age
- Disease stage
- Edema extent
- Imaging acquisition differences

### Cross-Weighting Appearance Changes
A tumor may appear:

- Subtle on T1
- Bright and well-defined on T1C+
- Diffuse on T2

The model must learn these modality-dependent visual transformations.

### Class Ambiguity
Some lesions do not fit clean diagnostic categories.

The presence of an “Other” category reflects real-world diagnostic uncertainty and acknowledges that radiological classification is not always binary or perfectly separable.

### Clinical Trust
High accuracy alone is insufficient in healthcare AI.

A clinically useful model must demonstrate:

- Spatial reasoning
- Transparent attention behavior
- Consistency
- Reproducibility
- Interpretability

NeuroSight AI is built around these principles rather than treating explainability as an afterthought.

---

## Research Significance

NeuroSight AI sits at the intersection of:

- Deep Learning
- Medical Imaging
- Computer Vision
- Explainable Artificial Intelligence

Its primary contribution is not only tumor classification performance, but the introduction of quantitatively validated attention alignment for clinically grounded medical AI systems.
