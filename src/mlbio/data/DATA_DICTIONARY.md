# WDBC data dictionary

## Why the dataset was created

The Wisconsin Diagnostic Breast Cancer (WDBC) dataset comes from research by William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian at the University of Wisconsin. The associated work was published in the early 1990s, and the dataset was donated to the UCI Machine Learning Repository in November 1995.

The original question was diagnostic: could a computer use quantitative descriptions of cell nuclei from a fine-needle aspirate (FNA) to distinguish a malignant breast mass from a benign one? An FNA uses a thin needle to remove cells from a mass for cytological examination. The source work developed an interactive computer system that extracted nuclear features from digitized FNA slides and used them for diagnostic classification.

The 1994 article describes a consecutive series of 569 patients used to develop the system. It also reports a separate set of 54 consecutive new patients used to test that system. The assignment table contains the 569-patient WDBC development dataset, not those 54 additional test cases. The evaluation in this assignment is therefore a new internal resampling analysis of the released development data, not a reproduction of the article's external test.

This context matters. The dataset concerns patients with a breast mass that had already been sampled. It was not collected to test population screening, cancer risk in asymptomatic people, prognosis, treatment response, or survival.

## From patient to table

The measurement chain is:

1. a breast mass is sampled by FNA;
2. cells are placed on a slide and digitized;
3. cell nuclei are identified in the image;
4. ten nuclear properties are computed;
5. each property is summarized as a mean, standard error, and "worst" value;
6. the resulting 30 numbers form one model input row;
7. the source diagnosis supplies the malignant/benign label.

Each row therefore represents one patient/sample, but the model never receives a patient, slide, or raw image directly. It receives derived numerical summaries produced after image acquisition, nucleus identification, and feature extraction. The supplied table is frozen from scikit-learn's `load_breast_cancer` copy of the UCI dataset; exact provenance and checksum are in `data/MANIFEST.json`.

## What the source does not establish

The released table does not include collection dates, institution or acquisition-batch identifiers, demographics, clinical histories, raw slides, segmentation masks, pathology reports, or prospective outcomes. It also does not expose how diagnosis was verified for each individual row. Consequently, this assignment can evaluate internal patient-level classification of the curated features, but it cannot independently audit label ascertainment, site effects, the upstream imaging pipeline, or performance in another clinical population.

## Metadata columns

| Column | Meaning |
|---|---|
| `sample_id` | Assignment-local stable row identifier; not the original clinical identifier |
| `diagnosis` | `malignant` or `benign`; encode malignant as 1 and benign as 0 |

## Feature columns

Ten properties of segmented cell nuclei are summarized in three ways, producing 30 features.

| Property | Interpretation |
|---|---|
| radius | Mean distance from the nuclear center to perimeter points |
| texture | Standard deviation of gray-scale values |
| perimeter | Nuclear boundary length |
| area | Nuclear area |
| smoothness | Local variation in radius lengths |
| compactness | $\mathrm{perimeter}^2 / \mathrm{area} - 1$ |
| concavity | Severity of concave portions of the contour |
| concave points | Number or extent of concave contour portions |
| symmetry | Nuclear shape symmetry |
| fractal dimension | Coastline-approximation measure of boundary complexity |

Prefixes identify the summary:

| Prefix | Meaning |
|---|---|
| `mean_` | Mean over nuclei represented in the sample |
| `*_error` | Standard-error summary distributed by the source dataset |
| `worst_` | Mean of the three largest values for that property |

Examples are `mean_radius`, `radius_error`, and `worst_radius`. Feature scales and physical interpretations differ; consult the UCI record for the source description. The dataset does not provide institution, acquisition batch, demographic variables, raw contours, or prospective clinical outcomes, so these cannot be audited or adjusted here.

## Required citation

Wolberg, W., Mangasarian, O., Street, N., and Street, W. Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. DOI: 10.24432/C5DW2B.

Street, W. N., Wolberg, W. H., and Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. *IS&T/SPIE Symposium on Electronic Imaging*, 1905, 861–870. DOI: 10.1117/12.148698.

Wolberg, W. H., Street, W. N., and Mangasarian, O. L. (1994). Machine learning techniques to diagnose breast cancer from image-processed nuclear features of fine needle aspirates. *Cancer Letters*, 77(2–3), 163–171. DOI: 10.1016/0304-3835(94)90099-X.
