# Parkinson's Disease Progression Prediction Competition

## Overview

The goal of this competition is to predict the course of Parkinson's Disease (PD) using protein abundance data. The complete set of proteins involved in PD remains an open research question, and any proteins that have predictive value are likely worth investigating further.

The core dataset consists of protein abundance values derived from mass spectrometry readings of cerebrospinal fluid (CSF) samples collected from several hundred patients. Each patient contributed multiple samples over the course of several years while also taking assessments to measure PD severity.

This is a time-series code competition: participants will receive test set data and make predictions using Kaggle's time-series API.

## Dataset Description

### Protein Data Files

#### `train_peptides.csv`
Mass spectrometry data at the peptide level. Peptides are the component subunits of proteins.

| Column | Description |
|--------|-------------|
| `visit_id` | ID code for the visit |
| `visit_month` | The month of the visit, relative to the first visit by the patient |
| `patient_id` | An ID code for the patient |
| `UniProt` | The UniProt ID code for the associated protein; there are often several peptides per protein |
| `Peptide` | The sequence of amino acids included in the peptide; see [amino acid code table](https://en.wikipedia.org/wiki/Amino_acid#Table_of_standard_amino_acid_abbreviations_and_properties) for relevant codes; some rare annotations may not be included in the table |
| `PeptideAbundance` | The frequency of the amino acid in the sample |

> Note: The test set may include peptides not found in the train set.

#### `train_proteins.csv`
Protein expression frequencies aggregated from the peptide level data.

| Column | Description |
|--------|-------------|
| `visit_id` | ID code for the visit |
| `visit_month` | The month of the visit, relative to the first visit by the patient |
| `patient_id` | An ID code for the patient |
| `UniProt` | The UniProt ID code for the associated protein |
| `NPX` | Normalized protein expression; the frequency of the protein's occurrence in the sample |

> Note: 
> - The test set may include proteins not found in the train set
> - NPX may not have a 1:1 relationship with the component peptides as some proteins contain repeated copies of a given peptide

### Clinical Data Files

#### `train_clinical_data.csv`

| Column | Description |
|--------|-------------|
| `visit_id` | ID code for the visit |
| `visit_month` | The month of the visit, relative to the first visit by the patient |
| `patient_id` | An ID code for the patient |
| `updrs_1` | The patient's score for Part 1 of the Unified Parkinson's Disease Rating Scale (mood and behavior) |
| `updrs_2` | The patient's score for Part 2 of the UPDRS |
| `updrs_3` | The patient's score for Part 3 of the UPDRS (motor function) |
| `updrs_4` | The patient's score for Part 4 of the UPDRS |
| `upd23b_clinical_state_on_medication` | Whether or not the patient was taking medication such as Levodopa during the UPDRS assessment |

> Note: Higher UPDRS scores indicate more severe symptoms. Each sub-section covers a distinct category of symptoms.
>
> Medication is expected to mainly affect the scores for Part 3 (motor function). These medications wear off fairly quickly (on the order of one day), so it's common for patients to take the motor function exam twice in a single month, both with and without medication.

#### `supplemental_clinical_data.csv`
Clinical records without any associated CSF samples. This data is intended to provide additional context about the typical progression of Parkinson's Disease.

- Uses the same columns as `train_clinical_data.csv`

### API Test Files

#### `example_test_files/`
Data intended to illustrate how the API functions. Includes the same columns delivered by the API (i.e., no UPDRS columns).

#### `amp_pd_peptide/`
Files that enable the API. The API is expected to:
- Deliver all of the data (less than 1,000 additional patients) in under five minutes
- Reserve less than 0.5 GB of memory

A brief demonstration of what the API delivers is available in the competition materials.

#### `public_timeseries_testing_util.py`
An optional file intended to make it easier to run custom offline API tests. See the script's docstring for details.

## Competition Task

Participants are challenged to:

1. Analyze the relationship between protein abundance in CSF samples and Parkinson's Disease progression
2. Build predictive models that can forecast the course of the disease based on protein biomarkers
3. Identify proteins that may have significant predictive value for PD progression

Successful models may provide valuable insights into the biological mechanisms of Parkinson's Disease and potentially contribute to improved diagnostic and therapeutic approaches.
