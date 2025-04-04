#!/bin/bash
# Dataset Data Preparation Script - Safe Copy Only
# Usage: ./prepare_dataset_data.sh <dataset_name>

# Check if dataset name argument is provided
if [[ $# -eq 0 ]]; then
    echo "Error: Please provide the dataset name as an argument"
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

DATASET_NAME="$1"
DATASET_NAME_LOWER=$(echo "$DATASET_NAME" | tr '[:upper:]' '[:lower:]')
DATASET_NAME_UPPER=$(echo "$DATASET_NAME" | tr '[:lower:]' '[:upper:]')

# Base directories
BASE_DIR="/data/users2/jwardell1/multimodal-subnetworks/groupedData"
SUBJECTS_FILE="subjects.txt"

# Source directories
SMRI_SOURCE_BASE="/data/qneuromark/Data/${DATASET_NAME_UPPER}/ZN_Neuromark/ZN_Prep_sMRI"
DWI_SOURCE_BASE="/data/qneuromark/Data/${DATASET_NAME_UPPER}/DTI_Data_BIDS/Raw_Data"

# Create target directories if they don't exist
mkdir -p "${BASE_DIR}/${DATASET_NAME_LOWER}_smri/images"
mkdir -p "${BASE_DIR}/${DATASET_NAME_LOWER}_dwi/images"

# Verify subjects file exists
if [[ ! -f "${SUBJECTS_FILE}" ]]; then
    echo "Error: Subjects file ${SUBJECTS_FILE} not found!"
    exit 1
fi

# Read subjects and process each one
while IFS= read -r SUBJECT_ID || [[ -n "$SUBJECT_ID" ]]; do
    # Skip empty lines
    [[ -z "$SUBJECT_ID" ]] && continue

    echo "Processing subject: ${SUBJECT_ID}"

    # ===== sMRI COPY =====
    SMRI_SOURCE="${SMRI_SOURCE_BASE}/${SUBJECT_ID}/VBM_modulated_SPM12_SM6.nii"
    SMRI_TARGET="${BASE_DIR}/${DATASET_NAME_LOWER}_smri/images/${SUBJECT_ID}_smri.nii"

    if [[ -f "${SMRI_SOURCE}" ]]; then
        echo "  Copying sMRI..."
        cp -v "${SMRI_SOURCE}" "${SMRI_TARGET}"
    else
        echo "  WARNING: sMRI not found at ${SMRI_SOURCE}"
    fi

    # ===== DWI COPY =====
    if [[ "${DATASET_NAME_LOWER}" == "fbirn" ]]; then
        # FBIRN-specific: Find the scanVisit directory dynamically
        DWI_SOURCE_DIR=$(find "${DWI_SOURCE_BASE}/${SUBJECT_ID}" -type d -name "scanVisit__*" | head -1)
        
        if [[ -n "${DWI_SOURCE_DIR}" ]]; then
            DWI_SOURCE="${DWI_SOURCE_DIR}/dti/dti_FA/tbdti_FA.nii.gz"
        else
            echo "  WARNING: No scanVisit directory found for ${SUBJECT_ID}"
            continue
        fi
    elif [[ "${DATASET_NAME_LOWER}" == "cobre" ]]; then
        # COBRE-specific: Find the Study directory dynamically
        STUDY_DIR=$(find "${DWI_SOURCE_BASE}/${SUBJECT_ID}" -type d -name "Study[0-9]*" | head -1)
        
        if [[ -n "${STUDY_DIR}" ]]; then
            DWI_SOURCE="${STUDY_DIR}/dwi_ori/dti_FA/tbdti_FA.nii.gz"
        else
            echo "  WARNING: No Study directory found for ${SUBJECT_ID}"
            continue
        fi
    else
        # Standard DWI path for non-FBIRN/non-COBRE datasets
        DWI_SOURCE="${DWI_SOURCE_BASE}/${SUBJECT_ID}/dti/dti_FA/tbdti_FA.nii.gz"
    fi

    DWI_TARGET="${BASE_DIR}/${DATASET_NAME_LOWER}_dwi/images/${SUBJECT_ID}_dwi.nii.gz"

    if [[ -f "${DWI_SOURCE}" ]]; then
        echo "  Copying DWI..."
        cp -v "${DWI_SOURCE}" "${DWI_TARGET}"
    else
        echo "  WARNING: DWI not found at ${DWI_SOURCE}"
    fi

done < "${SUBJECTS_FILE}"

echo "Data copy complete!"
