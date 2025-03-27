#!/bin/bash
# FBIRN Data Preparation Script - Safe Copy Only
# Usage: ./prepare_fbirn_data.sh

# Base directories
BASE_DIR="/data/users2/jwardell1/multimodal-subnetworks/groupedData"
SUBJECTS_FILE="subjects.txt"

# Source directories
SMRI_SOURCE_BASE="/data/qneuromark/Data/FBIRN/ZN_Neuromark/ZN_Prep_sMRI"
DWI_SOURCE_BASE="/data/qneuromark/Data/FBIRN/DTI_Data_BIDS/Raw_Data"

# Create target directories if they don't exist
mkdir -p "${BASE_DIR}/fbirn_smri/images"
mkdir -p "${BASE_DIR}/fbirn_dwi/images"

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
    SMRI_TARGET="${BASE_DIR}/fbirn_smri/images/${SUBJECT_ID}_smri.nii"
    
    if [[ -f "${SMRI_SOURCE}" ]]; then
        echo "  Copying sMRI..."
        cp -v "${SMRI_SOURCE}" "${SMRI_TARGET}"
    else
        echo "  WARNING: sMRI not found at ${SMRI_SOURCE}"
    fi
    
    # ===== DWI COPY =====
    # Find the scanVisit directory dynamically
    DWI_SOURCE_DIR=$(find "${DWI_SOURCE_BASE}/${SUBJECT_ID}" -type d -name "scanVisit__*" | head -1)
    
    if [[ -n "${DWI_SOURCE_DIR}" ]]; then
        DWI_SOURCE="${DWI_SOURCE_DIR}/dti/dti_FA/tbdti_FA.nii.gz"
        DWI_TARGET="${BASE_DIR}/fbirn_dwi/images/${SUBJECT_ID}_dwi.nii.gz"
        
        if [[ -f "${DWI_SOURCE}" ]]; then
            echo "  Copying DWI..."
            cp -v "${DWI_SOURCE}" "${DWI_TARGET}"
        else
            echo "  WARNING: DWI not found at ${DWI_SOURCE}"
        fi
    else
        echo "  WARNING: No scanVisit directory found for ${SUBJECT_ID}"
    fi
    
done < "${SUBJECTS_FILE}"

echo "Data copy complete!"
