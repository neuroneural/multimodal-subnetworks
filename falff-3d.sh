#!/bin/bash

module load afni

subjects_file="/data/users2/jwardell1/nshor_docker/examples/fbirn-project/FBIRN/subjects.txt"

input_dir="/data/users2/jwardell1/nshor_docker/examples/fbirn-project/FBIRN"

output_dir="/data/users2/jwardell1/nshor_docker/examples/fbirn-project/FBIRN/falff_output"

low_freq_min=0.01
low_freq_max=0.1

mask_file="/data/users2/jwardell1/nshor_docker/examples/fbirn-project/FBIRN/group_mean_masks/groupmeanmask_3mm.nii"

SUBJECTS=("000300655084")

#while IFS= read -r subject_id; do
for subject_id in $SUBJECTS; do
    echo "Processing subject: $subject_id"

    input_file="$input_dir/$subject_id/ses_01/processed/${subject_id}_alff.nii.gz_LFF+tlrc.BRIK"

    if [ -f "$input_file" ]; then
        echo "File found for subject $subject_id: $input_file"

        subject_output_dir="$output_dir/$subject_id"
        mkdir -p $subject_output_dir

        output_bandpass_file="$subject_output_dir/falff_low_freq_${subject_id}.nii.gz"
        if [ -f "$output_bandpass_file" ]; then
            echo "Removing existing file: $output_bandpass_file"
            rm -f $output_bandpass_file
        fi

        output_total_file="$subject_output_dir/falff_total_freq_${subject_id}.nii.gz"
        if [ -f "$output_total_file" ]; then
            echo "Removing existing file: $output_total_file"
            rm -f $output_total_file
        fi

        output_falff_file="$subject_output_dir/falff_3d_${subject_id}.nii.gz"
        if [ -f "$output_falff_file" ]; then
            echo "Removing existing file: $output_falff_file"
            rm -f $output_falff_file
        fi

        echo "Extracting low-frequency fluctuations for $subject_id..."
        3dTproject -prefix $output_bandpass_file \
            -bandpass $low_freq_min $low_freq_max \
            -input $input_file \
            -mask $mask_file

        echo "Computing total power for $subject_id..."
        3dcalc -prefix $output_total_file \
            -a $input_file -expr 'a'

        echo "Computing fALFF for subject $subject_id..."
        3dcalc -prefix $output_falff_file \
            -a $output_bandpass_file \
            -b $output_total_file \
            -expr "a / b"

        echo "Applying mask to fALFF output..."
        output_falff_masked_file="$subject_output_dir/falff_3d_masked_${subject_id}.nii.gz"
        3dcalc -prefix $output_falff_masked_file \
            -a $output_falff_file \
            -b $mask_file \
            -expr "a * b"


        echo "fALFF calculation completed for subject $subject_id. Results saved in $subject_output_dir."
    else
        echo "File $input_file not found for subject $subject_id, skipping..."
    fi
#done < "$subjects_file"
done
