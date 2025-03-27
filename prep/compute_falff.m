function compute_falff(dataset)
    % MATLAB script to compute fALFF for all subjects in a dataset
    % Usage: compute_falff('fbirn') % Replace 'fbirn' with your dataset name

    % Add REST toolbox to MATLAB path
    addpath(genpath('REST_V1.8_130615')); % Ensure REST toolbox is in your PWD

    % Define base directories
    base_dir = '/data/users2/jwardell1/nshor_docker/examples';
    output_dir = '/data/users2/jwardell1/multimodal-subnetworks/groupedData/fbirn_falff/images';

    % Define dataset-specific paths
    dataset_lower = lower(dataset); % Convert dataset name to lowercase
    dataset_upper = upper(dataset); % Convert dataset name to uppercase
    subjects_file = fullfile(base_dir, [dataset_lower '-project'], dataset_upper, 'subjects.txt');

    % Read subjects from the subjects.txt file
    subjects = readlines(subjects_file); % Read subjects into a string array
    subjects = subjects(~cellfun('isempty', subjects)); % Remove empty lines

    % Parameters for fALFF computation
    ASamplePeriod = 2; % TR (Repetition Time) in seconds
    LowCutoff = 0.01; % Lower frequency cutoff (e.g., 0.01 Hz)
    HighCutoff = 0.08; % Higher frequency cutoff (e.g., 0.08 Hz)
    TemporalMask = []; % No temporal masking (scrubbing)
    ScrubbingMethod = ''; % No scrubbing method needed
    CUTNUMBER = 10; % Default value for memory management

    % Mask file path
    mask_file = fullfile(base_dir, [dataset_lower '-project'], dataset_upper, 'group_mean_masks/groupmeanmask_3mm.nii');

    % Iterate over all subjects
    for i = 1:length(subjects)
        subject = subjects{i}; % Get current subject ID

        % Define output file path
        result_file = fullfile(output_dir, [subject '_fALFF.nii']);

        % Check if the output file already exists
        if exist(result_file, 'file')
            fprintf('Output file already exists for subject: %s. Skipping...\n', subject);
            continue; % Skip to the next subject
        end

        % Define fMRI file path
        fmri_file_gz = fullfile(base_dir, [dataset_lower '-project'], dataset_upper, subject, 'ses_01/processed', [subject '_rest.nii.gz']);
        fmri_file = fullfile(base_dir, [dataset_lower '-project'], dataset_upper, subject, 'ses_01/processed', [subject '_rest.nii']);

        % Unzip the fMRI file if it hasn't been unzipped already
        if ~exist(fmri_file, 'file')
            fprintf('Unzipping fMRI file for subject: %s\n', subject);
            gunzip(fmri_file_gz, fileparts(fmri_file_gz)); % Unzip the file
        end

        % Compute fALFF
        fprintf('Computing fALFF for subject: %s\n', subject);
        [fALFFBrain, Header] = f_alff(fmri_file, ASamplePeriod, HighCutoff, LowCutoff, mask_file, result_file, TemporalMask, ScrubbingMethod, [], CUTNUMBER);

        % Display completion message
        fprintf('fALFF computation completed for subject: %s\n', subject);
        fprintf('Result saved to: %s\n\n', result_file);
    end

    disp('fALFF computation completed for all subjects!');
end
