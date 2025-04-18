# wandb_utils.py
import wandb
import datetime
import torch # Needed for model watch potentially
import os # For checking API key environment variable

print("Wandb utilities module loaded.")

class WandbLogger:
    """A wrapper class for Weights & Biases logging."""

    def __init__(self, config):
        """
        Initializes the wandb run.

        Args:
            config (dict): The configuration dictionary.
        """
        self.config = config
        self.run = None
        self._project = config.get('wandb', {}).get('project', 'fmri-classification')
        self._entity = config.get('wandb', {}).get('entity', None) # Optional: your wandb username/team
        self._run_name_prefix = config.get('wandb', {}).get('run_name_prefix', config['model']['variant'])
        self._watch_freq = config.get('wandb', {}).get('watch_freq', 100) # Freq for wandb.watch

        # Check if WANDB_API_KEY is set, otherwise prompt login if needed
        # Wandb handles this internally mostly, but good to be aware
        api_key_set = os.getenv('WANDB_API_KEY') is not None
        # mode can be "online", "offline", "disabled"
        mode = config.get('wandb', {}).get('mode', "online" if api_key_set else None)

        try:
            self.run = wandb.init(
                project=self._project,
                entity=self._entity,
                config=config, # Log hyperparameters
                name=f'{self._run_name_prefix}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
                mode=mode, # Control online/offline/disabled status
                reinit=True
            )
            print(f"Wandb initialized successfully. Run name: {self.run.name}, Mode: {self.run.mode}")
            self._define_metrics() # Define metrics after successful init
        except Exception as e:
            print(f"Error initializing wandb: {e}. Logging will be disabled.")
            self.run = None # Ensure run is None if init fails

    @property
    def is_active(self):
        """Check if wandb run was successfully initialized."""
        return self.run is not None

    def _define_metrics(self):
        """Defines custom steps for metrics."""
        if not self.is_active: return
        try:
            self.run.define_metric("epoch") # Define epoch step
            # Link epoch-level metrics to the "epoch" step
            self.run.define_metric("train/loss_epoch", step_metric="epoch")
            self.run.define_metric("train/epoch_duration_sec", step_metric="epoch")
            self.run.define_metric("train/learning_rate", step_metric="epoch")
            self.run.define_metric("test/loss_epoch", step_metric="epoch")
            self.run.define_metric("test/accuracy", step_metric="epoch")
            # Batch-level metrics like "train/loss_interval" will use the default internal step
            print("Wandb metrics configured for epoch-based stepping.")
        except Exception as e:
             print(f"Warning: Could not define wandb metrics: {e}")


    def watch(self, model, log_type='all'):
        """
        Registers the model with wandb watch.

        Args:
            model (torch.nn.Module): The model to watch.
            log_type (str): Type of logging ('gradients', 'parameters', 'all', None).
        """
        if not self.is_active: return
        try:
            wandb.watch(model, log=log_type, log_freq=self._watch_freq)
            print(f"Wandb watching model (log='{log_type}', freq={self._watch_freq}).")
        except Exception as e:
             print(f"Warning: Could not initiate wandb.watch: {e}")

    def log_data_summary(self, data_summary_dict):
        """Logs dataset summary information (usually once)."""
        if not self.is_active: return
        try:
             self.run.log(data_summary_dict, step=0) # Log initial info at step 0
        except Exception as e:
             print(f"Warning: Failed to log data summary to wandb: {e}")

    def log_batch(self, metrics_dict, step):
        """
        Logs metrics associated with a training batch/step.

        Args:
            metrics_dict (dict): Dictionary of metric names and values.
            step (int): The global training step number.
        """
        if not self.is_active: return
        try:
            self.run.log(metrics_dict, step=step)
        except Exception as e:
             print(f"Warning: Failed to log batch metrics to wandb: {e}")

    def log_epoch(self, metrics_dict):
        """
        Logs metrics associated with an epoch end.
        Assumes 'epoch' key exists in metrics_dict for step inference.

        Args:
            metrics_dict (dict): Dictionary including 'epoch' and other epoch metrics.
        """
        if not self.is_active: return
        if 'epoch' not in metrics_dict:
             print("Warning: 'epoch' key missing in log_epoch dictionary. Cannot log to wandb.")
             return
        try:
            # No step argument needed here; uses 'epoch' value from dict
            self.run.log(metrics_dict)
        except Exception as e:
             print(f"Warning: Failed to log epoch metrics to wandb: {e}")


    def update_summary(self, key, value):
        """Updates a summary metric for the run."""
        if not self.is_active: return
        try:
             self.run.summary[key] = value
        except Exception as e:
             print(f"Warning: Failed to update wandb summary: {e}")


    def finish(self, exit_code=None):
        """Finishes the wandb run."""
        if not self.is_active: return
        try:
            self.run.finish(exit_code=exit_code)
            print("Wandb run finished.")
        except Exception as e:
             print(f"Warning: Error finishing wandb run: {e}")