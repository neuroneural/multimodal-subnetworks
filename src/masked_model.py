import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from copy import deepcopy

# Define layers we want to prune
PRUNE_LAYERS = (nn.Linear, nn.Conv3d)

class MultimodalSNIPMask(nn.Module):
    """
    Parametrization module that applies a modality-specific binary mask to weights.
    Registered via torch.nn.utils.parametrize.
    """
    def __init__(self, masks_dict):
        super().__init__()
        # Register masks as buffers (saved in state_dict, no gradients)
        self.keys = sorted(list(masks_dict.keys()))
        for mod_id, mask_tensor in masks_dict.items():
            self.register_buffer(f'mask_{mod_id}', mask_tensor)
        
        # State to control which mask is active. None = Identity (no mask)
        self.active_mod_id = None

    def forward(self, weight):
        if self.active_mod_id is None:
            return weight
        # Dynamic retrieval of the buffer corresponding to the active modality
        mask = getattr(self, f'mask_{self.active_mod_id}')
        return weight * mask

class MultiMaskSNIPWrapper(nn.Module):
    """
    Wrapper that implements Multimodal SNIP pruning.
    It generates unique pruning masks for different modalities and applies them dynamically.
    """
    def __init__(self, model, sparsity=0.9):
        super(MultiMaskSNIPWrapper, self).__init__()
        self.model = model
        self.sparsity = sparsity
        self.masks_registered = False

    def register_multimodal_masks(self, modalities, input_data, labels):
        """
        Initialization step (Run ONCE before training):
        1. Creates a temporary GPU copy of the model for SNIP calculation.
        2. Generates masks for each modality found in the input.
        3. Registers the masks as parametrizations on the main model.
        """
        # Determine target device (GPU the main model is on)
        target_device = next(iter(self.model.parameters())).device

        # Create temp model on same device (prevents messing with main model gradients)
        temp_model = deepcopy(self.model).to(target_device)
        temp_optimizer = torch.optim.SGD(temp_model.parameters(), 0.1)

        # 1. Generate Masks Dictionary: {mod_id: {layer_name: mask}}
        temp_mask_storage = {}
        unique_modalities = torch.unique(modalities).cpu().detach().tolist()

        for mod in unique_modalities:
            print(f"Generating SNIP masks for modality: {mod}")
            mask_idx = (modalities == mod)
            batch = (input_data[mask_idx].to(target_device), labels[mask_idx].to(target_device))

            masks_by_name = self._generate_mask_from_grad_scores(
                temp_model, temp_optimizer, batch, target_device
            )
            temp_mask_storage[mod] = masks_by_name

        # 2. Register Parametrizations on the actual model
        print("Registering Parametrizations...")
        for name, module in self.model.named_modules():
            if isinstance(module, PRUNE_LAYERS):
                layer_masks = {}
                has_masks = False
                for mod, mask_dict in temp_mask_storage.items():
                    if name in mask_dict:
                        layer_masks[mod] = mask_dict[name]
                        has_masks = True

                if has_masks:
                    snip_mask_module = MultimodalSNIPMask(layer_masks)
                    parametrize.register_parametrization(module, "weight", snip_mask_module)

        self.masks_registered = True

        # Cleanup to free GPU memory
        del temp_model
        del temp_optimizer
        if target_device.type == 'cuda':
            torch.cuda.empty_cache()
        print("Mask initialization complete. Temporary GPU model cleared.")

    def forward(self, input_data, modalities):
        if not self.masks_registered:
            return self.model(input_data)
        
        device = next(iter(self.model.parameters())).device 
        input_device = input_data.device
        assert device == input_device, f"Input data and model must be on the same device, got model device {device} and input device {input_device}"

        batch_size = input_data.shape[0]
        
        # Output container (Assuming Binary Classification [B, 1])
        final_outputs = torch.zeros(batch_size, 1, device=device) 
        
        unique_mods = torch.unique(modalities).cpu().tolist()

        for mod in unique_mods:
            mod_idx = (modalities == mod)
            sub_data = input_data[mod_idx]

            # A. Set the Active Modality
            self._set_active_modality(mod)

            # B. Forward Pass (Autograd tracks: output = weight * mask_mod)
            sub_output = self.model(sub_data)
            final_outputs[mod_idx] = sub_output

        # C. Reset to Identity (No mask)
        self._set_active_modality(None)
        
        return final_outputs

    def _set_active_modality(self, mod_id):
        """Iterates over modules to toggle the active mask state."""
        for module in self.model.modules():
            if parametrize.is_parametrized(module, "weight"):
                for param_module in module.parametrizations.weight:
                    if isinstance(param_module, MultimodalSNIPMask):
                        param_module.active_mod_id = mod_id

    def prepare_for_loading(self, modalities_list):
        """
        Pre-initializes structure for loading state_dict.
        Call this BEFORE loading a checkpoint.
        """
        print(f"Restoring parametrization structure for modalities: {modalities_list}")
        for name, module in self.model.named_modules():
            # Only process if it's a target layer and NOT already parametrized
            if isinstance(module, PRUNE_LAYERS) and not parametrize.is_parametrized(module, "weight"):
                # Get the actual shape of the weights for this specific layer
                weight_shape = module.weight.shape
                # Create dummy masks matching that shape
                dummy_masks = {
                    mod: torch.ones(weight_shape) 
                    for mod in modalities_list
                }
                # Register the parametrization
                snip_mask_module = MultimodalSNIPMask(dummy_masks)
                parametrize.register_parametrization(module, "weight", snip_mask_module)
        
        self.masks_registered = True

    # --- INTERNAL SNIP HELPERS ---
    def _generate_mask_from_grad_scores(self, model, optimizer, batch, target_device):
        scores_dict = self._calculate_scores(model, optimizer, batch)
        threshold = self._get_threshold_from_scores(scores_dict)

        masks = {}
        for name, values in scores_dict.items():
            masks[name] = (values > threshold).float().to(target_device)
        return masks

    def _calculate_scores(self, model, optimizer, batch):
        data, labels = batch

        model.train()
        optimizer.zero_grad()

        preds = model(data)
        loss = F.binary_cross_entropy_with_logits(preds, labels.float())
        loss.backward()

        scores_d = {}
        for name, module in model.named_modules():
            if isinstance(module, PRUNE_LAYERS) and module.weight.grad is not None:
                # SNIP score = |grad * weight|
                scores_d[name] = (module.weight.grad * module.weight.data).abs()
        return scores_d

    def _get_threshold_from_scores(self, scores_d):
        global_scores = torch.cat([torch.flatten(x) for x in scores_d.values()])
        num_params_to_keep = int(len(global_scores) * (1.0 - self.sparsity))
        if num_params_to_keep < 1: num_params_to_keep = 1
        topk_scores, _ = torch.topk(global_scores, num_params_to_keep, sorted=True)
        return topk_scores[-1]

    def register_masks_from_dense_checkpoints(self, unimodal_checkpoints_dict, snip_data):
        """
        Compute per-modality SNIP masks from fully-trained dense unimodal checkpoints.

        Loads each modality's dense checkpoint into a temp model and runs SNIP on those
        mature weights. Masks reflect genuine task structure rather than random-init noise.

        Args:
            unimodal_checkpoints_dict: {mod_id (int) -> state_dict from dense unimodal model}
            snip_data: (input_data, modalities_tensor, labels) minibatch
        """
        input_data, modalities_tensor, labels = snip_data
        target_device = next(iter(self.model.parameters())).device
        temp_mask_storage = {}

        for mod_id, state_dict in unimodal_checkpoints_dict.items():
            int_mod_id = int(mod_id)
            print(f"Computing SNIP mask for modality {int_mod_id} from dense checkpoint...")

            temp_model = deepcopy(self.model).to(target_device)
            temp_optimizer = torch.optim.SGD(temp_model.parameters(), 0.1)

            # Strip 'model.' prefix if present; skip any stray parametrization keys
            stripped_state = {
                (k[len('model.'):] if k.startswith('model.') else k): v
                for k, v in state_dict.items()
                if 'parametrizations' not in k
            }
            temp_model.load_state_dict(stripped_state, strict=False)

            # Use only samples for this modality in the SNIP batch
            mask_idx = (modalities_tensor == int_mod_id)
            if mask_idx.sum() == 0:
                print(f"  Warning: no samples for modality {int_mod_id} in snip_data, using full batch")
                mask_idx = torch.ones(len(modalities_tensor), dtype=torch.bool)

            batch = (
                input_data[mask_idx].to(target_device),
                labels[mask_idx].to(target_device),
            )
            temp_mask_storage[int_mod_id] = self._generate_mask_from_grad_scores(
                temp_model, temp_optimizer, batch, target_device
            )

            del temp_model, temp_optimizer
            if target_device.type == 'cuda':
                torch.cuda.empty_cache()

        print("Registering warmup masks...")
        for name, module in self.model.named_modules():
            if isinstance(module, PRUNE_LAYERS):
                layer_masks = {
                    mod_id: mask_dict[name]
                    for mod_id, mask_dict in temp_mask_storage.items()
                    if name in mask_dict
                }
                if layer_masks:
                    parametrize.register_parametrization(
                        module, "weight", MultimodalSNIPMask(layer_masks)
                    )

        self.masks_registered = True
        print("Warmup mask registration complete.")

    def initialize_from_unimodal_models(self, unimodal_models_dict, snip_data=None):
        """
        Initialize the multimodal sparse model from trained unimodal sparse models.
        1. Extracts pretrained masks and weights from each unimodal model.
        2. Optionally intersects pretrained masks with fresh SNIP masks computed
           on the pretrained weights (not fixed-init weights).
        3. Registers the combined masks as parametrizations.
        4. Averages the pretrained weights through the combined masks.

        Args:
            unimodal_models_dict: {modality_id -> trained unimodal model state_dict}
            snip_data: optional tuple (input_data, modalities, labels). When provided,
                       pretrained masks are intersected with SNIP masks computed on
                       each modality's pretrained weights.
        """
        print("Initializing multimodal model from unimodal models...")

        mod_id_list = list(unimodal_models_dict.keys())

        # Step 1: Extract pretrained masks AND weights from each unimodal model.
        # Checkpoints use 'model.' prefix; strip it to match self.model.named_modules().
        modality_masks = {}
        modality_weights = {}
        for mod_id, state_dict in unimodal_models_dict.items():
            modality_masks[mod_id] = {}
            modality_weights[mod_id] = {}
            for key, value in state_dict.items():
                stripped = key[len('model.'):] if key.startswith('model.') else key
                if 'parametrizations.weight' in stripped and 'mask_' in stripped:
                    layer_name = stripped.split('.parametrizations.weight')[0]
                    modality_masks[mod_id][layer_name] = value
                elif 'parametrizations.weight.original' in stripped:
                    layer_name = stripped.split('.parametrizations.weight')[0]
                    modality_weights[mod_id][layer_name] = value

        # Step 2: Run SNIP on pretrained weights and intersect with pretrained masks.
        if snip_data is not None:
            print("Running SNIP on pretrained weights to refine pretrained masks...")
            input_data, modalities_tensor, labels = snip_data
            target_device = next(iter(self.model.parameters())).device
            unique_mod_ints = [int(m) for m in torch.unique(modalities_tensor).cpu().tolist()]

            for mod_idx, mod_id in enumerate(mod_id_list):
                print(f"  SNIP for modality: {mod_id}")

                temp_model = deepcopy(self.model).to(target_device)
                temp_optimizer = torch.optim.SGD(temp_model.parameters(), 0.1)
                pretrained_state = {
                    layer_name + '.weight': w
                    for layer_name, w in modality_weights.get(mod_id, {}).items()
                }
                if pretrained_state:
                    temp_model.load_state_dict(pretrained_state, strict=False)

                if mod_idx in unique_mod_ints:
                    mask_idx = (modalities_tensor == mod_idx)
                else:
                    mask_idx = torch.ones(len(modalities_tensor), dtype=torch.bool)

                batch = (input_data[mask_idx].to(target_device), labels[mask_idx].to(target_device))
                snip_masks_for_mod = self._generate_mask_from_grad_scores(
                    temp_model, temp_optimizer, batch, target_device
                )

                for layer_name, pretrained_mask in modality_masks.get(mod_id, {}).items():
                    if layer_name in snip_masks_for_mod:
                        modality_masks[mod_id][layer_name] = torch.logical_and(
                            pretrained_mask.bool(),
                            snip_masks_for_mod[layer_name].bool()
                        ).float()

                del temp_model, temp_optimizer
                if target_device.type == 'cuda':
                    torch.cuda.empty_cache()
            print("SNIP intersection complete.")

        # Step 3: Register the refined masks on the multimodal model.
        print("Registering modality-specific masks...")
        for name, module in self.model.named_modules():
            if isinstance(module, PRUNE_LAYERS):
                layer_masks = {}
                has_masks = False
                for idx, mod_id in enumerate(mod_id_list):
                    if name in modality_masks[mod_id]:
                        layer_masks[idx] = modality_masks[mod_id][name].to(module.weight.device)
                        has_masks = True
                if has_masks:
                    snip_mask_module = MultimodalSNIPMask(layer_masks)
                    parametrize.register_parametrization(module, "weight", snip_mask_module)

        self.masks_registered = True

        # Step 4: Average pretrained weights through the combined masks.
        print("Merging pretrained weights with smart averaging...")
        for name, module in self.model.named_modules():
            if isinstance(module, PRUNE_LAYERS) and parametrize.is_parametrized(module, "weight"):
                device = module.parametrizations.weight.original.data.device

                combined_mask = torch.zeros_like(module.parametrizations.weight.original.data)
                merged_weights = torch.zeros_like(module.parametrizations.weight.original.data)
                count_matrix = torch.zeros_like(module.parametrizations.weight.original.data)

                for mod_id in mod_id_list:
                    if name in modality_masks[mod_id]:
                        mod_mask = modality_masks[mod_id][name].to(device)
                        combined_mask = torch.logical_or(combined_mask.bool(), mod_mask.bool()).float()
                        w = modality_weights[mod_id].get(name, module.parametrizations.weight.original.data)
                        merged_weights += w.to(device) * mod_mask
                        count_matrix += mod_mask

                averaged_weights = torch.where(
                    count_matrix > 0,
                    merged_weights / count_matrix,
                    torch.zeros_like(merged_weights)
                )
                module.parametrizations.weight.original.data = averaged_weights * combined_mask
                print(f"Layer {name}: combined mask sparsity = {1 - combined_mask.mean().item():.2%}")

        print("Initialization complete!")