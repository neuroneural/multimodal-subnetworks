import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from copy import deepcopy

# Define layers we want to prune
PRUNE_LAYERS = (nn.Linear, nn.Conv3d)

class MultimodalSNIPMask(nn.Module):
    """
    The parametrization module. It holds all masks for a specific layer
    and applies the correct one based on the active_mod_id.
    """
    def __init__(self, masks_dict):
        super().__init__()
        # We register masks as buffers so they move with .to(device) 
        # and don't get gradients.
        self.keys = sorted(list(masks_dict.keys()))
        for mod_id, mask_tensor in masks_dict.items():
            # Name format: mask_{mod_id}
            self.register_buffer(f'mask_{mod_id}', mask_tensor)
        
        # State to control which mask is active. None = Identity (no mask)
        self.active_mod_id = None

    def forward(self, weight):
        if self.active_mod_id is None:
            return weight
        
        # dynamic getattr to find the right buffer
        mask = getattr(self, f'mask_{self.active_mod_id}')
        
        # This multiplication is tracked by autograd. 
        # Backward pass will automatically mask the gradients!
        return weight * mask

class MultiMaskSNIPWrapper(nn.Module):
    def __init__(self, model, sparsity=0.9):
        super(MultiMaskSNIPWrapper, self).__init__()
        
        self.model = model
        self.sparsity = sparsity

        # Helper for SNIP calculation (frozen copy)
        self.cpu_model = deepcopy(model).to('cpu')
        self.cpu_optimizer = torch.optim.SGD(self.cpu_model.parameters(), 0.1)
        self.model_device = next(iter(model.parameters())).device
        
        # Flag to ensure we don't register twice
        self.masks_registered = False

    def register_multimodal_masks(self, modalities, input_data, labels):
        """
        1. Calculates SNIP scores.
        2. Organizes masks by layer.
        3. Registers the parametrization ONCE.
        """
        # 1. Generate Masks Dictionary: {mod_id: {layer_name: mask}}
        temp_mask_storage = {}
        unique_modalities = torch.unique(modalities).cpu().detach().tolist()
        
        for mod in unique_modalities:
            mask_idx = (modalities == mod)
            # print(f"mask_idx.shape: {mask_idx.shape}")
            mod_data = input_data[mask_idx]
            mod_labels = labels[mask_idx]
            
            print(f"Generating masks for modality: {mod}")
            batch = (mod_data, mod_labels)
            masks_by_name, _ = self.generate_mask_from_grad_scores(batch)
            temp_mask_storage[mod] = masks_by_name

        # 2. Register Parametrizations on the actual model
        print("Registering Parametrizations...")
        for name, module in self.model.named_modules():
            if isinstance(module, PRUNE_LAYERS):
                # Collect all masks for THIS specific module
                layer_masks = {}
                has_masks = False
                for mod, mask_dict in temp_mask_storage.items():
                    if name in mask_dict:
                        layer_masks[mod] = mask_dict[name]
                        has_masks = True
                
                if has_masks:
                    # Create the parametrization module
                    snip_mask_module = MultimodalSNIPMask(layer_masks)
                    # Register it! This moves module.weight -> module.parametrizations.weight.original
                    parametrize.register_parametrization(module, "weight", snip_mask_module)
        
        self.masks_registered = True
        print("Done.")

    def forward(self, input_data, modalities):
        if not self.masks_registered:
            print("WARNING: Masks not registered. Using full weights.")
            return self.model(input_data)

        model_device = next(iter(self.model.parameters())).device

        batch_size = input_data.shape[0]
        # Infer output shape or hardcode (e.g. [B, 1] for binary)
        final_outputs = torch.zeros(batch_size, 1, device=model_device) 
        
        unique_mods = torch.unique(modalities).cpu().tolist()

        for mod in unique_mods:
            mod_idx = (modalities == mod)
            sub_data = input_data[mod_idx]
            
            # A. Set the Active Modality
            # This updates the state of the MultimodalSNIPMask modules
            self._set_active_modality(mod)
            
            # B. Forward Pass 
            # The graph records: output = weight * mask_mod
            sub_output = self.model(sub_data)
            # print(f"Sub-output device: {sub_output.device}, final_outputs device: {final_outputs.device}")
            final_outputs[mod_idx] = sub_output
            
        # C. Reset to None (safety)
        self._set_active_modality(None)
        
        return final_outputs

    def _set_active_modality(self, mod_id):
        """Helper to iterate modules and update their mask state"""
        for module in self.model.modules():
            # Check if this module has our parametrization
            if parametrize.is_parametrized(module, "weight"):
                # We need to access the specific module we registered
                # It lives in module.parametrizations.weight[0] usually
                for param_module in module.parametrizations.weight:
                    if isinstance(param_module, MultimodalSNIPMask):
                        param_module.active_mod_id = mod_id

    # --- SNIP HELPERS (Updated with BCE fix) ---
    def generate_mask_from_grad_scores(self, batch):
        scores_dict = self._calculate_scores(batch)
        threshold = self._get_threshold_from_scores(scores_dict)
        masks = {}
        for name, values in scores_dict.items():
            masks[name] = (values > threshold).float().to(self.model_device)
        return masks, None
    
    def _calculate_scores(self, batch):
        data, labels = batch
        data, labels = data.to('cpu'), labels.to('cpu')
        
        self.cpu_model.to('cpu')
        self.cpu_model.train() # Ensure train mode
        self.cpu_optimizer.zero_grad()
        
        preds = self.cpu_model(data)
        
        # Binary Cross Entropy with Logits
        loss = F.binary_cross_entropy_with_logits(preds, labels.float())
        loss.backward()
        
        scores_d = {}
        for name, module in self.cpu_model.named_modules():
            if isinstance(module, PRUNE_LAYERS) and module.weight.grad is not None:
                scores_d[name] = (module.weight.grad * module.weight.data).abs()
        return scores_d

    def _get_threshold_from_scores(self, scores_d):
        global_scores = torch.cat([torch.flatten(x) for x in scores_d.values()])
        num_params_to_keep = int(len(global_scores) * (1.0 - self.sparsity))
        if num_params_to_keep < 1: num_params_to_keep = 1 # Safety
        topk_scores, _ = torch.topk(global_scores, num_params_to_keep, sorted=True)
        return topk_scores[-1]
    

    def prepare_for_loading(self, modalities_list):
        """
        Use this function to prepare the model for loading a state_dict with masks.
        Call this BEFORE loading a state_dict.
        It registers the parametrization structure with empty masks,
        so load_state_dict can successfully fill them with the saved values.

        Example:
        # 1. Initialize the fresh model and wrapper
        new_model = ResNet3D(...)
        wrapper = MultiMaskSNIPWrapper(new_model)

        # 2. Setup the structure (You must know which modalities you trained on!)
        # You don't need data, just the IDs (e.g., [0, 1])
        trained_modalities = [0, 1] 
        wrapper.prepare_for_loading(trained_modalities)

        # 3. Now load the weights
        state = torch.load("my_snip_model.pt")
        wrapper.load_state_dict(state)

        print("Model loaded successfully with masks restored!")

        """
        print(f"Restoring parametrization structure for modalities: {modalities_list}")
        
        # Create dummy masks just to initialize the architecture
        # The actual values will be overwritten by load_state_dict
        dummy_masks = {mod: torch.tensor([1.0]) for mod in modalities_list}
        
        for name, module in self.model.named_modules():
            if isinstance(module, PRUNE_LAYERS):
                # We register the parametrization with dummy data
                # Dimensions don't strictly matter for init, as load_state_dict 
                # will resize the buffers to match the saved file.
                snip_mask_module = MultimodalSNIPMask(dummy_masks)
                parametrize.register_parametrization(module, "weight", snip_mask_module)
        
        self.masks_registered = True