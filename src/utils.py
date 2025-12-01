import torch

def qnormalize(img, qmin=0.02, qmax=0.98):
    """Unit interval preprocessing with clipping"""
    qlow = torch.quantile(img, qmin)
    qhigh = torch.quantile(img, qmax)
    img = (img - qlow) / (qhigh - qlow)
    img = torch.clamp(img, 0, 1)  # Clip the values to be between 0 and 1
    return img


def crop_tensor(tensor, label, percentile=10):

    # Use torch.quantile instead of kthvalue for potentially faster operation
    threshold = torch.quantile(tensor.flatten(), percentile / 100)

    # Create a mask on the original device
    mask = tensor > threshold

    # If the mask is all False, return the original tensors
    if not torch.any(mask):
        return tensor, label

    # Find the bounding box (this part is already efficient)
    nonzero = torch.nonzero(mask)
    min_coords, _ = torch.min(nonzero, dim=0)
    max_coords, _ = torch.max(nonzero, dim=0)

    # Crop the original tensor and label using the bounding box
    slices = tuple(
        slice(min_coord.item(), max_coord.item() + 1)
        for min_coord, max_coord in zip(min_coords[2:], max_coords[2:])
    )
    cropped_tensor = tensor[(slice(None), slice(None)) + slices]
    cropped_label = label[(slice(None),) + slices]

    return cropped_tensor, cropped_label

class ProductScheduler:
    def __init__(self, scheduler1, scheduler2):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.initial_lr = scheduler1.optimizer.param_groups[0]["lr"]

    def step(self):
        lr1 = self.scheduler1.get_last_lr()[0]
        lr2 = self.scheduler2.get_last_lr()[0]
        combined_lr = lr1 * lr2
        self.scheduler1.step()
        self.scheduler2.step()
        self.scheduler1.optimizer.param_groups[0]["lr"] = combined_lr
        return combined_lr


def assert_equal_length(*args):
    """Enhanced version that shows which parameters have mismatched lengths"""
    if not all(len(arg) == len(args[0]) for arg in args):
        print("\nParameter length mismatch detected:")
        print("{:<15} {:<10}".format("Parameter", "Length"))
        print("-" * 25)
        param_names = [
            "cubesizes", "numcubes", "numvolumes", "weights",
            "databases", "collections", "epochs", 
            "prefetches", "attenuates"
        ]
        for name, arg in zip(param_names, args):
            print("{:<15} {:<10}".format(name, len(arg)))
        print()
        
        # Show first few elements of each list for comparison
        print("First few elements of each list:")
        for name, arg in zip(param_names, args):
            print(f"{name}: {arg[:3]}...")
        print()
        
        raise AssertionError("Not all parameter lists have the same length!")