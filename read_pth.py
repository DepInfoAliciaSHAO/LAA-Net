import torch
import os

def read_pth_file(file_path):
    """
    Reads a .pth file (PyTorch checkpoint) and extracts its contents.

    Args:
        file_path (str): The path to the .pth file.

    Returns:
        dict: A dictionary containing the loaded data from the .pth file,
              or None if an error occurs.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return None

    try:
        # Load the checkpoint
        # map_location='cpu' ensures it loads on CPU even if saved on GPU,
        # preventing potential issues if you don't have a GPU or CUDA setup.
        checkpoint = torch.load(file_path, map_location='cpu')
        print(f"Successfully loaded .pth file: {file_path}")

        # Print common keys and their types/values
        print("\n--- Contents of the .pth file ---")
        if isinstance(checkpoint, dict):
            for key, value in checkpoint.items():
                print(f"Key: '{key}'")
                if key == 'epoch':
                    print(f"  Epoch: {value}")
                elif key == 'state_dict':
                    print(f"  Type: {type(value)} (PyTorch model state_dict)")
                    # Print the number of keys in the state_dict, but not the whole dict
                    print(f"  Number of keys in state_dict: {len(value)}")
                elif key == 'optimizer':
                    print(f"  Type: {type(value)} (PyTorch optimizer state)")
                    # Can print some optimizer info, e.g., learning rate from param_groups
                    if hasattr(value, 'param_groups') and value.param_groups:
                        print(f"  Learning Rate (first group): {value.param_groups[0].get('lr', 'N/A')}")
                elif isinstance(value, torch.Tensor):
                    print(f"  Type: {type(value)} (PyTorch Tensor)")
                    print(f"  Shape: {value.shape}, Dtype: {value.dtype}")
                elif isinstance(value, (list, dict)):
                    print(f"  Type: {type(value)}")
                    # For large lists/dicts, print only a snippet
                    if len(str(value)) > 200:
                        print(f"  Value (truncated): {str(value)[:200]}...")
                    else:
                        print(f"  Value: {value}")
                else:
                    print(f"  Type: {type(value)}")
                    print(f"  Value: {value}")
                print("-" * 20) # Separator for readability
        else:
            print(f"The loaded .pth file is not a dictionary. Its type is: {type(checkpoint)}")
            print(f"Value: {checkpoint}")

        return checkpoint

    except Exception as e:
        print(f"An error occurred while reading the .pth file: {e}")
        print("This often happens if the file is corrupted, not a PyTorch checkpoint, or incompatible with your PyTorch version.")
        return None

read_pth_file("/home/alicia/LAA-net/LAA-Net/pretrained_weights/19-06_2.pth")
