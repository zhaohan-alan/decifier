import h5py

def print_hdf5_structure(file_path):
    """
    This function opens an HDF5 file, prints its structure (groups and datasets),
    and allows access to the data.

    Parameters:
    file_path (str): Path to the HDF5 file.
    """
    try:
        with h5py.File(file_path, 'r') as hdf_file:
            # Recursive function to print group structure
            def print_group(name, node):
                if isinstance(node, h5py.Dataset):
                    print(f"Dataset: {name}, Shape: {node.shape}, Dtype: {node.dtype}")
                elif isinstance(node, h5py.Group):
                    print(f"Group: {name}")
                    
            # Visit every node in the file and print details
            hdf_file.visititems(print_group)

            # Return the HDF5 file object to explore it further outside the function
            return hdf_file
    except Exception as e:
        print(f"Error: {e}")