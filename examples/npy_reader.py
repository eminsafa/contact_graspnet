import numpy as np

# Replace 'your_file.npy' with the actual path to your NPY file
file_path = '/home/juanhernandezvega/dev/contact_graspnet/test_data/0.npy'

# Load the NPY file
data = np.load(file_path, allow_pickle=True)

data.size