import numpy as np

def electrode_subset_batch(batch, max_n_electrodes, inplace=False):
    """
    Subset the electrodes to a maximum number of electrodes.

    Args:
        batch: dictionary from SingleSessionDataset with keys:
            'ieeg': {'data': torch.Tensor[batch_size, n_channels, n_samples], 'sampling_rate': int}
            'channels': {'id': np.array}
            'metadata': dict
        max_n_electrodes: int, the maximum number of electrodes to subset to
        inplace: boolean, if True, modify the batch dictionary in place

    Returns:
        batch: dictionary with subsetted data.
    """
    assert inplace, "electrode_subset_batch currently only supports inplace=True"

    electrode_data = batch['ieeg']['data']  # shape: (n_electrodes, n_samples)
    electrode_labels = batch['channels']['id']

    if len(electrode_labels) > max_n_electrodes:
        selected_indices = np.random.choice(len(electrode_labels), max_n_electrodes, replace=False)
        electrode_data = electrode_data[:, selected_indices, :]
        electrode_labels = electrode_labels[selected_indices]
    else:
        selected_indices = np.arange(len(electrode_labels))

    batch['ieeg']['data'] = electrode_data
    batch['channels']['id'] = electrode_labels

    return batch

