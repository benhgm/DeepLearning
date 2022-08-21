from torch.utils.data import Dataset


class UrbanSoundDataset(Dataset):
    def __init__(self, annotations_file, audio_dir):
        """
        Constructor

        Args:
            annotations_file (str): Path to the file containing all the annotations
            audio_dir (str): Path to the directory containing the audio data
        """

    def __len__(self):
        """
        Define how to calculate the total number of samples in the dataset
        """
        pass

    def __getitem__(self, index):
        """
        Define the method to get an item from the dataset
        """
        pass