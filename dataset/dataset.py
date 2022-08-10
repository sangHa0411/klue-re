from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, dataset):
        self.input_ids = dataset["input_ids"]
        self.attention_mask = dataset["attention_mask"]

        self.test_flag = False if "labels" in dataset else True
        if not self.test_flag :
            self.labels = dataset["labels"]
            
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]

        if self.test_flag :
            return input_ids, attention_mask
        else :
            labels = self.labels[idx]
            return input_ids, attention_mask, labels

