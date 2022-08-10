import torch

class PaddingCollator :
    def __init__(self, max_seq_length, tokenizer) :
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

    def __call__(self, data) :
        size = len(data)
        test_flag = True if len(data[0]) == 2 else False

        input_ids = []
        attention_masks = []
        labels = []

        for d in data :
            if test_flag :
                input_ids.append(d[0])
                attention_masks.append(d[1])
            else :
                input_ids.append(d[0])
                attention_masks.append(d[1])
                labels.append(d[2])

        batch_max_seq_length = max([len(x) for x in input_ids])
        batch_max_seq_length = min(self.max_seq_length, batch_max_seq_length)

        for i in range(size) :
            seq_size = len(input_ids[i])

            if seq_size < batch_max_seq_length :
                padding_size = batch_max_seq_length - seq_size
                input_ids[i] = input_ids[i] + [self.tokenizer.pad_token_id] * padding_size
                attention_masks[i] = attention_masks[i] + [0] * padding_size
            else :
                input_ids[i] = input_ids[i][:batch_max_seq_length]
                attention_masks[i] = attention_masks[i][:batch_max_seq_length]

        input_ids = torch.tensor(input_ids, dtype=torch.int32)
        attention_masks = torch.tensor(attention_masks, dtype=torch.int32)

        if not test_flag :
            labels = torch.tensor(labels, dtype=torch.int32)
            return input_ids, attention_masks, labels
        else :
            input_ids, attention_masks