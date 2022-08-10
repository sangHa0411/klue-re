from tqdm import tqdm

class Encoder :

    def __init__(self, args, relation_list, tokenizer):
        self.args = args
        self.label2id = {r: i for i, r in enumerate(relation_list)}
        self.tokenizer = tokenizer

    def __call__(self, dataset):
        test_flag = False if "labels" in dataset else True
        if not test_flag :
            sentences, labels = dataset["sentences"], dataset["labels"]
        else :
            sentences = dataset["sentences"]

        size = len(sentences)

        input_ids = []
        attention_masks = []
        label_ids = []
        for i in tqdm(range(size)) :
            data = self.tokenizer(sentences[i], 
                max_length=self.args.max_seq_length, 
                return_token_type_ids=False,
                truncation=True
            )

            input_ids.append(data["input_ids"])
            attention_masks.append(data["attention_mask"])

            if not test_flag :
               label_ids.append(self.label2id[labels[i]])

        if not test_flag :
            dataset = {"input_ids" : input_ids, "attention_mask" : attention_masks, "labels" : label_ids}
        else :
            dataset = {"input_ids" : input_ids, "attention_mask" : attention_masks}
        return dataset
