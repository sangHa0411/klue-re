import os
import json

class Loader :

    def __init__(self, data_dir, file_name) :
        self.data_dir = data_dir
        self.file_name = file_name
        self.test_flag = True if "test" in file_name else False

        self.subject_start_marker = "<subj>"
        self.subject_end_marker = "</subj>"
        self.object_start_marker = "<obj>"
        self.object_end_marker = "</obj>"

        path = os.path.join(data_dir, file_name)
        with open(path, "r") as f :
            self.dataset = json.load(f)

    def _add_token(self, sen, sub_entity, obj_entity):
        sub_entity_word = sub_entity["word"]
        sub_entity_start = sub_entity["start_idx"]
        sub_entity_end = sub_entity["end_idx"]

        obj_entity_word = obj_entity["word"]
        obj_entity_start = obj_entity["start_idx"]
        obj_entity_end = obj_entity["end_idx"]

        if obj_entity_start < sub_entity_start :
            sen = sen[:obj_entity_start] + self.object_start_marker + obj_entity_word + self.object_end_marker + sen[obj_entity_end+1:]
            sub_entity_start += 11
            sub_entity_end += 11
            sen = sen[:sub_entity_start] + self.subject_start_marker + sub_entity_word + self.subject_end_marker + sen[sub_entity_end+1:]
        else :
            sen = sen[:sub_entity_start] + self.subject_start_marker + sub_entity_word + self.subject_end_marker + sen[sub_entity_end+1:]
            obj_entity_start += 13
            obj_entity_end += 13
            sen = sen[:obj_entity_start] + self.object_start_marker + obj_entity_word + self.object_end_marker + sen[obj_entity_end+1:]
        return sen

    def _convert_dataset(self, dataset):
        sentences = []
        labels = []
        for d in dataset :
            sen = d["sentence"]
            sub_entity = d["subject_entity"]
            obj_entity = d["object_entity"]

            sen = self._add_token(sen, sub_entity, obj_entity)
            sentences.append(sen)

            if not self.test_flag :
                labels.append(d["label"])

        if self.test_flag :
            return sentences
        else :
            return sentences, labels

    def load(self):
        if self.test_flag :
            sentences = self._convert_dataset(self.dataset)
            dataset = {"sentences" : sentences}
        else :
            sentences, labels = self._convert_dataset(self.dataset)        
            dataset = {"sentences" : sentences, "labels" : labels}
        return dataset
