from tracking import Image
import torch

class FeatureTracker:
    def __init__(self):
        self.id_num = 1
        self.age = 0
        self.age_dict = {}
        self.reset = 10         # replace "noise" every X frames
        self.cos_dict = {}
        self.collection = {}
        self.threshold = 0.85   # Threshold for cosine similarity
        self.attn_dict = {}
        self.attn_age = 0
        self.attn_age_dict = {}

    def cossim(self, objects, index):
        for k in self.collection.keys():
            input1 = self.collection[k]
            input2 = objects[index]['features']
            cos = torch.nn.CosineSimilarity(dim=0)
            output = cos(input1, input2)
            self.cos_dict.update({k : output})

    def update(self, objects, index):
        maximum = max(self.cos_dict, key=self.cos_dict.get)
        if self.cos_dict[maximum] >= self.threshold:
            self.collection.update({maximum : objects[index]['features']})
            self.age_dict.setdefault(maximum, []).append(self.age)
            objects[index].update({'id': maximum})
        else:
            self.collection.update({self.id_num : objects[index]['features']})
            self.age_dict.setdefault(self.id_num, []).append(self.age)
            objects[index].update({'id': self.id_num})
            self.id_num += 1 

    def denoise(self):
        if self.age == self.reset:   # every X frames remove "noise"
            for k,v in self.age_dict.items():               
                if len(v) <= 1:  # considered "noise" if ID only appears Y times in X frames       
                    self.collection.pop(k, None)
                    self.cos_dict.pop(k, None)
            self.age_dict.clear()
            self.age -= self.reset

    def __call__(self, frame: Image):
        objs = frame.objects
        # if objects detected and ID-feature dictionary empty,
        # update empty dictionary w/ ID-feature pair, and append ID to frame.objects
        self.age += 1
        for n in range(len(objs)):
            if len(self.collection) < 1:
                self.collection.update({n + 1: objs[n]['features']})
                self.age_dict.setdefault(n + 1, []).append(self.age)
                objs[n].update({'id': n + 1})
                self.id_num += 1
                continue
            # if ID-feature dictionary is not empty and face is detected, 
            # calculate cosine similarity to each feature in ID-feature dictionary
            self.cossim(objs, n)
            # if the max cosine similarity of all id features < threshold, 
            # add new ID-feature pair and add new ID number to frame.object          
            self.update(objs, n)
        self.denoise()

        return frame
