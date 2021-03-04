import re
class Preprocess():
    def __init__(self, morpher):
        self.mc = morpher

    def work(self, sentence, stopwords, data):
        sentence = re.sub('\,|~|\"|=|<|>|\*|\'', '', sentence)
        sentence = re.sub('\(|\)', ',', sentence)
        sentence = re.sub('[0-9]+', 'num', sentence)
        sentence = re.sub(";+", ';', sentence)
        sentence = re.sub("[?]{2,}", '??', sentence)
        sentence = re.sub("[.]{2,}", '..', sentence)
        sentence = re.sub("[!]{2,}", '!!', sentence)
        #sentence = re.sub('[a-zA-Z]', '', sentence)
        temp_X = self.mc.morphs(sentence, norm=True, stem=True)
        temp_X = [word for word in temp_X if not word in stopwords]
        data.append(temp_X)
        return data

    def labeling(self, data, train):
        for i in range(len(data['Emotion'])):
            if data['Emotion'].iloc[i] == '슬픔':
                train.append([1, 0, 0, 0, 0])
            elif data['Emotion'].iloc[i] == '중립':
                train.append([0, 1, 0, 0, 0])
            elif data['Emotion'].iloc[i] == '행복':
                train.append([0, 0, 1, 0, 0])
            elif data['Emotion'].iloc[i] == '공포':
                train.append([0, 0, 0, 1, 0])
            elif data['Emotion'].iloc[i] == '분노':
                train.append([0, 0, 0, 0, 1])
        return train
