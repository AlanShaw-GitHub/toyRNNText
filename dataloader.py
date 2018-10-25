# from gensim.models import Word2Vec
import csv
# import jieba
import pickle
import re
import random
import numpy as np

_stopwords = ['[','？','：','“','”','，','（','）','、','」','「','《','》',']']
stopwords = "".join(_stopwords)
train_len = 100000
test_len = 1000
class Loader():

    # def __init__(self):
    #     self.max_tag_num = 10000
    #     self.max_ques_len = 20
    #     self.max_detail_len = 50
    #     self.batch_size = 64
    #     word2vec_path = '../model'
    #     file_path = '../train_data.csv'
    #     label_path = '../topic_2_id.csv'
    #     model = Word2Vec.load(word2vec_path)
    #     label = list(csv.reader(open(label_path, encoding='utf8').readlines()))[1:]
    #     self.label_dict = dict([(int(i[1]),i[0]) for i in label])
    #     self.freq_dict = dict([(int(i[1]),int(i[2])) for i in label])
    #     data = list(csv.reader(open(file_path,encoding='utf8').readlines()))[1:]
    #     data = np.array(data[:train_len])
    #     questions = data[:,1]
    #     self.o_questions = questions
    #     questions = [list(jieba.cut(re.sub(stopwords,'',i))) for i in questions]
    #     questions_dict = []
    #     for i in questions:
    #         questions_dict += i
    #     questions_dict = set(questions_dict)
    #     print('questions_dict len', len(questions_dict))
    #     self.questions_dict = dict(zip(range(1,len(questions_dict)+1),questions_dict))
    #     self.questions_dict[0] = 'UNK'
    #     self.reversed_questions_dict = {v: k for k, v in self.questions_dict.items()}
    #     self.embeddings = np.zeros([len(self.questions_dict),128])
    #     for i in range(len(questions_dict)):
    #         try:
    #             self.embeddings[i] = model.wv[self.questions_dict[i]]
    #         except:
    #             self.embeddings[i] = np.zeros([128])
    #     self.questions_len = np.array([len(i) if len(i) <= 20 else 20 for i in questions])
    #     self.questions = np.zeros([train_len,self.max_ques_len],dtype=int)
    #     for i in range(len(questions)):
    #         for j in range(self.max_ques_len):
    #             try:
    #                 tmp = self.reversed_questions_dict[questions[i][j]]
    #             except:
    #                 tmp = 0
    #             self.questions[i][j] = tmp
    #     tags = data[:,4]
    #     self.tags = np.array([[int(j) for j in i.split('|')] for i in tags])
    #     self.max_tag_len = max([len(i) for i in tags])
    #     pickle.dump([self.tags,self.max_tag_len,self.o_questions,self.questions_dict,self.questions_len,self.questions,self.reversed_questions_dict,self.embeddings,self.label_dict,self.freq_dict],open('./data.pickle','wb'))
    #     self.reset()

    def __init__(self):
        self.max_tag_num = 10000
        self.max_ques_len = 20
        self.max_detail_len = 50
        self.batch_size = 64
        
        self.tags,self.max_tag_len,self.o_questions,self.questions_dict,self.questions_len,self.questions,self.reversed_questions_dict,self.embeddings,self.label_dict,self.freq_dict = pickle.load(open('./data.pickle','rb'))
        self.reset()

    def reset(self):
        self.index = 0
        self.random = list(range(train_len-test_len))
        self.random_test = list(range(train_len-test_len,train_len))
        random.shuffle(self.random)
        random.shuffle(self.random_test)

    def generate(self):
        while True:
            if self.index + self.batch_size >= train_len - test_len:
                break
            o_questions = self.o_questions[self.random[self.index:self.index + self.batch_size]]
            questions = self.questions[self.random[self.index:self.index + self.batch_size]]
            ques_len = self.questions_len[self.random[self.index:self.index + self.batch_size]]
            _tags = self.tags[self.random[self.index:self.index + self.batch_size]]
            tags = np.zeros([self.batch_size,self.max_tag_num],dtype=float)
            for i in range(self.batch_size):
                for j in _tags[i]:
                    if j < self.max_tag_num:
                        tags[i][j-1] = 1.0
            for i in range(self.batch_size):
                for j in _tags[i]:
                    if j >= self.max_tag_num:
                        _tags[i].remove(j)

            self.index += self.batch_size
            yield questions,ques_len,tags,_tags,o_questions

    def generate_test(self):
        while True:
            if self.index + self.batch_size >= test_len:
                break
            o_questions = self.o_questions[self.random_test[self.index:self.index+self.batch_size]]
            questions = self.questions[self.random_test[self.index:self.index+self.batch_size]]
            ques_len = self.questions_len[self.random_test[self.index:self.index+self.batch_size]]
            _tags = self.tags[self.random_test[self.index:self.index + self.batch_size]]
            tags = np.zeros([self.batch_size,self.max_tag_num],dtype=float)
            for i in range(self.batch_size):
                for j in _tags[i]:
                    if j < self.max_tag_num:
                        tags[i][j-1] = 1.0
            for i in range(self.batch_size):
                for j in _tags[i]:
                    if j >= self.max_tag_num:
                        _tags[i].remove(j)

            self.index += self.batch_size
            yield questions,ques_len,tags,_tags,o_questions


if __name__ == '__main__':
    loader = Loader()
    x = loader.generate()
    print(x.__next__()[0])
