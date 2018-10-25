import sys
from dataloader import Loader
from model import Model
from dataloader import train_len
import time
import jieba
import cmd
import numpy as np
import re
import os
import tensorflow as tf

class Trainer(object):
    def __init__(self):
        self.loader = Loader()
        self.model = Model(self.loader)
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.learning_rates = tf.train.exponential_decay(5e-4, global_step,
                                                    decay_steps= 5 * int(train_len / self.loader.batch_size),
                                                    decay_rate=0.9, staircase=False)

        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rates)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rates)
        gradients = self.optimizer.compute_gradients(self.model.train_loss)
        capped_gradients = [(tf.clip_by_norm(grad, 5.), var) for grad, var in gradients if grad is not None]
        self.train_proc = self.optimizer.apply_gradients(capped_gradients, global_step=global_step)
        # self.train_proc = self.optimizer.minimize(self.model.loss, global_step=global_step)
        self.model_path = './model_path_large/'
        if not os.path.exists(self.model_path):
            print('create path: ', self.model_path)
            os.makedirs(self.model_path)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        init_proc = tf.global_variables_initializer()
        self.sess.run(init_proc)
        self.model_saver = tf.train.Saver()
        self.last_checkpoint = None

    def train(self):
        print('Trainnning begins......')
        best_epoch_acc = 100000000
        best_epoch_id = 0
        for i_epoch in range(500):
            t_begin = time.time()
            avg_batch_loss = self.train_one_epoch(i_epoch)
            t_end = time.time()
            print('Epoch %d ends. Average loss %.3f. %.3f seconds/epoch, learning_rates %.6f.' % (i_epoch, avg_batch_loss, t_end - t_begin,self.sess.run([self.learning_rates])[0]))

            print('=================================')
            print('valid set evaluation')
            valid_acc = self.evaluate(self.loader)
            print('=================================')

            if valid_acc < best_epoch_acc:
                best_epoch_acc = valid_acc
                best_epoch_id = i_epoch
                print('Saving new best model...')
                timestamp = time.strftime("%m%d%H%M%S", time.localtime())
                self.last_checkpoint = self.model_saver.save(self.sess, self.model_path + timestamp)
            else:
                if i_epoch - best_epoch_id >= 50:
                    print('Early stopped. Best loss %.3f at epoch %d' % (best_epoch_acc, best_epoch_id))
                    break

        print('=================================')
        print('Evaluating best model in file', self.last_checkpoint, '...')
        if self.last_checkpoint is not None:
            self.model_saver.restore(self.sess, self.last_checkpoint)
            self.evaluate(self.loader)
        else:
            print('ERROR: No checkpoint available!')

    def train_one_epoch(self, i_epoch):

        loss_sum = 0
        display_loss_sum = 0
        t1 = time.time()
        i_batch = 0

        self.loader.reset()
        for questions, ques_len, tags, _tags,o_questions in self.loader.generate():
            batch_data = dict()
            batch_data[self.model._questions] = questions
            batch_data[self.model.ques_len] = ques_len
            batch_data[self.model.tags] = tags
            batch_data[self.model.is_training] = True
            # Forward pass
            _, batch_loss, predict_tags, train_loss = self.sess.run(
               [self.train_proc, self.model.loss, self.model.predict_tags,self.model.train_loss], feed_dict=batch_data)

            if i_batch == 1:
                batch_data[self.model.is_training] = False
                # Forward pass
                _, batch_loss, predict_tags, train_loss = self.sess.run(
                [self.train_proc, self.model.loss, self.model.predict_tags,self.model.train_loss], feed_dict=batch_data)
                print('train_loss', train_loss)
                all_rank = []
                for i in range(len(predict_tags)):
                    dicts = dict(zip(range(1, len(predict_tags[i]) + 1), predict_tags[i]))
                    dicts = sorted(dicts.items(), key=lambda x: x[1], reverse=True)
                    all_rank.append(dicts)
                for i in range(3):
                    tmp_score = []
                    tmp_label = []
                    for j in range(5):
                        tmp_score.append(all_rank[i][j][1])
                        tmp_label.append(self.loader.label_dict[all_rank[i][j][0]])
                    gt_tags = [self.loader.label_dict[k] for k in _tags[i]]
                    print(o_questions[i])
                    print('predict score:',tmp_score)
                    print('predict label:',tmp_label)
                    print('gt label:',gt_tags)

            i_batch += 1
            loss_sum += batch_loss
            display_loss_sum += batch_loss

        avg_batch_loss = loss_sum / i_batch
        return avg_batch_loss

    def evaluate(self, data_loader):
        data_loader.reset()
        all_retrievd = 0
        loss_sum = 0

        for questions, ques_len, tags, _tags,o_questions in data_loader.generate_test():
            all_retrievd += 1
            batch_data = dict()
            batch_data[self.model._questions] = questions
            batch_data[self.model.ques_len] = ques_len
            batch_data[self.model.tags] = tags
            batch_data[self.model.is_training] = False
            # Forward pass
            batch_loss, predict_tags = self.sess.run(
                    [self.model.loss, self.model.predict_tags], feed_dict=batch_data)
            loss_sum += batch_loss

            if all_retrievd == 1:
                all_rank = []
                for i in range(len(predict_tags)):
                    dicts = dict(zip(range(1, len(predict_tags[i]) + 1), predict_tags[i]))
                    dicts = sorted(dicts.items(), key=lambda x: x[1], reverse=True)
                    all_rank.append(dicts)
                for i in range(3):
                    tmp_score = []
                    tmp_label = []
                    for j in range(5):
                        tmp_score.append(all_rank[i][j][1])
                        tmp_label.append(self.loader.label_dict[all_rank[i][j][0]])
                    gt_tags = [self.loader.label_dict[k] for k in _tags[i]]
                    print(o_questions[i])
                    print('predict score:',tmp_score)
                    print('predict label:',tmp_label)
                    print('gt label:',gt_tags)

        loss = loss_sum / all_retrievd
        print('=================================')
        print('loss:', loss)
        print('=================================')

        return loss

    def use_it(self,_questions,path = '1025023156',if_load = True):
        _stopwords = ['[','？','：','“','”','，','（','）','、','」','「','《','》',']']
        stopwords = "".join(_stopwords)
        if if_load:
            self.last_checkpoint = self.model_path + path
            self.model_saver.restore(self.sess, self.last_checkpoint)
        questions_ = list(jieba.cut(re.sub(stopwords,'',_questions)))
        ques_len = len(questions_) if len(questions_) <= 20 else 20
        questions = np.zeros([1,self.loader.max_ques_len],dtype=int)
        for i in range(self.loader.max_ques_len):
            try:
                tmp = self.loader.reversed_questions_dict[questions_[i]]
            except:
                tmp = 0
            questions[0][i] = tmp
        batch_data = dict()
        batch_data[self.model._questions] = questions
        batch_data[self.model.ques_len] = [ques_len]
        batch_data[self.model.is_training] = False
        # Forward pass
        predict_tags = self.sess.run(
                [self.model.predict_tags], feed_dict=batch_data)
        predict_tags = predict_tags[0]
        all_rank = []
        for i in range(len(predict_tags)):
            dicts = dict(zip(range(1, len(predict_tags[i]) + 1), predict_tags[i]))
            dicts = sorted(dicts.items(), key=lambda x: x[1], reverse=True)
            all_rank.append(dicts)
        for i in range(len(predict_tags)):
            tmp_score = []
            tmp_label = []
            for j in range(10):
                tmp_score.append(all_rank[i][j][1])
                tmp_label.append(self.loader.label_dict[all_rank[i][j][0]])
            print(_questions)
            print('predict score:',tmp_score)
            print('predict label:',tmp_label)

class miniSQL(cmd.Cmd):
    intro = 'Welcome to the toyRNNText.\nType help or ? to list commands.\n'

    def do_load(self,args):
        self.trainer = Trainer()
        self.trainer.last_checkpoint = self.trainer.model_path + args
        self.trainer.model_saver.restore(self.trainer.sess, self.trainer.last_checkpoint)
    def do_predict(self,args):
        self.trainer.use_it("".join(args),if_load = False)
    def do_quit(self,args):
        sys.exit()

    

if __name__ == '__main__':

    # trainer = Trainer()
    # trainer.train()
    miniSQL.prompt = 'toyRNNText > '
    miniSQL().cmdloop()
    # trainer.use_it("".join(sys.argv[2:]),sys.argv[1])
