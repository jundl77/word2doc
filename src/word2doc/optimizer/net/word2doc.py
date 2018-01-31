import os
import random
import time

from tqdm import tqdm
import numpy as np
import prettytable
import tensorflow as tf

from word2doc.util import constants
from word2doc.util import logger


class Word2Doc:
    def __init__(self):
        self.logger = logger.get_logger()
        self.n_batches = -1

        self.hyper_params = {
            'TIME': '',
            'TRAINING PARAMS': '',
            'loss_func': 'sampled_softmax_loss',
            'optimizer': 'adam',
            'epochs': 1,
            'batch_size': 1,
            'n_input': 4096,
            'n_neg_sample': 100,
            '': '',
            'LEARNING RATE': '',
            ' ': '',
            'REDUCTION LAYER': '',
            'n_reduction': 1,
            'reduction_activation': 'linear',
            '  ': '',
            'EMBEDDING LAYER': '',
            'n_embedding': 300,
            'embedding_activation': 'linear',
            '   ': '',
            'OUTPUT LAYER': '',
            'n_classes': 5000,
            'out_activation': 'softmax',
        }

    def log_hyper_params(self, id):
        self.hyper_params['TIME'] = id
        table = prettytable.PrettyTable(['Hyper Parameter', 'Value'])

        for key, val in self.hyper_params.items():
            table.add_row([key, val])

        self.logger.info(table)

    def create_run_log(self, id):
        path = os.path.join(constants.get_tensorboard_path(), id)
        os.makedirs(path)
        return path

    def load_data(self, path):
        self.logger.info('Load ' + path)

        data = np.load(path)

        labels = list()
        titles = dict()
        embeddings = list()
        context = list()

        doc_counter = 0
        for data_dict in data:
            titles[doc_counter] = data_dict['doc_title']
            ctx = data_dict['doc_window']

            e = data_dict['pivot_embeddings']

            for emb in e:
                labels.append([doc_counter])
                embeddings.append(emb)

                neg_ctx = self.negative_samples(data, ctx, 1)
                context.append(neg_ctx)

            doc_counter += 1

        return labels, embeddings, context, titles

    def negative_samples(self, data, doc_context, num):
        negative_samples = []

        while len(negative_samples) < num:
            sample = random.choice(data)

            if sample['doc_index'] not in doc_context:
                negative_samples.append(sample['pivot_embeddings'][0])

        return negative_samples

    def shuffle_data(self, x, y, c):
        shuffled = list(zip(x, y, c))
        random.shuffle(shuffled)

        x, y, c = zip(*shuffled)

        return x, y, c

    def get_batches(self, embeddings, target):
        data = list(zip(embeddings, target))

        batches = np.array_split(data, int(len(embeddings) / self.hyper_params['batch_size']))
        self.n_batches = len(batches)

        for batch in batches:
            x, y = zip(*batch)
            yield x, y

    def model(self, mode):
        start_time = time.time()
        self.logger.info('Compiling Model ... ')

        n_input = self.hyper_params['n_input']
        n_reduction = self.hyper_params['n_reduction']
        n_embedding = self.hyper_params['n_embedding']
        n_docs = self.hyper_params['n_classes']

        train_graph = tf.Graph()
        with train_graph.as_default():

            with tf.name_scope('input'):
                inputs = tf.placeholder(tf.float32, shape=(None, n_input), name='inputs')
            with tf.name_scope('labels'):
                labels = tf.placeholder(tf.int64, shape=(None, 1), name='labels')

            # # Reduction Layer 1
            # with tf.name_scope("redu_weights_1"):
            #     redu_w_1 = tf.Variable(tf.truncated_normal([n_input, 1300]))
            # with tf.name_scope("redu_biases_1"):
            #     redu_b_1 = tf.Variable(tf.zeros([1300]))
            # with tf.name_scope("redu_layer_1"):
            #     redu_layer_1 = tf.nn.relu_layer(inputs, weights=redu_w_1, biases=redu_b_1)
            #     tf.summary.histogram('redu_layer_1', redu_layer_1)
            #
            # # Reduction Layer 2
            # with tf.name_scope("redu_weights_2"):
            #     redu_w_2 = tf.Variable(tf.truncated_normal([1300, 700]))
            # with tf.name_scope("redu_biases_2"):
            #     redu_b_2 = tf.Variable(tf.zeros([700]))
            # with tf.name_scope("redu_layer_2"):
            #     redu_layer_2 = tf.nn.relu_layer(redu_layer_1, weights=redu_w_2, biases=redu_b_2)
            #     tf.summary.histogram('redu_layer_2', redu_layer_2)
            #
            # # Reduction Layer 3
            # with tf.name_scope("redu_weights_3"):
            #     redu_w_3 = tf.Variable(tf.truncated_normal([700, 500]))
            # with tf.name_scope("redu_biases_3"):
            #     redu_b_3 = tf.Variable(tf.zeros([500]))
            # with tf.name_scope("redu_layer_3"):
            #     redu_layer_3 = tf.nn.sampled_softmax_loss(
            #         weights=redu_w_3,
            #         biases=redu_b_3,
            #         inputs=inputs,
            #         num_sampled=self.hyper_params['n_neg_sample'],
            #         num_classes=n_docs)
            #     tf.summary.histogram('redu_layer_3', redu_layer_3)
            #
            # Reduction Layer 4
            # with tf.name_scope("redu_weights_4"):
            #     redu_w_4 = tf.Variable(tf.truncated_normal([500, 300]))
            # with tf.name_scope("redu_biases_4"):
            #     redu_b_4 = tf.Variable(tf.zeros([300]))
            # with tf.name_scope("redu_layer_4"):
            #     redu_layer_4 = tf.nn.relu_layer(redu_layer_3, weights=redu_w_4, biases=redu_b_4)
            #     tf.summary.histogram('redu_layer_4', redu_layer_4)
            #
            # # Reduction Layer 5
            # with tf.name_scope("redu_weights_5"):
            #     redu_w_5 = tf.Variable(tf.truncated_normal([300, 1]))
            # with tf.name_scope("redu_biases_5"):
            #     redu_b_5 = tf.Variable(tf.zeros([1]))
            # with tf.name_scope("redu_layer_5"):
            #     redu_layer_5 = tf.nn.relu_layer(redu_layer_4, weights=redu_w_5, biases=redu_b_5)
            #     redu_layer_5 = tf.cast(tf.round(redu_layer_5), tf.int64)
            #     redu_layer_5 = redu_layer_5[0]
            #     tf.summary.histogram('redu_layer_5', redu_layer_5)

            # Embedding Layer
            # embedding = tf.Variable(tf.random_uniform((n_docs, n_embedding), -1, 1))
            # embed = tf.nn.embedding_lookup(embedding, redu_layer_3)

            # Output Layer
            with tf.name_scope("softmax_weights"):
                softmax_w = tf.Variable(tf.truncated_normal((n_docs, n_input)))
            with tf.name_scope("softmax_biases"):
                softmax_b = tf.Variable(tf.zeros(n_docs), name="softmax_bias")

            # Differentiate between training and validation
            op = None
            if mode == "train":
                # Calculate the loss using negative sampling
                with tf.name_scope("loss"):
                    loss = tf.nn.sampled_softmax_loss(
                        weights=softmax_w,
                        biases=softmax_b,
                        labels=labels,
                        inputs=inputs,
                        num_sampled=self.hyper_params['n_neg_sample'],
                        num_classes=n_docs)

                    # Model
                    with tf.name_scope("cost"):
                        cost = tf.reduce_mean(loss)
                        tf.summary.scalar("cost", cost)
                    with tf.name_scope("optimizer"):
                        optimizer = tf.train.AdamOptimizer().minimize(cost)

                op = optimizer

            elif mode == "eval":
                with tf.name_scope('val_loss'):
                    logits = tf.matmul(inputs, tf.transpose(softmax_w))
                    logits = tf.nn.bias_add(logits, softmax_b)
                    labels_one_hot = tf.one_hot(labels, n_docs)
                    val_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=logits)
                    tf.summary.scalar('val_loss', val_loss)
                    pred = tf.argmax(logits, 1)

                with tf.name_scope('val_acc'):
                    correct_prediction = tf.equal(pred, tf.argmax(labels, 1))
                    val_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    tf.summary.scalar('val_acc', val_acc)

                op = pred

            elif mode == "pred":
                logits = tf.matmul(inputs, tf.transpose(softmax_w))
                logits = tf.nn.bias_add(logits, softmax_b)
                pred = tf.argmax(logits, 1)

                op = pred

            summary = tf.summary.merge_all()

        self.logger.info('Model compiled in {0} seconds'.format(time.time() - start_time))

        return {
            'graph': train_graph,
            'inputs': inputs,
            'labels': labels,
            'op': op,
            'summary': summary
        }

    def train(self):
        # Load data
        target, embeddings, context, titles = self.load_data(os.path.join(constants.get_word2doc_dir(), '1-wpp.npy'))

        # Shuffle data
        self.logger.info('Shuffling data..')
        embeddings, context, target = self.shuffle_data(embeddings, context, target)
        self.logger.info('Done shuffling data.')

        # Set up model
        model = self.model('train')
        model_id = str(int(round(time.time())))

        self.logger.info('Training model with hyper params:')
        self.log_hyper_params(model_id)

        train_graph = model['graph']
        inputs = model['inputs']
        labels = model['labels']
        op = model['op']
        summary_opt = model['summary']

        with train_graph.as_default():
            saver = tf.train.Saver()

        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())

            # Set up TensorBoard
            writer = tf.summary.FileWriter(self.create_run_log(model_id), sess.graph)

            self.logger.info("Starting training..")

            for epoch in range(1, self.hyper_params['epochs'] + 1):

                batches = self.get_batches(embeddings, target)

                self.logger.info("Epoch " + str(epoch) + "/" + str(self.hyper_params['epochs']))

                counter = 0
                with tqdm(total=self.n_batches) as pbar:
                    for batch in tqdm(batches):
                        feed = {inputs: batch[0], labels: batch[1]}
                        summary, _ = sess.run([summary_opt, op], feed_dict=feed)

                        # Update TensorBoard
                        writer.add_summary(summary, epoch * self.n_batches + counter)

                        # Update state
                        counter += 1
                        pbar.update()

            # save_path = saver.save(sess, "checkpoints/text8.ckpt")
            # embed_mat = sess.run(normalized_embedding)

    def predict(self):
        model = self.model('pred')

        pred_graph = model['graph']
        pred_op = model['op']

        with tf.Session(graph=pred_graph) as sess:
            sess.run(tf.global_variables_initializer())

        feed = {inputs: batch[0], labels: batch[1]}
        return sess.run([pred_op], feed_dict=feed)

