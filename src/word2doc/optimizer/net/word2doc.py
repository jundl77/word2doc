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
        self.saver = None

        self.hyper_params = {
            'TIME': '',
            'TRAINING PARAMS': '',
            'loss_func': 'sampled_softmax_loss',
            'optimizer': 'adam',
            'epochs': 100,
            'batch_size': 512,
            'n_input': 4096,
            'n_context_docs': 10,
            'n_neg_sample': 100,
            'EVALUATION PARAMS': '',
            'eval_fraction': 1,
            '': '',
            'LEARNING RATE': '',
            ' ': '',
            'EMBEDDING LAYER': '',
            'n_embedding': 512,
            'embedding_activation': 'relu',
            '  ': '',
            'OUTPUT LAYER': '',
            'n_classes': 5312646,
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
            doc_id = data_dict['doc_index']
            doc_id = doc_counter
            titles[doc_id] = data_dict['doc_title']
            ctx = data_dict['doc_window']

            e = data_dict['pivot_embeddings']

            for emb in e:
                labels.append([doc_id])
                embeddings.append(emb)

                # neg_ctx = self.negative_samples(data, ctx, 1)
                context.append(self.fill_contexts(ctx))

            doc_counter += 1

        return labels, embeddings, context, titles

    def fill_contexts(self, ctx):
        while len(ctx) < 10:
            ctx.append(0)

        return np.asarray(ctx)

    def normalize_context(self, ctx, labels):
        max_label = max(labels)
        return list(map(lambda c: int((c * self.hyper_params['n_classes']) / max_label), ctx))

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

    def set_num_batches(self, embeddings):
        batches = np.array_split(embeddings, int(len(embeddings) / self.hyper_params['batch_size']))
        self.n_batches = len(batches)

    def get_batches(self, embeddings, context, target):
        data = list(zip(embeddings, context, target))

        batches = np.array_split(data, int(len(embeddings) / self.hyper_params['batch_size']))
        self.n_batches = len(batches)

        for batch in batches:
            x, c, y = zip(*batch)
            yield x, c, y

    def model(self, mode):
        start_time = time.time()
        self.logger.info("Compiling train model ... ")

        n_input = self.hyper_params['n_input']
        n_context = self.hyper_params['n_context_docs']
        n_embedding = self.hyper_params['n_embedding']
        n_docs = self.hyper_params['n_classes']

        graph = tf.Graph()
        with graph.as_default():

            # Input
            with tf.name_scope('input'):
                inputs = tf.placeholder(tf.float32, shape=(None, n_input), name='inputs')
            with tf.name_scope('context'):
                context = tf.placeholder(tf.int64, shape=(None, n_context), name='contexts')
            with tf.name_scope('labels'):
                labels = tf.placeholder(tf.int64, shape=(None, 1), name='labels')

            # Input embedding
            embedded_input = tf.layers.dense(inputs, n_embedding, activation=tf.nn.relu, use_bias=True)

            # Context embeddings
            doc_embeddings = tf.get_variable("doc_embeddings", [n_docs, n_embedding], dtype=tf.float32)
            embedded_docs = tf.cast(tf.map_fn(lambda doc: tf.nn.embedding_lookup(doc_embeddings, doc), context),
                                   dtype=tf.float32)

            # Contact layers
            concat_embb = tf.concat([embedded_docs, tf.expand_dims(embedded_input, axis=1)], axis=1)
            embb_dim = n_input * n_context
            concat_embb = tf.reshape(concat_embb, [tf.shape(concat_embb)[0], embb_dim])

            # Merge layer
            merged_layer = tf.layers.dense(concat_embb, n_embedding, activation=tf.nn.relu, use_bias=True)

            # Output layer
            with tf.variable_scope("softmax_weights"):
                softmax_w = tf.Variable(tf.truncated_normal((n_docs, n_embedding)))
            with tf.variable_scope("softmax_biases"):
                softmax_b = tf.Variable(tf.zeros(n_docs), name="softmax_bias")

            self.saver = tf.train.Saver()

            op = None
            val_loss = [None]
            val_acc = None

            # Train model
            if mode == "train":
                with tf.name_scope("loss"):
                    loss = tf.nn.sampled_softmax_loss(
                        weights=softmax_w,
                        biases=softmax_b,
                        labels=labels,
                        inputs=merged_layer,
                        num_sampled=self.hyper_params['n_neg_sample'],
                        num_classes=n_docs)
                    tf.summary.scalar('train_loss', loss[0])
                with tf.name_scope("cost"):
                    cost = tf.reduce_mean(loss)
                    tf.summary.scalar("cost", cost)
                with tf.name_scope("optimizer"):
                    optimizer = tf.train.AdamOptimizer().minimize(cost)
                    op = optimizer

            # Eval model
            if mode == "eval":
                with tf.name_scope('val_loss'):
                    with tf.variable_scope("softmax_weights", reuse=True):
                        logits = tf.matmul(merged_layer, tf.transpose(softmax_w))
                    with tf.variable_scope("softmax_biases", reuse=True):
                        logits = tf.nn.bias_add(logits, softmax_b)
                    labels_one_hot = tf.one_hot(labels, self.hyper_params['n_classes'])
                    val_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=logits)
                    tf.summary.scalar('val_loss', val_loss[0])

                with tf.name_scope('val_acc'):
                    labels_flat = tf.map_fn(lambda l: l[0], labels)
                    correct_prediction = tf.equal(tf.argmax(logits, 1), labels_flat)
                    val_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    tf.summary.scalar('val_acc', val_acc)

            # Predict
            if mode == "predict":
                with tf.variable_scope("softmax_weights", reuse=True):
                    logits = tf.matmul(merged_layer, tf.transpose(softmax_w))
                with tf.variable_scope("softmax_biases", reuse=True):
                    logits = tf.nn.bias_add(logits, softmax_b)
                pred = tf.argmax(logits, 1)
                op = pred

            summary = tf.summary.merge_all()

        self.logger.info('Model compiled in {0} seconds'.format(time.time() - start_time))

        return {
            'graph': graph,
            'inputs': inputs,
            'context': context,
            'labels': labels,
            'op': op,
            'val_loss': val_loss[0],
            'val_acc': val_acc,
            'summary': summary
        }

    def train(self):
        # Load data
        target, embeddings, context, titles = self.load_data(os.path.join(constants.get_word2doc_dir(), '2-wpp.npy'))

        # Shuffle data
        self.logger.info('Shuffling data..')
        embeddings, context, target = self.shuffle_data(embeddings, context, target)
        self.logger.info('Done shuffling data.')

        # Set up model
        model = self.model("train")
        model_eval = self.model("eval")
        model_id = str(int(round(time.time())))

        self.logger.info('Training model with hyper params:')
        self.log_hyper_params(model_id)

        train_graph = model['graph']
        inputs = model['inputs']
        context_pl = model['context']
        labels = model['labels']
        op = model['op']
        summary_opt = model['summary']
        eval_summary_opt = model_eval['summary']

        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())

            # Set up TensorBoard
            writer = tf.summary.FileWriter(self.create_run_log(model_id), sess.graph)

            self.logger.info("Starting training..")

            for epoch in range(1, self.hyper_params['epochs'] + 1):

                batches = self.get_batches(embeddings, context, target)
                self.logger.info("Epoch " + str(epoch) + "/" + str(self.hyper_params['epochs']))

                counter = 0
                with tqdm(total=self.n_batches) as pbar:
                    for batch in tqdm(batches):
                        feed = {inputs: batch[0], context_pl: batch[1], labels: batch[2]}
                        summary, _ = sess.run([summary_opt, op], feed_dict=feed)

                        # Update train TensorBoard
                        writer.add_summary(summary, epoch * self.n_batches + counter)

                        # Update state
                        counter += 1
                        pbar.update()

                        # if counter % 1000 == 0:
                        #     summary = sess.run([eval_summary_opt], feed_dict=feed)
                        #
                        #     # Update eval TensorBoard
                        #     writer.add_summary(summary, epoch * self.n_batches + counter)

            self.saver.save(sess, os.path.join(constants.get_word2doc_dir(), "word2doc_model_5000_100e_2l_relu"))

    def eval(self):
        # Load data
        target, embeddings, context, titles = self.load_data(os.path.join(constants.get_word2doc_dir(), '1-wpp.npy'))

        # Shuffle data
        self.logger.info('Shuffling data..')
        embeddings, context, target = self.shuffle_data(embeddings, context, target)
        self.logger.info('Done shuffling data.')

        # Set up model
        model = self.model("eval")
        model_id = str(int(round(time.time()))) + "_eval"

        self.logger.info('Evaluating model..')
        self.log_hyper_params(model_id)

        train_graph = model['graph']
        inputs = model['inputs']
        labels = model['labels']
        acc = model['val_acc']
        loss = model['val_loss']
        summary_opt = model['summary']

        with tf.Session(graph=train_graph) as sess:
            self.saver.restore(sess, os.path.join(constants.get_word2doc_dir(), "word2doc_model_5000_100e_2l_relu"))

            # Set up TensorBoard
            writer = tf.summary.FileWriter(self.create_run_log(model_id), sess.graph)

            self.set_num_batches(embeddings)
            batches = self.get_batches(embeddings, target)
            n_eval = int(self.n_batches * self.hyper_params['eval_fraction'])

            self.logger.info("Starting evaluation across " + str(n_eval) + " (" +
                             str(self.hyper_params['eval_fraction'] * 100) + "%) randomly chosen elements")

            total_acc = 0
            total_loss = 0

            counter = 0
            with tqdm(total=n_eval) as pbar:
                for batch in tqdm(batches):

                    # Only do X% of data
                    if counter >= n_eval:
                        break

                    feed = {inputs: batch[0], labels: batch[1]}
                    summary, l, a = sess.run([summary_opt, loss, acc], feed_dict=feed)

                    # Update TensorBoard
                    writer.add_summary(summary, self.n_batches + counter)

                    # Update state
                    total_loss += l
                    total_acc += a
                    counter += 1
                    pbar.update()

            total_loss = total_loss / counter
            total_acc = total_acc / counter

            # Print results
            self.logger.info("Loss: " + str(total_loss) + " -- Accuracy: " + str(total_acc))

    def predict(self, x):
        model = self.model('predict')

        graph = model['graph']
        inputs = model['inputs']
        pred_op = model['op']

        with tf.Session(graph=graph) as sess:
            self.saver.restore(sess, os.path.join(constants.get_word2doc_dir(), "word2doc_model"))

            feed = {inputs: x}
            return sess.run([pred_op], feed_dict=feed)

