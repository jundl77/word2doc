import json
import os
import random
import sys
import time
from random import randint
from random import shuffle

import numpy as np
import prettytable
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import tqdm

from word2doc.retriever.doc_db import DocDB
from word2doc.util import constants
from word2doc.util import logger


class Word2Doc:
    """
    Word2Doc model is defined here.

    Possible modes:
    0: training
    1: testing with train data
    2: testing with eval data
    3: prediction
    """

    def __init__(self):
        self.logger = logger.get_logger()
        self.saver = None
        self.doc_db = DocDB(constants.get_db_path())
        self.data_mapping = {}
        self.ctx_data_mapping = {}
        self.title_mapping = {}
        self.label_titles = {}
        self.mapping_counter = 0
        self.ctx_mapping_counter = 0
        self.target_title_counter = 0
        self.doc_titles = self.doc_db.get_doc_ids()

        self.train_state = {
            'files_seen': [],
            'file_seen_last': '',
            'total_files': 0,
            'epoch': 0,
            'total_epochs': 0
        }

        self.hyper_params = {
            'TIME': '',
            'MODEL': '',
            'TRAINING PARAMS': '',
            'loss_func': 'sampled_softmax_loss',
            'optimizer': 'adam',
            'epochs': 30,
            'batch_size': 256,
            'eval_batch_size': 1,
            'n_input': 4096,
            'n_context_docs': 10,
            'n_neg_sample': 10,
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
            'n_classes': 5000,
            'out_activation': 'softmax',
        }

        self.predict_model = self.model_predict()

    def log_hyper_params(self, id, name):
        self.hyper_params['TIME'] = id
        self.hyper_params['MODEL'] = name
        table = prettytable.PrettyTable(['Hyper Parameter', 'Value'])

        for key, val in self.hyper_params.items():
            table.add_row([key, val])

        self.logger.info(table)

    def create_run_log(self, id):
        path = os.path.join(constants.get_tensorboard_path(), id)
        os.makedirs(path)
        return path

    def iter_files(self, path):
        """Walk through all files located under a root path."""
        if os.path.isfile(path):
            yield path
        elif os.path.isdir(path):
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    yield os.path.join(dirpath, f)
        else:
            raise RuntimeError('Path %s is invalid' % path)

    def load_train_data(self, path):
        self.logger.info('Load training data: ' + path)

        if os.path.isdir(path):
            files = [f for f in self.iter_files(path)]
            files = list(filter(lambda f: f.endswith("wpp.npy"), files))

            random.shuffle(files)
        else:
            files = [path]

        self.train_state['total_files'] = len(files)

        for f in files:
            self.logger.info("Doing file: " + f)
            self.train_state['file_seen_last'] = f

            # If we cannot read the file, stop
            try:
                data = np.load(f)
            except:
                yield None, None, None, None

            self.train_state['files_seen'].append(f)

            labels = list()
            titles = dict()
            embeddings = list()
            context = list()

            for data_dict in data:
                doc_id = data_dict['doc_index']
                doc_id = self.__map_data(doc_id)
                titles[doc_id] = data_dict['doc_title']
                ctx = data_dict['doc_window']

                e = data_dict['pivot_embeddings']

                for emb in e:
                    labels.append([doc_id])
                    embeddings.append(emb)

                    # neg_ctx = self.negative_samples(data, ctx, 1)
                    context.append(self.fill_contexts(ctx))

            yield labels, embeddings, context, titles

    def load_test_data(self, path):
        self.logger.info('Load test data: ' + path)

        data = np.load(path)

        labels = list()
        titles = dict()
        embeddings = list()
        context = list()

        for data_dict in data:
            doc_id = data_dict['doc_index']
            titles[doc_id] = data_dict['doc_title']
            ctx = data_dict['doc_window']

            e = data_dict['pivot_embeddings']

            labels.append([doc_id])
            embeddings.append(e)

            # neg_ctx = self.negative_samples(data, ctx, 1)
            context.append(self.fill_contexts(ctx))

        return labels, embeddings, context, titles

    def __map_data(self, id):
        if id in self.data_mapping:
            return self.data_mapping[id]

        self.data_mapping[id] = self.mapping_counter
        self.mapping_counter += 1
        return self.mapping_counter - 1

    def fill_contexts(self, ctx):
        while len(ctx) < 10:
            ctx.append(0)

        return np.asarray(ctx)

    def create_embedding_labels_file(self, model_name, embb_titles, embb_length):
        content = ''
        for title in embb_titles.values():
            content += title + "\n"

        for i in range(0, embb_length - len(embb_titles)):
            content += "NULL" + "\n"

        # Save for TensorBoard
        with open(os.path.join(constants.get_tensorboard_path(), model_name + ".tsv"), "w") as text_file:
            text_file.write(content)

        # Save for model
        with open(os.path.join(constants.get_word2doc_dir(), model_name + ".tsv"), "w") as text_file:
            text_file.write(content)

    def normalize_context(self, ctx):
        for sub_ctx in ctx:
            for num in sub_ctx:
                if num not in self.ctx_data_mapping:
                    self.ctx_data_mapping[num] = self.ctx_mapping_counter
                    self.title_mapping[self.ctx_mapping_counter] = self.doc_titles[num]
                    self.ctx_mapping_counter += 1

        i = 0
        for context in ctx:
            ctx[i] = list(map(lambda n: self.ctx_data_mapping[n], context))
            i += 1

        return ctx

    def normalize_test_context(self, testing_ctx):
        """Map testing context docs to the domain of the training context docs
        (i.e. [0, 5M] -> [0, 47k] with 5k training samples"""

        ctx = list(testing_ctx)
        i = 0

        for context in testing_ctx:

            # Map if possible (sometimes documents are not in the training domain range, so they have to be duplicated)
            single_ctx = list()
            for num in context:
                if num in self.ctx_data_mapping:
                    single_ctx.append(self.ctx_data_mapping[num])

            if len(single_ctx) == 0:
                ctx[i] = -1
                i += 1
                continue

            # Filling up contexts with duplicates
            for z in range(0, 10 - len(single_ctx)):
                index = randint(0, len(single_ctx) - 1)
                single_ctx.append(single_ctx[index])

            ctx[i] = single_ctx
            i += 1

        return ctx

    def update_target_titles(self, target, titles):
        target_new = list(target)

        for key, title in titles.items():
            if title not in self.label_titles.values():
                self.label_titles[self.target_title_counter] = title
                self.target_title_counter += 1

        self.logger.info("Updating target and title indices")
        i = 0
        with tqdm(total=len(target)) as pbar:
            for tar in tqdm(target):
                for key, val in self.label_titles.items():
                    if val == titles[tar[0]]:
                        target_new[i][0] = key
                        break
                i += 1
                pbar.update()

        return target_new

    def normalize_test_labels(self, target, titles, titles_test):
        i = 0
        target_new = list(target)
        with tqdm(total=len(target)) as pbar:
            for tar in tqdm(target):
                for key, val in titles.items():
                    if val == titles_test[tar[0]]:
                        target_new[i][0] = key
                        break
                i += 1
                pbar.update()

        return target_new

    def filter_test_data(self, x, y, c):
        zipped = list(zip(x, y, c))
        zipped = list(filter(lambda x: not x[2] == -1, zipped))

        x, y, c = zip(*zipped)

        return x, y, c

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

    def get_num_batches(self, embeddings):
        batches = np.array_split(embeddings, int(len(embeddings) / self.hyper_params['batch_size']))
        return len(batches)

    def get_batches(self, embeddings, context, target):
        batch_size = self.hyper_params['batch_size']

        data = list(zip(embeddings, context, target))

        batches = np.array_split(data, int(len(embeddings) / batch_size))

        for batch in batches:
            x, c, y = zip(*batch)
            yield x, c, y

    def get_eval_batches(self, embeddings, context, target):
        batch_size = self.hyper_params['eval_batch_size']
        data = list(zip(embeddings, context, target))
        return np.array_split(data, int(len(embeddings) / batch_size))

    # -----------------------------------------------------------------------------------------------------------------
    # DEFINE MODEL
    # -----------------------------------------------------------------------------------------------------------------

    def __model_inputs(self, mode):
        """Define input placeholders for the model."""

        n_input = self.hyper_params['n_input']
        n_context = self.hyper_params['n_context_docs']

        # Define input placeholders
        with tf.name_scope('input'):
            inputs = tf.placeholder(tf.float32, shape=(None, n_input), name='inputs')
        with tf.name_scope('context'):
            context = tf.placeholder(tf.int64, shape=(None, n_context), name='contexts')

        # Prediction mode
        if mode is 3:
            return inputs, context

        # Train or test mode
        else:
            with tf.name_scope('labels'):
                labels = tf.placeholder(tf.int64, shape=(None, 1), name='labels')

            # Test mode
            if mode > 0:
                return inputs, context, labels

            # Train mode
            else:
                with tf.name_scope('mode'):
                    mode_pl = tf.placeholder(tf.int32, shape=1, name='mode')

                return inputs, context, labels, mode_pl

    def __model_net(self, inputs, context, mode):
        """Define the net here."""

        def __apply_dropout(mode, tensor, dropout_value):
            """Applies dropout, but only if we are in training."""
            return tf.cond(tf.greater(mode, tf.constant(0)), lambda: tensor,
                           lambda: tf.nn.dropout(tensor, dropout_value))

        n_embedding = self.hyper_params['n_embedding']
        n_docs = self.hyper_params['n_classes']
        n_context = self.hyper_params['n_context_docs']

        with tf.variable_scope('net'):
            # Input embedding
            embedded_input = tf.layers.dense(inputs, n_embedding, activation=tf.nn.relu, use_bias=True)
            embedded_input = __apply_dropout(mode, embedded_input, 0.3)
            merged_layer = embedded_input

            # Context embeddings
            doc_embeddings = tf.get_variable("doc_embeddings", [47000, n_embedding], dtype=tf.float32)
            embedded_docs = tf.map_fn(lambda doc: tf.nn.embedding_lookup(doc_embeddings, doc), context,
                                      dtype=tf.float32)

            # Contact layers
            concat_embb = tf.concat([embedded_docs, tf.expand_dims(embedded_input, axis=1)], axis=1)
            #embb_dim = n_embedding * (n_context + 1)
            #concat_embb = tf.reshape(concat_embb, [tf.shape(concat_embb)[0], embb_dim])
            concat_embb = tf.reduce_mean(concat_embb, 1)
            concat_embb = __apply_dropout(mode, concat_embb, 0.3)

            # Merge layer
            merged_layer = tf.layers.dense(concat_embb, n_embedding, activation=tf.nn.relu, use_bias=True)
            merged_layer = __apply_dropout(mode, merged_layer, 0.3)

            # Output layer
            with tf.control_dependencies(None):
                with tf.name_scope("softmax_weights"):
                    softmax_w = tf.Variable(tf.truncated_normal((n_docs, n_embedding)))
                with tf.name_scope("softmax_biases"):
                    softmax_b = tf.Variable(tf.zeros(n_docs), name="softmax_bias")

        self.saver = tf.train.Saver()

        return merged_layer, softmax_w, softmax_b

    def __negative_sampling(self, softmax_w, softmax_b, labels, merged_layer):
        """Perform negative sampling and then apply the optimizer. This is for training only."""

        n_docs = self.hyper_params['n_classes']

        with tf.name_scope("loss"):
            loss = tf.nn.sampled_softmax_loss(
                weights=softmax_w,
                biases=softmax_b,
                labels=labels,
                inputs=merged_layer,
                num_sampled=self.hyper_params['n_neg_sample'],
                num_classes=n_docs)
            loss_summary = tf.summary.scalar('loss_ngs', loss[0])
        with tf.name_scope("cost"):
            cost = tf.reduce_mean(loss)
            cost_summary = tf.summary.scalar("cost_ngs", cost)
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer().minimize(cost)

        summary = tf.summary.merge([loss_summary, cost_summary])

        return optimizer, summary, cost, cost

    def __eval_loss_func(self, softmax_w, softmax_b, labels, merged_layer, mode, top_k):
        """Calculate loss over all data. Only use this for eval purposes."""

        def __eval_summary(mode, suffix, value):
            return tf.cond(tf.equal(mode, tf.constant(1)),
                lambda: tf.summary.scalar("train_" + suffix, value),
                lambda: tf.summary.scalar("val_" + suffix, value))

        with tf.name_scope('val_loss'):
            with tf.variable_scope("softmax_weights", reuse=True):
                logits = tf.matmul(merged_layer, tf.transpose(softmax_w))
            with tf.variable_scope("softmax_biases", reuse=True):
                logits = tf.nn.bias_add(logits, softmax_b)
            labels_one_hot = tf.one_hot(labels, self.hyper_params['n_classes'])
            val_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=logits)

            # Add loss to TensorBoard
            loss_summary = __eval_summary(mode, "loss", val_loss[0])

        with tf.name_scope('val_acc'):
            labels_flat = tf.map_fn(lambda l: l[0], labels)
            #correct_prediction = tf.equal(tf.argmax(logits, 1), labels_flat)
            #val_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            val_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=logits, targets=labels_flat, k=5), tf.float32))

            # Add acc to TensorBoard
            acc_summary = __eval_summary(mode, "acc", val_acc)

        # Create a fake optimizer that is never used, so that TF thinks we are returning the same data types.
        # Yes, it's a big hack.
        with tf.name_scope("fake_optimizer"):
            optimizer = tf.train.AdamOptimizer().minimize(tf.reduce_mean(val_loss))

        summary = tf.summary.merge([loss_summary, acc_summary])

        return optimizer, summary, val_loss[0], val_acc

    def model_train(self, top_k):
        start_time = time.time()
        self.logger.info("Compiling train model...")

        graph = tf.Graph()
        with graph.as_default():
            # Get input tensors
            inputs, context, labels, mode = self.__model_inputs(mode=0)

            # Define layers
            merged_layer, softmax_w, softmax_b = self.__model_net(inputs, context, mode[0])

            # Perform backprop or other optimization
            optimizer, summary, loss, acc = tf.cond(tf.greater(mode[0], tf.constant(0)),
                lambda: self.__eval_loss_func(softmax_w, softmax_b, labels, merged_layer, mode[0], top_k),
                lambda: self.__negative_sampling(softmax_w, softmax_b, labels, merged_layer))

        self.logger.info('Model compiled in {0} seconds'.format(time.time() - start_time))

        return {
            'graph': graph,
            'inputs': inputs,
            'context': context,
            'labels': labels,
            'mode': mode,
            'optimizer': optimizer,
            'loss': loss,
            'acc': acc,
            'summary': summary
        }

    def model_eval(self, top_k):
        start_time = time.time()
        self.logger.info("Compiling eval model...")

        graph = tf.Graph()
        with graph.as_default():
            # Get input tensors
            inputs, context, labels = self.__model_inputs(1)

            # Define layers
            merged_layer, softmax_w, softmax_b = self.__model_net(inputs, context, 1)

            # Calculate accuracy
            _, summary, loss, acc = self.__eval_loss_func(softmax_w, softmax_b, labels, merged_layer, 1, top_k)

        self.logger.info('Model compiled in {0} seconds'.format(time.time() - start_time))

        return {
            'graph': graph,
            'inputs': inputs,
            'context': context,
            'labels': labels,
            'loss': loss,
            'acc': acc,
            'summary': summary
        }

    def model_predict(self):
        start_time = time.time()
        self.logger.info("Compiling predict model...")

        graph = tf.Graph()
        with graph.as_default():
            # Get input tensors
            inputs, context = self.__model_inputs(3)

            # Define layers
            merged_layer, softmax_w, softmax_b = self.__model_net(inputs, context, 3)

            # Calculate prediction
            with tf.name_scope("softmax_weights"):
                logits = tf.matmul(merged_layer, tf.transpose(softmax_w))
            with tf.name_scope("softmax_biases"):
                logits = tf.nn.bias_add(logits, softmax_b)

            pred = tf.nn.top_k(logits, k=10, sorted=True)

            summary = tf.summary.merge_all()

        self.logger.info('Model compiled in {0} seconds'.format(time.time() - start_time))

        return {
            'graph': graph,
            'inputs': inputs,
            'context': context,
            'pred': pred,
            'summary': summary
        }

    # -----------------------------------------------------------------------------------------------------------------
    # DEFINE SESSIONS
    # -----------------------------------------------------------------------------------------------------------------

    def train(self, eval=True):
        model_name = "word2doc_mft"
        model_id = str(int(round(time.time())))
        log_path = self.create_run_log(model_id)

        # Set up model
        model = self.model_train(1)

        self.logger.info('Training model with hyper params:')
        self.log_hyper_params(model_id, model_name)

        # Extract relevant objects from tf model
        graph = model['graph']
        inputs_pl = model['inputs']
        context_pl = model['context']
        labels_pl = model['labels']
        mode_pl = model['mode']
        optimizer = model['optimizer']
        loss_op = model['loss']
        acc_op = model['acc']
        summary_op = model['summary']

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())

            # Set up TensorBoard
            writer = tf.summary.FileWriter(log_path, sess.graph)
            config_tb = projector.ProjectorConfig()

            # Config embeddings projector
            embedding = config_tb.embeddings.add()
            embedding.tensor_name = "doc_embeddings"

            self.train_state['total_epochs'] = self.hyper_params['epochs']

            for epoch in range(1, self.hyper_params['epochs'] + 1):

                self.train_state['epoch'] = epoch

                data_gen = self.load_train_data("./data/w2d-test")

                # If we have eval data, init the instance variables with the eval data
                if eval:
                    fake_data_gen = self.load_train_data("./data/w2d-test/3-wpp.npy")
                    for target, embeddings, context, titles in fake_data_gen:
                        self.normalize_context(context)

                # Load data
                for target, embeddings, context, titles in data_gen:
                    target = self.update_target_titles(target, titles)

                    # Unable to read a training file, so we are going to quit here and manually debug
                    if target is None:
                        self.logger.fatal('Error: Unable to read data')

                        if not len(self.train_state['files_seen']) == 0:
                            self.saver.save(sess, os.path.join(constants.get_word2doc_dir(), model_name + "_PARTIAL"))

                        # Save state before exiting
                        self.logger.info("State: " + str(self.train_state))
                        with open(os.path.join(constants.get_logs_dir(), model_name + "_PARTIAL_state" + '.json'),
                                  'w') as fp:
                            json.dump(self.train_state, fp, sort_keys=True, indent=4)
                        sys.exit()

                    if eval:
                        # Load testing data
                        target_eval, embeddings_eval, context_eval, titles_eval = self.load_test_data(
                            os.path.join(constants.get_word2doc_dir(), 'word2doc-test-bin-3.npy'))
                        context_eval = self.normalize_test_context(context_eval)
                        embeddings_eval, target_eval, context_eval = self.filter_test_data(embeddings_eval,
                                                                                           target_eval, context_eval)

                        target_eval = self.update_target_titles(target_eval, titles_eval)

                        eval_batches = self.get_eval_batches(embeddings_eval, context_eval, target_eval)

                    context = self.normalize_context(context)

                    # Shuffle data
                    self.logger.info('Shuffling data..')
                    embeddings, context, target = self.shuffle_data(embeddings, context, target)
                    self.logger.info('Done shuffling data.')

                    self.logger.info("Starting training..")

                    batches = self.get_batches(embeddings, context, target)
                    self.logger.info("Epoch " + str(epoch) + "/" + str(self.hyper_params['epochs']))

                    counter = 0
                    num_batches = self.get_num_batches(embeddings)
                    with tqdm(total=num_batches) as pbar:
                        for batch in tqdm(batches):

                            # Shuffle order of context docs within tensor
                            for b in batch[1]:
                                shuffle(b)

                            feed = {inputs_pl: batch[0], context_pl: batch[1], labels_pl: batch[2], mode_pl: [0]}
                            summary, op = sess.run([summary_op, optimizer], feed_dict=feed)

                            # Update train TensorBoard
                            writer.add_summary(summary, epoch * num_batches + counter)

                            # Update state
                            counter += 1
                            pbar.update()

                            # Perform eval every nth step
                            if counter % 10 == 0:

                                # Eval using training data
                                feed = {inputs_pl: batch[0], context_pl: batch[1], labels_pl: batch[2], mode_pl: [1]}
                                summary_train, loss, acc = sess.run([summary_op, loss_op, acc_op], feed_dict=feed)
                                writer.add_summary(summary_train, epoch * num_batches + counter)

                                # Eval using testing data
                                if eval:
                                    index = randint(0, len(eval_batches) - 1)
                                    x, c, y = zip(*eval_batches[index])
                                    feed = {inputs_pl: x, context_pl: c, labels_pl: y, mode_pl: [2]}
                                    summary_eval, loss, acc = sess.run([summary_op, loss_op, acc_op], feed_dict=feed)
                                    writer.add_summary(summary_eval, epoch * num_batches + counter)

                        # Save every nth step
                        if counter % 50 == 0:
                            self.saver.save(sess, os.path.join(constants.get_tensorboard_path(), model_name))

            # Save once for TensorBoard embeddings projector, and once for easy reuse
            self.saver.save(sess, os.path.join(constants.get_tensorboard_path(), model_name))
            self.saver.save(sess, os.path.join(constants.get_word2doc_dir(), model_name))

            self.create_embedding_labels_file(model_name, self.title_mapping, 2000)
            projector.visualize_embeddings(writer, config_tb)

    def eval(self):
        self.eval_impl(mode=1)

        total_acc = 0
        for i in range(0, 10):
            total_acc += self.eval_impl(mode=2)

        self.logger.info("Total testing accuracy: " + str(float(total_acc / 10)))

    def eval_impl(self, mode):

        # Set up model
        model = self.model_eval(10)
        model_id = str(int(round(time.time()))) + "_eval"

        self.logger.info('Evaluating model..')
        self.log_hyper_params(model_id, "test")

        # Extract relevant objects from tf model
        graph = model['graph']
        inputs_pl = model['inputs']
        context_pl = model['context']
        labels_pl = model['labels']
        loss_op = model['loss']
        acc_op = model['acc']
        summary_op = model['summary']

        with tf.Session(graph=graph) as sess:
            self.saver.restore(sess, os.path.join(constants.get_word2doc_dir(), "word2doc_test_avg"))

            # Load training data
            data_gen = self.load_train_data(os.path.join(constants.get_word2doc_dir(), '3-wpp.npy'))

            for target, embeddings, context, titles in data_gen:
                target = self.update_target_titles(target, titles)

                if mode == 2:
                    # Load testing data instead
                    target, embeddings, context_test, titles_test = self.load_test_data(
                        os.path.join(constants.get_word2doc_dir(), 'word2doc-test-400_normal.npy'))
                    context = self.normalize_test_context(context_test)
                    embeddings, target, context = self.filter_test_data(embeddings, target, context)
                    self.hyper_params['batch_size'] = 1

                    target = self.update_target_titles(target, titles_test)
                else:
                    context, ctx_titles = self.normalize_context(context)

                # Shuffle data
                self.logger.info('Shuffling data..')
                embeddings, context, target = self.shuffle_data(embeddings, context, target)
                self.logger.info('Done shuffling data.')

                self.saver.restore(sess, os.path.join(constants.get_word2doc_dir(), "word2doc_model_5000_100e_10ctx_dropout_v2"))

                num_batches = self.get_num_batches(embeddings)

                # Set batch size to 1 if we are using hand picked test set
                if mode == 2:
                    batches = self.get_eval_batches(embeddings, context, target)
                    self.hyper_params['eval_fraction'] = 1
                else:
                    batches = self.get_batches(embeddings, context, target)

                n_eval = int(num_batches * self.hyper_params['eval_fraction'])

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

                        # for b in batch[1]:
                        # shuffle(b)
                        if mode == 2:
                            x, c, y = zip(*batch)
                        else:
                            x, c, y = batch[0], batch[1], batch[2]

                        feed = {inputs_pl: x, context_pl: c, labels_pl: y}
                        summary, l, a = sess.run([summary_op, loss_op, acc_op], feed_dict=feed)

                        # Update state
                        total_loss += l
                        total_acc += a
                        counter += 1
                        pbar.update()

                total_loss = total_loss / counter
                total_acc = total_acc / counter

                # Print results
                if mode == 1:
                    self.logger.info("Train loss: " + str(total_loss) + " -- Train accuracy: " + str(total_acc))
                else:
                    self.logger.info("Test loss: " + str(total_loss) + " -- Test accuracy: " + str(total_acc))

                return total_acc

    def predict(self, x, c):
        model = self.predict_model

        graph = model['graph']
        inputs = model['inputs']
        context_pl = model['context']
        pred_op = model['pred']

        with tf.Session(graph=graph) as sess:
            self.saver.restore(sess, os.path.join(constants.get_word2doc_dir(), "word2doc_1p_72t"))

            feed = {inputs: x, context_pl: c}
            return sess.run([pred_op], feed_dict=feed)[0][1]
