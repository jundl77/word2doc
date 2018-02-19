import os
import random
import time

from random import shuffle
from random import randint
from tqdm import tqdm
import numpy as np
import prettytable
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

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

        self.hyper_params = {
            'TIME': '',
            'TRAINING PARAMS': '',
            'loss_func': 'sampled_softmax_loss',
            'optimizer': 'adam',
            'epochs': 250,
            'batch_size': 512,
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

    def load_train_data(self, path):
        self.logger.info('Load training data: ' + path)

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

    def fill_contexts(self, ctx):
        while len(ctx) < 10:
            ctx.append(0)

        return np.asarray(ctx)

    def normalize_context(self, ctx):
        ctx = list(ctx)

        i = 0
        mapping = {}
        for context in ctx:
            for num in context:
                if num not in mapping:
                    mapping[num] = i
                    i += 1

        i = 0
        for context in ctx:
            ctx[i] = list(map(lambda n: mapping[n], context))
            i += 1

        return ctx

    def normalize_test_context(self, old_ctx, new_ctx):
        """Map testing context docs to the domain of the training context docs
        (i.e. [0, 5M] -> [0, 47k] with 5k training samples"""

        # Create mapping
        old_ctx = list(old_ctx)
        i = 0
        mapping = {}
        for context in old_ctx:
            for num in context:
                if num not in mapping:
                    mapping[num] = i
                    i += 1

        # Map testing context domain to training context domain
        ctx = list(new_ctx)
        i = 0
        for context in new_ctx:

            # Map if possible (sometimes documents are not in the training domain range, so they have to be duplicated)
            single_ctx = list()
            for num in context:
                if num in mapping:
                    single_ctx.append(mapping[num])

            # Filling up contexts with duplicates
            for z in range(0, 10 - len(single_ctx)):
                index = randint(0, len(single_ctx) - 1)
                single_ctx.append(single_ctx[index])

            ctx[i] = single_ctx
            i += 1

        return ctx

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
            return tf.cond(tf.greater(mode, tf.constant(0)), lambda: tensor, lambda: tf.nn.dropout(tensor, dropout_value))

        n_embedding = self.hyper_params['n_embedding']
        n_docs = self.hyper_params['n_classes']
        n_context = self.hyper_params['n_context_docs']

        with tf.variable_scope('net'):
            # Input embedding
            embedded_input = tf.layers.dense(inputs, n_embedding, activation=tf.nn.relu, use_bias=True)
            embedded_input = __apply_dropout(mode, embedded_input, 0.3)

            # Context embeddings
            doc_embeddings = tf.get_variable("doc_embeddings", [47000, n_embedding], dtype=tf.float32)
            embedded_docs = tf.map_fn(lambda doc: tf.nn.embedding_lookup(doc_embeddings, doc), context, dtype=tf.float32)

            # Contact layers
            concat_embb = tf.concat([embedded_docs, tf.expand_dims(embedded_input, axis=1)], axis=1)
            embb_dim = n_embedding * (n_context + 1)
            concat_embb = tf.reshape(concat_embb, [tf.shape(concat_embb)[0], embb_dim])
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

    def __eval_loss_func(self, softmax_w, softmax_b, labels, merged_layer, mode):
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
            correct_prediction = tf.equal(tf.argmax(logits, 1), labels_flat)
            val_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # Add acc to TensorBoard
            acc_summary = __eval_summary(mode, "acc", val_acc)

        # Create a fake optimizer that is never used, so that TF thinks we are returning the same data types.
        # Yes, it's a big hack.
        with tf.name_scope("fake_optimizer"):
            optimizer = tf.train.AdamOptimizer().minimize(tf.reduce_mean(val_loss))

        summary = tf.summary.merge([loss_summary, acc_summary])

        return optimizer, summary, val_loss[0], val_acc

    def model_train(self):
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
                lambda: self.__eval_loss_func(softmax_w, softmax_b, labels, merged_layer, mode[0]),
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

    def model_eval(self):
        start_time = time.time()
        self.logger.info("Compiling eval model...")

        graph = tf.Graph()
        with graph.as_default():
            # Get input tensors
            inputs, context, labels = self.__model_inputs(1)

            # Define layers
            merged_layer, softmax_w, softmax_b = self.__model_net(inputs, context, 1)

            # Calculate accuracy
            _, summary, loss, acc = self.__eval_loss_func(softmax_w, softmax_b, labels, merged_layer, 1)

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
            pred = tf.argmax(logits, 1)

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

    def train(self, eval=False):
        data_name = "3-wpp.npy"
        model_name = "word2doc_model_5000_100e_10ctx_dropout_v2"
        model_id = str(int(round(time.time())))
        log_path = self.create_run_log(model_id)

        # Load data
        target, embeddings, context, titles = self.load_train_data(os.path.join(constants.get_word2doc_dir(), data_name))

        if eval:
            target_eval, embeddings_eval, context_eval, titles_eval = self.load_test_data(
                os.path.join(constants.get_word2doc_dir(), 'word2doc-test-bin-3.npy'))
            context_eval = self.normalize_test_context(context, context_eval)
            eval_batches = self.get_eval_batches(embeddings_eval, context_eval, target_eval)

        context = self.normalize_context(context)

        # Shuffle data
        self.logger.info('Shuffling data..')
        embeddings, context, target = self.shuffle_data(embeddings, context, target)
        self.logger.info('Done shuffling data.')

        # Set up model
        model = self.model_train()

        self.logger.info('Training model with hyper params:')
        self.log_hyper_params(model_id)

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
            config = projector.ProjectorConfig()

            # Config embeddings projector
            embedding = config.embeddings.add()
            doc_embeddings = tf.get_variable("doc_embeddings", [47000, 512], dtype=tf.float32)
            embedding.tensor_name = doc_embeddings.name

            self.logger.info("Starting training..")

            for epoch in range(1, self.hyper_params['epochs'] + 1):

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
                        summary, _ = sess.run([summary_op, optimizer], feed_dict=feed)

                        # Update train TensorBoard
                        writer.add_summary(summary, epoch * num_batches + counter)

                        # Update state
                        counter += 1
                        pbar.update()

                        # Perform eval every nth step
                        if counter % 20 == 0:

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
                        if counter % 100 == 0:
                            self.saver.save(sess, os.path.join(log_path, model_name))

            # Save once for TensorBoard embeddings projector, and once for easy reuse
            self.saver.save(sess, os.path.join(log_path, model_name))
            self.saver.save(sess, os.path.join(constants.get_word2doc_dir(), model_name))
            projector.visualize_embeddings(writer, config)

    def eval(self):
        self.eval_impl(mode=1)

        total_acc = 0
        for i in range(0, 10):
            total_acc += self.eval_impl(mode=2)

        self.logger.info("Total testing accuracy: " + str(float(total_acc / 10)))

    def eval_impl(self, mode):

        # Load training data
        target, embeddings, context, titles = self.load_train_data(os.path.join(constants.get_word2doc_dir(), '3-wpp.npy'))

        if mode == 2:
            # Load testing data instead
            target, embeddings, context_test, titles = self.load_test_data(
                os.path.join(constants.get_word2doc_dir(), 'word2doc-test-bin-3.npy'))
            context = self.normalize_test_context(context, context_test)
            self.hyper_params['batch_size'] = 1
        else:
            context = self.normalize_context(context)

        # Shuffle data
        self.logger.info('Shuffling data..')
        embeddings, context, target = self.shuffle_data(embeddings, context, target)
        self.logger.info('Done shuffling data.')

        # Set up model
        model = self.model_eval()
        model_id = str(int(round(time.time()))) + "_eval"

        self.logger.info('Evaluating model..')
        self.log_hyper_params(model_id)

        # Extract relevant objects from tf model
        graph = model['graph']
        inputs_pl = model['inputs']
        context_pl = model['context']
        labels_pl = model['labels']
        loss_op = model['loss']
        acc_op = model['acc']
        summary_op = model['summary']

        with tf.Session(graph=graph) as sess:
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
        pred_op = model['op']

        with tf.Session(graph=graph) as sess:
            self.saver.restore(sess,
                               os.path.join(constants.get_word2doc_dir(), "word2doc_model_5000_2_200e_10ctx_dropout"))

            feed = {inputs: x, context_pl: c}
            return sess.run([pred_op], feed_dict=feed)
