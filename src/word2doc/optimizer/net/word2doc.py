import os
import random
import time

from random import shuffle
from random import randint
from tqdm import tqdm
import numpy as np
import prettytable
import tensorflow as tf

from word2doc.util import constants
from word2doc.util import logger


class Word2Doc:
    def __init__(self):
        self.logger = logger.get_logger()
        self.saver = None

        self.hyper_params = {
            'TIME': '',
            'TRAINING PARAMS': '',
            'loss_func': 'sampled_softmax_loss',
            'optimizer': 'adam',
            'epochs': 200,
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
            'n_classes': 5000,
            'out_activation': 'softmax',
        }

        self.predict_model = self.model("predict", False)

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
        data = list(zip(embeddings, context, target))

        batches = np.array_split(data, int(len(embeddings) / self.hyper_params['batch_size']))

        for batch in batches:
            x, c, y = zip(*batch)
            yield x, c, y

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

        if mode is "predict":
            return inputs, context
        else:
            with tf.name_scope('labels'):
                labels = tf.placeholder(tf.int64, shape=(None, 1), name='labels')
            with tf.name_scope('is_eval'):
                is_eval = tf.placeholder(tf.bool, shape=(None, 1), name='is_eval')

            return inputs, context, labels, is_eval

    def __model_net(self, inputs, context, mode):
        """Define the net here."""

        n_embedding = self.hyper_params['n_embedding']
        n_docs = self.hyper_params['n_classes']
        n_context = self.hyper_params['n_context_docs']

        with tf.variable_scope('net'):
            # Input embedding
            embedded_input = tf.layers.dense(inputs, n_embedding, activation=tf.nn.relu, use_bias=True)
            if mode == "train":
                embedded_input = tf.nn.dropout(embedded_input, 0.3)

            # Context embeddings
            doc_embeddings = tf.get_variable("doc_embeddings", [47000, n_embedding], dtype=tf.float32)
            embedded_docs = tf.map_fn(lambda doc: tf.nn.embedding_lookup(doc_embeddings, doc), context,
                                      dtype=tf.float32)
            if mode == "train":
                embedded_docs = tf.nn.dropout(embedded_docs, 0.3)

            # Contact layers
            concat_embb = tf.concat([embedded_docs, tf.expand_dims(embedded_input, axis=1)], axis=1)
            embb_dim = n_embedding * (n_context + 1)
            concat_embb = tf.reshape(concat_embb, [tf.shape(concat_embb)[0], embb_dim])
            if mode == "train":
                concat_embb = tf.nn.dropout(concat_embb, 0.3)

            # Merge layer
            merged_layer = tf.layers.dense(concat_embb, n_embedding, activation=tf.nn.relu, use_bias=True)
            if mode == "train":
                merged_layer = tf.nn.dropout(merged_layer, 0.3)

            # Output layer
            with tf.variable_scope("softmax_weights"):
                softmax_w = tf.Variable(tf.truncated_normal((n_docs, n_embedding)))
            with tf.variable_scope("softmax_biases"):
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
            tf.summary.scalar('train_loss', loss[0])
        with tf.name_scope("cost"):
            cost = tf.reduce_mean(loss)
            tf.summary.scalar("cost", cost)
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer().minimize(cost)

        return optimizer

    def __eval_loss_func(self, softmax_w, softmax_b, labels, merged_layer):
        """Calculate loss over all data. Only use this for eval purposes."""

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

        return [val_loss, val_acc]

    def model(self):
        start_time = time.time()
        self.logger.info("Compiling model...")

        graph = tf.Graph()
        with graph.as_default():
            inputs, context, labels, is_eval = self.__model_inputs("train")

            merged_layer, softmax_w, softmax_b = tf.cond(is_eval,
                                                         lambda: self.__model_net(inputs, context, "eval"),
                                                         lambda: self.__model_net(inputs, context, "train"))

            res = tf.cond(is_eval,
                         lambda: self.__eval_loss_func(softmax_w, softmax_b, labels, merged_layer),
                         lambda: self.__negative_sampling(softmax_w, softmax_b, labels, merged_layer))

            summary = tf.summary.merge_all()

        self.logger.info('Model compiled in {0} seconds'.format(time.time() - start_time))

        return {
            'inputs': inputs,
            'context': context,
            'labels': labels,
            'is_eval': is_eval,
            'res': res,
            'summary': summary
        }

    def model_predict(self):
        start_time = time.time()
        self.logger.info("Compiling predict model...")

        graph = tf.Graph()
        with graph.as_default():
            inputs, context = self.__model_inputs("predict")
            merged_layer, softmax_w, softmax_b = self.__model_net(inputs, context, "predict")

            with tf.variable_scope("softmax_weights", reuse=True):
                logits = tf.matmul(merged_layer, tf.transpose(softmax_w))
            with tf.variable_scope("softmax_biases", reuse=True):
                logits = tf.nn.bias_add(logits, softmax_b)
            pred = tf.argmax(logits, 1)

            summary = tf.summary.merge_all()

        self.logger.info('Model compiled in {0} seconds'.format(time.time() - start_time))

        return {
            'inputs': inputs,
            'context': context,
            'pred': pred,
            'summary': summary
        }

    # -----------------------------------------------------------------------------------------------------------------
    # DEFINE SESSION
    # -----------------------------------------------------------------------------------------------------------------

    def train(self):
        # Load data
        target, embeddings, context, titles = self.load_test_data(
            os.path.join(constants.get_word2doc_dir(), '3-wpp.npy'))

        context = self.normalize_context(context)

        # Shuffle data
        self.logger.info('Shuffling data..')
        embeddings, context, target = self.shuffle_data(embeddings, context, target)
        self.logger.info('Done shuffling data.')

        # Set up model
        model = self.train_model()
        model_id = str(int(round(time.time())))

        self.logger.info('Training model with hyper params:')
        self.log_hyper_params(model_id)

        graph = model['graph']
        inputs = model['inputs']
        context_pl = model['context']
        labels = model['labels']
        op = model['train_op']
        summary_opt = model['train_summary']
        eval_summary_opt = model['eval_summary']

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())

            # Set up TensorBoard
            writer = tf.summary.FileWriter(self.create_run_log(model_id), sess.graph)

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

                        feed = {inputs: batch[0], context_pl: batch[1], labels: batch[2]}
                        summary, _ = sess.run([summary_opt, op], feed_dict=feed)

                        # Update train TensorBoard
                        writer.add_summary(summary, epoch * num_batches + counter)

                        # Update state
                        counter += 1
                        pbar.update()

                        if counter % 1000 == 0:
                            summary = sess.run([eval_summary_opt], feed_dict=feed)

                            # Update eval TensorBoard
                            writer.add_summary(summary, epoch * self.n_batches + counter)

            self.saver.save(sess,
                            os.path.join(constants.get_word2doc_dir(), "word2doc_model_5000_2_200e_10ctx_dropout"))

    def eval(self):
        self.eval_impl("train")

        total_acc = 0
        for i in range(0, 10):
            total_acc += self.eval_impl("test")

        self.logger.info("Total testing accuracy: " + str(float(total_acc / 10)))

    def eval_impl(self, mode):

        # Load training data
        target, embeddings, context, titles = self.load_train_data(
            os.path.join(constants.get_word2doc_dir(), '3-wpp.npy'))

        if not mode == "train":
            # Load testing data instead
            target, embeddings, context_test, titles = self.load_test_data(
                os.path.join(constants.get_word2doc_dir(), 'word2doc-test-bin-3.npy'))
            context = self.normalize_test_context(context, context_test)
        else:
            context = self.normalize_context(context)

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
        context_pl = model['context']
        labels = model['labels']
        acc = model['val_acc']
        loss = model['val_loss']
        summary_opt = model['summary']

        with tf.Session(graph=train_graph) as sess:
            self.saver.restore(sess,
                               os.path.join(constants.get_word2doc_dir(), "word2doc_model_5000_2_200e_10ctx_dropout"))

            # Set up TensorBoard
            writer = tf.summary.FileWriter(self.create_run_log(model_id), sess.graph)

            # Set batch size to 1 if we are using hand picked test set
            if not mode == "train":
                self.hyper_params['batch_size'] = 1
                self.hyper_params['eval_fraction'] = 1

            num_batches = self.get_num_batches(embeddings)
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
                    feed = {inputs: batch[0], context_pl: batch[1], labels: batch[2]}
                    summary, l, a = sess.run([summary_opt, loss, acc], feed_dict=feed)

                    # Update TensorBoard
                    writer.add_summary(summary, num_batches + counter)

                    # Update state
                    total_loss += l
                    total_acc += a
                    counter += 1
                    pbar.update()

            total_loss = total_loss / counter
            total_acc = total_acc / counter

            # Print results
            if mode == "train":
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
