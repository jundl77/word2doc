from keras.callbacks import TensorBoard


class LoggableTensorBoard(TensorBoard):
    """A TensorBoard subclass that lets you log custom values"""
    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None,
                 custom_log_func=None):
        TensorBoard.__init__(self,
                             log_dir=log_dir,
                             histogram_freq=histogram_freq,
                             batch_size=batch_size,
                             write_graph=write_graph,
                             write_grads=write_grads,
                             write_images=write_images,
                             embeddings_freq=embeddings_freq,
                             embeddings_layer_names=embeddings_layer_names,
                             embeddings_metadata=embeddings_metadata)

        self.custom_log_func = custom_log_func

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch. Modified to add custom log values so that they also
        show up on TensorBoard"""
        if self.custom_log_func is not None:
            logs.update(self.custom_log_func(self, epoch, logs))

        super(LoggableTensorBoard, self).on_epoch_end(epoch, logs)

