"""Library for finetuning pretrained models."""
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


def encode_traces(trace_set, encoder):
    """Encodes traces into a numpy matrix."""
    with tf.device('CPU:0'):
        embedding = encoder.predict(trace_set.to_tensor(), verbose=0)
    return embedding


def finetune_classification_head(encoder, train_sets, epochs, num_classes=2, trace_level=True, lr=1e-3, verbose=0, return_embedding=False, embedding_list=None):
    """Finetunes a classification head."""
    if trace_level:
        if embedding_list is None:
            embedding_list = [encode_traces(trace_set, encoder) for trace_set in train_sets]
        label_list = [tf.cast(np.max(trace_set.label, axis=-1), tf.int64) for trace_set in train_sets]
    else:
        if embedding_list is None:
            embedding_list = [tf.reshape(encode_traces(trace_set, encoder), (-1, encoder.output_shape[-1])) for trace_set in train_sets]
        label_list = [tf.cast(tf.reshape(trace_set.label, (-1, )), tf.int64) for trace_set in train_sets]

    embedding = tf.concat(embedding_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    readout = keras.Sequential([
        keras.layers.Dense(num_classes),
    ])

    readout.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.legacy.Adam(lr),
    )

    #TODO: Customize class weights if necessary.
    class_weights = None

    if trace_level:
        readout.fit(embedding, label, batch_size=256, epochs=100, verbose=verbose, class_weight=class_weights,)
    else:
        readout.fit(embedding, label, batch_size=1024, epochs=2, verbose=verbose, class_weight=class_weights,)

    classifier = keras.Sequential([
        encoder,
        readout,
    ])
    if return_embedding:
        return classifier, embedding_list
    else:
        return classifier