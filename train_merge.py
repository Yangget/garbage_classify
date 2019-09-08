# -*- coding: utf-8 -*-
import multiprocessing
import os
import shutil
from glob import glob

import numpy as np
from keras import backend
from keras.callbacks import TensorBoard, Callback
from keras.layers import Dropout
from keras.layers import Flatten, Dense, Input, concatenate, PReLU
from keras.models import Model
from keras.optimizers import Adam

from data_gen_label import data_flow
from models.densenet import DenseNet201
from models.mobilenet_v2 import MobileNetV2
from models.xception import Xception
import efficientnet.keras as efn
from warmup_cosine_decay_scheduler import WarmUpCosineDecayScheduler
backend.set_image_data_format('channels_last')


def model_fn(FLAGS, objective, optimizer, metrics):

    input_layer = Input(shape=(FLAGS.input_size, FLAGS.input_size, 3))

    base_model1 = Xception(weights="imagenet",
                           include_top=False,
                           pooling='avg',
                           input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                           classes=FLAGS.num_classes)
    base_model2 = MobileNetV2(weights='imagenet',
                              include_top=False,
                              pooling='avg',
                              alpha=1.4,
                              input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                              classes=FLAGS.num_classes,
                              )
    base_model3 = DenseNet201(weights="imagenet",
                              include_top=False,
                              pooling='avg',
                              input_shape=(FLAGS.input_size, FLAGS.input_size, 3),
                              classes=FLAGS.num_classes)
    x1 = base_model1(inputs=input_layer)
    x2 = base_model2(inputs=input_layer)
    x3 = base_model3(inputs=input_layer)

    # 值得修改的地方
    x1 = Dense(FLAGS.num_classes, activation='relu')(x1)
    x2 = Dense(FLAGS.num_classes, activation='relu')(x2)
    x3 = Dense(FLAGS.num_classes, activation='relu')(x3)
    x = concatenate([x1, x2, x3], axis=1)

    predictions = Dense(FLAGS.num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(loss=objective, optimizer=optimizer, metrics=metrics)
    return model


class LossHistory(Callback):
    def __init__(self, FLAGS):
        super(LossHistory, self).__init__( )
        self.FLAGS = FLAGS

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        save_path = os.path.join(self.FLAGS.train_local, 'weights_%03d_%.4f.h5' % (epoch, logs.get('val_acc')))
        self.model.save_weights(save_path)
        if self.FLAGS.train_url.startswith('s3://'):
            save_url = os.path.join(self.FLAGS.train_url, 'weights_%03d_%.4f.h5' % (epoch, logs.get('val_acc')))
            shutil.copyfile(save_path, save_url)
        print('save weights file', save_path)

        if self.FLAGS.keep_weights_file_num > -1:
            weights_files = glob(os.path.join(self.FLAGS.train_local, '*.h5'))
            if len(weights_files) >= self.FLAGS.keep_weights_file_num:
                weights_files.sort(key=lambda file_name: os.stat(file_name).st_ctime, reverse=True)
                for file_path in weights_files[self.FLAGS.keep_weights_file_num:]:
                    os.remove(file_path)  # only remove weights files on local path


def train_model(FLAGS):
    # 注意采用的归一化的方式
    preprocess_input = efn.preprocess_input

    train_sequence, validation_sequence = data_flow(FLAGS.data_local, FLAGS.batch_size,
                                                    FLAGS.num_classes, FLAGS.input_size, preprocess_input)

    optimizer = Adam(lr=FLAGS.learning_rate)

    objective = 'categorical_crossentropy'
    metrics = ['accuracy']

    model = model_fn(FLAGS, objective, optimizer, metrics)

    if FLAGS.restore_model_path != '' and os.path.exists(FLAGS.restore_model_path):
        if FLAGS.restore_model_path.startswith('s3://'):
            restore_model_name = FLAGS.restore_model_path.rsplit('/', 1)[1]
            shutil.copyfile(FLAGS.restore_model_path, '/cache/tmp/' + restore_model_name)
            model.load_weights('/cache/tmp/' + restore_model_name)
            os.remove('/cache/tmp/' + restore_model_name)
        else:
            model.load_weights(FLAGS.restore_model_path)
        print("LOAD OK!!!")
    if not os.path.exists(FLAGS.train_local):
        os.makedirs(FLAGS.train_local)

    log_local = '../log_file/'


    tensorBoard = TensorBoard(log_dir=log_local)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, mode='auto')

    sample_count = len(train_sequence) * FLAGS.batch_size
    epochs = FLAGS.max_epochs
    warmup_epoch = 5
    batch_size = FLAGS.batch_size
    learning_rate_base = FLAGS.learning_rate
    total_steps = int(epochs * sample_count / batch_size)
    warmup_steps = int(warmup_epoch * sample_count / batch_size)

    warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
                                            total_steps=total_steps,
                                            warmup_learning_rate=0,
                                            warmup_steps=warmup_steps,
                                            hold_base_rate_steps=0,
                                            )

    history = LossHistory(FLAGS)
    model.fit_generator(
        train_sequence,
        steps_per_epoch=len(train_sequence),
        epochs=FLAGS.max_epochs,
        verbose=1,
        callbacks=[history, tensorBoard, warm_up_lr],
        validation_data=validation_sequence,
        max_queue_size=10,
        workers=int(multiprocessing.cpu_count( ) * 0.7),
        use_multiprocessing=True,
        shuffle=True
    )

    print('training done!')

    if FLAGS.deploy_script_path != '':
        from save_model import save_pb_model
        save_pb_model(FLAGS, model)

    if FLAGS.test_data_url != '':
        print('test dataset predicting...')
        from eval import load_test_data
        img_names, test_data, test_labels = load_test_data(FLAGS)
        test_data = preprocess_input(test_data)
        predictions = model.predict(test_data, verbose=0)

        right_count = 0
        for index, pred in enumerate(predictions):
            predict_label = np.argmax(pred, axis=0)
            test_label = test_labels[index]
            if predict_label == test_label:
                right_count += 1
        accuracy = right_count / len(img_names)
        print('accuracy: %0.4f' % accuracy)
        metric_file_name = os.path.join(FLAGS.train_local, 'metric.json')
        metric_file_content = '{"total_metric": {"total_metric_values": {"accuracy": %0.4f}}}' % accuracy
        with open(metric_file_name, "w") as f:
            f.write(metric_file_content + '\n')
    print('end')
