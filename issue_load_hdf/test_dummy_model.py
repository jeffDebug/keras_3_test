"""
This test is run with keras 3.3.3, tensorflow 2.16.1 and torch 2.3.0+cu118.
"""

import sys
import keras

import keras.ops as kops
import numpy as np

from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Flatten, Input
from keras.models import Model, load_model

TEST_INPUT_CHANNEL = 3
TEST_INPUT_SIZE = 3
TEST_OUTPUT_SIZE = 1
BATCH_SIZE = 32


def make_test_model():
    inp = Input(
        (TEST_INPUT_CHANNEL, TEST_INPUT_SIZE, TEST_INPUT_SIZE),
        batch_size=BATCH_SIZE,
        name="Input",
    )
    bn0 = BatchNormalization(axis=1, scale=False, name="bn_0")(inp)

    conv0 = Conv2D(
        256,
        (2, 2),
        activation="relu",
        data_format="channels_first",
        name="conv2d_0",
    )(bn0)
    conv1 = Conv2D(
        128,
        (2, 2),
        activation="relu",
        data_format="channels_first",
        name="conv2d_1",
    )(bn0)

    concat0 = Concatenate(axis=1, name="concat_0")([conv1, conv0])
    bn1 = BatchNormalization(axis=1, scale=False, name="bn_1")(concat0)

    conv2 = Conv2D(
        64,
        (2, 2),
        activation="relu",
        data_format="channels_first",
        name="conv2d_2",
    )(bn1)
    flat0 = Flatten(name="flat_0")(conv2)
    oup = Dense(TEST_OUTPUT_SIZE, activation="relu", name="output")(flat0)

    model = Model(inputs=inp, outputs=oup)
    return model


def f_test(x):
    return kops.expand_dims(kops.max(x, axis=(1, 2, 3)), axis=1)


if "__main__" == __name__:
    savepath = sys.argv[1]

    # create the keras model
    model = make_test_model()
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.MeanSquaredError(),
    )
    model.summary()

    # train the model
    print(f"\nStart training", flush=True)
    x = np.random.randn(1024, TEST_INPUT_CHANNEL, TEST_INPUT_SIZE, TEST_INPUT_SIZE)
    y = f_test(x)
    model.fit(x, y, batch_size=BATCH_SIZE, epochs=5)

    # save the model
    print(f"\nStart saving model {savepath}", flush=True)
    model.save(savepath, overwrite=True)
    print(f"Finished saving {savepath}", flush=True)

    # load the model
    try:
        print(f"\nStart loading model {savepath}", flush=True)
        model_loaded = load_model(savepath)
        print(f"Done loading model {savepath}", flush=True)
        model_loaded.summary()
    except Exception:
        import traceback

        print(f"\nloading model {savepath} failed.", flush=True)
        print(traceback.format_exc(), flush=True)
