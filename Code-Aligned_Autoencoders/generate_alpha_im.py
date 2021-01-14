import os.path
import tensorflow as tf

from datetime import datetime
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tqdm import trange
import numpy as np

from filtering import threshold_otsu
from decorators import image_to_tensorboard, timed
from metrics import CohenKappa
from config import get_config_kACE
import datasets
from change_priors import patched_alpha

# instead of "from tensorflow_addons.metrics import CohenKappa" due to
# https://github.com/tensorflow/addons/pull/675


class ChangePrior:
    """docstring for ChangePrior."""

    def __init__(self, **kwargs):
        """
            Input:
                translation_spec - dict with keys 'f_X', 'f_Y'.
                                   Values are passed as kwargs to the
                                   respective ImageTranslationNetwork's
                cycle_lambda=2 - float, loss weight
                cross_lambda=1 - float, loss weight
                l2_lambda=1e-3 - float, loss weight
                learning_rate=1e-5 - float, initial learning rate for
                                     ExponentialDecay
                clipnorm=None - gradient norm clip value, passed to
                                tf.clip_by_global_norm if not None
                logdir=None - path to log directory. If provided, tensorboard
                              logging of training and evaluation is set up at
                              'logdir/'
        """
        #learning_rate = kwargs.get("learning_rate", 1e-5)
        #lr_all = ExponentialDecay(
        #    learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True
        #)
        #self._optimizer_all = tf.keras.optimizers.Adam(lr_all)
        #lr_k = ExponentialDecay(
        #    learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True
        #)
        #self._optimizer_k = tf.keras.optimizers.Adam(lr_k)
        #self.clipnorm = kwargs.get("clipnorm", None)

        # To keep a history for a specific training_metrics,
        # add `self.metrics_history[name] = []` in subclass __init__
        #self.train_metrics = {}
        #self.difference_img_metrics = {"ACC_di": tf.keras.metrics.Accuracy()} #{"AUC": tf.keras.metrics.AUC()}
        #self.change_map_metrics = {
        #    "ACC": tf.keras.metrics.Accuracy(),
        #    "cohens kappa": CohenKappa(num_classes=2),
            # 'F1': tfa.metrics.F1Score(num_classes=2, average=None)
        #}
        #assert not set(self.difference_img_metrics) & set(self.change_map_metrics)
        # If the metric dictionaries shares keys, the history will not work
        #self.metrics_history = {
        #    **{key: [] for key in self.change_map_metrics.keys()},
        #    **{key: [] for key in self.difference_img_metrics.keys()},
        #}

        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.channels = {"x": kwargs.get("channel_x"), "y": kwargs.get("channel_y")}

        # Flag used in image_to_tensorboard decorator
        self._save_images = tf.Variable(False, trainable=False)

        logdir = kwargs.get("logdir", None)
        if logdir is not None:
            self.log_path = logdir
            self.tb_writer = tf.summary.create_file_writer(self.log_path)
            self._image_dir = tf.constant(os.path.join(self.log_path, "images"))
        else:
            self.tb_writer = tf.summary.create_noop_writer()

        self.evaluation_frequency = tf.constant(
            kwargs.get("evaluation_frequency", 1), dtype=tf.int64
        )
        self.epoch = tf.Variable(0, dtype=tf.int64)

    @image_to_tensorboard()
    def print_image(self, x):
        return x

    def print_all_input_images(self, evaluation_dataset):
        tf.summary.experimental.set_step(self.epoch + 1)
        self._save_images.assign(True)
        for x, y, z in evaluation_dataset.batch(1):
            self.print_image(x, name="x")
            self.print_image(y, name="y")
            self.print_image(tf.cast(z, dtype=tf.float32), name="Ground_Truth")
        self._save_images.assign(False)
        tf.summary.flush(self.tb_writer)

    def print_alpha_prior(self, evaluation_dataset, CONFIG):
        tf.summary.experimental.set_step(self.epoch + 1)
        self._save_images.assign(True)
        for x, y, z in evaluation_dataset.batch(1):
            print("x.shape = ", x.shape)
            print("y.shape = ", y.shape)
            print("z.shape = ", z.shape)
            alpha = patched_alpha(np.squeeze(x, axis=0), 
                np.squeeze(y, axis=0), 20, 1, **CONFIG)
            print("Org alpha.shape = ", alpha.shape)
            self.print_image(x, name="x")
            self.print_image(y, name="y")
            alpha = alpha[np.newaxis,...]
            print("1 alpha.shape = ", alpha.shape)
            alpha = alpha[..., np.newaxis]
            print("2 alpha.shape = ", alpha.shape)
            #self.print_image(alpha, name="alpha")
            self.print_image(tf.cast(alpha, dtype=tf.float32), name="alpha")
            self.print_image(tf.cast(z, dtype=tf.float32), name="Ground_Truth")
        self._save_images.assign(False)
        tf.summary.flush(self.tb_writer)


def test(DATASET="Texas", CONFIG =None):
    if CONFIG is None:
        CONFIG = get_config(DATASET)
    _, _, EVALUATE, _ = datasets.fetch(DATASET, **CONFIG)
    cd = ChangePrior(**CONFIG)
    #cd.print_all_input_images(EVALUATE)
    cd.print_alpha_prior(EVALUATE, CONFIG)


if __name__ == "__main__":
    #DATASET = "Polmak-LS5-PGNLM_C-stacked" 
    polmak_list = ["Polmak-LS5-S2", "Polmak-LS5-S2-collocate", 
    "Polmak-LS5-S2-warp", "Polmak-A2-S2", "Polmak-A2-S2-collocate", "Polmak-LS5-PGNLM_A", 
    "Polmak-LS5-PGNLM_C", "Polmak-LS5-PGNLM_A-stacked", "Polmak-LS5-PGNLM_C-stacked"]
    process_list = ["Polmak-LS5-S2"] #["Polmak-Pal-RS2_010817-collocate"]
    for DATASET in process_list:
        CONFIG = get_config_kACE(DATASET)
        if DATASET in polmak_list:
            CONFIG["channel_y"] = [2, 1, 0]
            if DATASET in ["Polmak-LS5-S2_ONLY_align"]:
                from filtering import decorated_median_filter, decorated_gaussian_filter
                CONFIG["final_filter"] = decorated_median_filter("z_median_filtered_diff")
        if DATASET in ["Polmak-Pal-RS2_010817-collocate"]:
            CONFIG["channel_x"] = [0, 1]
            CONFIG["channel_y"] = [0, 1, 3]
        if DATASET in ["Polmak-LS5-S2"]:
            CONFIG["channel_x"] = [0, 1, 2]
            CONFIG["channel_y"] = [2, 1, 0]
        else:
            CONFIG = get_config_kACE(DATASET)
        # Change
        CONFIG["logdir"] = f"logs/{DATASET}/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-alpha"
        print("11 January - test alpha dataset: ", DATASET)
        test(DATASET, CONFIG)
