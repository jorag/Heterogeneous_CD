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
    polmak_list = ["Polmak-LS5-S2", "Polmak-LS5-S2-collocate", 
    "Polmak-LS5-S2-warp", "Polmak-A2-S2", "Polmak-A2-S2-collocate", "Polmak-LS5-PGNLM_A", 
    "Polmak-LS5-PGNLM_C", "Polmak-LS5-PGNLM_A-stacked", "Polmak-LS5-PGNLM_C-stacked"]
    #process_list = ["Polmak-LS5-S2"] #["Polmak-Pal-RS2_010817-collocate"]
    process_list = polmak_list
    for DATASET in process_list:
        CONFIG = get_config_kACE(DATASET)
        suffix = ""

        if DATASET in polmak_list:
            print("Usinging Polmak processing dict")
            load_options = dict()
            load_options["norm_type"] = "_clip_norm" #  "_norm01" # 
            suffix += load_options["norm_type"]
            load_options["debug"] = True
            load_options["row_shift"] = int(0)
            load_options["col_shift"] = int(0)
            load_options["reduce"] = True
        else:
            load_options = None
        
        if DATASET in ["Polmak-LS5-S2_ONLY_align"]:
            print("Not using AUC!")
            from filtering import decorated_median_filter, decorated_gaussian_filter
            CONFIG["final_filter"] = decorated_median_filter("z_median_filtered_diff")
        if DATASET in ["Polmak-Pal-RS2_010817-collocate"]:
            CONFIG["channel_x"] = [0]
            CONFIG["channel_y"] = [0, 1, 3]
        
        # Set x-channels
        if DATASET in ["Polmak-LS5-S2", "Polmak-LS5-S2-collocate", "Polmak-LS5-S2-warp", 
            "Polmak-LS5-PGNLM_A", "Polmak-LS5-PGNLM_C", "Polmak-LS5-PGNLM_A-stacked", "Polmak-LS5-PGNLM_C-stacked"]:
            CONFIG["channel_x"] = [0, 1, 2] # LS5 RGB
        elif DATASET in ["Polmak-A2-S2", "Polmak-A2-S2-collocate"]:
            CONFIG["channel_x"] = [2, 1, 0] # ALOS AVNIR
        elif DATASET in ["Polmak-LS5-S2-NDVI"]:
            CONFIG["channel_x"] = [0] # NDVI
        
        # Set y-channels
        if DATASET in ["Polmak-LS5-S2", "Polmak-LS5-S2-collocate", "Polmak-LS5-S2-warp", 
            "Polmak-A2-S2", "Polmak-A2-S2-collocate"]:
            CONFIG["channel_y"] = [2, 1, 0] # S2 RGB
        if DATASET in ["Polmak-LS5-PGNLM_A", "Polmak-LS5-PGNLM_C", "Polmak-LS5-PGNLM_A-stacked", "Polmak-LS5-PGNLM_C-stacked"]:
            CONFIG["channel_y"] = [0, 1, 2] # C11, C22, C33
        if DATASET in ["Polmak-LS5-PGNLM_A-stacked", "Polmak-LS5-PGNLM_C-stacked"]:
            CONFIG["channel_y"] = [7, 1, 2] # Red, C22, C33

        # Set dataset shift
        if DATASET in ["Polmak-LS5-S2", "Polmak-LS5-S2-NDVI", "Polmak-LS5-S2-warp"]:
            load_options["col_shift"] = int(2)
            suffix += "_shift_col" + str(load_options["col_shift"])
        elif DATASET in ["Polmak-LS5-PGNLM_A", "Polmak-LS5-PGNLM_C", "Polmak-LS5-PGNLM_A-stacked", "Polmak-LS5-PGNLM_C-stacked"]:
            load_options["col_shift"] = int(6)
            suffix += "_shift_col" + str(load_options["col_shift"])

        if load_options is not None and load_options["row_shift"] != 0:
            if load_options["row_shift"]:
                suffix += "_shift_row" + str(load_options["row_shift"])
        
        # Check if suffix should be added 
        if load_options is not None and load_options["reduce"]:
            CONFIG["logdir"] = f"logs/reduce/{DATASET}/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-alpha" + suffix
        else:
            CONFIG["logdir"] = f"logs/{DATASET}/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-alpha"+ suffix

        # Add load_options to CONFIG dict if it exists
        if load_options is not None:
            CONFIG["load_options"] = load_options



        test(DATASET, CONFIG)
