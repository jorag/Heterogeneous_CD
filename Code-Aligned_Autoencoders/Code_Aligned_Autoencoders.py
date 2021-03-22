import os
import gc

# Set loglevel to suppress tensorflow GPU messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import datasets
from change_detector import ChangeDetector
from image_translation import ImageTranslationNetwork
from change_priors import Degree_matrix, ztz, image_in_patches, Degree_matrix_fixed_krnl
from config import get_config_kACE
from decorators import image_to_tensorboard
import numpy as np
from datetime import datetime


class Kern_AceNet(ChangeDetector):
    def __init__(self, translation_spec, **kwargs):
        """
                Input:
                    translation_spec - dict with keys 'enc_X', 'enc_Y', 'dec_X', 'dec_Y'.
                                       Values are passed as kwargs to the
                                       respective ImageTranslationNetworks
                    cycle_lambda=2 - float, loss weight
                    cross_lambda=1 - float, loss weight
                    l2_lambda=1e-3 - float, loss weight
                    kernels_lambda - float, loss weight
                    learning_rate=1e-5 - float, initial learning rate for
                                         ExponentialDecay
                    clipnorm=None - gradient norm clip value, passed to
                                    tf.clip_by_global_norm if not None
                    logdir=None - path to log directory. If provided, tensorboard
                                  logging of training and evaluation is set up at
                                  'logdir/timestamp/' + 'train' and 'evaluation'
        """

        super().__init__(**kwargs)

        self.cycle_lambda = kwargs.get("cycle_lambda", 0.2)
        self.cross_lambda = kwargs.get("cross_lambda", 0.1)
        self.recon_lambda = kwargs.get("recon_lambda", 0.1)
        self.l2_lambda = kwargs.get("l2_lambda", 1e-6)
        self.kernels_lambda = kwargs.get("kernels_lambda", 1)
        self.min_impr = kwargs.get("minimum improvement", 1e-2)
        self.last_losses = []
        self.patience = kwargs.get("patience", 10) + 1
        self.aps = kwargs.get("affinity_patch_size", 20)
        # 
        self.difference_basis = kwargs.get("difference_basis", "translated")
        self.domain_diff_bw_x = kwargs.get("domain_diff_bw_x", tf.constant(3, dtype=tf.float32))
        self.domain_diff_bw_y = kwargs.get("domain_diff_bw_y", tf.constant(3, dtype=tf.float32))
        # Global kernel width
        self.krnl_width_x = kwargs.get("krnl_width_x", None) 
        self.krnl_width_y = kwargs.get("krnl_width_y", None) 
        print("Kernel width x = ", self.krnl_width_x)
        print("Kernel width y = ", self.krnl_width_y)
        # Check how codes should be aligned
        self.patch_size = kwargs.get("patch_size", 100)
        if self.patch_size >= self.aps:
            print("Using centre crop to align codes...")
            self.centre_crop_frac = self.aps / self.patch_size
            print("Patch size: ", self.patch_size, 
                ", affinity patch size: ", self.aps, 
                ", centre crop fraction: ", self.centre_crop_frac)
            self.align_option = "centre_crop"
        else:
            self.align_option = "full"

        # encoders of X and Y
        self._enc_x = ImageTranslationNetwork(
            **translation_spec["enc_X"], name="enc_X", l2_lambda=self.l2_lambda
        )
        self._enc_y = ImageTranslationNetwork(
            **translation_spec["enc_Y"], name="enc_Y", l2_lambda=self.l2_lambda
        )

        # decoder of X and Y
        self._dec_x = ImageTranslationNetwork(
            **translation_spec["dec_X"], name="dec_X", l2_lambda=self.l2_lambda
        )
        self._dec_y = ImageTranslationNetwork(
            **translation_spec["dec_Y"], name="dec_Y", l2_lambda=self.l2_lambda
        )

        self.loss_object = tf.keras.losses.MeanSquaredError()

        self.train_metrics["cycle_x"] = tf.keras.metrics.Sum(name="cycle_x MSE sum")
        self.train_metrics["cross_x"] = tf.keras.metrics.Sum(name="cross_x MSE sum")
        self.train_metrics["recon_x"] = tf.keras.metrics.Sum(name="recon_x MSE sum")
        self.train_metrics["cycle_y"] = tf.keras.metrics.Sum(name="cycle_y MSE sum")
        self.train_metrics["cross_y"] = tf.keras.metrics.Sum(name="cross_y MSE sum")
        self.train_metrics["recon_y"] = tf.keras.metrics.Sum(name="recon_y MSE sum")
        self.train_metrics["krnls"] = tf.keras.metrics.Sum(name="krnls MSE sum")
        self.train_metrics["l2"] = tf.keras.metrics.Sum(name="l2 MSE sum")
        self.train_metrics["total"] = tf.keras.metrics.Sum(name="total MSE sum")

        # Track kernel loss history for use in early stopping
        self.metrics_history["krnls"] = []

    def save_all_weights(self):
        self._enc_x.save_weights(self.log_path + "/weights/_enc_x/")
        self._enc_y.save_weights(self.log_path + "/weights/_enc_y/")
        self._dec_x.save_weights(self.log_path + "/weights/_dec_x/")
        self._dec_y.save_weights(self.log_path + "/weights/_dec_y/")

    def load_all_weights(self, folder):
        self._enc_x.load_weights(folder + "/weights/_enc_x/")
        self._enc_y.load_weights(folder + "/weights/_enc_y/")
        self._dec_x.load_weights(folder + "/weights/_dec_x/")
        self._dec_y.load_weights(folder + "/weights/_dec_y/")

    @image_to_tensorboard()
    def enc_x(self, inputs, training=False):
        """ Wraps encoder call for TensorBoard printing and image save """
        return self._enc_x(inputs, training)

    @image_to_tensorboard()
    def dec_x(self, inputs, training=False):
        return self._dec_x(inputs, training)

    @image_to_tensorboard()
    def enc_y(self, inputs, training=False):
        return self._enc_y(inputs, training)

    @image_to_tensorboard()
    def dec_y(self, inputs, training=False):
        return self._dec_y(inputs, training)

    def early_stopping_criterion(self):
        self.last_losses = np.array(self.metrics_history["krnls"][-self.patience :])
        diffs = self.last_losses[:-1] - self.last_losses[-1]
        going_down = tf.reduce_all(diffs < self.min_impr)
        going_down = False
        tf.print("kernels_loss", self.last_losses[-1])
        return going_down

    @tf.function
    def __call__(self, inputs, training=False):
        x, y = inputs
        tf.debugging.Assert(tf.rank(x) == 4, [x.shape])
        tf.debugging.Assert(tf.rank(y) == 4, [y.shape])

        if training:
            x_code, y_code = self._enc_x(x, training), self._enc_y(y, training)
            x_hat, y_hat = self._dec_x(y_code, training), self._dec_y(x_code, training)
            x_dot, y_dot = (
                self._dec_x(self._enc_y(y_hat, training), training),
                self._dec_y(self._enc_x(x_hat, training), training),
            )
            x_tilde, y_tilde = (
                self._dec_x(x_code, training),
                self._dec_y(y_code, training),
            )
            # zx_t_zy = ztz(image_in_patches(x_code, 20), image_in_patches(y_code, 20))
            if self.align_option in ["centre_crop", "center_crop"]:
                # Crop X % of pixels from the centre of the patch
                zx_t_zy = ztz(
                    tf.image.central_crop(x_code, self.centre_crop_frac), 
                    tf.image.central_crop(y_code, self.centre_crop_frac)
                )
            elif self.align_option in ["full", "no_crop"]:
                # Align code of entire patches - will cause memory issues if patches are too large
                zx_t_zy = ztz(x_code, y_code)
            retval = [x_hat, y_hat, x_dot, y_dot, x_tilde, y_tilde, zx_t_zy]

        else:
            x_code, y_code = self.enc_x(x, name="code_x"), self.enc_y(y, name="code_y")
            x_tilde, y_tilde = (
                self.dec_x(x_code, name="x_tilde"),
                self.dec_y(y_code, name="y_tilde"),
            )
            x_hat, y_hat = (
                self.dec_x(y_code, name="x_hat"),
                self.dec_y(x_code, name="y_hat"),
            )
            # Check if translated images or original should be used as basis
            if self.difference_basis in ["translated", "tilde"]:
                difference_img = self._difference_img(x_tilde, y_tilde, x_hat, y_hat)
            elif self.difference_basis in ["original"]:
                difference_img = self._difference_img(x, y, x_hat, y_hat)
            retval = difference_img

        return retval

    @tf.function
    def _train_step(self, x, y, clw):
        """
        Input:
        x - tensor of shape (bs, ps_h, ps_w, c_x)
        y - tensor of shape (bs, ps_h, ps_w, c_y)
        clw - cross_loss_weight, tensor of shape (bs, ps_h, ps_w, 1)
        """
        with tf.GradientTape() as tape:
            x_hat, y_hat, x_dot, y_dot, x_tilde, y_tilde, ztz = self(
                [x, y], training=True
            )
            if self.align_option in ["centre_crop", "center_crop"]: 
                # Crop X % of pixels from the centre of the patch
                if self.krnl_width_x is None or self.krnl_width_y is None:
                    Kern = 1.0 - Degree_matrix(
                        tf.image.central_crop(x, self.centre_crop_frac), tf.image.central_crop(y, self.centre_crop_frac)
                    )
                elif self.krnl_width_x is not None or self.krnl_width_y is not None:
                    Kern = 1.0 - Degree_matrix_fixed_krnl(
                        tf.image.central_crop(x, self.centre_crop_frac), tf.image.central_crop(y, self.centre_crop_frac),
                        self.krnl_width_x, self.krnl_width_y
                    )
            elif self.align_option in ["full", "no_crop"]:
                # Align code of entire patches - will cause memory issues if patches are too large
                if self.krnl_width_x is None or self.krnl_width_y is None:
                    Kern = 1.0 - Degree_matrix(x, y)
                elif self.krnl_width_x is not None or self.krnl_width_y is not None:
                    # Use global kernel size 
                    Kern = 1.0 - Degree_matrix_fixed_krnl(x, y, self.krnl_width_x, self.krnl_width_y)

            kernels_loss = self.kernels_lambda * self.loss_object(Kern, ztz)
            l2_loss_k = sum(self._enc_x.losses) + sum(self._enc_y.losses)
            targets_k = (
                self._enc_x.trainable_variables + self._enc_y.trainable_variables
            )
            gradients_k = tape.gradient(kernels_loss + l2_loss_k, targets_k)
            if self.clipnorm is not None:
                gradients_k, _ = tf.clip_by_global_norm(gradients_k, self.clipnorm)

            self._optimizer_k.apply_gradients(zip(gradients_k, targets_k))

        with tf.GradientTape() as tape:
            x_hat, y_hat, x_dot, y_dot, x_tilde, y_tilde, ztz = self(
                [x, y], training=True
            )
            l2_loss = (
                sum(self._enc_x.losses)
                + sum(self._enc_y.losses)
                + sum(self._dec_x.losses)
                + sum(self._dec_y.losses)
            )
            cycle_x_loss = self.cycle_lambda * self.loss_object(x, x_dot)
            cross_x_loss = self.cross_lambda * self.loss_object(y, y_hat, clw)
            recon_x_loss = self.recon_lambda * self.loss_object(x, x_tilde)
            cycle_y_loss = self.cycle_lambda * self.loss_object(y, y_dot)
            cross_y_loss = self.cross_lambda * self.loss_object(x, x_hat, clw)
            recon_y_loss = self.recon_lambda * self.loss_object(y, y_tilde)

            total_loss = (
                cycle_x_loss
                + cross_x_loss
                + recon_x_loss
                + cycle_y_loss
                + cross_y_loss
                + recon_y_loss
                + l2_loss
            )

            targets_all = (
                self._enc_x.trainable_variables
                + self._enc_y.trainable_variables
                + self._dec_x.trainable_variables
                + self._dec_y.trainable_variables
            )

            gradients_all = tape.gradient(total_loss, targets_all)

            if self.clipnorm is not None:
                gradients_all, _ = tf.clip_by_global_norm(gradients_all, self.clipnorm)
            self._optimizer_all.apply_gradients(zip(gradients_all, targets_all))
        self.train_metrics["cycle_x"].update_state(cycle_x_loss)
        self.train_metrics["cross_x"].update_state(cross_x_loss)
        self.train_metrics["recon_x"].update_state(recon_x_loss)
        self.train_metrics["cycle_y"].update_state(cycle_y_loss)
        self.train_metrics["cross_y"].update_state(cross_y_loss)
        self.train_metrics["recon_y"].update_state(recon_y_loss)
        self.train_metrics["krnls"].update_state(kernels_loss)
        self.train_metrics["l2"].update_state(l2_loss)
        self.train_metrics["total"].update_state(total_loss)


def test(DATASET="Texas", CONFIG=None):
    """
    1. Fetch data (x, y, change_map)
    2. Compute/estimate A_x and A_y (for patches)
    3. Compute change_prior
    4. Define dataset with (x, A_x, y, A_y, p). Choose patch size compatible
       with affinity computations.
    5. Train CrossCyclicImageTransformer unsupervised
        a. Evaluate the image transformations in some way?
    6. Evaluate the change detection scheme
        a. change_map = threshold [(x - f_y(y))/2 + (y - f_x(x))/2]
    """
    if CONFIG is None:
        CONFIG = get_config_kACE(DATASET)
    
    
    print(f"Loading {DATASET} data")
    x_im, y_im, EVALUATE, (C_X, C_Y) = datasets.fetch(DATASET, **CONFIG)
    if tf.test.is_gpu_available() and not CONFIG["debug"]:
        C_CODE = 3
        print("here")
        TRANSLATION_SPEC = {
            "enc_X": {"input_chs": C_X, "filter_spec": [50, 50, C_CODE]},
            "enc_Y": {"input_chs": C_Y, "filter_spec": [50, 50, C_CODE]},
            "dec_X": {"input_chs": C_CODE, "filter_spec": [50, 50, C_X]},
            "dec_Y": {"input_chs": C_CODE, "filter_spec": [50, 50, C_Y]},
        }
    else:
        print("why here?")
        C_CODE = 1
        TRANSLATION_SPEC = {
            "enc_X": {"input_chs": C_X, "filter_spec": [C_CODE]},
            "enc_Y": {"input_chs": C_Y, "filter_spec": [C_CODE]},
            "dec_X": {"input_chs": C_CODE, "filter_spec": [C_X]},
            "dec_Y": {"input_chs": C_CODE, "filter_spec": [C_Y]},
        }
    print("Change Detector Init")
    cd = Kern_AceNet(TRANSLATION_SPEC, **CONFIG)
    print("Training")
    training_time = 0
    cross_loss_weight = tf.expand_dims(tf.zeros(x_im.shape[:-1], dtype=tf.float32), -1)
    for epochs in CONFIG["list_epochs"]:
        CONFIG.update(epochs=epochs)
        tr_gen, dtypes, shapes = datasets._training_data_generator(
            x_im[0], y_im[0], cross_loss_weight[0], CONFIG["patch_size"]
        )
        TRAIN = tf.data.Dataset.from_generator(tr_gen, dtypes, shapes)
        TRAIN = TRAIN.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        tr_time, _ = cd.train(TRAIN, evaluation_dataset=EVALUATE, **CONFIG)
        for x, y, _ in EVALUATE.batch(1):
            alpha = cd([x, y])
        cross_loss_weight = 1.0 - alpha
        training_time += tr_time

    cd.save_all_weights()
    cd.final_evaluate(EVALUATE, **CONFIG)
    # Alternate evaluation
    cd.alt_evaluate(EVALUATE, **CONFIG)
    final_kappa = cd.metrics_history["cohens kappa"][-1]
    final_acc = cd.metrics_history["ACC"][-1]
    performance = (final_kappa, final_acc)
    timestamp = cd.timestamp
    epoch = cd.epoch.numpy()
    speed = (epoch, training_time, timestamp)

    # Save config parameters
    with open(os.path.join(CONFIG["logdir"], "config.txt"), "w+") as f:
        print(CONFIG, file=f)

    del cd
    gc.collect()

    return performance, speed


if __name__ == "__main__":
    polmak_list = ["Polmak-LS5-S2", "Polmak-LS5-S2-collocate", "Polmak-LS5-S2-NDVI",
    "Polmak-LS5-S2-warp", "Polmak-A2-S2", "Polmak-A2-S2-collocate", "Polmak-LS5-PGNLM_A", 
    "Polmak-LS5-PGNLM_C", "Polmak-LS5-PGNLM_A-stacked", "Polmak-LS5-PGNLM_C-stacked", "Polmak-Pal-RS2_010817-collocate",
    "Polmak-Air05-Air10-align_sub0", "Polmak-Air05-Air15-align_sub0", "Polmak-Air10-Air15-align_sub0",
    "Polmak-Air05-S2-align_sub0", "Polmak-Air15-S2-align_sub0"]
    
    #process_list = ["Polmak-Pal-RS2_010817-collocate"] # ["Polmak-LS5-S2"] # [ "Polmak-LS5-PGNLM_A", "Polmak-LS5-PGNLM_C"] # ["Polmak-Pal-RS2_010817-collocate"]
    #process_list = polmak_list
    process_list = ["Polmak-Air05-Air10-align_sub0", "Polmak-Air10-Air15-align_sub0"] #"Polmak-Air05-Air15-align_sub0"] #, 
    #process_list = ["Polmak-Air05-S2-align_sub0", "Polmak-Air15-S2-align_sub0"] 
    # ["Polmak-Air05-Air10-align_sub0", "Polmak-Air05-Air15-align_sub0"]
    # process_list = ["Polmak-LS5-S2", "Polmak-A2-S2", "Polmak-A2-S2-collocate", "Polmak-LS5-PGNLM_A", 
    #"Polmak-LS5-PGNLM_C"]
    for DATASET in process_list:
        if DATASET in ["Polmak-Pal-RS2_010817-collocate", "Polmak-LS5-S2-NDVI", "Polmak-Air05-S2-align_sub0", "Polmak-Air15-S2-align_sub0"]: # "Polmak-Pal-RS2_010817-collocate", 
            print("Skipping dataset: "+DATASET)
            continue
        else:
            print(DATASET)
        CONFIG = get_config_kACE(DATASET)
        
        # Basis for difference image
        CONFIG["difference_basis"] = "original" #"translated"
        # Bandwidth for domain difference images
        CONFIG["domain_diff_bw_x"] = tf.constant(3.0, dtype=tf.float32)  # tf.constant(3, dtype=tf.float32)
        CONFIG["domain_diff_bw_y"] = tf.constant(3.0, dtype=tf.float32) # tf.constant(3, dtype=tf.float32)
        #CONFIG["krnl_width_x"] = 1.0 
        #CONFIG["krnl_width_y"] = 1.0

        # Suffix to add to log output name
        suffix = "_NOTILDE" # "_sigma25pct" # "_domdiffBW3_kwx0p50_kwy0p50" # "_NOTILDE_kwx1p0_kwy1p0" # 
        #suffix = ""
        print(suffix)
        CONFIG["patch_size"] = 20
        CONFIG["batch_size"] = 13
        CONFIG["batches"] = 500
        CONFIG["affinity_patch_size"] = 20
        print("Patch Size (ps): ", CONFIG["patch_size"])
        print("Affinity Patch Size (aps): ", CONFIG["affinity_patch_size"])
        print("number of Patches In Batch (pib): ", CONFIG["batch_size"])
        print("number of Batches Ier Epoch (bie): ", CONFIG["batches"])
        suffix += "_ps"+str(CONFIG["patch_size"])+"_aps"+str(CONFIG["affinity_patch_size"])
        suffix += "_pib"+str(CONFIG["batch_size"])+"_bie"+str(CONFIG["batches"])
        #suffix += "_"+str(CONFIG["batches"])+"_batches"
        
        if DATASET in polmak_list:
            print("Usinging Polmak processing dict")
            load_options = dict()
            load_options["norm_type"] = "_clip_norm" # "_norm01" # 
            suffix += load_options["norm_type"]
            load_options["debug"] = True
            load_options["row_shift"] = int(0)
            load_options["col_shift"] = int(0)
            load_options["reduce"] = False
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
            CONFIG["logdir"] = f"logs/reduce/{DATASET}/" + datetime.now().strftime("%Y%m%d-%H%M") + suffix
        else:
            CONFIG["logdir"] = f"logs/{DATASET}/" + datetime.now().strftime("%Y%m%d-%H%M") + suffix

        # Add load_options to CONFIG dict if it exists
        if load_options is not None:
            CONFIG["load_options"] = load_options

        test(DATASET, CONFIG)
