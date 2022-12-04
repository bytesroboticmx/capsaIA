import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

import io
from PIL import Image
import matplotlib.pyplot as plt

import config


def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex // ncols
    assert nindex == nrows * ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (
        array.reshape(nrows, ncols, height, width, intensity)
        .swapaxes(1, 2)
        .reshape(height * nrows, width * ncols, intensity)
    )
    # (384, 480, 1) -> (1, 384, 480, 1) or (384, 480, 3) -> (1, 384, 480, 3)
    return tf.expand_dims(result, 0)


def read_and_close():
    # save and close the figure
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    # unsqueeze otherwise tf.summary err "ValueError: Tensor  must have rank 4.  Received rank 3, shape (500, 800, 4)"
    im = np.array(Image.open(buf))[np.newaxis, :]  # (500, 800, 4) -> (1, 500, 800, 4)
    return im


def get_outs(name, model, x, y, loss, string):
    dic_scalars = {}

    # todo-high: add another method/variable to our metric wrappers
    # which will expose intermediate variables e.g. z_mean, z_var
    # for vae, such that we can log them?

    if name in ["vae"]:
        out = model(x)
        y_hat, risk = out.y_hat, out.epistemic
        dic_imgs = {"rec": rec, "risk (mse)": risk}
        dic_scalars = {
            "mean of risk (mse)": tf.reduce_mean(risk),
            "std of risk (mse)": tf.math.reduce_std(risk),
        }

    elif name in ["dropout", "ensemble"]:
        out = model(x)
        y_hat, risk = out.y_hat, out.epistemic
        dic_imgs = {"y_hat": y_hat, "risk (std)": risk}
        dic_scalars = {
            "mean of risk (std)": tf.reduce_mean(risk),
            "std of risk (std)": tf.math.reduce_std(risk),
        }

    elif name in ["mve", "compatibility_mve"]:
        ### for BatchVisCallback

        # todo-high: we use batch callback with MVE, but the 'on_train_batch_end' does not have a val_loss
        # in the keras dict. So in order to log and plot the val_loss we need to calculate it ourselves.
        # So besides the (y_hat, risk) here will also need the loss dict, but we don't want to run forward
        # twice e.g. 'call' to calculate the former and then 'loss_fn()' to calculate the latter
        # because this is extremely inefficient. One way how we can achieve this, is by modifying each
        # wrapper's 'loss_fn' to also return the (y_hat, risk) pair in addition to the keras dict containing
        # losses. But need to consider how useful is this functionally in general to modify the current
        # behavior of this method's

        # loss, y_hat, risk = model.loss_fn(x, y, return_var=True)
        # dic_imgs = {"y_hat": y_hat, "risk (variance)": risk}
        # dic_scalars = {
        #     "mean of risk (variance)": tf.reduce_mean(risk),
        #     "std of risk (variance)": tf.math.reduce_std(risk),
        # }

        # # combined wit the reason above and because 'get_outs' is used for both train and test data
        # # (see 'run_and_save'), inside 'get_outs' we need to know what data we're currently processing
        # # ('train' or 'test') in order to name the entry in the loss dict accordingly
        # loss = {"loss" if string == "train" else "val_loss": loss}

        ### for EpochVisCallback
        out = model(x)
        y_hat, risk = out.y_hat, out.aleatoric
        dic_imgs = {"y_hat": y_hat, "risk (variance)": risk}
        dic_scalars = {
            "mean of risk (variance)": tf.reduce_mean(risk),
            "std of risk (variance)": tf.math.reduce_std(risk),
        }

    elif name in ["base"]:
        y_hat = model(x, training=False)
        dic_imgs = {"y_hat": y_hat}

    dic_scalars.update(loss)
    dic_imgs["x"], dic_imgs["y"] = x, y

    return dic_imgs, dic_scalars


# https://github.com/aamini/evidential-deep-learning/blob/main/neurips2020/trainers/deterministic.py
class VisCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        checkpoints_path,
        logs_path,
        model_name,
        xy_train,
        xy_test,
        optional_func,
        is_sample_different,
    ):

        self.save_dir = checkpoints_path

        train_log_dir = os.path.join(logs_path, "train-epoch")
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        val_log_dir = os.path.join(logs_path, "val-epoch")
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        train_bs_log_dir = os.path.join(logs_path, "train-batch")
        self.train_bs_summary_writer = tf.summary.create_file_writer(train_bs_log_dir)

        # if the dict is not empty init the writer
        if bool(optional_func):
            optional_plots_log_dir = os.path.join(logs_path, "optional-plots")
            self.optional_plots_writer = tf.summary.create_file_writer(
                optional_plots_log_dir
            )

        self.is_sample_different = is_sample_different
        if is_sample_different:
            # datasets
            self.xy_train, self.xy_test = xy_train, xy_test
        else:
            # one batch of the dataset
            self.xy_train, self.xy_test = (
                iter(xy_train).get_next(),
                iter(xy_test).get_next(),
            )

        self.model_name = model_name
        self.iter = 0
        self.min_vloss = float("inf")
        self.optional_func = optional_func

    def _save_summary(self, dic_imgs, dic_scalars):
        # x (32, 128, 160, 3), y (32, 128, 160, 1)

        for k, v in dic_scalars.items():
            # both plot val and train losses on the same card in tfboard -- val_mse_loss -> mse_loss
            # replace only if current key is loss (and not std for example). Can still easily
            # differentiate logs (tran vs val) based on the name of their tf.writer
            if "loss" in k:
                k = k if "val_" not in k else k.replace("val_", "")
            tf.summary.scalar(k, v, step=self.iter)

        idx = np.random.choice(config.BS, 9)
        for k, v in dic_imgs.items():
            tf.summary.image(
                k, gallery(tf.gather(v, idx).numpy()), max_outputs=1, step=self.iter
            )

    def _save(self, name):
        save_path = os.path.join(self.save_dir, name)
        if self.model_name == "base":
            # Functional model or Sequential model
            self.model.save(save_path)
        else:
            self.model.save_weights(save_path, save_format="tf")

    @staticmethod
    def _get_keras_dict(logs):
        val_loss_names = [i for i in logs if i.startswith("val_")]
        loss_names = [i for i in logs if i not in val_loss_names]
        loss = {k: v for k, v in logs.items() if k in loss_names}
        vloss = {k: v for k, v in logs.items() if k in val_loss_names}
        return loss, vloss

    # assumes ds is already shuffled
    def _get_batch(self, xy, num_samples):
        if self.is_sample_different:
            num_batches = num_samples // config.BS
            x_, y_ = zip(*xy.take(num_batches))
            x_, y_ = tf.concat(x_, 0), tf.concat(y_, 0)
        else:
            x_, y_ = xy
        assert x_.shape == (num_samples, 128, 160, 3)
        assert y_.shape == (num_samples, 128, 160, 1)
        return x_, y_

    # note: getting keras dict either from 'on_train_batch_end' or from 'on_epoch_end' so they are not
    # the losses that are calculated from the x_train_batch/y_train_batch or x_test_batch/y_test_batch (which we sample below).
    # Only for the case of EpochVisCallback (used e.g. in mve), we don't use keras dict provided to us (from logs),
    # rather make run the model on the sampled data and use the loss *on that exact data* that the model outputs.
    def run_and_save(self, logs):
        # note: vloss can be an empty dict (e.g., if using BatchVisCallback)
        loss, vloss = self._get_keras_dict(logs)

        x_train_batch, y_train_batch = self._get_batch(self.xy_train, config.BS)
        dic_imgs, dic_scalars = get_outs(
            self.model_name, self.model, x_train_batch, y_train_batch, loss, "train"
        )
        with self.train_summary_writer.as_default():
            self._save_summary(dic_imgs, dic_scalars)

        x_test_batch, y_test_batch = self._get_batch(self.xy_test, config.BS)
        dic_imgs, dic_scalars = get_outs(
            self.model_name, self.model, x_test_batch, y_test_batch, vloss, "val"
        )
        with self.val_summary_writer.as_default():
            self._save_summary(dic_imgs, dic_scalars)

        # note: expects one of the keys to be called exactly "loss", e.g. "return {'loss':some_loss_value}" in "train_step"
        # val_loss = dic_scalars["val_wrapper_loss"]
        # todo-med: if multiple val wrapper losses (e.g., ensemble) it picks the loss first of the 1st member, this is not
        # ideal and should be changed to taking their mean
        # todo-high: val_wrapper_loss
        if self.model_name == "ensemble":
            val_loss = [
                v
                for k, v in dic_scalars.items()
                if ("compiled_loss" in k) and ("val" in k)
            ][0]
        else:
            val_loss = [
                v
                for k, v in dic_scalars.items()
                if ("wrapper_loss" in k) and ("val" in k)
            ][0]
        if val_loss < self.min_vloss:
            self.min_vloss = (
                vloss.numpy() if isinstance(val_loss, np.ndarray) else val_loss
            )
            self._save("{:0.6f}vloss_{}iter.tf".format(self.min_vloss, self.iter))

        if bool(self.optional_func):
            with self.optional_plots_writer.as_default():
                for func_name, [func, iter] in self.optional_func.items():
                    if self.iter % iter == 0:
                        # the optional_func's are already partially initialized (args to these funcs are provided in model.fit(callbacks=...), so suffice to call them like that
                        # the only thing we need to provide here is the model (since it changes every epoch so we cannot just provide this arg inside model.fit(callbacks=...)
                        func(model=self.model)
                        im = read_and_close()
                        tf.summary.image(func_name, im, max_outputs=1, step=self.iter)


class EpochVisCallback(VisCallback):
    def __init__(
        self,
        checkpoints_path,
        logs_path,
        model_name,
        xy_train,
        xy_test,
        optional_func={},
        is_sample_different=True,
    ):
        super().__init__(
            checkpoints_path,
            logs_path,
            model_name,
            xy_train,
            xy_test,
            optional_func,
            is_sample_different,
        )

        self.model_name = model_name
        self.optional_func = optional_func

    def on_train_batch_end(self, batch, logs):
        # note: this setup doesn't record initial loss (of the untrained model)
        loss, _ = self._get_keras_dict(logs)
        with self.train_bs_summary_writer.as_default():
            for k, v in loss.items():
                tf.summary.scalar(k, v, step=self.iter)

        self.iter += 1

    def on_epoch_end(self, epoch, logs):
        self.run_and_save(logs)


class BatchVisCallback(VisCallback):
    """
    More granular val_loss downside is slower training due to additionally running the model
    (every n steps on_train_batch_end, instead of on on_epoch_end)
    """

    def __init__(
        self,
        checkpoints_path,
        logs_path,
        model_name,
        xy_train,
        xy_test,
        optional_func={},
        is_sample_different=True,
    ):
        super().__init__(
            checkpoints_path,
            logs_path,
            model_name,
            xy_train,
            xy_test,
            optional_func,
            is_sample_different,
        )

    def on_train_batch_end(self, batch, logs):
        loss, _ = self._get_keras_dict(logs)
        with self.train_bs_summary_writer.as_default():
            for k, v in loss.items():
                tf.summary.scalar(k, v, step=self.iter)

        # iter % 100 is to log dic_imgs and dic_scalars and we don't wan to do it every single batch bc it involves
        # re-running the model. But remember that in train.py we specify optional_func and at what iter we want to run them
        # it could be e.g. 782 or 3910 these numbers ofc do not evaluate to True in '% 100 == 0' (so it would never log them).
        # Thus we add the second condition, according to which if any of desired iters specified in optional_func
        # evaluates to True, then we run 'self.run_and_save'
        iters_optional_func = [
            int(func_iter[1]) for func_iter in list(self.optional_func.values())
        ]
        if (self.iter % 100 == 0) or any(
            [self.iter % i == 0 for i in iters_optional_func]
        ):
            self.run_and_save(logs)

        self.iter += 1


def get_checkpoint_callback(checkpoints_path):
    itters_per_ep = config.N_TRAIN / config.BS
    total_itters = itters_per_ep * config.EP
    save_itters = int(total_itters // 10)  # save 10 times during training
    # save_ep = int(save_itters / itters_per_ep)
    # last_saved_ep = round(save_itters * 10 // itters_per_ep)
    print("total_itters:", total_itters)
    print("save_itters:", save_itters)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        # note: tf tutorial saves all checkpoints to same folder https://www.tensorflow.org/tutorials/keras/save_and_load#checkpoint_callback_options
        # Alternatively can save checkpoints to different folders to create a separate "checkpoint" file for every saved weights -- filepath=os.path.join(checkpoints_path, 'ep_{epoch:02d}', 'weights.tf')
        filepath=os.path.join(checkpoints_path, "ep_{epoch:02d}_weights.tf"),
        save_weights_only=True,
        # monitor='loss', # val_loss
        save_best_only=False,
        # mode='auto',
        save_freq=save_itters,
    )

    return checkpoint_callback
