import tensorflow as tf



class ImputationLoss(tf.keras.losses.Loss):
    def __init__(self, use_r2_loss=True, **kwargs):
        super(ImputationLoss, self).__init__(**kwargs)
        self.ce_loss_obj = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
        self.kld_loss_obj = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM)
        self.use_r2_loss = use_r2_loss

    def calculate_Minimac_R2(self, pred_alt_allele_probs, gt_alt_af):
        mask = tf.logical_or(tf.equal(gt_alt_af, 0.0), tf.equal(gt_alt_af, 1.0))
        gt_alt_af = tf.where(mask, 0.5, gt_alt_af)
        denom = gt_alt_af * (1.0 - gt_alt_af)
        denom = tf.where(denom < 0.01, 0.01, denom)
        r2 = tf.reduce_mean(tf.square(pred_alt_allele_probs - gt_alt_af), axis=0) / denom
        r2 = tf.where(mask, tf.zeros_like(r2), r2)
        return r2

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)

        cat_loss = self.ce_loss_obj(y_true, y_pred)
        kl_loss = self.kld_loss_obj(y_true, y_pred)

        print("CE",cat_loss)
        print("KL",  kl_loss)

        total_loss = cat_loss + kl_loss

        if self.use_r2_loss:
            batch_size = tf.shape(y_true)[0]
            group_size = 4
            num_full_groups = batch_size // group_size
            num_remainder_samples = batch_size % group_size

            y_true_grouped = tf.reshape(y_true[:num_full_groups * group_size], (num_full_groups, group_size) + tuple(y_true.shape[1:]))
            y_pred_grouped = tf.reshape(y_pred[:num_full_groups * group_size], (num_full_groups, group_size) + tuple(y_pred.shape[1:]))

            r2_loss = 0.0
            for i in range(num_full_groups):
                gt_alt_af = tf.cast(tf.math.count_nonzero(tf.argmax(y_true_grouped[i], axis=-1), axis=0), tf.int32) / group_size
                gt_alt_af = tf.cast(gt_alt_af, tf.float32)
                pred_alt_allele_probs = tf.reduce_sum(y_pred_grouped[i][:, :, 1:], axis=-1)
                r2_loss += -tf.reduce_sum(self.calculate_Minimac_R2(pred_alt_allele_probs, gt_alt_af)) * tf.cast(group_size, tf.float32)

            if num_remainder_samples > 0:
                remainder_start_index = num_full_groups * group_size
                y_true_remainder = y_true[remainder_start_index:]
                y_pred_remainder = y_pred[remainder_start_index:]

                gt_alt_af = tf.cast(tf.math.count_nonzero(tf.argmax(y_true_remainder, axis=-1), axis=0), tf.int32) / num_remainder_samples
                gt_alt_af = tf.cast(gt_alt_af, tf.float32)
                pred_alt_allele_probs = tf.reduce_sum(y_pred_remainder[:, :, 1:], axis=-1)
                r2_loss += -tf.reduce_sum(self.calculate_Minimac_R2(pred_alt_allele_probs, gt_alt_af)) * tf.cast(num_remainder_samples, tf.float32)
            total_loss += r2_loss
            print('R2 loss', r2_loss)
        return total_loss

import numpy as np
B, L, C = 10, 1000, 4
missing_label = -1

# 随机生成整数标签
rng = np.random.default_rng(42)
y_true_int = rng.integers(0, C, size=(B, L), dtype=np.int32)

# # 制造缺失：每 3 个位点缺 1 个
# mask_missing = (np.arange(L) % 3 == 0)
# y_true_int[:, mask_missing] = missing_label

# 构造 one-hot 版 y_true（缺失位点全 0）
y_true_onehot = np.zeros((B, L, C), dtype=np.float32)
for b in range(B):
    for l in range(L):
        label = y_true_int[b, l]
        if label != missing_label:
            y_true_onehot[b, l, label] = 1.0

# 构造 y_pred（模拟模型输出）
logits = rng.normal(size=(B, L, C)).astype(np.float32)

# 转 TensorFlow 常量
y_true_tf = tf.constant(y_true_onehot)
y_pred_tf = tf.constant(logits)

# 测试
loss_obj = ImputationLoss(use_r2_loss=True)
loss = loss_obj(y_true_tf, y_pred_tf)
print("TF loss =", float(loss))
