import tensorflow as tf
from tensorflow.python.ops.parallel_for.gradients import jacobian, batch_jacobian
def jacobian_loss(Y,X):
    J = batch_jacobian(Y,X,use_pfor=False)
    return tf.reduce_sum(tf.reduce_mean(tf.square(J),axis=0))


