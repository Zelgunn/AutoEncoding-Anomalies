import tensorflow as tf
from tensorflow.python.training import training_ops
from tensorflow_probability.python.math import diag_jacobian


class SGLD(tf.optimizers.Optimizer):
    def __init__(self,
                 learning_rate=1e-3,
                 dataset_batch_count=1,
                 burn_in_steps=25,
                 preconditioner_decay_rate=0.95,
                 diagonal_bias=1e-8,
                 name="SGLD",
                 **kwargs
                 ):
        super(SGLD, self).__init__(name, **kwargs)

        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)

        self._set_hyper("dataset_batch_count", dataset_batch_count)
        self._set_hyper("burn_in_steps", burn_in_steps)
        self._set_hyper("preconditioner_decay_rate", preconditioner_decay_rate)
        self._set_hyper("diagonal_bias", diagonal_bias)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "rms", initializer="ones")

    def _resource_apply_dense(self, grad, var, apply_state):
        rms = self.get_slot(var, "rms")
        grad = self._apply_noisy_update(rms, grad, var)

        operation = training_ops.resource_apply_gradient_descent(
            var=var.handle,
            alpha=self._get_hyper("learning_rate", grad.dtype),
            delta=grad,
            use_locking=self._use_locking
        )

        # learning_rate = self._get_hyper("learning_rate", grad.dtype)
        # operation = var.assign_add(- learning_rate * grad)

        return operation

    def _resource_apply_sparse(self, grad, var, indices, apply_state):
        rms = self.get_slot(var, "rms")
        grad = self._apply_noisy_update(rms, grad, var, indices=indices)
        grad *= self._get_hyper("learning_rate", grad.dtype)

        operation = self._resource_scatter_add(
            x=var,
            i=indices,
            v=-grad
        )

        return operation

    def _apply_noisy_update(self, momentum, gradient, variable, indices=None):
        gradient_dtype = gradient.dtype
        gradient_shape = tf.shape(gradient)

        decay = self._get_hyper("decay", dtype=gradient_dtype)
        diagonal_bias = self._get_hyper("diagonal_bias", dtype=gradient_dtype)
        dataset_batch_count = self._get_hyper("dataset_batch_count", dtype=gradient_dtype)

        noise_stddev = tf.cond(
            pred=self.iterations < self.burn_in_steps,
            true_fn=lambda: tf.zeros([], dtype=gradient_dtype),
            false_fn=lambda: tf.cast(tf.math.rsqrt(self.learning_rate), gradient_dtype)
        )

        new_momentum = momentum * decay + (1.0 - decay) * tf.square(gradient)

        preconditioner = tf.math.rsqrt(new_momentum + diagonal_bias)

        _, preconditioner_grads = diag_jacobian(
            xs=variable,
            ys=preconditioner,
            parallel_iterations=10
        )

        gradient_mean = preconditioner * gradient * dataset_batch_count
        gradient_mean = 0.5 * (gradient_mean - preconditioner_grads[0])
        noise_stddev *= tf.sqrt(preconditioner) * 1e-3
        # noise_stddev = tf.sqrt(noise_stddev)

        noisy_gradient = tf.random.normal(mean=gradient_mean, stddev=noise_stddev,
                                          shape=gradient_shape, dtype=gradient_dtype)

        if indices is None:
            momentum.assign(new_momentum)
        else:
            self._resource_scatter_update(x=momentum, i=indices, v=new_momentum)

        return noisy_gradient

    @property
    def burn_in_steps(self):
        return self._get_hyper("burn_in_steps", dtype=tf.int64)

    @property
    def learning_rate(self):
        return self._get_hyper("learning_rate")

    def get_config(self):
        config = super(SGLD, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._serialize_hyperparameter("decay"),

            "dataset_batch_count": self._serialize_hyperparameter("dataset_batch_count"),
            "burn_in_steps": self._serialize_hyperparameter("burn_in_steps"),
            "preconditioner_decay_rate": self._serialize_hyperparameter("preconditioner_decay_rate"),
            "diagonal_bias": self._serialize_hyperparameter("diagonal_bias"),
        })
        return config
