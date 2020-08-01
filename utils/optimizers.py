import tensorflow as tf
import numpy as np

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer


class BLsgd_v2(optimizer.Optimizer):
    def __init__(self, learning_rate=0.01, gamma=0.9, beta=0.999, betaL=0.8, momentum=0.0, epsilon=1e-8,
                 use_locking=False, name="BLsgd_v2"):
        super(BLsgd_v2, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._gamma = gamma
        self._beta = beta
        self._epsilon = epsilon
        self._lr_t = None
        self._gamma_t = None
        self._beta_t = None
        self._epsilon_t = None
        self._momentum = momentum
        self._betaL = betaL

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr)
        self._gamma_t = ops.convert_to_tensor(self._gamma)
        self._beta_t = ops.convert_to_tensor(self._beta)
        self._betaL_t = ops.convert_to_tensor(self._betaL)

        self._epsilon_t = ops.convert_to_tensor(self._epsilon)
        self._momentum_t = ops.convert_to_tensor(self._momentum)

    def _create_slots(self, var_list):
        self.first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(initial_value=np.array([1.0], dtype="float32"), name="iterations",
                                       colocate_with=self.first_var)

        for v in var_list:
            self._zeros_slot(v, "momentum", self._name)
            self._zeros_slot(v, "pr_g", self._name)
            self._zeros_slot(v, "dt", self._name)
            self._create_non_slot_variable(initial_value=np.array([1.0], dtype="float32"), name=v.name + "/Ls",
                                           colocate_with=v)

    def _resource_apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        gamma_t = math_ops.cast(self._gamma_t, var.dtype.base_dtype)
        beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        momentum_t = math_ops.cast(self._momentum_t, var.dtype.base_dtype)
        betaL_t = math_ops.cast(self._betaL_t, var.dtype.base_dtype)

        # Getting the updated t value required for correction.
        t_new = self._get_non_slot_variable("iterations")

        pr_g = self.get_slot(var, "pr_g")

        dt = self.get_slot(var, "dt")
        new_dt = dt.assign(grad - pr_g)

        l = tf.norm(pr_g) / (tf.norm(new_dt) + epsilon_t)

        #         Lt = self._get_or_make_slot(var, val=None, slot_name="Ls", op_name=self._name)
        Lt = self._get_non_slot_variable(var.name + "/Ls")

        new_Ls = Lt.assign((betaL_t * Lt) + ((1 - betaL_t) * l))

        new_lr = tf.cond(tf.less(t_new, 2), lambda: lr_t, lambda: lr_t * new_Ls)

        # Getting the moment value and calculating the corrected updated value.
        mt = self.get_slot(var, "momentum")
        mt_new = mt.assign(momentum_t * mt - new_lr * grad)

        with ops.control_dependencies([new_lr]):
            new_prg = pr_g.assign(grad)

        var_update = state_ops.assign_add(var, mt_new)

        return control_flow_ops.group(*[var_update, mt_new, new_prg, new_dt, new_Ls])

    def _finish(self, update_ops, name_scope):
        it = self._get_non_slot_variable("iterations")
        new_it = it.assign(it + 1)
        return control_flow_ops.group(*(update_ops + [new_it]), name=name_scope)


class BLadam_v2(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, gamma=0.9, beta=0.999, betaL=0.8, epsilon=1e-8, use_locking=False,
                 name="BLadam_v2"):
        super(BLadam_v2, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._gamma = gamma
        self._beta = beta
        self._epsilon = epsilon
        self._lr_t = None
        self._gamma_t = None
        self._beta_t = None
        self._epsilon_t = None
        self._betaL = betaL

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr)
        self._gamma_t = ops.convert_to_tensor(self._gamma)
        self._beta_t = ops.convert_to_tensor(self._beta)
        self._betaL_t = ops.convert_to_tensor(self._betaL)
        self._epsilon_t = ops.convert_to_tensor(self._epsilon)

    def _create_slots(self, var_list):
        self.first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(initial_value=np.array([1.0], dtype="float32"), name="iterations",
                                       colocate_with=self.first_var)

        #         print(first_var)
        for v in var_list:
            self._zeros_slot(v, "mt", self._name)
            self._zeros_slot(v, "vt", self._name)
            self._zeros_slot(v, "pr_g", self._name)
            self._zeros_slot(v, "dt", self._name)
            #             self._get_or_make_slot(v, val=tf.Variable(1.0),
            #                                     slot_name="Ls", op_name=self._name)
            #              self._get_or_make_slot_with_initializer(v, init, v_shape[:1], dtype, "accumulator", self._name)
            self._create_non_slot_variable(initial_value=np.array([1.0], dtype="float32"), name=v.name + "/Ls",
                                           colocate_with=v)

    def _resource_apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        gamma_t = math_ops.cast(self._gamma_t, var.dtype.base_dtype)
        beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        betaL_t = math_ops.cast(self._betaL_t, var.dtype.base_dtype)

        # Getting the updated t value required for correction.
        t_new = self._get_non_slot_variable("iterations")
        #         t_new = t.assign(t + 1)

        # Getting the moment value and calculating the corrected updated value.
        mt = self.get_slot(var, "mt")
        mt_new = mt.assign((gamma_t * mt) + ((1 - gamma_t) * grad))
        corr_mt_new = mt_new / (1 - tf.math.pow(gamma_t, t_new))

        pr_g = self.get_slot(var, "pr_g")

        dt = self.get_slot(var, "dt")
        new_dt = dt.assign(grad - pr_g)

        # Getting the velocity value and calculating the corrected updated value.
        vt = self.get_slot(var, "vt")
        vt_new = vt.assign((beta_t * vt) + ((1 - beta_t) * tf.square(grad)))
        corr_vt_new = vt_new / (1 - tf.pow(beta_t, t_new))

        l = tf.norm(pr_g) / (tf.norm(new_dt) + epsilon_t)

        #         Lt = self._get_or_make_slot(var, val=None, slot_name="Ls", op_name=self._name)
        Lt = self._get_non_slot_variable(var.name + "/Ls")

        new_Ls = Lt.assign((betaL_t * Lt) + ((1 - betaL_t) * l))

        new_lr = tf.cond(tf.less(t_new, 2), lambda: lr_t, lambda: lr_t * new_Ls)

        # new_lr = lr_t * new_Ls

        # Updating weights based on the final update equation.
        update_val = (new_lr * corr_mt_new) / tf.sqrt(corr_vt_new + epsilon_t)

        with ops.control_dependencies([new_lr]):
            new_prg = pr_g.assign(grad)

        var_update = state_ops.assign_sub(var, update_val)
        return control_flow_ops.group(*[var_update, mt_new, vt_new, new_prg, new_dt, new_Ls])

    def _finish(self, update_ops, name_scope):
        it = self._get_non_slot_variable("iterations")
        new_it = it.assign(it + 1)
        return control_flow_ops.group(*(update_ops + [new_it]), name=name_scope)


class BLrmsprop_v2(optimizer.Optimizer):

    def __init__(self, learning_rate=0.01, gamma=0.9, beta=0.999, betaL=0.8, epsilon=1e-8, use_locking=False,
                 name="BLrmsprop_v2"):
        super(BLrmsprop_v2, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._gamma = gamma
        self._beta = beta
        self._epsilon = epsilon
        self._lr_t = None
        self._gamma_t = None
        self._beta_t = None
        self._epsilon_t = None
        self._betaL = betaL

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr)
        self._gamma_t = ops.convert_to_tensor(self._gamma)
        self._beta_t = ops.convert_to_tensor(self._beta)
        self._betaL_t = ops.convert_to_tensor(self._betaL)
        self._epsilon_t = ops.convert_to_tensor(self._epsilon)

    def _create_slots(self, var_list):
        self.first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(initial_value=np.array([1.0], dtype="float32"), name="iterations",
                                       colocate_with=self.first_var)

        #         print(first_var)
        for v in var_list:
            self._zeros_slot(v, "mt", self._name)
            self._zeros_slot(v, "vt", self._name)
            self._zeros_slot(v, "pr_g", self._name)
            self._zeros_slot(v, "dt", self._name)
            #             self._get_or_make_slot(v, val=tf.Variable(1.0),
            #                                     slot_name="Ls", op_name=self._name)
            #              self._get_or_make_slot_with_initializer(v, init, v_shape[:1], dtype, "accumulator", self._name)
            self._create_non_slot_variable(initial_value=np.array([1.0], dtype="float32"), name=v.name + "/Ls",
                                           colocate_with=v)

    def _resource_apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        gamma_t = math_ops.cast(self._gamma_t, var.dtype.base_dtype)
        beta_t = math_ops.cast(self._beta_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        betaL_t = math_ops.cast(self._betaL_t, var.dtype.base_dtype)

        # Getting the updated t value required for correction.
        t_new = self._get_non_slot_variable("iterations")
        #         t_new = t.assign(t + 1)

        pr_g = self.get_slot(var, "pr_g")

        dt = self.get_slot(var, "dt")
        new_dt = dt.assign(grad - pr_g)

        # Getting the velocity value and calculating the corrected updated value.
        vt = self.get_slot(var, "vt")
        vt_new = vt.assign((beta_t * vt) + ((1 - beta_t) * tf.square(grad)))

        l = tf.norm(pr_g) / (tf.norm(new_dt) + epsilon_t)

        #         Lt = self._get_or_make_slot(var, val=None, slot_name="Ls", op_name=self._name)
        Lt = self._get_non_slot_variable(var.name + "/Ls")

        new_Ls = Lt.assign((betaL_t * Lt) + ((1 - betaL_t) * l))

        # new_lr = lr_t * new_Ls
        new_lr = tf.cond(tf.less(t_new, 2), lambda: lr_t, lambda: lr_t * new_Ls)

        # Updating weights based on the final update equation.
        update_val = (new_lr * grad) / tf.sqrt(vt_new + epsilon_t)

        with ops.control_dependencies([new_lr]):
            new_prg = pr_g.assign(grad)

        var_update = state_ops.assign_sub(var, update_val)
        return control_flow_ops.group(*[var_update, vt_new, new_dt, new_Ls, new_prg])

    def _finish(self, update_ops, name_scope):
        it = self._get_non_slot_variable("iterations")
        new_it = it.assign(it + 1)
        return control_flow_ops.group(*(update_ops + [new_it]), name=name_scope)
