import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors


class Forecaster:
    def __init__(
        self,
        max_iter=100,
        target_col=0,
        horizon=12,
    ):
        self.optimizer_ = (
            tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)
            if optimizer is None
            else optimizer
        )
        self.mse_loss_ = tf.keras.losses.MeanSquaredError()
        self.tolerance_ = tf.constant(tolerance)
        self.max_iter = max_iter
        
        self.target_col = target_col
        self.horizon = horizon
        
        self.step_weights = step_weights
        
        self.MISSING_MAX_BOUND = np.inf
        self.MISSING_MIN_BOUND = -np.inf
        

    def fit(self, model):
        """Fit a new counterfactual explainer to the model parameters
        ----------
        model : keras.Model
            The model
        """
        self.model_ = model
        return self

    def predict(self, x):
        """Compute the difference between the desired and actual forecasting predictions
        ---------
        x : Variable
            Variable of the sample
        """

        return self.model_(x)
     
        def compute_loss(
            self,
            original_sample,
            z_search,
            step_weights,
            max_bound,
            min_bound,
            n_iter=None,
        ):
            loss = tf.zeros(shape=())
            pred = self.model_(z_search)

            forecast_margin_loss = self.margin_mse(pred, max_bound, min_bound)
            loss += self.pred_margin_weight * forecast_margin_loss

            # weighted_ape for each changeable variable
            for z_idx in self.z_change_idx:
                weighted_steps_loss = self.weighted_ape(
                    tf.cast(original_sample[:, :, z_idx], tf.float32),
                    tf.cast(z_search[:, :, z_idx], tf.float32),
                    tf.cast(step_weights[:, :, z_idx], tf.float32),
                )
                loss += self.weighted_steps_weight * weighted_steps_loss

            return loss, forecast_margin_loss, weighted_steps_loss

    
    
    def transform(
        self,
        x,
        max_bound_lst=None,
        min_bound_lst=None,
        clip_range_inputs=None,
        hist_value_inputs=None,
    ):
        try:
            print(
                f"Validating threshold input: {len(max_bound_lst)==x.shape[0] or len(min_bound_lst)==x.shape[0]}"
            )
        except:
            print("Wrong parameter inputs, at least one threshold should be provided.")

        result_samples = np.empty(x.shape)
        losses = np.empty(x.shape[0])
        # `weights_all` needed for debugging
        weights_all = np.empty((x.shape[0], 1, x.shape[1], x.shape[2]))
        
        for i in range(x.shape[0]):
            # if i % 25 == 0:
            print(f"{i} samples been transformed.")

            x_sample = np.expand_dims(x[i], axis=0)
            if self.step_weights == "unconstrained":
                step_weights = np.zeros(x_sample.shape)
            elif self.step_weights == "uniform":
                step_weights = np.ones(x_sample.shape)
            elif self.step_weights in ["meal", "meal_time"]:
                step_weights = get_meal_weights(x_sample)
            # if self defined arrays as input
            elif isinstance(self.step_weights, np.ndarray):
                step_weights = self.step_weights
            else:
                raise NotImplementedError(
                    "step_weights not implemented, please choose 'unconstrained', 'meal_time' or 'uniform'."
                )
            
            # Check the condition of desired CF: upper and lower bound
            max_bound = (
                max_bound_lst[i] if max_bound_lst != None else self.MISSING_MAX_BOUND
            )
            min_bound = (
                min_bound_lst[i] if min_bound_lst != None else self.MISSING_MIN_BOUND
            )
            
            cf_sample, loss = self._transform_sample(
                x_sample, step_weights, max_bound, min_bound#, clip_ranges, hist_values
            )

    def _transform_sample(
        self, x, step_weights, max_bound, min_bound
    ):
        #split into target and exog
        variables = list()
        target_idx = [self.target_col]
        exog_idx = [i for i in list(range(x.shape[2])) if i not in target_idx]
        print(f"target_idx, exog_idx:{target_idx, exog_idx}")
        self.target_idx, self.exog_idx = target_idx, exog_idx
        for dim in range(x.shape[2]):
            v = tf.Variable( #???
                        np.expand_dims(x[:, :, dim], axis=2),
                        dtype=tf.float32,
                        name="var" + str(dim),
            )
            variables.append(v)
        it = 0 #???
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch([variables[i] for i in self.exog_idx])
            loss, forecast_margin_loss, weighted_steps_loss = self.compute_loss(
                x,
                tf.concat(variables, axis=2),
                step_weights,
                max_bound,
                min_bound,
                n_iter=it,
            )
        print(f"watched variables:{[var.name for var in tape.watched_variables()]}")


def get_meal_weights(x_sample, activity_threshold=0):
    # for all the variables in x_sample => 0 - weights for all positive values (i.e., larger than the threshold); more effective for bolus insulin and carbs intake

    # custom_step_weights has the same dimension as all the input variables (index needed);
    # but then only the weights for `z_change_idx` will be called
    custom_step_weights = (
        np.asarray(x_sample <= activity_threshold, dtype=np.float32) * 1
    )
    return custom_step_weights
