#statistical tf
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
def compute_loss(max_bound, min_bound, pred):
    mse_loss_ = tf.keras.losses.MeanSquaredError(
        reduction=tf.keras.losses.Reduction.SUM
    )
    dist_max = mse_loss_(max_bound, pred)
    dist_min = mse_loss_(min_bound, pred)
    loss = dist_max + dist_min
    return loss
    
max_bound = np.transpose(max_bound)
min_bound = np.transpose(min_bound)
X_test_exog = random.choice(dataset.X_test_exog)
#X = dataset.X_train_exog
#X = random.choice(X)
#X = tf.Variable(tf.convert_to_tensor(X, dtype=tf.float32), dtype=tf.float32)
grad_X_e = tf.convert_to_tensor(X_test_exog, dtype=tf.float32)
grad_X_e = tf.Variable(grad_X_e, dtype=tf.float32)

X = random.choice(dataset.X_train_exog)
y = random.choice(dataset.X_train_target)
print(X, y)
mod = sm.tsa.SARIMAX(endog=np.asarray(y), exog=np.asarray(X), order=(1,0,0))
#res = mod.fit(disp=False)
mod = mod.fit(disp=False)
start_params = mod.params
print(start_params)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.05,  epsilon=1e-07,)
with tf.GradientTape() as tape:
    tape.watch(grad_X_e)
    # Calculate the value of the function and record the gradient
    #pred = tf_model(tf.expand_dims(grad_X_e, axis=0))
    temp = grad_X_e.numpy()
    pred = mod.forecast(horizon, start_params=start_params, exog=temp)
    #pred = mod.forecast(horizon, start_params=start_params, exog=np.asarray(grad_X_e))
    pred = tf.convert_to_tensor(pred, dtype=tf.float32)
    #pred = tf.squeeze(pred)
    #pred = tf.Variable(pred, dtype=tf.float32)
    loss = compute_loss(max_bound, min_bound, pred)
    print(loss)

max_iter = 100
it = 0
while (tf.reduce_any(pred>max_bound) or tf.reduce_any(pred<min_bound)) and (it<max_iter):
    #change (X_e)
    print(";lyf;khf",loss, grad_X_e)
    gradient = tape.gradient(loss, grad_X_e)
    print(gradient)
    if gradient is None:
        print("no gradient")
        #break

    # Use the Adam optimizer to update the value of x
    optimizer.apply_gradients([(gradient, grad_X_e)])
    
    with tf.GradientTape() as tape:
        tape.watch(grad_X_e)
        # Calculate the value of the function and record the gradient
        temp = grad_X_e.numpy()
        pred = mod.forecast(horizon, start_params=start_params, exog=temp)
        print(pred)
        pred = tf.convert_to_tensor(pred, dtype=tf.float32)
        #pred = tf.Variable(pred, dtype=tf.float32)
        #pred = tf.squeeze(pred)
        loss = compute_loss(max_bound, min_bound, pred)
        print(loss)

    # Record the current value of x
    print(f"Iteration {it}, Loss: {loss.numpy()}, Grad_X_e: {grad_X_e.numpy()}")
    it += 1
    
print("Optimized value of x:", grad_X_e.numpy())
final_pred = tf_model(tf.expand_dims(grad_X_e, axis=0))
print("Value of the function at the optimized point:", final_pred.numpy())
print("ADGAB", tf.squeeze(final_pred, axis=-1).numpy()[:,:][0])


#TEST

#regression
import statsmodels.api as sm

def mean_squared_error(y_true, y_predicted):
    # Calculating the loss or cost
    cost = np.sum((y_true-y_predicted)**2) / len(y_true)
    return cost
    
orig_train = orig_train.dropna()
orig_test = orig_test.dropna()
endog = orig_train[orig_train.patient_id==591].glucose
exog = orig_train[orig_train.patient_id==591].drop(['glucose', 'time', 'patient_id'], axis=1)
exog_pred = orig_test[orig_test.patient_id==591].drop(['glucose', 'time', 'patient_id'], axis=1)
endog_pred = orig_test[orig_test.patient_id==591].glucose
print(orig_test, endog_pred)
print(np.asarray(endog), np.asarray(exog))
forest = sm.OLS(np.asarray(endog), np.asarray(exog)).fit()
pred = forest.predict(np.asarray(exog_pred))
print(pred)
max_iter = 100
it = 0
learning_rate = 0.0001
gradient=lambda v: 4 * v**3 - 10 * v - 3
exog_pred_change = np.asarray(exog_pred)
print("max_min", max_bound, min_bound)
while ((pred>max_bound).any() or (pred<min_bound).any()) and (it<max_iter):
    #change (X_e)
    print(it)
    diff = -learning_rate * gradient(exog_pred_change)
    exog_pred_change += diff
    it += 1
    pred = forest.predict(exog_pred_change)
    print(pred)
    
print(pred)
