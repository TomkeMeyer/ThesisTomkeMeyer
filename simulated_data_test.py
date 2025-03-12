import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.ensemble import RandomForestRegressor

a = [1,2,1,1,2,3,3,2,1,2,1,1,2,3,3,2,1,0,0,1,2,3,2,2,1,0,0,3,2,1,1,0,1,0,0,1,2,3,2,2,1,0,0,3,2,1,1,0]
b = [-1,-3,-4,-1,-3,-4,0,6,6,0,3,5,-5,0,4,-4,3,3,3,3,0,0,0,-4,-4,4,2,0,6,6,0,3,5,-5,0,4,-4,3,3,3,3,0,0,0,-4,-4,4,2]
t = []
for i,j in zip(a,b):
    t.append(4*(i**2)+3*j)
    
a_e = [1,2,3,2,3,1,3,2,0,0,1,1]
b_e = [3,-4,-5,5,3,2,0,0,-6,-5,5,4]
t_e = []
for i,j in zip(a_e,b_e):
    t_e.append(4*(i**2)+3*j)
    
print(t_e)

#gru forecaster?
data  = np.row_stack((np.array(t), np.array(a), np.array(b)))
data = pd.DataFrame(data)
data = np.transpose(np.array(data))

print("----------------------Regression--------------------")
#regression
#print(data)
data  = np.row_stack((np.array(t), np.array(a), np.array(b)))
data = pd.DataFrame(data)
data = np.transpose(np.array(data))
X = data[:, 1:]
y = data[:, 0]
X = np.array(X).astype(np.float32)
y = np.array(y).astype(np.float32)
#y = np.expand_dims(y, axis=0)
data_e  = np.row_stack((np.array(t_e), np.array(a_e), np.array(b_e)))
data_e = pd.DataFrame(data_e)
data_e = np.transpose(np.array(data_e))
X_e = data_e[:, 1:]
y_e = data_e[:, 0]
X_e = np.array(X_e)
y_e = np.array(y_e)
print(X,y,X_e,y_e)
#forest = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True).fit(X, y)
tf_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Assuming regression
])
tf_model.compile(optimizer='adam', loss='mse')
tf_model.fit(X, y, epochs=100, verbose=0)
max_bound = tf.constant([5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1], dtype=tf.float32)
min_bound = tf.constant([3, 3, 2, 2, 1, 1, 0, 0, -1, -1, -1, -1], dtype=tf.float32)
it = 0
max_iter = 100
grad_X_e = tf.convert_to_tensor(X_e, dtype=tf.float32)
max_bound = tf.convert_to_tensor(max_bound, tf.float32)
min_bound = tf.convert_to_tensor(min_bound, tf.float32)
def compute_loss(max_bound, min_bound, pred):
    mse_loss_ = tf.keras.losses.MeanSquaredError(
        reduction=tf.keras.losses.Reduction.SUM
    )
    dist_max = mse_loss_(max_bound, pred)
    dist_min = mse_loss_(min_bound, pred)
    loss = dist_max + dist_min
    return loss

grad_X_e = tf.Variable(grad_X_e, dtype=tf.float32)

with tf.GradientTape() as tape:
    tape.watch(grad_X_e)
    # Calculate the value of the function and record the gradient
    pred = tf_model(tf.expand_dims(grad_X_e, axis=0))
    pred = tf.squeeze(pred)
    loss = compute_loss(max_bound, min_bound, pred)
#pred = tf_model(grad_X_e)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.05,  epsilon=1e-07,)
while (tf.reduce_any(pred>max_bound) or tf.reduce_any(pred<min_bound)) and (it<max_iter):
    #change (X_e)
    gradient = tape.gradient(loss, grad_X_e)
    if gradient is None:
        print("no gradient")
        break

    # Use the Adam optimizer to update the value of x
    optimizer.apply_gradients([(gradient, grad_X_e)])
    
    with tf.GradientTape() as tape:
        tape.watch(grad_X_e)
        # Calculate the value of the function and record the gradient
        pred = tf_model(tf.expand_dims(grad_X_e, axis=0))
        pred = tf.squeeze(pred)
        loss = compute_loss(max_bound, min_bound, pred)
    # Record the current value of x
    print(f"Iteration {it}, Loss: {loss.numpy()}, Grad_X_e: {grad_X_e.numpy()}")
    it += 1
    
print("Optimized value of x:", grad_X_e.numpy())
final_pred = tf_model(tf.expand_dims(grad_X_e, axis=0))
print("Value of the function at the optimized point:", final_pred.numpy())
print("ADGAB", tf.squeeze(final_pred, axis=-1).numpy()[:,:][0])
#exit()
test = []    
for i,j in zip(grad_X_e.numpy()[:,0],grad_X_e.numpy()[:,1]):
    test.append(4*(i**2)+3*j)

plt.figure(figsize=(12,6))
plt.plot(grad_X_e.numpy()[:,0], label='exog1')#data[look_back+1:][0]
plt.plot(grad_X_e.numpy()[:,1], label='exog2')#data[look_back+1:][0]
plt.plot(tf.squeeze(final_pred, axis=-1).numpy()[:,:][0], label='target', color='r')
plt.plot(min_bound, label='min_bound', color='black')
plt.plot(max_bound, label='max_bound', color='black')
#plt.plot(test, label='actual', color='g')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time-Series Forecasting')
plt.legend()
plt.show() 
