import tensorflow as tf
import pandas as pd
import mouse

data = pd.read_csv("database.csv")
print(data.head())

ind_var = data[['l_c_x', 'l_c_y', 'l_r_x', 'l_r_y', 'l_t_x', 'l_t_y', 'l_l_x', 'l_l_y','l_b_x', 'l_b_y', 'r_c_x', 'r_c_y', 'r_r_x', 'r_r_y', 'r_t_x', 'r_t_y','r_l_x', 'r_l_y', 'r_b_x', 'r_b_y']]
d_var = data[['x','y']]

X = tf.keras.layers.Input(shape=[20])
H = tf.keras.layers.Dense(6, activation="relu")(X)
H = tf.keras.layers.Dense(4, activation="relu")(H)
Y = tf.keras.layers.Dense(2)(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

model.fit(ind_var, d_var, epochs=1000)

model.save('train_model.h5')

