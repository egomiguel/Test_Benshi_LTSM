import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from itertools import product
from dataclasses import dataclass

path = "data\\"

numerical_features = ['stock_initial', 'stock_received', 'stock_distributed', 'stock_adjustment', 'stock_end',
                    'average_monthly_consumption', 'stock_stockout_days', 'stock_ordered']


def read_and_process_data():
    train = pd.read_csv(path + 'Train.csv')
    train.describe()

    # Checking for outliers
    Q1 = train.quantile(0.01)
    Q3 = train.quantile(0.99)
    IQR = Q3 - Q1

    # Finding the Upper Bound and the Lower Bound
    lower_bound = Q1 - (2.0 * IQR)
    upper_bound = Q3 + (2.0 * IQR)

    print("Lower Bound")
    print(lower_bound)

    print("Upper Bound")
    print(upper_bound)

    df = train[~((train < lower_bound) | (train > upper_bound)).any(axis=1)]

    print(train.shape)
    print(df.shape)

    df.info()
    '''
        There are 35473 observations in all columns except the column "stock_ordered" looks like it's missing some values.
    '''

    df['stock_ordered'] = df.groupby(['site_code', 'product_code'])['stock_ordered'].transform(lambda x: x.fillna(x.mean()))

    '''
        Missing values were filled in with the mean for each product by city.
    '''

    date = df[['year', 'month']].drop_duplicates().reset_index(drop=True)
    location = df[['region', 'district', 'site_code', 'product_code']].drop_duplicates().reset_index(drop=True)
    combination = pd.merge(date.assign(j=1), location.assign(j=1)).drop(columns='j')

    '''
        Expanding the data set for all months, the products that are not distributed at a specific month will be filled with zero.
    '''

    df = pd.merge(combination, df, how='left')

    df['day'] = 1
    df['ds'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df.drop(columns=['year', 'month', 'day'])

    '''
        Organizing some columns to be able to study the csv file better.
    '''

    df = df[['site_code', 'product_code'] + df.drop(columns=['site_code', 'product_code']).columns.tolist()]

    '''
        Ordering the data set, this ordering is necessary to be able to generate new features in the correct order.
    '''

    df = df.sort_values(by=['site_code', 'product_code', 'ds']).reset_index(drop=True)

    df[numerical_features] = df[numerical_features].fillna(0)

    df.info()

    return df


df = read_and_process_data()

site_code_list = df['site_code'].unique()
product_code_list = df['product_code'].unique()
important_cols = ['stock_distributed', 'stock_initial', 'stock_received', 'stock_adjustment',
                  'stock_end', 'average_monthly_consumption', 'stock_stockout_days', 'stock_ordered']


def cross_product():
    return product(site_code_list, product_code_list)


def reordering_by_columns(df):
    new_df = pd.DataFrame(index=df['ds'].unique())
    new_df['ds'] = df['ds'].unique()
    cont = 0
    cross_product_list = cross_product()
    submission = pd.DataFrame(columns=['ID', 'prediction'])

    for site_code, product_code in cross_product_list:
        if len(df[(df['site_code'] == site_code) & (df['product_code'] == product_code)]) != 0:
            cols = [col + "_" + str(cont) for col in important_cols]
            new_df = new_df.reset_index(drop=True)
            new_df[cols] = df[(df['site_code'] == site_code) & (df['product_code'] == product_code)][important_cols].reset_index(drop=True)
            cont += 1

            date1 = "2019X7X" + site_code + "X" + product_code
            date2 = "2019X8X" + site_code + "X" + product_code
            date3 = "2019X9X" + site_code + "X" + product_code

            dict = {'ID': [date1, date2, date3],
                    'prediction' : None}
            df2 = pd.DataFrame(dict)
            submission = submission.append(df2, ignore_index=True)


    '''
        Generating date Features
    '''

    new_df['month'] = new_df['ds'].dt.month
    new_df['quarter'] = new_df['ds'].dt.quarter

    new_df['weekend'] = new_df['ds'].dt.days_in_month - np.busday_count(
        new_df['ds'].dt.date.values.astype('datetime64[D]'),
        (new_df['ds'].dt.date + pd.DateOffset(months=1)).values.astype('datetime64[D]')
    )

    return new_df, submission


df, submission = reordering_by_columns(df)

inference_date = pd.to_datetime('2019-01-01')

inference = df.loc[(df["ds"] >= inference_date), :]
inference = (inference.drop(columns=['ds'])).to_numpy()
inference = tf.convert_to_tensor(inference)
inference = tf.reshape(inference, [1, 6, -1])

df = df.drop(columns=['ds'])
df_shape = df.shape
new_features_amount = 3


@dataclass
class G:
    WINDOW_SIZE = 6
    LABEL_SIZE = 3
    LABEL_COLUMNS = (df_shape[1] - new_features_amount) // len(important_cols)
    BATCH_SIZE = 16


def split_window(features, data_size, label_size):

    tSize = len(important_cols)
    inputs = features[:, 0:data_size, :]
    labels = features[:, data_size:, 0:-new_features_amount:tSize]

    inputs.set_shape([None, data_size, None])
    labels.set_shape([None, label_size, None])

    return inputs, labels


def plot_loss_acc(history):
    loss = history.history['loss']
    epochs = range(len(loss))
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.title('Loss')
    plt.figure()
    plt.show()


dataset = tf.keras.utils.timeseries_dataset_from_array(data=df, targets=None, sequence_length=G.WINDOW_SIZE + G.LABEL_SIZE,
                                                       sampling_rate=1, sequence_stride=1, batch_size=G.BATCH_SIZE, shuffle=True)

dataset = dataset.map(lambda x: split_window(x, G.WINDOW_SIZE, G.LABEL_SIZE))


mc = ModelCheckpoint('model_LTSM.h5', monitor='loss', mode='min', verbose=1, save_best_only=True)


def create_model_LSTM():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, activation='relu', padding='causal', input_shape=[G.WINDOW_SIZE, df_shape[1]]))
    model.add(tf.keras.layers.LSTM(512, return_sequences=True))
    model.add(tf.keras.layers.LSTM(512))
    model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(G.LABEL_SIZE * G.LABEL_COLUMNS, kernel_initializer=tf.initializers.zeros()))
    model.add(tf.keras.layers.Reshape([G.LABEL_SIZE, G.LABEL_COLUMNS]))
    model.compile(loss="mse", optimizer='adam')
    return model


model = create_model_LSTM()
history = model.fit(dataset, epochs=3000, callbacks=[mc])
plot_loss_acc(history)


new_model = tf.keras.models.load_model('model_LTSM.h5')
y_pred = new_model.predict(inference)
result = y_pred.flatten('F')
submission['prediction'] = result

submission.to_csv(f'data/submission.csv', index=False)