import tensorflow as tf

train_samples_file_path=tf.keras.utils.get_file("trainingSamples.csv",
                        "file:///E:/workspaces/NaiveRecSys/src/main/resources/webroot/sampledata/trainingSamples.csv")
test_samples_file_path=tf.keras.utils.get_file("trainingSamples.csv",
                                               "file:///E:/workspaces/NaiveRecSys/src/main/resources/webroot/sampledata/testSamples.csv")
##加载知道目录数据为tf dataset
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name='label',
        na_value='0',
        num_epochs=1,
        ignore_errors=True
    )
    return  dataset

train_dataset = get_dataset(train_samples_file_path)
test_dataset = get_dataset(test_samples_file_path)

##movieId embedding feature
movie_col=tf.feature_column.categorical_column_with_identity(key='movieId',num_buckets=1001)
movie_emb_col=tf.feature_column.embedding_column(movie_col,10)
##userId embedding feature
user_col=tf.feature_column.categorical_column_with_identity(key='userId',num_buckets=30001)
user_emb_col=tf.feature_column.embedding_column(user_col,10)

##define input for keras  model
inputs={
    'movieId':tf.keras.layers.Input(name='movieId',shape=(),dtype='int32'),
    'userId':tf.keras.layers.Input(name='userId',shape=(),dtype='int32')
}

##神经网络CF模型1：双塔模型的底层使用embedding，特征交叉层使用MLP
def neural_model_1(feature_inputs,item_feature_columns,user_feature_columns,hidden_units):
    item_tower=tf.keras.layers.DenseFeatures(item_feature_columns)(feature_inputs)
    user_tower=tf.keras.layers.DenseFeatures(user_feature_columns)(feature_inputs)
    interact_layer=tf.keras.layers.concatenate([item_tower,user_tower])
    for num_nodes in hidden_units:
        interact_layer=tf.keras.layers.Dense(num_nodes,activation='relu')(interact_layer)
    output_layer=tf.keras.layers.Dense(1,activation='sigmoid')(interact_layer)
    neural_cf_model=tf.keras.Model(feature_inputs,output_layer)
    return  neural_cf_model

##neural cf 模型2：双塔模型的底层使用embedding+mlp,讲双塔结果进行点击处理作为输出层
def neural_model_2(feature_inputs,item_feature_columns,user_feature_columns,hidden_units):
    item_tower=tf.keras.layers.DenseFeatures(item_feature_columns)(feature_inputs)
    for num_nodes in hidden_units:
        item_tower=tf.keras.layers.Dense(num_nodes,activation='relu')(item_tower)

    user_tower=tf.keras.layers.DenseFeatures(user_feature_columns)(feature_inputs)
    for num_nodes in hidden_units:
        user_tower=tf.keras.layers.Dense(num_nodes,activation='relu')(user_tower)

    output =tf.keras.layers.Dot(axes=1)([item_tower,user_tower])
    output =tf.keras.layers.Dense(1,activation='sigmoid')(output)
    neural_cf_model=tf.keras.Model(feature_inputs,output)
    return neural_cf_model

##neural cf model architecture
model =neural_model_1(inputs,[movie_emb_col],[user_emb_col],[10,10])

##编译模型，设置损失函数，优化器，以及模型评估指标
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy',tf.keras.metrics.AUC(curve='ROC'),tf.keras.metrics.AUC(curve='PR')]
)

##训练模型
model.fit(train_dataset,epochs=5)
##evaluate the model
test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset)
print('\n\nTest Loss {}, Test Accuracy {}, Test ROC AUC {}, Test PR AUC {}'.format(test_loss, test_accuracy,
                                                                                   test_roc_auc, test_pr_auc))
# print some predict results
predictions = model.predict(test_dataset)
##zip([迭代器1，迭代器2...])函数将多个迭代器的元素合并成元组型元素的单个迭代器
for prediction, goodRating in zip(predictions[:12], list(test_dataset)[0][1][:12]):
    print("Predicted good rating: {:.2%}".format(prediction[0]),
          " | Actual rating label: ",
          ("Good Rating" if bool(goodRating) else "Bad Rating"))


##模型保存
tf.keras.models.save_model(
    model,
    'E:/workspaces/NaiveRecSys/src/main/resources/webroot/modeldata/neuralcf/002',
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)
