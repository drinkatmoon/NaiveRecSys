##深度兴趣网络

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
# split as test dataset and training dataset
train_dataset = get_dataset(train_samples_file_path)
test_dataset = get_dataset(test_samples_file_path)

##config
RECENT_MOVIES=5
EMBEDDING_SIZE=10

#define input for keras model
inputs = {
    'movieAvgRating': tf.keras.layers.Input(name='movieAvgRating', shape=(), dtype='float32'),
    'movieRatingStddev': tf.keras.layers.Input(name='movieRatingStddev', shape=(), dtype='float32'),
    'movieRatingCount': tf.keras.layers.Input(name='movieRatingCount', shape=(), dtype='int32'),
    'userAvgRating': tf.keras.layers.Input(name='userAvgRating', shape=(), dtype='float32'),
    'userRatingStddev': tf.keras.layers.Input(name='userRatingStddev', shape=(), dtype='float32'),
    'userRatingCount': tf.keras.layers.Input(name='userRatingCount', shape=(), dtype='int32'),
    'releaseYear': tf.keras.layers.Input(name='releaseYear', shape=(), dtype='int32'),

    'movieId': tf.keras.layers.Input(name='movieId', shape=(), dtype='int32'),
    'userId': tf.keras.layers.Input(name='userId', shape=(), dtype='int32'),
    'userRatedMovie1': tf.keras.layers.Input(name='userRatedMovie1', shape=(), dtype='int32'),
    'userRatedMovie2': tf.keras.layers.Input(name='userRatedMovie2', shape=(), dtype='int32'),
    'userRatedMovie3': tf.keras.layers.Input(name='userRatedMovie3', shape=(), dtype='int32'),
    'userRatedMovie4': tf.keras.layers.Input(name='userRatedMovie4', shape=(), dtype='int32'),
    'userRatedMovie5': tf.keras.layers.Input(name='userRatedMovie5', shape=(), dtype='int32'),

    'userGenre1': tf.keras.layers.Input(name='userGenre1', shape=(), dtype='string'),
    'userGenre2': tf.keras.layers.Input(name='userGenre2', shape=(), dtype='string'),
    'userGenre3': tf.keras.layers.Input(name='userGenre3', shape=(), dtype='string'),
    'userGenre4': tf.keras.layers.Input(name='userGenre4', shape=(), dtype='string'),
    'userGenre5': tf.keras.layers.Input(name='userGenre5', shape=(), dtype='string'),
    'movieGenre1': tf.keras.layers.Input(name='movieGenre1', shape=(), dtype='string'),
    'movieGenre2': tf.keras.layers.Input(name='movieGenre2', shape=(), dtype='string'),
    'movieGenre3': tf.keras.layers.Input(name='movieGenre3', shape=(), dtype='string'),
}
# user id embedding feature
user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=30001)
user_emb_col = tf.feature_column.embedding_column(user_col, EMBEDDING_SIZE)
# genre features vocabulary
genre_vocab = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western', 'Documentary',
               'Sci-Fi', 'Drama', 'Thriller',
               'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery', 'Children', 'Musical']
# user genre embedding feature
user_genre_col = tf.feature_column.categorical_column_with_vocabulary_list(key="userGenre1",
                                                                           vocabulary_list=genre_vocab)
user_genre_emb_col = tf.feature_column.embedding_column(user_genre_col, EMBEDDING_SIZE)
# item genre embedding feature
item_genre_col = tf.feature_column.categorical_column_with_vocabulary_list(key="movieGenre1",
                                                                           vocabulary_list=genre_vocab)
item_genre_emb_col = tf.feature_column.embedding_column(item_genre_col, EMBEDDING_SIZE)


candidate_movie_col=[tf.feature_column.numeric_column(key='movieId',default_value=0),]
recent_rate_col=[
    tf.feature_column.numeric_column(key='userRatedMovie1',default_value=0),
    tf.feature_column.numeric_column(key='userRatedMovie2',default_value=0),
    tf.feature_column.numeric_column(key='userRatedMovie3',default_value=0),
    tf.feature_column.numeric_column(key='userRatedMovie4',default_value=0),
    tf.feature_column.numeric_column(key='userRatedMovie5',default_value=0)
]

#user profile
user_profile=[
    user_emb_col,
    user_genre_emb_col,
    tf.feature_column.numeric_column('userRatingCount'),
    tf.feature_column.numeric_column('userAvgRating'),
    tf.feature_column.numeric_column('userRatingStddev')
]

#context features
context_features=[
    item_genre_emb_col,
    tf.feature_column.numeric_column('releaseYear'),
    tf.feature_column.numeric_column('movieRatingCount'),
    tf.feature_column.numeric_column('movieAvgRating'),
    tf.feature_column.numeric_column('movieRatingStddev')
]

candidate_layer = tf.keras.layers.DenseFeatures(candidate_movie_col)(inputs)
user_behaviors_layer=tf.keras.layers.DenseFeatures(recent_rate_col)(inputs)
user_profile_layer=tf.keras.layers.DenseFeatures(user_profile)(inputs)
context_features_layer=tf.keras.layers.DenseFeatures(context_features)(inputs)

# Activation Unit
movie_emb_layer = tf.keras.layers.Embedding(input_dim=1001,output_dim=EMBEDDING_SIZE,mask_zero=True)# mask zero
user_behaviors_emb_layer=movie_emb_layer(user_behaviors_layer)
candidate_emb_layer = movie_emb_layer(candidate_layer)

candidate_emb_layer = tf.squeeze(candidate_emb_layer,axis=1)##axis可以用来指定要删掉的为1的维度
repeated_candidate_emb_layer = tf.keras.layers.RepeatVector(RECENT_MOVIES)(candidate_emb_layer) ##将输入重复RECENT_MOVIES次
##计算两个输入张量的差
activation_sub_layer = tf.keras.layers.Subtract()([user_behaviors_emb_layer,repeated_candidate_emb_layer])
##计算两个张量的元素积
activation_product_layer=tf.keras.layers.Multiply()([user_behaviors_emb_layer,repeated_candidate_emb_layer])
##连接输入张量列表,除了连接轴axis外，其他的尺寸必须相同
activation_all=tf.keras.layers.concatenate([activation_sub_layer,user_behaviors_emb_layer,repeated_candidate_emb_layer,activation_product_layer],axis=-1)

##定义中间层激活函数
activation_unit = tf.keras.layers.Dense(32)(activation_all) ##定义一个32维的输出空间
#参数化的RELU
activation_unit = tf.keras.layers.PReLU()(activation_unit)
activation_unit = tf.keras.layers.Dense(1, activation='sigmoid')(activation_unit)
##将输入展开
activation_unit=tf.keras.layers.Flatten()(activation_unit)
activation_unit = tf.keras.layers.RepeatVector(EMBEDDING_SIZE)(activation_unit)
##根据给定的模式置换输入的维度
activation_unit = tf.keras.layers.Permute((2,1))(activation_unit)
activation_unit=tf.keras.layers.Multiply()([user_behaviors_emb_layer,activation_unit])

##sum pooling层，对embedding进行加权（activation_all中含有了权重）求和
##Lambda，将任意表达式封装为layer对象
##backend,后端引擎，是一个专门的，优化的张量操作库
user_behaviors_pooled_layers=tf.keras.layers.Lambda(lambda x:tf.keras.backend.sum(x,axis=1))(activation_all)

##定义一个全连接层
concat_layer=tf.keras.layers.concatenate([user_profile_layer, user_behaviors_pooled_layers,
                             candidate_emb_layer, context_features_layer])
##定义输出层
output_layer = tf.keras.layers.Dense(128)(concat_layer)
output_layer = tf.keras.layers.PReLU()(output_layer)
output_layer = tf.keras.layers.Dense(64)(output_layer)
output_layer = tf.keras.layers.PReLU()(output_layer)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(output_layer)

##定义一个模型，包含从inputs导output_layer层的计算的所有网络层
model = tf.keras.Model(inputs,output_layer)

#配置模型,配置模型的损失函数（名），模型优化器（名称），模型评估标准
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy',tf.keras.metrics.AUC(curve='ROC'),tf.keras.metrics.AUC(curve='PR')]
)

##训练模型
model.fit(train_dataset,epochs=5)
##评估模型
test_loss,test_accuracy,test_roc_auc,test_pr_auc = model.evaluate(test_dataset)
print('\n\nTest Loss {}, Test Accuracy {}, Test ROC AUC {}, Test PR AUC {}'.format(test_loss, test_accuracy,
                                                                                   test_roc_auc, test_pr_auc))

# print some predict results
predictions = model.predict(test_dataset)
for prediction, goodRating in zip(predictions[:12], list(test_dataset)[0][1][:12]):
    print("Predicted good rating: {:.2%}".format(prediction[0]),
          " | Actual rating label: ",
          ("Good Rating" if bool(goodRating) else "Bad Rating"))

