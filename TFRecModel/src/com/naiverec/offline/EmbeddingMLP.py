import tensorflow as tf


##获取训练数据集
training_samples_file_path = tf.keras.utils.get_file("trainingSamples.csv",
                            "file:///D:/workspaces/NaiveRecSys/src/main/resources/webroot/sampledata/trainingSamples.csv")
##获取测试数据集
test_samples_file_path = tf.keras.utils.get_file("testSamples.csv",
                                                     "file:///D:/workspaces/NaiveRecSys/src/main/resources/webroot/sampledata/testSamples.csv")

print(training_samples_file_path )
print(test_samples_file_path )
##定义一个加载文件到tf的方法
def get_dataset(file_path):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name='label',
        na_value='0',
        num_epochs=1,
        ignore_errors=True
    )
    return dataset


##加载训练与测试数据集
train_dataset = get_dataset(training_samples_file_path)
test_dataset = get_dataset(test_samples_file_path)
##定义电影的genres 特征值集合
genre_vocab=['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western', 'Documentary',
             'Sci-Fi', 'Drama', 'Thriller',
             'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery', 'Children', 'Musical']
##定义一个map字典用于one-hot编码
GENRE_FEATURES={
    'userGenre1':genre_vocab,
    'userGenre2':genre_vocab,
    'userGenre3':genre_vocab,
    'userGenre4':genre_vocab,
    'userGenre5':genre_vocab,
    'movieGenre1':genre_vocab,
    'movieGenre2':genre_vocab,
    'movieGenre3':genre_vocab,
}
##定义一个集合存放所有类别型特征
categorical_columns = []
for feature,vocab in GENRE_FEATURES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature,vocabulary_list=vocab)
    emb_col =tf.feature_column.embedding_column(cat_col,10)
    # print(emb_col)
    categorical_columns.append(emb_col)

##movieId embedding 特征
movie_col=tf.feature_column.categorical_column_with_identity(key='movieId',num_buckets=1001)
movie_emb_col=tf.feature_column.embedding_column(movie_col,10)
categorical_columns.append(movie_emb_col)
##userId embedding  特征
user_col = tf.feature_column.categorical_column_with_identity(key='userId',num_buckets=30001)
user_emb_col = tf.feature_column.embedding_column(user_col,10)
categorical_columns.append(user_emb_col)
##所有数值型特征
numrical_columns=[
    tf.feature_column.numeric_column('releaseYear'),
    tf.feature_column.numeric_column('movieRatingCount'),
    tf.feature_column.numeric_column('movieAvgRating'),
    tf.feature_column.numeric_column('movieRatingStddev'),
    tf.feature_column.numeric_column('userRatingCount'),
    tf.feature_column.numeric_column('userAvgRating'),
    tf.feature_column.numeric_column('userRatingStddev')
]
##定义embedding+MLP model架构
model=tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(numrical_columns+categorical_columns),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
##编译模型，定义模型的loss function，optimizer，以及evaluation metrics
model.compile(
    loss='binary_crossentropy', ##损失函数使用二分交叉熵
    optimizer='adam',
    metrics=['accuracy',tf.keras.metrics.AUC(curve='ROC'),tf.keras.metrics.AUC(curve='PR')]
)

##训练模型
model.fit(train_dataset,epochs=5)
##evaluate the model
test_loss,test_accuracy,test_roc_auc,test_pr_auc = model.evaluate(test_dataset)
print("test loss:{},test accuracy:{},test roc auc:{},test pr auc:{}".format(test_loss,test_accuracy,test_roc_auc,test_pr_auc))
##输出一些预测结果
predictions = model.predict(test_dataset)
print(predictions)
for prediction,goodrating in zip(predictions[:12], list(test_dataset)[0][1][:12]):
    print("predicted good rating:{:.2%}".format(prediction[0]),
          "| actual rating label:",
          ("good rating" if bool(goodrating) else "bad rating"))
