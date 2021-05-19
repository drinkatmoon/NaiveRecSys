import tensorflow as tf

##导入本地训练以及测试样本路径
train_samples_file_path = tf.keras.utils.get_file("trainingSamples.csv",
                                                  "file:///D:/workspaces/NaiveRecSys/src/main/resources/webroot/sampledata/trainingSamples.csv")
test_samples_file_path = tf.keras.utils.get_file("trainingSamples.csv",
                                                  "file:///D:/workspaces/NaiveRecSys/src/main/resources/webroot/sampledata/testSamples.csv")

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
# genre features vocabulary
genre_vocab = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western', 'Documentary',
               'Sci-Fi', 'Drama', 'Thriller',
               'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery', 'Children', 'Musical']

GENRE_FEATURES = {
    'userGenre1': genre_vocab,
    'userGenre2': genre_vocab,
    'userGenre3': genre_vocab,
    'userGenre4': genre_vocab,
    'userGenre5': genre_vocab,
    'movieGenre1': genre_vocab,
    'movieGenre2': genre_vocab,
    'movieGenre3': genre_vocab
}

#all categorical features
categorical_columns = []
for feature,vocab in GENRE_FEATURES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(key=feature,vocabulary_list=vocab)
    emb_col = tf.feature_column.embedding_column(cat_col,10)
    categorical_columns.append(emb_col)

movie_col = tf.feature_column.categorical_column_with_identity(key='movieId',num_buckets=1001)
movie_emb_col = tf.feature_column.embedding_column(movie_col,10)
categorical_columns.append(movie_emb_col)

user_col = tf.feature_column.categorical_column_with_identity(key='userId',num_buckets=10001)
user_emb_col = tf.feature_column.embedding_column(user_col,10)
categorical_columns.append(user_emb_col)

# all numerical features
numerical_columns = [tf.feature_column.numeric_column('releaseYear'),
                     tf.feature_column.numeric_column('movieRatingCount'),
                     tf.feature_column.numeric_column('movieAvgRating'),
                     tf.feature_column.numeric_column('movieRatingStddev'),
                     tf.feature_column.numeric_column('userRatingCount'),
                     tf.feature_column.numeric_column('userAvgRating'),
                     tf.feature_column.numeric_column('userRatingStddev')]