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

##定义模型的输入层
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

    'userGenre1': tf.keras.layers.Input(name='userGenre1', shape=(), dtype='string'),
    'userGenre2': tf.keras.layers.Input(name='userGenre2', shape=(), dtype='string'),
    'userGenre3': tf.keras.layers.Input(name='userGenre3', shape=(), dtype='string'),
    'userGenre4': tf.keras.layers.Input(name='userGenre4', shape=(), dtype='string'),
    'userGenre5': tf.keras.layers.Input(name='userGenre5', shape=(), dtype='string'),
    'movieGenre1': tf.keras.layers.Input(name='movieGenre1', shape=(), dtype='string'),
    'movieGenre2': tf.keras.layers.Input(name='movieGenre2', shape=(), dtype='string'),
    'movieGenre3': tf.keras.layers.Input(name='movieGenre3', shape=(), dtype='string'),
}
##定义movieId embedding feature
movie_col = tf.feature_column.categorical_column_with_identity(key='movieId',num_buckets=1001)
movie_emb_col = tf.feature_column.embedding_column(movie_col,10)
movie_ind_col=tf.feature_column.indicator_column(movie_col) ##movieId indicator column
##定义userId embedding feature
user_col = tf.feature_column.categorical_column_with_identity(key='userId',num_buckets=30001)
user_emb_col=tf.feature_column.embedding_column(user_col,10)
user_ind_col=tf.feature_column.indicator_column(user_col)

# genre features vocabulary
genre_vocab = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western', 'Documentary',
               'Sci-Fi', 'Drama', 'Thriller',
               'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery', 'Children', 'Musical']

##定义user genre embedding feature
user_genre_col=tf.feature_column.categorical_column_with_vocabulary_list(key='userGenre1',vocabulary_list=genre_vocab)
user_genre_emb_col=tf.feature_column.embedding_column(user_genre_col,10)
##用户类别指标特征列
user_genre_ind_col=tf.feature_column.indicator_column(user_genre_col)

##商品 genre embedding feature
item_genre_col=tf.feature_column.categorical_column_with_vocabulary_list(key='movieGenre1',vocabulary_list=genre_vocab)
item_genre_emb_col =tf.feature_column.embedding_column(item_genre_col,10)
##商品类别指标特征列
item_genre_ind_col = tf.feature_column.indicator_column(item_genre_col)


##FM(factorization mechine) first-order 所用项目特征列，不含embedding以及concatenate
fm_first_order_columns=[movie_ind_col,user_ind_col,user_genre_ind_col,item_genre_ind_col]

##定义deepFM的deep部分的特征列
deep_feature_columns=[
    tf.feature_column.numeric_column("releaseYear"),
    tf.feature_column.numeric_column("movieRatingCount"),
    tf.feature_column.numeric_column("movieAvgRating"),
    tf.feature_column.numeric_column("movieRatingStddev"),
    tf.feature_column.numeric_column("userRatingCount"),
    tf.feature_column.numeric_column("userAvgRating"),
    tf.feature_column.numeric_column("userRatingStddev"),
    movie_emb_col,
    user_emb_col
]

##开始通过keras定义模型的各层
