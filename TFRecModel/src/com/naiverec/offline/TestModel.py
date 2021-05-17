import tensorflow as tf

##载入mnist数据
mnist=tf.keras.datasets.mnist
##划分训练集和测试集
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train,x_test = x_train/255.0,x_test/255.0
##定义模型结构和参数
model = tf.keras.models.Sequential([
    ##输入层28*28维矩阵
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    ##dropout说的是在神经网络中进行前向传导的过程,让某个神经元的激活值以一定的概率p,让其停止工作
    tf.keras.layers.Dropout(0.2),
    ##输出层采用softmax模型,处理多分类问题
    tf.keras.layers.Dense(10,activation='softmax')
])

##定义模型的优化方法,损失函数和评估指标
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
##训练模型,进行5轮迭代更新
model.fit(x_train,y_train,epochs=5)
##模型评估 verbose该参数的值用于控制日志显示的方式
model.evaluate(x_test,y_test,verbose=2)