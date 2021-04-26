package com.naiverec.featureeng

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.{SparkConf, sql}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object FeatureEngineering {

  def oneHotEncoderExample(movieSamples: DataFrame): Unit = {
    //samples样本集中的每一条数据代表一部电影的信息，其中movieId为电影id
    val samplesWithIdNumber = movieSamples.withColumn("movieIdNumber", col("movieId").cast(sql.types.IntegerType))
    //利用Spark的机器学习库Spark MLlib创建One-hot编码器
    val oneHotEncoder = new OneHotEncoderEstimator()
      .setInputCols(Array("movieIdNumber"))
      .setOutputCols(Array("movieIdVector"))
      .setDropLast(false)
    //训练One-hot编码器，并完成从id特征到One-hot向量的转换
    val oneHotEncoderSample = oneHotEncoder.fit(samplesWithIdNumber).transform(samplesWithIdNumber)
    //打印最终样本的数据结构
    oneHotEncoderSample.printSchema()
    //打印10条样本查看结果
    oneHotEncoderSample.show(10)
  }

  def array2vec:UserDefinedFunction = udf{
        //使用Vectors.sparse构造稀疏向量，第一个参数为向量长度，第2个参数为顺序向量，第三个为值向量
        //如(1.0,0.0,1.0,3.0) 的稀疏格式为(4,[0,2,3],[1.0,1.0,3.0]),顺序向量里没有出现的索引对应的值为0
    (a: Seq[Int], length: Int) => Vectors.sparse(length,a.sortWith(_<_).toArray,Array.fill[Double](a.length)(1.0))
  }

  /**
   * 多热编码示例
   *
   * @param samples
   */
  def multiHotEncoderExample(samples: DataFrame): Unit = {
    val samplesWithGenre = samples.select(col("movieId"), col("title"),
      explode(split(col("genres"), "\\|").cast("array<string>")).as("genre"))
    samplesWithGenre.show()
    //使用stringIndexer编码器编码时，优先编码出现频率最高的标签，从0.0开始
    val genreIndexer = new StringIndexer().setInputCol("genre").setOutputCol("genreIndex")
    val stringIndexerModel = genreIndexer.fit(samplesWithGenre)
    val genreIndexSamples = stringIndexerModel.transform(samplesWithGenre)
      .withColumn("genreIndexInt",col("genreIndex").cast(sql.types.IntegerType))
    genreIndexSamples.show()
    var indexSize = genreIndexSamples.agg(max(col("genreIndexInt"))).head().getAs[Int](0)+1
    println(indexSize)
    val processedSamples = genreIndexSamples.groupBy(col("movieId")).agg(collect_list("genreIndexInt").as("genreIndexes"))
      .withColumn("indexSize", typedLit(indexSize)) //通过typedLit可以添加list，seq，Map类型的常量列
    processedSamples.show()
    val finalSample = processedSamples.withColumn("vector", array2vec(col("genreIndexes"), col("indexSize")))
    finalSample.show()
  }

  def double2vec:UserDefinedFunction = udf{
    (value:Double) => Vectors.dense(value)
  }

  /**
   * 数值型特征处理
   * 1.避免不同特征之间区间范围相差过大而对模型产生较大影响，将不同特征范围压缩到一个区间
   * 2.调大区分度，可以采用分桶方式
   *
   * @param ratingSamples
   */
  def ratingFeatures(ratingSamples: DataFrame): Unit = {
    ratingSamples.printSchema()
    ratingSamples.show()
    //利用打分表ratings计算电影的平均分、被打分次数等数值型特征
    val movieFeatures = ratingSamples.groupBy("movieId")
      .agg(count(lit(1)).as("ratingCount"),
        avg(col("rating")).as("avgRating"),
        variance(col("rating")).as("ratingVar"))
      .withColumn("avgRatingVec", double2vec(col(("avgRating"))))
    movieFeatures.show()
    //分桶处理，创建QuantileDiscretizer进行分桶，将打分次数这一特征分到100个桶中

    //归一化处理，创建MinMaxScaler进行归一化，将平均得分进行归一化

    //创建一个pipeline，依次执行两个特征处理过程

    //打印最终结果
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    val conf = new SparkConf().setMaster("local[*]")
      .setAppName("FeatureEngineering")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder().config(conf).getOrCreate()
    val movieResourcesPath = this.getClass.getResource("/webroot/sampledata/movies.csv")
    val movieSamples = spark.read.format("csv").option("header", "true").load(movieResourcesPath.getPath)
    println("Raw Movie Samples:")
    movieSamples.printSchema()
    movieSamples.show(20)
    //对类别型数据进行独热编码
    println("OneHotEncoder Example:")
    oneHotEncoderExample(movieSamples)

    //对含多值的类别型数据，如标签进行多热编码
    println("MultiHotEncoder Example:")
    multiHotEncoderExample(movieSamples)

    println("Numberical features Example:")
    val ratingResourcePath = this.getClass.getResource("/webroot/sampledata/ratings.csv")
    val ratingSamples = spark.read.format("csv").option("header", "true").load(ratingResourcePath.getPath)
    ratingFeatures(ratingSamples)
  }
}

