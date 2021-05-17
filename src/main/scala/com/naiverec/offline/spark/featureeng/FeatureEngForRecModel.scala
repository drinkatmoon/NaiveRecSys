package com.naiverec.offline.spark.featureeng

import org.apache.log4j.{Level, Logger}

import org.apache.spark.SparkConf
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType

import scala.collection.immutable.ListMap
import scala.collection.mutable

object FeatureEngForRecModel {
  val NUMBER_PRECISION = 2 //小数精度

  def addSampleLabel(ratingSamples: DataFrame): DataFrame = {
    ratingSamples.show(10,true)
    ratingSamples.printSchema()
    //统计数量
    val sampleCount = ratingSamples.count()
    //计算各类评分占比
    ratingSamples.groupBy(col("rating")).count().orderBy(col("rating"))
      .withColumn("percentage",col("count")/sampleCount).show(100,truncate = true)
    //将评分数据作为label数据,并转化为二分类问题
    ratingSamples.withColumn("label",when(col("rating")>=3.5,1).otherwise(0))
//    ratingSamples
  }

  def addMovieFeatures(movieSamples: DataFrame, ratingSamples: DataFrame): DataFrame = {
    //添加电影基础特征
    val samplesWithMovies1 = ratingSamples.join(movieSamples, Seq("movieId"), "left")
    println("samplesWithMovies1.show:")
    samplesWithMovies1.show()
    //自定义函树获取发行年份
    val extractReleaseYearUdf = udf({
      (title:String)=>{
        if(null == title  || title.trim.length<6){
          1990
        }else{
          var yearString = title.trim.substring(title.length-5,title.length-1)
          yearString.toInt
        }
      }
    })
    val extractTitleUdf = udf({ (title: String) => {
      title.trim.substring(0, title.length - 6).trim
    }
    })
    //对训练数据的title进行处理
    val samplesWithMovies2 = samplesWithMovies1.withColumn("releaseYear", extractReleaseYearUdf(col("title")))
      .withColumn("title", extractTitleUdf(col("title")))
      .drop("title")//删除原title

    //补充电影3级分类数据
    val samplesWithMovie3 = samplesWithMovies2.withColumn("movieGenre1", split(col("genres"), "\\|").getItem(0))
      .withColumn("movieGenre2", split(col("genres"), "\\|").getItem(1))
      .withColumn("moivieGenre3", split(col("genres"), "\\|").getItem(2))
    //计算电影评分特征(评分次数,平均分,标准差)
    val movieRatingFeatures = samplesWithMovie3.groupBy("movieId")
      .agg(count(lit(1)).as("movieRatingCount"),
        format_number(avg(col("rating")), NUMBER_PRECISION).as("movieAvgRating"),
        stddev(col("rating")).as("movieRatingStddev"))
      .na.fill(0).withColumn("movieRatingStddev", format_number(col("movieRatingStddev"), NUMBER_PRECISION))
    //补充电影的评分特征
    println("samplesWithMovie3.printSchema:")
    samplesWithMovie3.printSchema()
    println("movieRatingFeatures.printSchema:")
    movieRatingFeatures.printSchema()
    val samplesWithMovie4 = samplesWithMovie3.join(movieRatingFeatures, Seq("movieId"), "left")
    samplesWithMovie4.printSchema()
    samplesWithMovie4.show(10,false)
    samplesWithMovie4
  }

  val extractGenres: UserDefinedFunction = udf { (genreArray: Seq[String]) => {
    val genreMap = mutable.Map[String, Int]()
    genreArray.foreach((element:String) => {
      val genres = element.split("\\|")
      genres.foreach((oneGenre:String) => {
        genreMap(oneGenre) = genreMap.getOrElse[Int](oneGenre, 0)  + 1
      })
    })
    val sortedGenres = ListMap(genreMap.toSeq.sortWith(_._2 > _._2):_*)
    sortedGenres.keys.toSeq
  }}
  def addUserFeatures(ratingSamples: DataFrame):DataFrame = {
    val samplesWithUserFeatures = ratingSamples.withColumn("userPositiveHistory", collect_list(when(col("label") === 1, col("movieId")).otherwise(lit(null)))
      .over(Window.partitionBy(col("userId")).orderBy(col("timestamp")).rowsBetween(-100, -1)))
      .withColumn("userPositiveHistory", col("userPositiveHistory"))
      .withColumn("userRatedMovie1", col("userPositiveHistory").getItem(0))
      .withColumn("userRatedMovie2", col("userPositiveHistory").getItem(1))
      .withColumn("userRatedMovie3", col("userPositiveHistory").getItem(2))
      .withColumn("userRatedMovie4", col("userPositiveHistory").getItem(3))
      .withColumn("userRatedMovie5", col("userPositiveHistory").getItem(4))
      .withColumn("userRatingCount", count(lit(1)).over(Window.partitionBy(col("userId")).orderBy("timestamp").rowsBetween(-100, -1)))
      .withColumn("userAvgReleaseYear", avg("releaseYear").over(Window.partitionBy("userId").orderBy("timestamp").rowsBetween(-100, -1)).cast(IntegerType))
      .withColumn("userReleaseYearStddev", stddev(col("releaseYear")).over(Window.partitionBy("userId").orderBy(col("timestamp")).rowsBetween(-100, -1)))
      .withColumn("userAvgRating", format_number(avg("rating").over(Window.partitionBy("userId").orderBy("timestamp").rowsBetween(-100, -1)), NUMBER_PRECISION))
      .withColumn("userRatingStddev", stddev(col("rating")).over(Window.partitionBy("userId").orderBy(col("timestamp")).rowsBetween(-100, -1)))
      .withColumn("userGenres", extractGenres(collect_list(when(col("label") === 1, col("genres")).otherwise(lit(null)))
        .over(Window.partitionBy("userId").orderBy(col("timestamp")).rowsBetween(-100, -1)))).na.fill(0)
      .withColumn("userRatingStddev", format_number(col("userRatingStddev"), NUMBER_PRECISION))
      .withColumn("userReleaseYearStddev", format_number(col("userReleaseYearStddev"), NUMBER_PRECISION))
      .withColumn("userGenre1", col("userGenres").getItem(0))
      .withColumn("userGenre2", col("userGenres").getItem(1))
      .withColumn("userGenre3", col("userGenres").getItem(2))
      .withColumn("userGenre4", col("userGenres").getItem(3))
      .withColumn("userGenre5", col("userGenres").getItem(4))
      .drop("genres", "userGenres", "userPositiveHistory")
      .filter(col("userRatingCount") > 1)

    samplesWithUserFeatures.printSchema()
    samplesWithUserFeatures.show(100,false)

    samplesWithUserFeatures
  }

  def splitAndSaveTrainingTestSamples(samples: DataFrame, savePath: String): Unit = {
    //生成一个少量的样本集
    val smallSamples = samples.sample(0.1)
    //将训练集与测试集按8:2拆分
    val Array(training,test) = smallSamples.randomSplit(Array(0.8, 0.2))
    val samplesResourcesPath = this.getClass.getResource(savePath)
    training.repartition(1).write.option("header","true").mode(SaveMode.Overwrite).csv(samplesResourcesPath+"/trainingSamples")
    test.repartition(1).write.option("header","true").mode(SaveMode.Overwrite).csv(samplesResourcesPath+"/testSamples")

  }

  def extractAndSaveUserFeaturesToRedis(samplesWithUF: DataFrame): Unit = {
    samplesWithUF
  }

  def extractAndSaveMovieFeaturesToRedis(samplesWithUserFeatures: DataFrame): Unit = {}

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("featureEngForRecModel")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder().config(conf).getOrCreate()
    val movieResourcesPath = this.getClass.getResource("/webroot/sampledata/movies.csv")
    val movieSamples = spark.read.format("csv").option("header", "true").load(movieResourcesPath.getPath)
    val ratingsResourcesPath = this.getClass.getResource("/webroot/sampledata/ratings.csv")
    val ratingSamples = spark.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)

    val ratingSamplesWithLabel = addSampleLabel(ratingSamples)
    println("ratingSamplesWithLabel.show:")
    ratingSamplesWithLabel.show(10,true)

    val samplesWithMovieFeatures = addMovieFeatures(movieSamples, ratingSamplesWithLabel)
    val samplesWithUserFeatures = addUserFeatures(samplesWithMovieFeatures)
    //save samples as csv format:将训练样本保存为csv格式文件
    splitAndSaveTrainingTestSamples(samplesWithUserFeatures, "/webroot/sampledata")

    //保存用户特征与物品特征到redis库以便线上使用
    extractAndSaveUserFeaturesToRedis(samplesWithUserFeatures)
    extractAndSaveMovieFeaturesToRedis(samplesWithUserFeatures)
  }

}
