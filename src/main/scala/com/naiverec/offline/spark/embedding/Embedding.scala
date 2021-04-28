package com.naiverec.offline.spark.embedding


import java.io.{BufferedWriter, File, FileWriter}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.mllib.feature.Word2Vec
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import redis.clients.jedis.Jedis
import redis.clients.jedis.params.SetParams

import scala.collection.mutable
import scala.util.Random
import scala.util.control.Breaks.{break,breakable}

object Embedding {

  val redisEndpoint = "localhost"
  val redisPort = 6379

  def processItemSequence(spark: SparkSession, rawSampleDataPath: String): RDD[Seq[String]] = {
    val ratingsResourcesPath = this.getClass.getResource(rawSampleDataPath)
    val ratingSamples = spark.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)
    ratingSamples.printSchema()
    //定义一个udf函数，按照用户的tiemstamp排序
    val sortUdf:UserDefinedFunction = udf {
      (rows:Seq[Row]) =>{
        rows.map{
          case Row(movieId:String,timestamp:String) =>(movieId,timestamp)
        }.sortBy{ case (_,timestamp)=>timestamp}
          .map{case (movieId,_)=>movieId}
      }
    }
    //把原始的rating数据处理成序列数据
    val userSeq = ratingSamples.where("rating>=3.5") //过滤掉评分在3.5以下的评分记录
      .groupBy("userId") //按照userId进行分组聚合
      .agg(sortUdf(collect_list(struct("movieId", "timestamp"))) as "movieIds")
      //把所有movieId连接起来方便后续word2vec模型处理
      .withColumn("movieIdStr", array_join(col("movieIds"), " "))

    //把序列数据筛选出来,丢掉其他过程数据
    val value = userSeq.select("movieIdStr").rdd.map(r => r.getAs[String]("movieIdStr").split(" ").toSeq)
    value.take(10).foreach(println)
    value
  }


  def trainItem2vec(spark: SparkSession, samples: RDD[Seq[String]], embLength: Int, embOutputFilename: String, saveToRedis: Boolean, redisKeyPrefix: String) = {
    //设置模型参数
    val word2vec = new Word2Vec()
      .setVectorSize(embLength) //设定生成的embedding向量的维度
      .setWindowSize(5) //设定在序列数据上采样的滑动窗口大小
      .setNumPartitions(10) //设定训练时的迭代次数

    //训练模型
    val model = word2vec.fit(samples)
    //训练结束,用模型查找与item592最相似的20个item
    val synonyms = model.findSynonyms("592", 20)
    for((synoym,consineSimilarity) <- synonyms){
      println(s"$synoym $consineSimilarity")
    }
    //保存模型
    val embFolderPath = this.getClass.getResource("/webroot/modeldata/")
    val file = new File(embFolderPath.getPath + embOutputFilename)
    val bw = new BufferedWriter(new FileWriter(file))
    for(movieId <- model.getVectors.keys){
      bw.write(movieId+":"+model.getVectors(movieId).mkString(" ")+"\n")
    }
    bw.close()

    if(saveToRedis){
      //创建redis client
      val redisClient = new Jedis(redisEndpoint, redisPort)
      val params = SetParams.setParams()
      //设置过期时间(ttl)为24小时
      params.ex(60*60&24)
      //遍历存储embedding向量
      for(movieId <- model.getVectors.keys){
        //key的形式为前缀+movieId，例如i2vEmb:361
        //value的形式是由Embedding向量生成的字符串，例如 "0.1693846 0.2964318 -0.13044095 0.37574086 0.55175656 0.03217995 1.327348 -0.81346786 0.45146862 0.49406642"
        redisClient.set(redisKeyPrefix+":"+movieId,model.getVectors(movieId).mkString(" "),params)
      }
      redisClient.close()
    }

  }

  def generateTransitionMatrix(samples: RDD[Seq[String]]) = {

    val pairSamples = samples.flatMap[(String, String)](sample => {
      var pairSeq = Seq[(String, String)]()
      var previousItem: String = null
      sample.foreach((element: String) => {
        if (previousItem != null) {
          pairSeq = pairSeq :+ (previousItem, element)
        }
        previousItem = element
      })
      pairSeq
    })

    //统计影片对的数量
    val pairCountMap = pairSamples.countByValue()
    pairCountMap.take(10).foreach(println)
    var pairTotalCount = 0L
    //使用双层map建图
    val transitionCountMatrix = mutable.Map[String, mutable.Map[String, Long]]()
    val itemCountMap = mutable.Map[String,Long]()
    pairCountMap.foreach(pair=>{
      val pairItems = pair._1
      val count = pair._2
      if(!transitionCountMatrix.contains(pairItems._1)){
        transitionCountMatrix(pairItems._1) = mutable.Map[String,Long]()
      }
      transitionCountMatrix(pairItems._1)(pairItems._2) = count
      itemCountMap(pairItems._1) = itemCountMap.getOrElse[Long](pairItems._1,0) + count
      pairTotalCount = pairTotalCount + count
    })
    transitionCountMatrix.take(10).foreach(println)
    itemCountMap.take(10).foreach(println)
    println(s"pairTotalCount:$pairTotalCount")
    //将权重图转化为概率跳转图
    val transitionMatrix = mutable.Map[String, mutable.Map[String, Double]]()
    val itemDistribution = mutable.Map[String, Double]()
    transitionCountMatrix.foreach{
      case (itemAId,transitionMap) =>
        transitionMatrix(itemAId) = mutable.Map[String,Double]()
        transitionMap.foreach{case (itemBId,transitionCount)=>  transitionMatrix(itemAId)(itemBId) = transitionCount.toDouble/itemCountMap(itemAId)}
    }
    transitionMatrix.take(10).foreach(println)
    //某节点的总权重占所有节点权重的比例
    itemCountMap.foreach{case (itemId,itemCount) => itemDistribution(itemId) = itemCount.toDouble/pairTotalCount}
    itemDistribution.take(10).foreach(println)
    (transitionMatrix,itemDistribution)
  }

  def oneRandomWalk(transitionMatrix: mutable.Map[String, mutable.Map[String, Double]], itemDistribution: mutable.Map[String, Double], sampleLength: Int): Seq[String] = {
    val sample = mutable.ListBuffer[String]()
    //采取首个元素
    val randomDouble = Random.nextDouble()
    var firstItem = ""
    var accumulateProb:Double = 0D
    breakable{
      //根据物品出现的概率,随机决定起始点
      for((item,prob) <- itemDistribution){
        accumulateProb += prob
        if(accumulateProb >= randomDouble){
          firstItem = item
          break()
        }
      }
    }
    sample.append(firstItem)
    var curElement = firstItem
    breakable{
      //通过随机游走产生长度为samplelength的样本
      for(_ <- 1 until sampleLength){
        if(!itemDistribution.contains(curElement) || !transitionMatrix.contains(curElement)){
          break()
        }
        //从curElement到下一跳的转移概率向量
        val probDistribution = transitionMatrix(curElement)
        val randomDouble = Random.nextDouble()
        breakable{
          //根据转移概率向量,随机决定下一跳的物品
          for((item,prob) <- probDistribution){
            if(randomDouble >= prob){
              curElement = item
              break()
            }
          }}
        sample.append(curElement)
      }
    }
    Seq(sample.toList: _*)
  }

  /**
   * 采用随机游走方式进行采样
   *
   * @param transitionMatrix 转移概率矩阵
   * @param itemDistribution item出现次数的分布
   * @param sampleCount 要采取的样本量
   * @param sampleLength 单个样本的序列长度
   * @return
   */
  def randomWalk(transitionMatrix: mutable.Map[String, mutable.Map[String, Double]], itemDistribution: mutable.Map[String, Double], sampleCount: Int, sampleLength: Int) = {
    val samples = mutable.ListBuffer[Seq[String]]()
    for(_ <- 1 to sampleCount){
      val sampleSeq = oneRandomWalk(transitionMatrix,itemDistribution,sampleLength)
//      println(sampleSeq)
      samples.append(sampleSeq)
    }
    //_*表示将:前集合转化为参数序列
    Seq(samples.toList:_*)
  }

  def grapEmb(spark: SparkSession, samples: RDD[Seq[String]], embLength: Int, embOutputFilename: String, saveToRedis: Boolean, redisKeyPrefix: String)  = {
    //生成转移概率矩阵
    val transitionMatrixAndItemDis = generateTransitionMatrix(samples)
    println(transitionMatrixAndItemDis._1.size)
    println(transitionMatrixAndItemDis._2.size)

    val sampleCount = 20000
    val sampleLength = 10
    val newSamples = randomWalk(transitionMatrixAndItemDis._1, transitionMatrixAndItemDis._2, sampleCount, sampleLength)
    newSamples.take(10).foreach(println)
    val rddSamples = spark.sparkContext.parallelize(newSamples)
    trainItem2vec(spark,rddSamples,embLength,embOutputFilename,saveToRedis,redisKeyPrefix)
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("ctrModel")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder.config(conf).getOrCreate()
    val rawSampleDataPath = "/webroot/sampledata/ratings.csv"
    val embLength = 10
    println(Random.nextDouble())
    val samples = processItemSequence(spark, rawSampleDataPath)
//    trainItem2vec(spark,samples,embLength,"item2vecEmb.csv",saveToRedis=false,"i2vEmb")
    grapEmb(spark,samples,embLength,"item2vecEmb.csv",saveToRedis=false,"i2vEmb")
  }
}


