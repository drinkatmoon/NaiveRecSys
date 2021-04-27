package com.naiverec.offline.spark.embedding

import java.util.logging.{Level, Logger}

import org.apache.spark.SparkConf
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._

object Embedding {

  def processItemSequence(spark: SparkSession, rawSampleDataPath: String): Unit = {

    val ratingsResourcesPath = this.getClass.getResource(rawSampleDataPath)
    val ratingSamples = spark.read.format("csv").option("header", "true").load(ratingsResourcesPath.getPath)
    //定义一个udf函数，按照用户的tiemstamp排序
    val sortUdf:UserDefinedFunction = udf {
      (rows:Seq[Row]) =>{

      }
    }


  }


  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARNING)

    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("ctrModel")
      .set("spark.submit.deployMode", "client")

    val spark = SparkSession.builder.config(conf).getOrCreate()
    val rawSampleDataPath = "/webroot/sampledata/ratings.csv"
    val embLength = 10

    processItemSequence(spark,rawSampleDataPath)

  }
}


