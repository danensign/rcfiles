package com.whatever

import scala.collection.JavaConverters._

class MyProperties(resourceName: String) {
  lazy val props: java.util.Properties = {
    val p = new java.util.Properties()
    val stream = getClass.getResourceAsStream("/" + resourceName)
    p.load(stream)
    p
  }
  def apply(key: String): Option[String] = Option(props.getProperty(key))
  def propertyNames = props.propertyNames.asScala
}

object MyProperties {
  def apply(resourceName: String = "Props") = new MyProperties(resourceName)
}

package com.whatever
import com.google.gson.{Gson, GsonBuilder}
import org.apache.spark.sql.DataFrame

abstract class Feature {

  private def cachePath(): String = {
    val dir = MyProperties()("Feature.cachepath").get
    new java.io.File(dir).mkdir()
    dir + "/" + featureNameWithPackage
  }

  private def featureNameWithPackage: String = getClass.toString.replace("class ", "")

  private def computeAndCache(): DataFrame = {
    vprint("Computing and caching")
    val result = transform(dependencies)
    result.write.mode("overwrite").parquet(cachePath)
    result
  }

  private def readFromCache(): DataFrame = {
    if (new java.io.File(cachePath).exists) {
      vprint("Reading from cache")
      SparkForFeatures().sqlContext.read.format(MyProperties()("Feature.cacheformat").get).load(cachePath)
    } else {
      computeAndCache()
    }
  }

  def vprint(message: String) = {
    if (verbose) {
      println(getClass.toString.replace("class ", "") + ": " + message)
    }
  }

  def json(): String = {
    val jr = new JsonRep(
      featureNameWithPackage,
      dataFrame.dtypes.map(x => Array(x._1, x._2, "This is a field named " + x._1)),
      dependencies.map(_.featureNameWithPackage).toArray,
      java.util.Calendar.getInstance.getTime.toString, // creationDatetime
      System.getProperty("user.name") + "@" + java.net.InetAddress.getLocalHost, // creator
      System.getProperty("user.name"), // owner
      Array[String](), // tags
      "" // description
    )
    new GsonBuilder().setPrettyPrinting.create.toJson(jr)
  }

  private case class JsonRep(featureName: String,
                             schema: Array[Array[String]],
                             dependencies: Array[String],
                             creationDatetime: String,
                             creator: String,
                             owner: String,
                             tags: Array[String],
                             description: String)
  def dependencies(): Array[Feature]
  def transform(features: Array[Feature]): DataFrame
}

object Feature {
  def apply(featureName: String): Feature = {
    Class.forName(featureName).newInstance().asInstanceOf[Feature]
  }
}

object SparkForFeatures {
  lazy val spark = {
    Logger.getRootLogger.setLevel(Level.ERROR)
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    val sp = SparkSession.builder
      .master("local")
      .appName(MyProperties()("spark.featurebrowser.name").get)
      .config("spark.executor.memory", MyProperties()("spark.executor.memory").get)
      .getOrCreate()
    sp.sparkContext.setLogLevel("ERROR")
    sp
  }

  def apply(): SparkSession = spark
}
import java.util.zip.{GZIPInputStream, GZIPOutputStream}
import java.io.{FileOutputStream, PrintWriter}
import org.apache.spark.sql.DataFrame

abstract class DemoCsvFeature extends ExternalFeature {
  // No need for caching for this class
  override def dataFrame(): DataFrame = {
    this.transform(this.dependencies)
  }

  override def transform(dataFrames: Array[Feature]): DataFrame = {
    val tmpFilename =
      MyProperties()("Feature.cachepath").get + "/" + getClass.toString.replace("class ", "") + ".tmp.csv.gz"
    vprint("Writing tmp csv to " + tmpFilename)
    new PrintWriter(new GZIPOutputStream(new FileOutputStream(tmpFilename))) {
      write(scala.io.Source.fromInputStream(
        new GZIPInputStream(getClass.getResourceAsStream("/" + csvLoc))).getLines() mkString "\n")
      close
    }

    customFilter(
      SparkForFeatures().read
        .format("csv")
        .option("header", "true")
        .option("mode", "DROPMALFORMED")
        .load(tmpFilename))
  }

  val csvLoc: String
}

object LibraryTester {
  def main(args: Array[String]) {
    val feature = Feature(args(0)).dataFrame.show
  }
}
import java.io.PrintWriter
import java.io.FileInputStream
import java.util.zip.ZipInputStream

def readClassesFromJar(jarName: String): Array[String] = {
    println("Processing jar " + jarName)
    val zip = new ZipInputStream(new FileInputStream(jarName))
    Stream.continually(zip.getNextEntry)
      .takeWhile(_ != null)
      .filter(!_.isDirectory)
      .filter(_.getName.endsWith("class"))
      .map { entry =>
        val className = entry.toString.replace(".class", "").replaceAll("/", ".")
        try {
          Class.forName(className).newInstance().asInstanceOf[TheType].generateSomeString
        } catch {
          case e: Exception => {
            println("Failed on " + className + " because: " + e)
            None
          }
        }
      }
      .filter(_ != None)
}

def emptydf(spark: SparkSession): DataFrame = {
  import spark.implicits._
  Seq.empty[(String, Int)].toDF("key", "value")
}
