/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions.{col, explode, size}
import org.scalatest.flatspec.AnyFlatSpec

class AlbertForSequenceClassificationTestSpec extends AnyFlatSpec {

  import ResourceHelper.spark.implicits._

  "AlbertForSequenceClassification" should "correctly load custom model with extracted signatures" taggedAs SlowTest in {

    val ddd = Seq(
      "John Lenon was born in London and lived in Paris. My name is Sarah and I live in London.",
      "Rare Hendrix song draft sells for almost $17,000.",
      "EU rejects German call to boycott British lamb .",
      "TORONTO 1996-08-21",
      " carbon emissions have come down without impinging on our growth. .  . .",
      "\\u2009.carbon emissions have come down without impinging on our growth .\\u2009.\\u2009."
    ).toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val classifier = AlbertForSequenceClassification
      .pretrained()
      .setInputCols(Array("token", "document"))
      .setOutputCol("label")
      .setCaseSensitive(false)
      .setCoalesceSentences(false)

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, classifier))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.select("label").show(20, truncate = false)
    pipelineDF.select("document.result", "label.result").show(20, truncate = false)
    pipelineDF
      .withColumn("doc_size", size(col("document")))
      .withColumn("label_size", size(col("label")))
      .where(col("doc_size") =!= col("label_size"))
      .select("doc_size", "label_size", "document.result", "label.result")
      .show(20, truncate = false)

    val totalDocs = pipelineDF.select(explode($"document.result")).count.toInt
    val totalLabels = pipelineDF.select(explode($"label.result")).count.toInt

    println(s"total tokens: $totalDocs")
    println(s"total embeddings: $totalLabels")

    assert(totalDocs == totalLabels)
  }

  "AlbertForSequenceClassification" should "be saved and loaded correctly" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "John Lenon was born in London and lived in Paris. My name is Sarah and I live in London",
      "Rare Hendrix song draft sells for almost $17,000.",
      "EU rejects German call to boycott British lamb .",
      "TORONTO 1996-08-21"
    ).toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val classifier = AlbertForSequenceClassification.pretrained()
      .setInputCols(Array("token", "document"))
      .setOutputCol("label")
      .setCaseSensitive(true)

    val pipeline = new Pipeline().setStages(Array(document, tokenizer, classifier))

    val pipelineModel = pipeline.fit(ddd)
    val pipelineDF = pipelineModel.transform(ddd)

    pipelineDF.select("label.result").show(false)

    Benchmark.time("Time to save AlbertForSequenceClassification pipeline model") {
      pipelineModel.write.overwrite().save("./tmp_forsequence_pipeline")
    }

    Benchmark.time("Time to save AlbertForSequenceClassification model") {
      pipelineModel.stages.last.asInstanceOf[AlbertForSequenceClassification].write.overwrite().save("./tmp_forsequence_model")
    }

    val loadedPipelineModel = PipelineModel.load("./tmp_forsequence_pipeline")
    loadedPipelineModel.transform(ddd).select("label.result").show(false)

    val loadedSequenceModel = AlbertForSequenceClassification.load("./tmp_forsequence_model")
    println(loadedSequenceModel.getClasses.mkString("Array(", ", ", ")"))

  }

  "AlbertForSequenceClassification" should "benchmark test" taggedAs SlowTest in {

    val conll = CoNLL()
    val training_data = conll.readDataset(ResourceHelper.spark, "src/test/resources/conll2003/eng.train")

    val classifier = AlbertForSequenceClassification.pretrained()
      .setInputCols(Array("token", "document"))
      .setOutputCol("class")
      .setCaseSensitive(true)

    val pipeline = new Pipeline()
      .setStages(Array(
        classifier
      ))

    val pipelineDF = pipeline.fit(training_data).transform(training_data)
    Benchmark.time("Time to save AlbertForSequenceClassification results") {
      pipelineDF.write.mode("overwrite").parquet("./tmp_sequence_classifier")
    }

    pipelineDF.select("label").show(20, truncate = false)
    pipelineDF.select("document.result", "label.result").show(20, truncate = false)
    pipelineDF
      .withColumn("doc_size", size(col("document")))
      .withColumn("label_size", size(col("label")))
      .where(col("doc_size") =!= col("label_size"))
      .select("doc_size", "label_size", "document.result", "label.result")
      .show(20, truncate = false)

    val totalDocs = pipelineDF.select(explode($"document.result")).count.toInt
    val totalLabels = pipelineDF.select(explode($"label.result")).count.toInt

    println(s"total tokens: $totalDocs")
    println(s"total embeddings: $totalLabels")

    assert(totalDocs == totalLabels)
  }
}
