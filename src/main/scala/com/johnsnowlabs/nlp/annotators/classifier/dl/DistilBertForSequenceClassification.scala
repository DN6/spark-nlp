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

import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

import java.io.File

/**
 * DistilBertForSequenceClassification can load DistilBERT Models with sequence classification/regression head on top
 * (a linear layer on top of the pooled output) e.g. for multi-class document classification tasks.
 *
 * Pretrained models can be loaded with `pretrained` of the companion object:
 * {{{
 * val sequenceClassifier = DistilBertForSequenceClassification.pretrained()
 *   .setInputCols("token", "document")
 *   .setOutputCol("label")
 * }}}
 * The default model is `"distilbert_base_sequence_classifier_imdb"`, if no name is provided.
 *
 * For available pretrained models please see the [[https://nlp.johnsnowlabs.com/models?task=Text+Classification Models Hub]].
 *
 * Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. The Spark NLP Workshop
 * example shows how to import them [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]].
 * and the [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/DistilBertForSequenceClassificationTestSpec.scala DistilBertForSequenceClassificationTestSpec]].
 *
 * ==Example==
 * {{{
 * import spark.implicits._
 * import com.johnsnowlabs.nlp.base._
 * import com.johnsnowlabs.nlp.annotator._
 * import org.apache.spark.ml.Pipeline
 *
 * val documentAssembler = new DocumentAssembler()
 *   .setInputCol("text")
 *   .setOutputCol("document")
 *
 * val tokenizer = new Tokenizer()
 *   .setInputCols("document")
 *   .setOutputCol("token")
 *
 * val sequenceClassifier = DistilBertForSequenceClassification.pretrained()
 *   .setInputCols("token", "document")
 *   .setOutputCol("label")
 *   .setCaseSensitive(true)
 *
 * val pipeline = new Pipeline().setStages(Array(
 *   documentAssembler,
 *   tokenizer,
 *   sequenceClassifier
 * ))
 *
 * val data = Seq("John Lenon was born in London and lived in Paris. My name is Sarah and I live in London").toDF("text")
 * val result = pipeline.fit(data).transform(data)
 *
 * result.select("label.result").show(false)
 * +--------------------+
 * |result              |
 * +--------------------+
 * |[neg, neg]          |
 * |[pos, pos, pos, pos]|
 * +--------------------+
 * }}}
 *
 * @see [[DistilBertForSequenceClassification]] for sequence-level classification
 * @see [[https://nlp.johnsnowlabs.com/docs/en/annotators Annotators Main Page]] for a list of transformer based classifiers
 * @param uid required uid for storing annotator to disk
 * @groupname anno Annotator types
 * @groupdesc anno Required input and expected output annotator types
 * @groupname Ungrouped Members
 * @groupname param Parameters
 * @groupname setParam Parameter setters
 * @groupname getParam Parameter getters
 * @groupname Ungrouped Members
 * @groupprio param  1
 * @groupprio anno  2
 * @groupprio Ungrouped 3
 * @groupprio setParam  4
 * @groupprio getParam  5
 * @groupdesc param A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
 * */
class DistilBertForSequenceClassification(override val uid: String)
  extends AnnotatorModel[DistilBertForSequenceClassification]
    with HasBatchedAnnotate[DistilBertForSequenceClassification]
    with WriteTensorflowModel
    with HasCaseSensitiveProperties {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  def this() = this(Identifiable.randomUID("DistilBertForSequenceClassification"))

  /**
   * Input Annotator Types: DOCUMENT, TOKEN
   *
   * @group anno
   */
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.DOCUMENT, AnnotatorType.TOKEN)

  /**
   * Output Annotator Types: CATEGORY
   *
   * @group anno
   */
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.CATEGORY

  /** @group setParam */
  def sentenceStartTokenId: Int = {
    $$(vocabulary)("[CLS]")
  }

  /** @group setParam */
  def sentenceEndTokenId: Int = {
    $$(vocabulary)("[SEP]")
  }

  /**
   * Vocabulary used to encode the words to ids with WordPieceEncoder
   *
   * @group param
   * */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary")

  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)


  /**
   * Labels used to decode predicted IDs back to string tags
   *
   * @group param
   * */
  val labels: MapFeature[String, Int] = new MapFeature(this, "labels")

  /** @group setParam */
  def setLabels(value: Map[String, Int]): this.type = set(labels, value)

  /**
   * Returns labels used to train this model
   */
  def getClasses: Array[String] = {
    $$(labels).keys.toArray
  }

  /** Instead of 1 class per sentence (if inputCols is '''sentence''') output 1 class per document by averaging probabilities in all sentences.
   * Due to max sequence length limit in almost all transformer models such as BERT (512 tokens), this parameter helps feeding all the sentences
   * into the model and averaging all the probabilities for the entire document instead of probabilities per sentence. (Default: true)
   *
   * @group param
   * */
  val coalesceSentences = new BooleanParam(this, "coalesceSentences", "If sets to true the output of all sentences will be averaged to one output instead of one output per sentence. Default to true.")

  /** @group setParam */
  def setCoalesceSentences(value: Boolean): this.type = set(coalesceSentences, value)

  /** @group getParam */
  def getCoalesceSentences: Boolean = $(coalesceSentences)

  /** ConfigProto from tensorflow, serialized into byte array. Get with `config_proto.SerializeToString()`
   *
   * @group param
   * */
  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): DistilBertForSequenceClassification.this.type = set(this.configProtoBytes, bytes)

  /** @group getParam */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  /** Max sentence length to process (Default: `128`)
   *
   * @group param
   * */
  val maxSentenceLength = new IntParam(this, "maxSentenceLength", "Max sentence length to process")

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    require(value <= 512, "DistilBERT models do not support sequences longer than 512 because of trainable positional embeddings.")
    require(value >= 1, "The maxSentenceLength must be at least 1")
    set(maxSentenceLength, value)
    this
  }

  /** @group getParam */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  /**
   * It contains TF model signatures for the laded saved model
   *
   * @group param
   * */
  val signatures = new MapFeature[String, String](model = this, name = "signatures")

  /** @group setParam */
  def setSignatures(value: Map[String, String]): this.type = {
    if (get(signatures).isEmpty)
      set(signatures, value)
    this
  }

  /** @group getParam */
  def getSignatures: Option[Map[String, String]] = get(this.signatures)

  private var _model: Option[Broadcast[TensorflowDistilBertClassification]] = None

  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, tensorflowWrapper: TensorflowWrapper): DistilBertForSequenceClassification = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowDistilBertClassification(
            tensorflowWrapper,
            sentenceStartTokenId,
            sentenceEndTokenId,
            configProtoBytes = getConfigProtoBytes,
            tags = $$(labels),
            signatures = getSignatures,
            $$(vocabulary)
          )
        )
      )
    }

    this
  }

  /** @group getParam */
  def getModelIfNotSet: TensorflowDistilBertClassification = _model.get.value


  /** Whether to lowercase tokens or not
   *
   * @group setParam
   * */
  override def setCaseSensitive(value: Boolean): this.type = {
    if (get(caseSensitive).isEmpty)
      set(this.caseSensitive, value)
    this
  }

  setDefault(
    batchSize -> 8,
    maxSentenceLength -> 128,
    caseSensitive -> true,
    coalesceSentences -> false
  )

  /**
   * takes a document and annotations and produces new annotations of this annotator's annotation type
   *
   * @param batchedAnnotations Annotations that correspond to inputAnnotationCols generated by previous annotators if any
   * @return any number of annotations processed for every input annotation. Not necessary one to one relationship
   */
  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {
    batchedAnnotations.map(annotations => {
      val sentences = SentenceSplit.unpack(annotations).toArray
      val tokenizedSentences = TokenizedWithSentence.unpack(annotations).toArray

      if (tokenizedSentences.nonEmpty) {
        getModelIfNotSet.predictSequence(
          tokenizedSentences,
          sentences,
          $(batchSize),
          $(maxSentenceLength),
          $(caseSensitive),
          $(coalesceSentences),
          $$(labels)
        )
      }
      else {
        Seq.empty[Annotation]
      }
    }

    )
  }


  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModelV2(path, spark, getModelIfNotSet.tensorflowWrapper, "_distilbert_classification", DistilBertForSequenceClassification.tfFile, configProtoBytes = getConfigProtoBytes)
  }

}

trait ReadablePretrainedDistilBertForSequenceModel extends ParamsAndFeaturesReadable[DistilBertForSequenceClassification] with HasPretrained[DistilBertForSequenceClassification] {
  override val defaultModelName: Some[String] = Some("distilbert_base_sequence_classifier_imdb")

  /** Java compliant-overrides */
  override def pretrained(): DistilBertForSequenceClassification = super.pretrained()

  override def pretrained(name: String): DistilBertForSequenceClassification = super.pretrained(name)

  override def pretrained(name: String, lang: String): DistilBertForSequenceClassification = super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): DistilBertForSequenceClassification = super.pretrained(name, lang, remoteLoc)
}

trait ReadDistilBertForSequenceTensorflowModel extends ReadTensorflowModel {
  this: ParamsAndFeaturesReadable[DistilBertForSequenceClassification] =>

  override val tfFile: String = "distilbert_classification_tensorflow"

  def readTensorflow(instance: DistilBertForSequenceClassification, path: String, spark: SparkSession): Unit = {

    val tf = readTensorflowModel(path, spark, "_distilbert_classification_tf", initAllTables = false)
    instance.setModelIfNotSet(spark, tf)
  }

  addReader(readTensorflow)

  def loadSavedModel(tfModelPath: String, spark: SparkSession): DistilBertForSequenceClassification = {

    val f = new File(tfModelPath)
    val savedModel = new File(tfModelPath, "saved_model.pb")

    require(f.exists, s"Folder $tfModelPath not found")
    require(f.isDirectory, s"File $tfModelPath is not folder")
    require(
      savedModel.exists(),
      s"savedModel file saved_model.pb not found in folder $tfModelPath"
    )

    val vocabPath = new File(tfModelPath + "/assets", "vocab.txt")
    require(vocabPath.exists(), s"Vocabulary file vocab.txt not found in folder $tfModelPath/assets/")

    val vocabResource = new ExternalResource(vocabPath.getAbsolutePath, ReadAs.TEXT, Map("format" -> "text"))
    val words = ResourceHelper.parseLines(vocabResource).zipWithIndex.toMap

    val labelsPath = new File(tfModelPath + "/assets", "labels.txt")
    require(labelsPath.exists(), s"Labels file labels.txt not found in folder $tfModelPath/assets/")

    val labelsResource = new ExternalResource(labelsPath.getAbsolutePath, ReadAs.TEXT, Map("format" -> "text"))
    val labels = ResourceHelper.parseLines(labelsResource).zipWithIndex.toMap

    val (wrapper, signatures) = TensorflowWrapper.read(tfModelPath, zipped = false, useBundle = true)

    val _signatures = signatures match {
      case Some(s) => s
      case None => throw new Exception("Cannot load signature definitions from model!")
    }

    /** the order of setSignatures is important if we use getSignatures inside setModelIfNotSet */
    new DistilBertForSequenceClassification()
      .setVocabulary(words)
      .setLabels(labels)
      .setSignatures(_signatures)
      .setModelIfNotSet(spark, wrapper)
  }
}

/**
 * This is the companion object of [[DistilBertForSequenceClassification]]. Please refer to that class for the documentation.
 */
object DistilBertForSequenceClassification extends ReadablePretrainedDistilBertForSequenceModel with ReadDistilBertForSequenceTensorflowModel
