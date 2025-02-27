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

package com.johnsnowlabs.nlp.annotators.btm

import com.johnsnowlabs.storage.{RocksDBConnection, StorageReader}

class TMEdgesReader(
                     override protected val connection: RocksDBConnection,
                     override protected val caseSensitiveIndex: Boolean
                   ) extends StorageReader[Int] {

  override def emptyValue: Int = -1

  override def fromBytes(source: Array[Byte]): Int = {
    BigInt(source).toInt
  }

  def lookup(index: (Int, Int)): Option[Int] = {
    super.lookup(index.toString())
  }

  override protected def readCacheSize: Int = 50000

}
