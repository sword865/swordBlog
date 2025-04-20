+++
topics = ["spark"]
keywords = ["RDD","DataFrame","MLlib"]
author = "sword865"
type = "post"
tags =  ["RDD","DataFrame","MLlib"]
date = "2017-03-12T15:49:45+08:00"
title = "比较一下spark2的DataFrame和RDD"
description = "研究一下Spark2.x中成为主流的DataSet,DataFrame与原来的RDD之间的差别"

+++

前段时间把spark集群升级到2.x，使用起来感觉相对1.x的版本最大的改动就是DataFrame正式开始替代RDD成为主流，包括我们最常用到的mllib的官方文档也提到：

    In the Spark 2.x releases, MLlib will add features to the DataFrames-based API to reach feature parity with the RDD-based API.
    After reaching feature parity (roughly estimated for Spark 2.2), the RDD-based API will be deprecated.
    The RDD-based API is expected to be removed in Spark 3.0.


#### RDD的结构

RDD可以看成是一个分布式的无序列表，这个列表内的元素是一个object，RDD并不关心每个object的内部结构。因此所有操作都必须对这个object进行，不利于算子的复用。

比起DataFrame，RDD更方便我们对数据做一些底层的操作，也可以用于unstructured的数据。

#### DataFrame的结构

DataFrame不同于RDD，框架会去了解object中的数据是什么样的结构，这样每个算子就可以单独实现在某个列上，复用起来就更加简单。

因为DataFrame比RDD多个更多的限制，对内部的元素也有了更多的了解，可以使用SQL语句进行操作，因此也就可以在对DataFrame进行操作时使用Spark SQL的Catalyst优化器进行优化。

Catalyst一个易于扩展的查询优化器，同时支持基于规则(rule-based)和基于代价(cost-based)的优化方法，我们可以基于相关API自己定义优化规则。

最后，Spark的Tungsten目前还只支持DataFrame API, 因此在使用RDD时不能享受到Tungsten带来的效率优化。（Tungsten做的优化概括起来说就是由Spark自己来管理内存而不是使用JVM，这样可以避免JVM GC带来的性能损失）

#### DataSet数据结构

前面提到DataFrame每一个record对应了一个Row。而Dataset的定义更加宽松，每一个record对应了一个任意的类型。实际上，从源码中可以看到，DataFrame就是Dataset的一种特例。

    package object sql {
        ...
        type DataFrame = Dataset[Row]
    }

DataSet和DataFrame可以通过df.as和ds.toDF方法方便的进行转化。

不同于Row是一个泛化的无类型JVM object, Dataset是由一系列的强类型JVM object组成的，因此DataSet可以在编译时进行类型检查。

比起RDD，DataSet的API也以Spark SQL引擎为基础，因此在对DataSet进行操作时，同样可以从Catalyst优化器中受益。

基本上，我觉得DataSet集合了RDD和DataSet两者的优点。

#### 关于效率

最后，在效率上，在使用RDD的API时候，使用Python明显比Scala要慢上很多（据我们测试是慢了2倍以上）。但是在使用DataFame时，这个缺陷就不复存在了，换句话说，喜欢Python或者放不下各种Python扩展的同志们可以把Python写起来了，哈哈。这里放个国外网友测试的效率比较吧：

<img src="../images/Spark_Dataframe_Official_Benchmark.png" />

可以看到，速度上大致是 Scala DF = Python DF > Scala RDD > Python RDD，并且DF优势很显著。

#### 其他参考资料

[探索Spark Tungsten的秘密](https://github.com/hustnn/TungstenSecret)

[Spark 2.0介绍：在Spark SQL中定义查询优化规则](https://www.iteblog.com/archives/1706.html)

http://www.infoq.com/cn/articles/2015-Review-Spark

http://stackoverflow.com/questions/37301226/difference-between-dataset-api-and-dataframe

https://databricks.com/blog/2016/07/14/a-tale-of-three-apache-spark-apis-rdds-dataframes-and-datasets.html

https://0x0fff.com/spark-dataframes-are-faster-arent-they/


