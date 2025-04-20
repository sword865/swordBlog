+++
date = '2018-11-08T23:51:35+08:00'
title = 'NPE问题与一些语言中的解决方案'
author = "sword865"
type = "post"
tags = ["java", "haskell"]
topics = ["编程语言"]
description = "研究一下Spark2.x中成为主流的DataSet,DataFrame与原来的RDD之间的差别"


+++

# NPE问题与一些语言中的解决方案

NPE(NullPointerException)是一个很烦人的问题，这里简单列举了一些语言中对NPE的处理。

## 1. 通过语法标记进行检查

### Kotlin

Kotlin要求可以为null的变量必需在定义时声明，同时在读取该类型变量属性时必须进行空值判断。例：String 和 String?

```kotlin
var a: String = "abc"
a = null // compilation error, a can not be null

var b: String? = "abc"
b = null // ok

val l = b.length // compiler error: variable 'b' can be null
val l = if (b != null) b.length else -1 // ok
```

### Jetbrains annotations for Java

IntelliJ IDEA提供了一些工具，比如可以对@NotNull的参数进行检查，当出现null赋值时在IDE中会给出提示。

```java
import org.jetbrains.annotations.NotNull;
import java.util.ArrayList;

public class Test{
    public void foo(@NotNull Object param){
        int i = param.hashCode();
    }
    
    public void test(){
        foo(null); // warn in IntelliJ IDEA
    }
}
```

（类似的，FindBugs也提供了@Nonnull注释，用于检查）

### Lombok for Java

Lombok通过在编译时改写字节码对原始代码进行优化，其中的@NonNull，会自动插入运行时检查代码，发现错误抛出异常。

```java
public NonNullExample(@NonNull Person person) {
  super("Hello");
  this.name = person.getName();
}
```

等价于

```java
public NonNullExample(Person person) {
  if (person == null) {
    throw new NullPointerException("person is marked @NonNull but is null");
  }
  super("Hello");
  this.name = person.getName();
}
```

## 2. 用更好的错误处理代替null

空值通常都是由错误导致的无法赋值，因此更好的错误处理也是NPE的一种应对。

### Rust：基于Result错误处理

Rust通过Result类型提供了强大的错误处理机制。

### 基于Monad处理错误

Scala等FP语言基于Monad(Option, Either, Try...)提供了错误处理，其中Optional是最基础的一种。在Option中，定义了专门的None来表示计算失败。这样，在得不到结果时，就会得到None，因此在后续的使用中可以使用isDefined先判断是否有值，再进行处理：

```scala
val name: Option[String] = request getParameter "name"
if(name.isDefined){
    //do some stuff with name.get
}
```

但是这么写很不方便，还是更推荐使用flatMap乃至for推导式来进行计算。(for-yield推导式其实就是flatmap和map的语法糖)

```scala
val upper = for {
  name <- request getParameter "name"
  trimmed <- Some(name.trim)
  upper <- Some(trimmed.toUpperCase) if trimmed.length != 0
} yield upper
println(upper getOrElse "")
```

由于Option只能用None表示失败，不能记录错误信息，所以scala中还提供了Either用来携带更多的信息。

### Optional in Java

Java中的Optional跟scala里的Option是很相似的，同样提供了flatMap操作。但是因为没有for推导式，用起来就感觉不太方便。另外，Java中也缺少可以携带错误信息的Either。
