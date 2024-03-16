# spark demo

## install spark

### Update the System

```shell
apt update -y
apt upgrade -y
```

### Install Java JDK

1. install java

    ```shell
    apt-get install default-jdk -y
    ```

2. check java version

    ```shell
    java --version
    ```

3. my java version

    ```
    openjdk 11.0.17 2022-10-18
    OpenJDK Runtime Environment (build 11.0.17+8-post-Ubuntu-1ubuntu220.04)
    OpenJDK 64-Bit Server VM (build 11.0.17+8-post-Ubuntu-1ubuntu220.04, mixed mode, sharing)
    ```

### Install Scala

1. install Scala

    ```shell
    apt-get install scala -y
    ```

2. check java version

    ```shell
    scala -version
    ```

3. my Scala version

    ```
    Scala code runner version 2.11.12 -- Copyright 2002-2017, LAMP/EPFL
    ```

4. test Scala

    ```shell
    scala
    ```

5. test Scala version
    ```shell
    Welcome to Scala 2.11.12 (OpenJDK 64-Bit Server VM, Java 11.0.17).
    Type in expressions for evaluation. Or try :help.
    
    scala>
    ```

### Install Apache Spark

1. Install Apache Spark 2024/03/16 can work
   ```shell
   wget https://archive.apache.org/dist/spark/spark-3.3.1/spark-3.3.1-bin-hadoop3.tgz
   ```
2. tar it
   ```shell
   tar -xvzf spark-3.3.1-bin-hadoop3.tgz
   ```

3. tar it
   ```shell
   mv spark-3.3.1-bin-hadoop3 /mnt/spark
   ```

4. add it to path
   ```shell
   vim ~/.bashrc
   ```

   add this
   ```
   export SPARK_HOME=/mnt/spark
   export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
   ```

   source it
   ```shell
   source ~/.bashrc
   ```

### Start Apache Spark

1. Spark-master
   ```shell
   start-master.sh
   stop-master.sh
   ```
   sample output
   ```
   starting org.apache.spark.deploy.master.Master, logging to /mnt/spark/logs/spark-root-org.apache.spark.deploy.master.Master-1-spark.out
   ```
2. Spark-worker
   ```shell
   start-worker.sh
   start-worker.sh
   ```
   sample output
   ```
   starting org.apache.spark.deploy.worker.Worker, logging to /mnt/spark/logs/spark-root-org.apache.spark.deploy.worker.Worker-1-spark.out
   ```
3. Spark-submit
   ```shell
   spark-submit --master spark://xunhaoz-Destop:7077 ./train/clustering/BisectingKMeans.py
   ```
   sample output
   ```
   ...
   +------------+--------------------+----------+
   |          id|     scaler_features|prediction|
   +------------+--------------------+----------+
   |Efxl-07-1604|[0.90045248868778...|         0|
   |Efxl-07-1605|[0.57918552036199...|         0|
   |Efxl-07-1606|[0.60180995475113...|         1|
   |Efxl-07-1607|[0.56108597285067...|         1|
   |Efxl-07-1608|[0.66515837104072...|         0|
   |Efxl-07-1609|[0.61538461538461...|         1|
   |Efxl-07-1610|[0.66515837104072...|         1|
   |Efxl-07-1611|[0.74660633484162...|         0|
   |Efxl-07-1612|[0.59276018099547...|         1|
   |Efxl-07-1613|[0.41628959276018...|         1|
   |Efxl-07-1614|[0.76470588235294...|         0|
   |Efxl-07-1615|[0.73755656108597...|         1|
   |Efxl-07-1616|[0.62443438914027...|         1|
   |Efxl-07-1617|[0.80542986425339...|         1|
   |Efxl-07-1618|[0.84615384615384...|         1|
   |Efxl-07-1619|[0.82352941176470...|         0|
   |Efxl-07-1620|[0.65158371040723...|         1|
   |Efxl-07-1621|[0.54298642533936...|         1|
   |Efxl-07-1622|[0.52036199095022...|         1|
   |Efxl-07-1623|[0.63348416289592...|         0|
   +------------+--------------------+----------+
   ...
   ```

## spark port

1. job port from 4040 auto increment
2. master port from 8080 auto increment
3. worker port from 8081 auto increment
4. submit port from 7077

## python version
Python 3.9.18

## python requirements