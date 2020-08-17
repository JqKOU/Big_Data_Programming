# Big data programming: from single node to cluster computing 
### General method to run algorithms: Single node computing  
Single node architecture consists of CPU, memory, and disk. Algorithms run in CPU, CPU has access to memory, but it does not have direct access to the disk. So, data in the disk will be read into the memory and the data will fully fit in memory, which means memory should be big enough to hold data from disk. Once the data is in the memory, algorithms in CPU can access data and run successfully.

### When data too big to fit in memory: Memory swap
If the data is too big to fit into the memory at once, we could use the memory swap method to read data. To be more specific, we bring only a portion of data into memory at a time, process it in batches, and write back results to disk. For example, when designing a database system, database files are normally too big to fit into memory, but we could still manage to search, update and delete data by designing indexing schema to minimize the number of memory swaps.

### When memory swap is not sufficient: Big data
Sometimes even a smart indexed memory swap is not sufficient. For example, we crawled 10 billion webpages (not big number, around 20% of total webpages), assume the average size of a webpage are 20 KB (not big webpage), so the size of all the webpages would be 10 billion x 20 KB = 200 TB, and all the data are stored on a single disk. If we want to do memory swap, which means we want to read data from disk into memory, and the read speed (disk read bandwidth) is 50 MB/sec (a very good speed). So the time to read 200 TB data are 4 million seconds which equals more than 46 days. Even reading data take so long, let alone to do something useful with the data. So we need a better solution.

### How to handle big data: Cluster computing 
If the data set is too big and normal smart indexing memory swap doesn't work, we could split the large dataset into smaller chunks, then have a lot of smaller (cheap) computers, and assign each computer a chunk of the data, let these computers work at the same time try to solve the problem. All these computers are connected by Ethernet cables and inexpensive switches. This is the idea of cluster computing.

### Challenges of cluster computing 
Cluster computing has its own challenge. One major challenge is that nodes (cheap computer) can fail, once node in which data stored failed, we could not read data back; moreover, if nodes fail during a long-running computation, we may have to start the computation all over again; another challenge is the network could be a bottleneck because a complex computation might need to move a lot of data, which can slow the computation down; finally, distributed programming (break data into chunks, assign chunks to different machines, write code to coordinate machine) is hard even for a sophisticated programmer.

### Ways to make cluster computing easy: Hadoop and Spark 
Hadoop and Spark computation frameworks are solutions to the challenges of cluster computing and make cluster computing easy. As a programmer, we don't need to worry about which node fails, what if it fails in the middle of computing and don't need to worry about the complexity of distributed programming by using Hadoop and Spark.

### About this repository 
This repository is a collection of big data programming examples using either Hadoop or Spark which include: PageRank, similar items (minHash/LSH algorithms), recommendation system (content-based and collaborative filtering) and machine learning algorithms (Naive Bayes Classification).


