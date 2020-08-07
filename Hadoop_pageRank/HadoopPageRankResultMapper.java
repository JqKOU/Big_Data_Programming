import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class HadoopPageRankResultMapper extends Mapper<LongWritable, Text, DoubleWritable, Text> {

     @Override
	 public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
    	 
    	 String[] parts = value.toString().split("\t");

    	 Double PageRank = Double.parseDouble(parts[1]);
    	 String node = parts[0];

    	 context.write(new DoubleWritable(PageRank), new Text(node));   	 
     }    
}