import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class HadoopPageRankInitMapper extends Mapper<LongWritable, Text, Text, Text> {
	
	/* Job#1 mapper will simply parse a line of the input graph creating a map with key-value(s) pairs.
     * Input format is the following (separator is TAB):
     * 
     *     <nodeA>    <nodeB>
     * 
     * which denotes an edge going from <nodeA> to <nodeB>.
     * skip comment lines (denoted by the # characters at the beginning of the line)
    */
     @Override
	 public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
    	 
    	 
    	 if ( value == null || value.charAt(0) == '#' ) {
    		 return;
    	 }
    	 
    	 int tabIndex = value.find("\t");
    	 String nodeA = Text.decode(value.getBytes(), 0, tabIndex);
    	 String nodeB = Text.decode(value.getBytes(), tabIndex + 1, value.getLength() - (tabIndex + 1));
    	 context.write(new Text(nodeA), new Text(nodeB));
    	 
     }
     
}