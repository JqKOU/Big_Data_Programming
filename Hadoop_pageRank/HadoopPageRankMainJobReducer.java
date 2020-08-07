import java.io.IOException;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class HadoopPageRankMainJobReducer extends Reducer<Text, Text, Text, Text> {
    
    /* PageRank calculation algorithm (reducer)
     * Input file format has 2 kind of records (separator is TAB):
     * 
     * One record composed by the collection of links of each page:
     * 
     *     <title>   |<link1>,<link2>,<link3>,<link4>, ... , <linkN>
     *     
     * Another record composed by the linked page, the page rank contribution
     * from the source page
     *
     *     <link>    <page-rank>
     */
    @Override
    public void reduce(Text key, Iterable<Text> values, Context context) 
    		throws IOException, InterruptedException {
    	
    	if ( values == null ) {
    		return;
    	}
    	
    	String links = "";
        double receivedContribution = 0.0;
        
        for (Text value : values) {
 
            String content = value.toString();
            if (content.startsWith("|")) {
                links += content.substring("|".length());
            } else {
            	receivedContribution += Double.parseDouble(content);
            }

        }
        
        double newPageRank = receivedContribution;
        context.write(key, new Text(newPageRank + "\t" + links));
        
    }

}