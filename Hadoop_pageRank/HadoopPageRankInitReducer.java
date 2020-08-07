import java.io.IOException;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class HadoopPageRankInitReducer extends Reducer<Text, Text, Text, Text> {
	
	/* Output format is the following (separator is TAB):
     * 
     *     <page>    <page-rank>    <link1>,<link2>,<link3>,<link4>,...,<linkN>
     *     
     * <page-rank> is an initial value, simple = 1/N
     * where N is the total number of nodes in the Web   
     */
	
	// to make it simple, hard-code TOTAL_WEB_PAGES for now. Note we have to use 
	// pageRankTinyWeb.txt as the input document because of this hard-coded value
	public static final long TOTAL_WEB_PAGES = 4;
	
	@Override
	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
	        	        
	    boolean first = true;
	    String links = (1.0 / TOTAL_WEB_PAGES) + "\t";

	    for (Text value : values) {
	    	if (!first) links += ",";
	    	links += value.toString();
	    	first = false;
	    }
	    
	    context.write(key, new Text(links));
	}
	
	
}