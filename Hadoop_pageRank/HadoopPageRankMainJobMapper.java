import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class HadoopPageRankMainJobMapper extends Mapper<LongWritable, Text, Text, Text> {
    
    /* PageRank calculation algorithm (mapper)
     * Input file format (separator is TAB):
     * 
     *     <page>    <page-rank>    <link1>,<link2>,<link3>,<link4>,... ,<linkN>
     * 
     * Output has 2 kind of records:
     * One record composed by the collection of links of each page:
     *     
     *     <page>   |<link1>,<link2>,<link3>,<link4>, ... , <linkN>
     *     
     * Another record composed by the linked page, and the pagerank contribution this 
     * page gets from the source page (for example, A go to be B, C, D. A has a pageRank 
     * of 0.3, and B get A's contribuition of 0.3/3 = 0.1:
     *     B	     0.1
     *     <link>    <page-rank> 
     */
    @Override
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
             
    	if ( value == null || value.getLength() == 0 ) {
    		return;
    	}
    	
        int tabIdx1 = value.find("\t");
        int tabIdx2 = value.find("\t", tabIdx1 + 1);
        
        // extract tokens from the current line
        String page = Text.decode(value.getBytes(), 0, tabIdx1);
        String pageRank = Text.decode(value.getBytes(), tabIdx1 + 1, tabIdx2 - (tabIdx1 + 1));
        String outlinks = Text.decode(value.getBytes(), tabIdx2 + 1, value.getLength() - (tabIdx2 + 1));
        
        // calculate contribution to each target page
        
        String[] allNextPages = outlinks.split(",");
        if ( allNextPages == null || allNextPages.length == 0 ) {
        	return;
        }
        
        double currentPR = Double.parseDouble(pageRank.toString());
        int totalNumOfNextPages = allNextPages.length;
        for (String nextPage : allNextPages) { 
        	Text rankContribution = new Text(currentPR/totalNumOfNextPages + "");
            context.write(new Text(nextPage), rankContribution); 
        }
        
        // put the original links so the reducer is able to produce the correct output
        context.write(new Text(page), new Text("|" + outlinks));
        
    }
    
}