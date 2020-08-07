import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class HadoopPageRank extends Configured implements Tool {
	
	public int run(String[] args) throws Exception {
		
		// house keeping
        int iteration = 0;
        Path inputPath = new Path(args[0]);
        Path basePath = new Path(args[1]);
        FileSystem fs = FileSystem.get(getConf());
        fs.delete(basePath, true); 
        fs.mkdirs(basePath);
        
		// configure initial job
		Job initJob = Job.getInstance(getConf(),"pageRank");
		initJob.setJarByClass(HadoopPageRank.class);
        initJob.setMapperClass(HadoopPageRankInitMapper.class);
        initJob.setReducerClass(HadoopPageRankInitReducer.class);
		initJob.setOutputKeyClass(Text.class);
		initJob.setOutputValueClass(Text.class);
		initJob.setInputFormatClass(TextInputFormat.class);
		
		Path outputPath = new Path(basePath, "iteration_" + iteration);
		FileInputFormat.addInputPath(initJob, inputPath);
		FileOutputFormat.setOutputPath(initJob, outputPath);
		
        // let initJob run and wait for finish
		if ( !initJob.waitForCompletion(true) ) {
			return -1;
		}
		
		// calculate the page ranks 
		int totalIterations = Integer.parseInt(args[2]);
		while ( iteration < totalIterations ) {
			
			iteration ++;
			inputPath = outputPath; // new input is the old output
			outputPath = new Path(basePath, "iteration_" + iteration);
			
			Job mainJob = Job.getInstance(getConf(),"Iteration " + iteration);
			mainJob.setJarByClass(HadoopPageRank.class);
			mainJob.setMapperClass(HadoopPageRankMainJobMapper.class);
			mainJob.setReducerClass(HadoopPageRankMainJobReducer.class);
			mainJob.setOutputKeyClass(Text.class);
			mainJob.setOutputValueClass(Text.class);
			mainJob.setInputFormatClass(TextInputFormat.class);
			FileInputFormat.setInputPaths(mainJob, inputPath);
			FileOutputFormat.setOutputPath(mainJob, outputPath);
			
			if ( !mainJob.waitForCompletion(true) ) {
				return -1;
			}
            
		}
		
		// collect the result, highest rank first - you will need to finish this up
		Job resultJob = Job.getInstance(getConf(),"final result");
		
		//here is the place you will need to add a final Map/Reduce code
		resultJob.setJarByClass(HadoopPageRank.class);
		resultJob.setMapperClass(HadoopPageRankResultMapper.class);
		resultJob.setOutputKeyClass(DoubleWritable.class);
		resultJob.setOutputValueClass(Text.class);	
		//end
		
		FileInputFormat.setInputPaths(resultJob, outputPath);		
		FileOutputFormat.setOutputPath(resultJob,new Path(basePath, "result"));
		
		if ( !resultJob.waitForCompletion(true) ) {
			return -1;
		}
		
		return 0;
	}
	
	public static void main(String[] args) throws Exception {
		
		int exitCode = ToolRunner.run(new HadoopPageRank(), args);
		System.exit(exitCode);
	}
	
}

