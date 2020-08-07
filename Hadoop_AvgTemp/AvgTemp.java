import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.log4j.Logger;

public class AvgTemp {
	final static Logger log = Logger.getLogger(AvgTemp.class);
	
	public static class AvgTempMapper extends 
	   Mapper<LongWritable, Text, Text, IntWritable> {
		
		
		private static final int MISSING = 9999;
		
		@Override
		public void map(LongWritable key, Text value, Context context)
		   throws IOException, InterruptedException {
			
			String line = value.toString();
			String year = line.substring(15, 19);
			int airTemperature;
			if (line.charAt(87) == '+') { 
				airTemperature = Integer.parseInt(line.substring(88, 92));
			} else {
				airTemperature = Integer.parseInt(line.substring(87, 92));
			}
			
			String quality = line.substring(92, 93);
			if (airTemperature != MISSING && quality.matches("[01459]")) {
				context.write(new Text(year), new IntWritable(airTemperature));
			}
		}
	}
	
	
	public static class AvgTempReducer extends 
	   Reducer<Text, IntWritable, Text, IntWritable> {
		
		@Override
		public void reduce(Text key, Iterable<IntWritable> values, Context context)
		   throws IOException, InterruptedException {
			
			int sumTemp = 0;
			int count = 0;
			for (IntWritable value : values) {
				sumTemp += value.get();
				count += 1;
			}
			context.write(key, new IntWritable(sumTemp/count));
		}
	}
		
	public static void main(String[] args) throws Exception {
		
		Configuration conf = new Configuration();
		Job job = Job.getInstance(conf, "Average Temperature");
		
		job.setJobName("Home_Work_2_1");
		log.debug("Job Name" + job.getJobName());
		
		job.setJarByClass(AvgTemp.class);
		job.setMapperClass(AvgTempMapper.class);
		job.setReducerClass(AvgTempReducer.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);
			
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		
		log.debug("Name"+job.getPartitionerClass().getName());
		/**/
		
		System.exit(job.waitForCompletion(true) ? 0 : 1);
		log.debug("Number" + job.getCounters());
		
  }
}

