
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.StringTokenizer;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;

public class SparkNaiveBayes {

	private static final String LABEL_SEPARATOR = "|";
	private static final String TESTING_URI = "file:///Users/**omitted for privacy purpose**/NB_test_doc*.txt";
	//private static final String CATEGORIES_URI = "file:///Users/**omitted for privacy purpose**/*"; //don't know why this uri didn't work 
	private static final String TRAINING_URI = "file:///Users/**omitted for privacy purpose**/NB_training_doc*.txt";
	

	public static void main(String[] args) throws IOException {
		
		// initializing spark
		SparkSession spark = SparkSession.builder().config("spark.master","local[*]").getOrCreate();
		JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());
		sc.setLogLevel("WARN");
		
		// read the categories file that maps text categories to numerical ones
		System.out.println("==> read the categories file that maps text categories to numerical ones");
		HashMap<String, Integer> categories = getCategoryMap("Resources/NB_categories.txt"); //original URI didn't work import file to the system instead 
		Broadcast<HashMap> allCategories = sc.broadcast(categories);


		// read the training documents
		JavaPairRDD<String,String> documents = sc.wholeTextFiles(TRAINING_URI);
		System.out.println("==> original training set");
		System.out.println(documents.take((int)documents.count()).toString());
		
		// each training document starts with the label
		// get the label, and change it to an integer
		JavaPairRDD<String, Tuple2<Integer,String>> trainingDocs = documents.mapValues( new Function<String,Tuple2<Integer,String>>() {
			public Tuple2<Integer,String> call(String line) throws Exception {
				if ( line == null || line.length() == 0 ) return null;
				if ( line.indexOf(LABEL_SEPARATOR) < 0 )  return null;
				String label = line.substring(0, line.indexOf(LABEL_SEPARATOR));
				
				if ( allCategories.getValue().containsKey(label) == false ) {
					// missing label
					return null;
				}
				String content = line.substring(line.indexOf(LABEL_SEPARATOR)+1);
				return new Tuple2(allCategories.getValue().get(label),content);
			}
		});
		
		System.out.println("==> original training set change label to integer");
		System.out.println(trainingDocs.take((int)trainingDocs.count()).toString());
		
		// create a dataframe for training documents
		StructType docSchema = new StructType(
			new StructField[] {
				DataTypes.createStructField("label", DataTypes.IntegerType, false),
				DataTypes.createStructField("text", DataTypes.StringType, false)
			}
		);
		Dataset<Row> trainingSet = spark.createDataFrame(
			trainingDocs.map( new Function<Tuple2<String, Tuple2<Integer,String>>, Row> () {
				@Override
				public Row call(Tuple2<String, Tuple2<Integer,String>> record) {
					return RowFactory.create(record._2()._1(), record._2()._2());
				}
			} ), docSchema);
		// trainingSet.show(false);
		
		// tokenize the training set
		Tokenizer tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words");
		Dataset<Row> trainingSetTokenized = tokenizer.transform(trainingSet);
		// trainingSetTokenized.show(false);

		// remove stopwords etc, can use Stanford NLP library if needed
		StopWordsRemover remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered");
		Dataset<Row> trainingSetStopWordsRemoved = remover.transform(trainingSetTokenized);
		//trainingSetStopWordsRemoved.show(false);
		
		// fit a CountVectorizerModel from the corpus
		CountVectorizer vectorizer = new CountVectorizer().setInputCol("filtered").setOutputCol("features");
		CountVectorizerModel cvm = vectorizer.fit(trainingSetStopWordsRemoved);
		System.out.println("vocab size = " + cvm.vocabulary().length);
		for (int i = 0; i < cvm.vocabulary().length; i ++ ) {
			System.out.print(cvm.vocabulary()[i] + "(" + i + ") ");
		}
		System.out.println();
		Dataset<Row> featurizedTrainingSet = cvm.transform(trainingSetStopWordsRemoved);
		System.out.println("===> final featured training set");
		featurizedTrainingSet.show(false);
		
		
		// create naive bayes model and train it
		NaiveBayes nb = new NaiveBayes();
		NaiveBayesModel model = nb.fit(featurizedTrainingSet.select("label", "features"));
		// NaiveBayesModel model = nb.train(featurizedTrainingSet.select("label", "features"));
		
		// study the model
		System.out.println("model.getFeaturesCol() = " + model.getFeaturesCol());
		System.out.println("model.getLabelCol() = " + model.getLabelCol());
		System.out.println("model.getModelType() = " + model.getModelType());
		System.out.println("model.getPredictionCol() = " + model.getPredictionCol());
		System.out.println("model.getProbabilityCol() = " + model.getProbabilityCol());
		System.out.println("model.getRawPredictionCol() = " + model.getRawPredictionCol());
		System.out.println("model.numFeatures() = " + model.numFeatures());

		
		
		// read the testing documents
		System.out.println("===> read the testing documents");
		JavaPairRDD<String,String> testdocuments = sc.wholeTextFiles(TESTING_URI);
		System.out.println(testdocuments.take((int)documents.count()).toString());
		
		
		//create testing data frame
		//testing set schema with "file name" and "text" two columns
		StructType schema = new StructType(
				new StructField[] {
						DataTypes.createStructField("file_name", DataTypes.StringType, false),
						DataTypes.createStructField("text",DataTypes.StringType,false)
				});
			
		System.out.println("===> testing dataframe");
		Dataset<Row> testingSet = spark.createDataFrame(
				testdocuments.map( new Function<Tuple2<String, String>, Row>() {
					@Override
					public Row call(Tuple2<String, String> record) {
						return RowFactory.create(record._1().substring(record._1().lastIndexOf("/")+1), record._2());
					}
				} ), schema);
		testingSet.show(true);
		
		
		// tokenize the testing set
		System.out.println("===> tokenize the testing set");
		Dataset<Row> testingSetTokenized = tokenizer.transform(testingSet);
		trainingSetTokenized.show(false);

		// remove stopwords etc
		System.out.println("===> remove stopwords etc");
		Dataset<Row> testingSetStopWordsRemoved = remover.transform(testingSetTokenized);
	    trainingSetStopWordsRemoved.show(false);
		
		// fit a CountVectorizerModel from the corpus by using cvm from ** training set **
		Dataset<Row> featurizedTestingSet = cvm.transform(testingSetStopWordsRemoved);
		System.out.println("===> final featured testing set");
		featurizedTestingSet.show(false);
		
		
		Dataset<Row> predictions = model.transform(featurizedTestingSet);
		predictions.show();
		System.out.println("===> final prediction");
		predictions.select("file_name","prediction").show();
				

		allCategories.unpersist();
		allCategories.destroy();
		sc.close();
	}
	

	//category map function
	private static HashMap getCategoryMap(String filePath) {
		
		HashMap<String, Integer> categories = new HashMap<String,Integer>();
		BufferedReader br = null;
		
		try {
			br = new BufferedReader(new FileReader("Resources/NB_categories.txt"));
			String line = br.readLine();
			
			while (line != null) {
				
				StringTokenizer st = new StringTokenizer(line);
				String categoryText = st.nextToken();
				Integer categoryIndex = new Integer(st.nextToken());
				categories.put(categoryText, categoryIndex);
				
				line = br.readLine();
				
			}
			
		} catch(Exception e) { // handle it the way you want
			System.out.println(e.getMessage());
		} finally {
			if ( br != null ) {
				try {
					br.close();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
		
		return categories;
	}


}

