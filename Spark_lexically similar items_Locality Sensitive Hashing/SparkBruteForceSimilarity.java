import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
//import org.apache.spark.api.java.function.Function2;
//import org.apache.spark.api.java.function.MapFunction;
//import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.linalg.SparseVector;
//import org.apache.spark.ml.linalg.Matrix;
//import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.HashSet;
import java.util.Set;



import scala.Tuple2;

// this is the example in Chap 3, Example 3.6
// for the related homework question, see SparkBruteForceSimilarity

public class SparkBruteForceSimilarity {

	private static final String FILE_URI = "file:///Users/jianqunkou/Desktop/bigdata/hw5/data/LSH_*.txt";
	private static final double sizeAdj = 1.0;
	
	public static void main(String[] args) {
		
		// initializing spark
		SparkSession spark = SparkSession.builder().config("spark.master","local[*]").getOrCreate();
		JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());
		sc.setLogLevel("WARN");
		
		// create RDD by using text files
		JavaPairRDD<String,String> documents = sc.wholeTextFiles(FILE_URI);
		// System.out.println(documents.take((int)documents.count()).toString());
		
		// convert original documents into shingle representation
		class ShinglesCreator implements Function<String,String[]> { 		
			@Override
			public String[] call(String text) throws Exception {
				return ShingleUtils.getTextShingles(text);
			}			
		}
		JavaPairRDD<String,String[]> shinglesDocs = documents.mapValues(new ShinglesCreator());
		shinglesDocs.values().foreach(new VoidFunction<String[]>() {
		    public void call(String[] shingles) throws Exception {
		        for ( int i = 0; i < shingles.length; i ++ ) {
		        	System.out.print(shingles[i] + "|");		        	
		        }
		        System.out.println();
		    }
		});	
		

		// create characteristic matrix representation of each document
		StructType schema = new StructType(
				new StructField[] {
						DataTypes.createStructField("file_path", DataTypes.StringType, false),
						DataTypes.createStructField("file_content",DataTypes.createArrayType(DataTypes.StringType, false),false)
				});
		Dataset<Row> df = spark.createDataFrame(
				shinglesDocs.map( new Function<Tuple2<String, String[]>, Row>() {
					@Override
					public Row call(Tuple2<String, String[]> record) {
						return RowFactory.create(record._1().substring(record._1().lastIndexOf("/")+1), record._2());
					}
				} ), schema);
		df.show(true);
		
		CountVectorizer vectorizer = new CountVectorizer().setInputCol("file_content").setOutputCol("feature_vector").setBinary(true);
		CountVectorizerModel cvm = vectorizer.fit(df);
		Broadcast<Integer> vocabSize = sc.broadcast(cvm.vocabulary().length);
		
		System.out.println("vocab size = " + cvm.vocabulary().length);
		for (int i = 0; i < vocabSize.value(); i ++ ) {
			System.out.print(cvm.vocabulary()[i] + "(" + i + ") ");
		}
		System.out.println();
		
		Dataset<Row> characteristicMatrix = cvm.transform(df);
		characteristicMatrix.show();

		JavaRDD<Row> cm = characteristicMatrix.select("file_path","feature_vector").toJavaRDD();
		
		// check the content of this RDD
		System.out.println("checking content of RDD cm:");
		System.out.println(cm.take((int)cm.count()).toString());
		
		//cartersian() and filter()
		JavaPairRDD<Row,Row> pairedCM = cm.cartesian(cm).filter( new Function<Tuple2<Row,Row>,Boolean> () {
			public Boolean call(Tuple2<Row,Row> row) {
				String leftFileName = row._1().getString(0);
				String rightFileName = row._2().getString(0);
				if ( leftFileName.compareTo(rightFileName) < 0 ) {
					return true;
					} else return false;
				}
			} );

		System.out.println("checking content of RDD pairedCM:");
		System.out.println(pairedCM.take((int)pairedCM.count()).toString());
		
		// calculate Jaccard similarity using shingles information
		JavaPairRDD<String,Double> pairwiseComparison = pairedCM.mapToPair( new PairFunction<Tuple2<Row,Row>,String,Double> () {
			public Tuple2<String, Double> call(Tuple2<Row,Row> row) {
				String leftFileName = row._1().getString(0);
				String rightFileName = row._2().getString(0);
				SparseVector leftFeatureVector = (SparseVector)row._1().get(1);
				SparseVector rightFeatureVector = (SparseVector)row._2().get(1);
				double[] leftFeature = leftFeatureVector.toDense().toArray();
				double[] rightFeature = rightFeatureVector.toDense().toArray();
				double union = 0.0;
				double intersection = 0.0;
				for (int i = 0; i < leftFeature.length; i ++ ) {
					if ( leftFeature[i] == rightFeature[i] ) {
						if (leftFeature[i] != 0.0 ) {
							intersection ++;
							union ++;
							}
						} else {
							union ++;
							}
					}

				return new Tuple2<String, Double>(leftFileName + "-" +

				rightFileName, intersection/union);

				}

				} );
		
		System.out.println("===> FINAL:");
		System.out.println(pairwiseComparison.take((int)pairwiseComparison.count()).toString());
}}

