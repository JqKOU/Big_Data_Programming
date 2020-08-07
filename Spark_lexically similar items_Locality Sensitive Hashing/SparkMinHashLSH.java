import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.CountVectorizerModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;


import scala.Tuple2;


public class SparkMinHashLSH {

	private static final String FILE_URI = "file:///Users/jianqunkou/Desktop/big data/hw5/LSH_*.txt";
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
		characteristicMatrix.show(false);

		
		
		
		
		// create minhashSignature for each document
		Broadcast<Double> sSize = sc.broadcast(sizeAdj);
		JavaPairRDD<String,List<Integer>> minhashSignature = characteristicMatrix.toJavaRDD().mapToPair(new PairFunction<Row, String, List<Integer>>() {
			
		    public Tuple2<String, List<Integer>> call(Row row) throws Exception {
		    	
		    	int signatureMatrixRowNum = (int)(vocabSize.value()*sSize.value());
		    	MinHashHelper mh = new MinHashHelper(vocabSize.value()); // vocabSize.value(): number of rows the signature matrix
		    	double[] characteristicMatrixRow = ((Vector)row.getAs("feature_vector")).toArray();

		    	List<Integer> signatureVector = new ArrayList<Integer>();
		    	for ( int i = 0; i < signatureMatrixRowNum; i ++ ) {
		    		int[] p = mh.getPermutation();
		    		int flag = Integer.MAX_VALUE;
		    		for ( int j = 0; j < characteristicMatrixRow.length; j ++ ) {
		    			if ( characteristicMatrixRow[j] == 1.0 && flag > p[j] ) {
		    				flag = p[j];
		    			}
		    		}
		    		signatureVector.add(flag);
		    	}
		    	
		        return new Tuple2<String, List<Integer>>((String) row.getAs("file_path"), signatureVector);
		    }
		});
		System.out.println("Minhash signatures:");
		System.out.println(minhashSignature.take((int)minhashSignature.count()).toString());

		
		// LSH implementation using hashCode
		class LocalSensitiveHashingImpl implements PairFlatMapFunction <Tuple2<String, List<Integer>>, String, String> {
			
			private final int ROWS  = 16;  // r factor

		    public Iterator<Tuple2<String, String>> call(Tuple2<String, List<Integer>> signatureColumn) throws Exception {
		    	
		    	String documentName = signatureColumn._1;
		    	int BANDS = signatureColumn._2.size()/ROWS;  // b factor
		    	String signatureVector = signatureColumn._2.toString().replaceAll("[^\\d.]", "");  // only the signature left here
		    	
		    	List<Tuple2<String, String>> lsh = new ArrayList<>();
		    	for ( int i = 0; i < BANDS; i++ ) {
		    		String singleBand = signatureVector.substring(i*ROWS, (i+1)*ROWS);
		    		lsh.add(new Tuple2<>("BAND-" + i + "-[" + singleBand.hashCode() + "]", documentName));
		    	}
		        return lsh.iterator(); 
		    }
		}
		JavaPairRDD<String,String> minhashResult = minhashSignature.flatMapToPair(new LocalSensitiveHashingImpl());
		System.out.println("Minhash results:");
		System.out.println(minhashResult.take((int)minhashResult.count()).toString());
				
		// finally
		JavaPairRDD<String,String> similarDocuments = minhashResult.reduceByKey(new Function2<String,String,String>() {
			public String call(final String document1,final String document2) {
				return document1 + "," + document2;
			}
		}).filter(new Function<Tuple2<String, String>,Boolean>() {
			public Boolean call(Tuple2<String, String> bucketDocument) {
				if ( bucketDocument._2.contains(",") ) return true;
				else return false;
			}
		});
		System.out.println("===> FINAL:"+ similarDocuments.take((int)similarDocuments.count()).toString());
		
		vocabSize.unpersist();
		vocabSize.destroy();
		sSize.unpersist();
		sSize.destroy();
		sc.close();

	}

}
