import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;

// this is the code for the PPT, but not the homework

public class SparkRecommendationMovie {

	// original data files
	private static final String USER_URI = "file:///f:/my talks and teaching/New folder/examples/datasets/u.data";
	// private static final String MOVIE_URI = "file:///f:/my talks and teaching/New folder/examples/datasets/u.item";
	private static final String MOVIE_URI = "f:\\my talks and teaching\\New folder\\examples\\datasets\\u.item";
	
	// let us focus on this user, can change to any other user
	private static final int user = 356;
	
	public static void main(String[] args) {
		
		SparkSession spark = SparkSession.builder().config("spark.master","local[*]").getOrCreate();
		JavaSparkContext sc = new JavaSparkContext(spark.sparkContext());
		sc.setLogLevel("WARN");
		
	    JavaRDD<Movie> movieRdd = spark.read().textFile(MOVIE_URI).javaRDD().map(
	    		
				new Function<String, Movie>() {
					public Movie call(String moviedata) {
						String[] tmpStrs = moviedata.split("\\|");
	   					if ( tmpStrs != null && tmpStrs.length >= 2 && 
	   						 tmpStrs[0] != null && tmpStrs[1] != null ) {
	   						return new Movie(tmpStrs[0],tmpStrs[1]);
	   					} else return null;
					}
				}
		
	    );	    
	    // create a DataFrame representing movies
	 	Dataset<Row> movieDS = spark.createDataFrame(movieRdd.rdd(), Movie.class);
	 	movieDS.show();
		movieDS.createOrReplaceTempView("movies");	
				
		JavaRDD<Rating> ratingsRdd = spark.read().textFile(USER_URI).javaRDD().map(
				
				new Function<String,Rating>() {
					public Rating call(String userRating) {
						String[] tmpStrs = userRating.split("\t");
						if ( tmpStrs != null && tmpStrs.length >= 3 && 
							 tmpStrs[0] != null && tmpStrs[1] != null && tmpStrs[2] != null ) {
							int userId = Integer.parseInt(tmpStrs[0]);
							int movieId = Integer.parseInt(tmpStrs[1]);
							double rating = Double.parseDouble(tmpStrs[2]);
							return new Rating(userId, movieId, rating);
						} else return null;
					}
				}
				
		);
		System.out.println(ratingsRdd.take(20).toString());
		
		// create a DataFrame representing ratings, but using a different way
	 	StructType ratingSchema = new StructType(
				new StructField[] {
						DataTypes.createStructField("userId", DataTypes.IntegerType, false),
						DataTypes.createStructField("movieId", DataTypes.IntegerType, false),
						DataTypes.createStructField("rating",DataTypes.DoubleType,false)
				});
		
		Dataset<Row> originalRatingsMatrix = spark.createDataFrame(
				ratingsRdd.map( new Function<Rating, Row>() {
					@Override
					public Row call(Rating record) {
						return RowFactory.create(record.user(), record.product(), record.rating());
					}
				} ), ratingSchema); 
		originalRatingsMatrix.show(40, true);
		
		originalRatingsMatrix.createOrReplaceTempView("originalRatings");
		spark.sql("select m.movieTitle,p.userId,p.rating from originalRatings p,movies m "
				+ "where p.userId = " + user + " and p.movieId = m.movieId order by p.rating desc").show(40);

 	    // Build the recommendation model using ALS, make changes to below parameters to adjust
		int rank = 5;
	    int numIterations = 15;
	    double lambda = 0.01;
	    MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(ratingsRdd), rank, numIterations, lambda);	    
	    
	    // prepare input parameter for predictions 
	    JavaRDD<Tuple2<Object, Object>> userMovie = ratingsRdd.map(
	    		
	    		new Function<Rating, Tuple2<Object, Object>>() {
	    			public Tuple2<Object, Object> call(Rating r) {
	    				return new Tuple2<Object, Object>(r.user(), r.product());
	    			}
	    		}
	    		
	    );
	    
	    // use the model to do the predictions
	    JavaRDD<Rating> predictionRdd = model.predict(JavaRDD.toRDD(userMovie)).toJavaRDD();
	    System.out.println(predictionRdd.take(20).toString());
	    
	    // check the predicted ratings for the same user, just to compare
	    Dataset<Row> predictedRatingsMatrix = spark.createDataFrame(
	    		predictionRdd.map( new Function<Rating, Row>() {
					@Override
					public Row call(Rating record) {
						return RowFactory.create(record.user(), record.product(), record.rating());
					}
				} ), ratingSchema); 
	    predictedRatingsMatrix.createOrReplaceTempView("predictedRatings");
		spark.sql("select m.movieTitle,p.userId,p.rating from predictedRatings p,movies m "
				+ "where p.userId = " + user + " and p.movieId = m.movieId order by p.rating desc").show(140);
		
	    // create RDD to hold all original user ratings
	    JavaPairRDD<Tuple2<Integer, Integer>, Double> userRatings = 
	    		JavaPairRDD.fromJavaRDD(ratingsRdd.map(
	    			new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
	    				public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r) {
	    					return new Tuple2<Tuple2<Integer, Integer>, Double>(
	    							new Tuple2<Integer, Integer>(r.user(), r.product()), r.rating());
	    				}
	    			}
	    		));
	    System.out.println(" ==> " + userRatings.take(20).toString());
	    
	    // create RDD to hold predictions, using the same structure
	    JavaPairRDD<Tuple2<Integer, Integer>, Double> userRatingPredictions =
	    	      JavaPairRDD.fromJavaRDD(predictionRdd.map(
	    	        new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
	    	          public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r){
	    	            return new Tuple2<Tuple2<Integer, Integer>, Double>(
	    	              new Tuple2<Integer, Integer>(r.user(), r.product()), r.rating());
	    	          }
	    	        }
	    	));
	    System.out.println(" ==> " + userRatingPredictions.take(20).toString());
	    
	    // create a RDD to compare original ratings with predicted values
	    JavaRDD<Tuple2<Double, Double>> ratesAndPreds = 
	    		userRatingPredictions.join(userRatings).values();
	    System.out.println(" ==> " + ratesAndPreds.take(20).toString());
	    
	    // calculate MSE to evaluate the model
	    double MSE = JavaDoubleRDD.fromRDD(ratesAndPreds.map(
	    		new Function<Tuple2<Double, Double>, Object>() {
	    			public Object call(Tuple2<Double, Double> pair) {
	    				Double err = pair._1() - pair._2();
	    				return err * err;
	    	        }
	    	      }
	    	).rdd()).mean();
	    System.out.println("Mean Squared Error = " + MSE);
	    
	    spark.close();

	    	    
	}

}

