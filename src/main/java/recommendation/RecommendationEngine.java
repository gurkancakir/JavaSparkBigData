package recommendation;

import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.DoubleFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import com.jasongoodwin.monads.Try;

import model.Movie;
import model.User;
import model.UserProductTuple;
import scala.Tuple2;


public class RecommendationEngine {
	
	public static final String moviesPath = "data/movies.dat";
	public static final String usersPath = "data/users.dat";
	public static final String ratingsPath = "data/ratings.dat";


	public static void main(String[] args) {
		
		/**
		 * Create SQL contex
		 */
		JavaSparkContext jsc = new JavaSparkContext("local", "Recommendation Engine");
		SQLContext sqlContext = new SQLContext(jsc);		
		
		/**
		 * Load Movie data
		 */
		 JavaRDD<Movie> movieRDD =  jsc.textFile(moviesPath).map(new Function<String, Movie>() {
			 public Movie call(String line) throws Exception {
					String[] movieArr = line.split("::");
					Integer movieId =  Integer.parseInt(Try.ofFailable(() -> movieArr[0]).orElse("-1"));
					return new Movie(movieId, movieArr[1], movieArr[2]);
				}
		 }).cache();
		 
		 
		System.out.println(movieRDD.first()); 
		
		 JavaRDD<User> userRDD = jsc.textFile(usersPath).map(new Function<String, User>() {

			@Override
			public User call(String line) throws Exception {
				String[] userArr = line.split("::");
				Integer userId = Integer.parseInt(Try.ofFailable(() -> userArr[0]).orElse("-1"));
				Integer age = Integer.parseInt(Try.ofFailable(() -> userArr[2]).orElse("-1"));
				Integer occupation = Integer.parseInt(Try.ofFailable(() -> userArr[3]).orElse("-1"));
				return new User(userId, userArr[1], age, occupation, userArr[4]);
			}
		}).cache();
		 
		 
		 JavaRDD<Rating> ratingRDD = jsc.textFile(ratingsPath).map(new Function<String, Rating>() {

			@Override
			public Rating call(String line) throws Exception {
				String[] ratingArr = line.split("::");
				Integer userId = Integer.parseInt(Try.ofFailable(() -> ratingArr[0]).orElse("-1"));
				Integer movieId = Integer.parseInt(Try.ofFailable(() -> ratingArr[1]).orElse("-1"));
				Double rating = Double.parseDouble(Try.ofFailable(() -> ratingArr[2]).orElse("-1"));
				return new Rating(userId, movieId, rating);
			}
		}).cache();
		 
		 System.out.println("Total number of user  : " + userRDD.count());
		 System.out.println("Total number of rating  : " + ratingRDD.count());
		 
		/**
		 * Group ratings by product key 
		 */
		 
		JavaPairRDD<Integer, Iterable<Rating>> ratingsGroupByProduct = ratingRDD.groupBy(new Function<Rating, Integer>() {
            @Override
            public Integer call(Rating rating) throws Exception {
                return rating.product();
            }
        });
		
		System.out.println("Total number of movies rated   : " + ratingsGroupByProduct.count());
		
		/**
		 * Group ratings by user id
		 */
		JavaPairRDD<Integer, Iterable<Rating>> ratingsGroupByUser = ratingRDD.groupBy(rating -> {
			return rating.user();
		});
		 
		System.out.println("Total number of users who rated movies   : " + ratingsGroupByUser.count());
		
		
		/**
		 * Users DF
		 */
		
		DataFrame usersDF = sqlContext.createDataFrame(userRDD, User.class);
		usersDF.registerTempTable("users");
		
		usersDF.printSchema();
		
		System.out.println("Total Number of users df : " + usersDF.count());
		
		DataFrame filteredUsersDF = sqlContext.sql("select * from users where users.userId in (11,12)");
		
		Row[] filteredUsers  = filteredUsersDF.collect();
		
		for(Row row : filteredUsers){
			System.out.print("UserId : " + row.getAs("userId"));
			System.out.print("	Gender : " + row.getAs("gender"));
			System.out.print("	Age : " + row.getAs("age"));
			System.out.print("	Occupation : " + row.getAs("occupation"));
			System.out.println("	Zip : " + row.getAs("zip"));
		}
		
		
		/**
		 * Ratings DF
		 * 
		 */
		StructType structType = new StructType(new StructField[]{DataTypes.createStructField("user", DataTypes.IntegerType, true),
				DataTypes.createStructField("product", DataTypes.IntegerType, true),
				DataTypes.createStructField("rating", DataTypes.DoubleType, true)});
		
		

		JavaRDD<Row> ratingRowRdd = ratingRDD.map(new Function<Rating, Row>() {

			@Override
			public Row call(Rating rating) throws Exception {
				return new RowFactory().create(rating.user() , rating.product() , rating.rating());
			}
		});
		
		
		DataFrame schemaPeople = sqlContext.createDataFrame(ratingRowRdd, structType);
		schemaPeople.registerTempTable("ratings");
		
		schemaPeople.printSchema();

		DataFrame teenagers = sqlContext.sql("SELECT * FROM ratings WHERE ratings.user = 1 and product in (938,919)");

		System.out.println("Number of rows : (user = 1 and product = 938 ) : " + teenagers.count());
		
		Row[] filteredDF = teenagers.collect();
		
		for(Row row : filteredDF){
			System.out.print("UserId : " + row.getAs("user"));
			System.out.print("	MovieId : " + row.getAs("product"));
			System.out.println("	Rating : " + row.getAs("rating"));
		}
			
		
		/**
		 * Movie DF
		 */
		
		DataFrame moviesDF = sqlContext.createDataFrame(movieRDD, Movie.class);
		moviesDF.registerTempTable("movies");
		
		moviesDF.printSchema();
		
		System.out.println("Total Number of movies df : " + usersDF.count());
		
		DataFrame filteredMoviesDF = sqlContext.sql("select * from movies where movies.movieId in (19,4000)");
		
		Row[] filteredMovies  = filteredMoviesDF.collect();
		
		for(Row row : filteredMovies){
			System.out.print("MovieId : " + row.getAs("movieId"));
			System.out.print("	Title : " + row.getAs("title"));
			System.out.println("	Genres : " + row.getAs("genres"));
		}
		
		/***
		 * Get the max, min ratings along with the count of users who have
		// rated a movie.
		 */
		DataFrame ratingsDF = sqlContext.sql("select movies.title, movierates.maxr, movierates.minr, movierates.cntu  " +
					" from(SELECT ratings.product, max(ratings.rating) as maxr, " +
					" min(ratings.rating) as minr,count(distinct user) as cntu  " + 
					" FROM ratings group by ratings.product ) movierates " + 
					" join movies on movierates.product=movies.movieId " + 
					" order by movierates.cntu desc ");
		
		ratingsDF.show();
		
		
		/**
		 * how the top 10 most-active users and how many times they rated
		// a movie
		 * 
		 */
		DataFrame top10MostActive = sqlContext.sql("SELECT ratings.user, count(*) as ct from ratings group by ratings.user order by ct desc limit 10");
		
		for(Row row : top10MostActive.collectAsList()){
			System.out.println(row);
		}
		
		/**
		 * Find the movies that user 4169 rated higher than 4
		 */
		DataFrame idUser4169 = sqlContext.sql("SELECT ratings.user, ratings.product,ratings.rating, movies.title FROM ratings JOIN movies ON movies.movieId=ratings.product "
				+ "where ratings.user=4169 and ratings.rating > 4");
		
		idUser4169.show();
		
		//Split rating into training and test
        JavaRDD<Rating>[] ratingSplits = ratingRDD.randomSplit(new double[] { 0.8, 0.2 });

        JavaRDD<Rating> trainingRatingRDD = ratingSplits[0].cache();
        JavaRDD<Rating> testRatingRDD = ratingSplits[1].cache();

        long numOfTrainingRating = trainingRatingRDD.count();
        long numOfTestingRating = testRatingRDD.count();

        System.out.println("Number of training Rating : " + numOfTrainingRating);
        System.out.println("Number of training Testing : " + numOfTestingRating);
        
        //Create prediction model (Using ALS)
        ALS als = new ALS();
        MatrixFactorizationModel model = als.setRank(20).setIterations(10).run(trainingRatingRDD);
        
        //Get the top 5 movie predictions for user 4169
        Rating[] recommendedsFor4169 = model.recommendProducts(4169, 5);
        System.out.println("Recommendations for 4169");
        for (Rating ratings : recommendedsFor4169) {
            System.out.println("Product id : " + ratings.product() + "-- Rating : " + ratings.rating());
        }
        
        //Get user product pair from testRatings
        JavaPairRDD<Integer, Integer> testUserProductRDD = testRatingRDD.mapToPair(new PairFunction<Rating, Integer, Integer>() {
            @Override
            public Tuple2<Integer, Integer> call(Rating rating) throws Exception {
                return new Tuple2<Integer, Integer>(rating.user(), rating.product());
            }
        });
 
        JavaRDD<Rating> predictionsForTestRDD = model.predict(testUserProductRDD);
        
        System.out.println("Test predictions");
        predictionsForTestRDD.take(10).stream().forEach(rating -> {
            System.out.println("Product id : " + rating.product() + "-- Rating : " + rating.rating());
        });
        
        //We will compare the test predictions to the actual test ratings
        JavaPairRDD<UserProductTuple, Double> predictionsKeyedByUserProductRDD = predictionsForTestRDD.mapToPair(new PairFunction<Rating, UserProductTuple, Double>() {
            @Override
            public Tuple2<UserProductTuple, Double> call(Rating rating) throws Exception {
                return new Tuple2<UserProductTuple, Double>(new UserProductTuple(rating.user(), rating.product()),rating.rating());
            }
        });
        
        
        JavaPairRDD<UserProductTuple, Double> testKeyedByUserProductRDD = testRatingRDD.mapToPair(new PairFunction<Rating, UserProductTuple, Double>() {
            @Override
            public Tuple2<UserProductTuple, Double> call(Rating rating) throws Exception {
                return new Tuple2<UserProductTuple, Double>(new UserProductTuple(rating.user(), rating.product()),rating.rating());
            }
        });
        
        JavaPairRDD<UserProductTuple, Tuple2<Double,Double>> testAndPredictionsJoinedRDD  = testKeyedByUserProductRDD.join(predictionsKeyedByUserProductRDD);
        
        testAndPredictionsJoinedRDD.take(10).forEach(k ->{
            System.out.println("UserID : " + k._1.getUserId() + "||ProductId: " + k._1.getProductId() + "|| Test Rating : " + k._2._1 + "|| Predicted Rating : " + k._2._2);
        });
        
        
        //Find false positives
        JavaPairRDD<UserProductTuple, Tuple2<Double,Double>> falsePositives =  testAndPredictionsJoinedRDD.filter(new Function<Tuple2<UserProductTuple,Tuple2<Double,Double>>, Boolean>() {

            @Override
            public Boolean call(Tuple2<UserProductTuple, Tuple2<Double, Double>> k) throws Exception {
                
                return k._2._1 <= 1 && k._2._2 >=4;
            }
        });
        
        System.out.println("testAndPredictionsJoinedRDD  count : " + testAndPredictionsJoinedRDD.count());
        System.out.println("False positives count : " + falsePositives.count());
        
        //Find absolute differences between the predicted and actual targets.
        JavaDoubleRDD meanAbsoluteError = testAndPredictionsJoinedRDD.mapToDouble(new DoubleFunction<Tuple2<UserProductTuple,Tuple2<Double,Double>>>() {

            @Override
            public double call(Tuple2<UserProductTuple, Tuple2<Double, Double>> v1) throws Exception {
                return Math.abs(v1._2._1 - v1._2._2) ;
            }
        });
        
        System.out.println("Mean : " + meanAbsoluteError.mean());
		
	}//main

}