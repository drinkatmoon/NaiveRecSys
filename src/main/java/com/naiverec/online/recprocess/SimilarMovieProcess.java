package com.naiverec.online.recprocess;

import com.naiverec.online.datamanager.DataManager;
import com.naiverec.online.datamanager.Movie;
import org.apache.avro.generic.GenericData;

import javax.xml.crypto.Data;
import java.util.*;

public class SimilarMovieProcess {
    /**
     * 获取推荐影片集合
     * @param movieId 输入影片id
     * @param size 相似商品大小
     * @param model 相似度计算model
     * @return 相似电源列表
     */
    public static List<Movie> getRecList(int movieId, int size, String model) {
        //查询影片基础信息
        Movie movie = DataManager.getInstance().getMovieById(movieId);
        if(null == movie){
            return new ArrayList<>();
        }
        List<Movie> candidates = candidateGenerator(movie);
        //精排序
        ranker(movie,candidates,model);

        return null;
    }

    /**
     * 基于指定model计算出相似度进行排序
     * @param movie
     * @param candidates
     * @param model
     * @return
     */
    private static List<Movie> ranker(Movie movie, List<Movie> candidates, String model){
        HashMap<Movie,Double> candidateScoreMap = new HashMap<>();
        for(Movie candidate:candidates){
            double similarity;
            switch (model){
                case "emb":
                    similarity = calculateEmbSimilarScore(movie,candidate);
                    break;
                default:
                    similarity = calculateSimilarScore(movie,candidate);
            }
            candidateScoreMap.put(candidate,similarity);
        }
        List<Movie> rankedList = new ArrayList<>();
        candidateScoreMap.entrySet().stream().sorted(Map.Entry.comparingByValue(Comparator.reverseOrder())).forEach(m->rankedList.add(m.getKey()));
        return rankedList;
    }

    /**
     * 采用基于影片标签属性的单策略召回方法生成候选集
     * @param movie
     * @return
     */
    private static List<Movie> candidateGenerator(Movie movie) {
        Map<Integer,Movie> candidateMap = new HashMap<>();
        for(String genre :movie.getGenres() ){
            List<Movie> oneCandidates = DataManager.getInstance().getMoviesByGenre(genre, 100, "rating");
            for(Movie candidate:oneCandidates){
                candidateMap.put(candidate.getMovieId(),candidate);
            }
        }
        candidateMap.remove(movie.getMovieId());
        return new ArrayList<>(candidateMap.values());
    }

    /**
     * 采用多路召回方法生成候选集
     * @param movie
     * @return
     */
    public static List<Movie> multipleRetrievalCandidates(Movie movie){
        if(null ==movie ){
            return  null;
        }
        //按电影分类标签召回
        HashSet<String> genres = new HashSet<>(movie.getGenres());
        HashMap<Integer,Movie> candidateMap = new HashMap<>();
        for(String genre:genres){
            List<Movie> oneCandidates = DataManager.getInstance().getMoviesByGenre(genre,20,"rating");
            for(Movie candidate: oneCandidates){
                candidateMap.put(candidate.getMovieId(),candidate);
            }
        }
        //根据评分高低召回候选集
        List<Movie> hignRatingCandidates = DataManager.getInstance().getMovies(100, "rating");
        for(Movie candidate:hignRatingCandidates){
            candidateMap.put(candidate.getMovieId(),candidate);
        }
        //根据发行年限（流行度）召回候选集
        List<Movie> latestCandidates = DataManager.getInstance().getMovies(100, "releaseYear");
        for(Movie candidate:latestCandidates){
            candidateMap.put(candidate.getMovieId(),candidate);
        }
        candidateMap.remove(movie.getMovieId());
        return new ArrayList<>(candidateMap.values());
    }

    /**
     * 基于embedding召回候选集
     * @param movie
     * @param size
     * @return
     */
    public static List<Movie> retrievalCandidatesByEmbedding(Movie movie, int size){
        if(null==movie || null ==movie.getEmb()){
            return  null;
        }
        List<Movie> allCandidates = DataManager.getInstance().getMovies(10000, "rating");
        HashMap<Movie,Double> movieScoreMap = new HashMap<>();
        for(Movie candidate:allCandidates){
            calculateEmbSimilarScore(movie,candidate);
        }
        return null;
    }

    /**
     * 基于emb相似度计算得分
     * @param movie
     * @param candidate
     */
    private static double calculateEmbSimilarScore(Movie movie, Movie candidate) {
        if(null == movie || null == candidate){
            return -1;
        }
        return  movie.getEmb().calculateSimilarity(candidate.getEmb());
    }

    /**
     * 基于影片特征标签与评分计算物品间的相似度
     * @param movie
     * @param candidate
     * @return
     */
    private static double calculateSimilarScore(Movie movie, Movie candidate) {
        //统计与候选影片同样类别标签数量
        int sameGenreCount = 0;
        for(String genre:movie.getGenres()){
            if(candidate.getGenres().contains(genre)){
                sameGenreCount++;
            }
        }
        //计算标签相似度
        double genreSimilarity = (double)sameGenreCount/(movie.getGenres().size() + candidate.getGenres().size())/2;
        //计算得分占总分比例
        double ratingScore = candidate.getAverageRating()/5;
        //设定权重
        double similarityWeight = 0.7;
        double ratingScoreWeight = 0.3;

        return genreSimilarity * similarityWeight + ratingScoreWeight * ratingScore;
    }
}
