package com.naiverec.online.recprocess;

import com.naiverec.online.datamanager.DataManager;
import com.naiverec.online.datamanager.Movie;
import org.apache.avro.generic.GenericData;

import java.util.ArrayList;
import java.util.List;

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
        List<Movie> candidates = candidateGenerator(movieId);
        //精排序
        ranker(movie,candidates,model);

        return null;
    }

    private static List<Movie> ranker(Movie movie, List<Movie> candidates, String model) {
        return null;
    }

    private static List<Movie> candidateGenerator(int movieId) {

        return null;
    }
}
