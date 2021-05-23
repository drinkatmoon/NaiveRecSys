package com.naiverec.online.recprocess;

import com.naiverec.online.datamanager.DataManager;
import com.naiverec.online.datamanager.Movie;
import com.naiverec.online.datamanager.RedisClient;
import com.naiverec.online.datamanager.User;
import com.naiverec.online.util.Config;
import com.naiverec.online.util.Utility;
import org.json.JSONArray;
import org.json.JSONObject;
import org.mortbay.util.ajax.JSON;

import java.util.*;

import static com.naiverec.online.util.HttpClient.asyncSinglePostRequest;

public class RecForYouProcess {
    /**
     * 获取推荐的电影列表
     * @param userId 被推荐用户id
     * @param size 相似商品数量
     * @param model 用于计算相似度的模型名称
     * @return 相似电影集合
     */
    public static List<Movie> getRecList(int userId, int size, String model) {
        User user = DataManager.getInstance().getUserById(userId);
        if(null == user){
            return new ArrayList<>();
        }
        final int CONDIDATE_SIZE=800;
        //获取候选电影集合
        List<Movie> candidates = DataManager.getInstance().getMovies(CONDIDATE_SIZE,"rating");
        //如果数据源是存储在redis中，从redis加载用户embedding特征
        if(Config.EMB_DATA_SOURCE.equals(Config.DATA_SOURCE_REDIS)){
            String userEmbKey = "uEmb:"+userId;
            String userEmb = RedisClient.getInstance().get(userEmbKey);
            if(null != userEmb){
                user.setEmb(Utility.parseEmbStr(userEmb));
            }
        }
        if(Config.IS_LOAD_USER_FEATURE_FROM_REDIS){
            String userFeatureKey = "uf:"+userId;
            Map<String,String> userFeatures = RedisClient.getInstance().hgetAll(userFeatureKey);
            if(null != userFeatures){
                user.setUserFeatures(userFeatures);
            }
        }
        List<Movie> rankedList = ranker(user,candidates,model);
        return  null;
    }

    /**
     * 对候选集进行排序
     * @param user 输入user
     * @param candidates 候选影片集
     * @param model 模型名称
     * @return
     */
    private static List<Movie> ranker(User user, List<Movie> candidates, String model) {
        HashMap<Movie,Double> candidateScoreMap =new HashMap<>();
        switch (model){
            case "emb":
                for(Movie candidate:candidates){
                    double similarity = calculateEmbSimilarScore(user,candidate);
                    candidateScoreMap.put(candidate,similarity);
                }
                break;
            case "neuralcf":
                callNeuralCFTFServing(user,candidates,candidateScoreMap);
                break;
            default:
                //默认排序
                for(int i=0;i<candidates.size();i++){
                    candidateScoreMap.put(candidates.get(i),(double)candidates.size()-i);
                }
        }
        List<Movie> rankedList = new ArrayList<>();
        candidateScoreMap.entrySet().stream().sorted(Map.Entry.comparingByValue(Comparator.reverseOrder())).forEach(m->rankedList.add(m.getKey()));
        return rankedList;
    }

    /**
     * 调用TF Serving以获取NeuralCF model接口结果
     * @param user input user
     * @param candidates 候选集
     * @param candidateScoreMap 保存预测得分带scoreMap中
     */
    private static void callNeuralCFTFServing(User user, List<Movie> candidates, HashMap<Movie, Double> candidateScoreMap) {
        if(null == user || null == candidates || candidates.size()==0){
            return;
        }
        JSONArray instances = new JSONArray();
        for(Movie m :candidates){
            JSONObject instance = new JSONObject();
            instance.put("movieId",m.getMovieId());
            instance.put("userId",user.getUserId());
            instances.put(instance);
        }
        JSONObject instancesRoot = new JSONObject();
        instancesRoot.put("instances",instances);
        //need to confirm the tf serving end point
        String predictionScores =asyncSinglePostRequest("http://localhost:8051/v1/models/recmodel:predict",instancesRoot.toString());
        System.out.println("send user" + user.getUserId() + " request to tf serving.");

        JSONObject predictionsObject = new JSONObject(predictionScores);
        JSONArray scores =predictionsObject.getJSONArray("predictions");
        for(int i =0;i<candidates.size();i++){
            candidateScoreMap.put(candidates.get(i),scores.getJSONArray(i).getDouble(0));
        }
    }

    private static double calculateEmbSimilarScore(User user, Movie candidate) {
        if(null == user || null == candidate || null == user.getEmb()){
            return -1;
        }
        return user.getEmb().calculateSimilarity(candidate.getEmb());
    }
}
