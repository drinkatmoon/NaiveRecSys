package com.naiverec.online.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.naiverec.online.datamanager.DataManager;
import com.naiverec.online.datamanager.Movie;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

/**
 * MovieService, return information of a specific movie
 * MovieService需要继承Jetty的HttpServlet
 */

public class MovieService extends HttpServlet {
    //实现servlet中的get method
    protected void doGet(HttpServletRequest request,
                         HttpServletResponse response) throws IOException {
        try {
            //该接口返回json对象，所以设置json类型
            response.setContentType("application/json");
            response.setStatus(HttpServletResponse.SC_OK);
            response.setCharacterEncoding("UTF-8");
            response.setHeader("Access-Control-Allow-Origin", "*");

            //获得请求中的id参数，转换为movie id
            String movieId = request.getParameter("id");

            //从数据库中获取该movie的数据对象
            Movie movie = DataManager.getInstance().getMovieById(Integer.parseInt(movieId));

            //使用fasterxml.jackson库把movie对象转换成json对象
            if (null != movie) {
                ObjectMapper mapper = new ObjectMapper();
                String jsonMovie = mapper.writeValueAsString(movie);
                //返回json对象
                response.getWriter().println(jsonMovie);
            }else {
                response.getWriter().println("");
            }

        } catch (Exception e) {
            e.printStackTrace();
            response.getWriter().println("");
        }
    }
}
