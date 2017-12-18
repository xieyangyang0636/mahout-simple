package cn.yano.mahout.taste.simple.kmeans.main;


import cn.yano.mahout.taste.simple.kmeans.util.MathUtil;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;

/**
 * Created by xieyy23076 on 2017/12/18.
 */
public class Kmeans {

    private static Logger log = LoggerFactory.getLogger(Kmeans.class);
    final static int K = 3;
    final static double THRESHOLD = 0.01;
    public static void main(String[] args) throws IOException {
        String inputPath = "data/item.csv";
        List<Vector> data = MathUtil.readFileToVector(inputPath);
        List<Vector> randomPoints = MathUtil.chooseRandomPoints(data, K);
        for (Vector vector : randomPoints) {
            log.info("Init Point center: {}", vector);
        }

    }
}
