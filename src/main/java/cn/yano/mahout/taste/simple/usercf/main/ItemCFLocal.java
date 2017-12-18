package cn.yano.mahout.taste.simple.usercf.main;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Created by xieyy23076 on 2017/12/18.
 */
public class ItemCFLocal {

    private static Logger log = LoggerFactory.getLogger(ItemCFLocal.class);
    final static int RECOMMENDER_NUM = 3;

    /**
     *  Taste由以下五个主要的组件组成：
     *  1、DataModel         用户喜好信息的抽象接口
     *  2、UserSimilarity    用于计算用户之间的相似度
     *  3、ItemSimilarity    用于计算内容之间的相似度
     *  4、UserNeighborhood  用于基于用户相似度的推荐方法中计算邻居用户
     *  5、Recommender       是推荐引擎的抽象接口，Taste中的核心组件
     */
    public static void main(String[] args) {
        // 文件路径
        String inputPath = "data/item.csv";
        try {
            /**
             *  加载数据
             *  1、内存    GenericDataModel
             *  2、文件    FileDataModel
             *      a、每一行包含一个用户Id，物品Id，用户喜好
             *      b、逗号隔开或者Tab隔开
             *      c、zip 和 gz 文件会自动解压缩（Mahout 建议在数据量过大时采用压缩的数据存储）
             *  3、数据库  JDBCDataModel
             *      a、用户ID和物品ID 列需要是BIGINT而且非空
             *      b、用户喜好值列需要是FLOAT
             *      c、建议在用户ID和物品ID上建索引
             * **/
            DataModel dataModel = new FileDataModel(new File(inputPath));
            /**
             *  相似度计算
             *  1、欧几里德距离        EuclideanDistanceSimilarity
             *      a、原理：利用欧式距离d定义的相似度s，s=1 / (1+d)
             *      b、范围：[0,1]，值越大，说明d越小，也就是距离越近，则相似度越大。
             *      c、说明：同皮尔森相似度一样，该相似度也没有考虑重叠数对结果的影响，同样地，Mahout通过增加一个枚举类型（Weighting）的参数来使得重叠数也成为计算相似度的影响因子。
             *  2、皮尔森相关系数      PearsonCorrelationSimilarity
             *      a、原理：用来反映两个变量线性相关程度的统计量
             *      b、范围：[-1,1]，绝对值越大，说明相关性越强，负相关对于推荐的意义小。
             *      c、说明：1、 不考虑重叠的数量；2、 如果只有一项重叠，无法计算相似性（计算过程被除数有n-1）；3、 如果重叠的值都相等，也无法计算相似性（标准差为0，做除数）。
             *  3、余弦相似度          UncenteredCosineSimilarity or PearsonCorrelationSimilarity
             *      a、原理：多维空间两点与所设定的点形成夹角的余弦值。
             *      b、范围：[-1,1]，值越大，说明夹角越大，两点相距就越远，相似度就越小。
             *      c、说明：在数学表达中，如果对两个项的属性进行了数据中心化，计算出来的余弦相似度和皮尔森相似度是一样的，在mahout中，实现了数据中心化的过程，所以皮尔森相似度值也是数据中心化后的余弦相似度。另外在新版本中，Mahout提供了UncenteredCosineSimilarity类作为计算非中心化数据的余弦相似度。
             *  4、Spearman秩相关系数  SpearmanCorrelationSimilarity
             *      a、原理：Spearman秩相关系数通常被认为是排列后的变量之间的Pearson线性相关系数。
             *      b、范围：{-1.0,1.0}，当一致时为1.0，不一致时为-1.0。
             *      c、说明：计算非常慢，有大量排序。针对推荐系统中的数据集来讲，用Spearman秩相关系数作为相似度量是不合适的。
             *  5、曼哈顿距离          CityBlockSimilarity
             *      a、原理：曼哈顿距离的实现，同欧式距离相似，都是用于多维数据空间距离的测度。
             *      b、范围：[0,1]，同欧式距离一致，值越小，说明距离值越大，相似度越大。
             *      c、说明：比欧式距离计算量少，性能相对高。
             *  6、Tanimoto系数        TanimotoCoefficientSimilarity
             *      a、原理：又名广义Jaccard系数，是对Jaccard系数的扩展。
             *      b、范围：[0,1]，完全重叠时为1，无重叠项时为0，越接近1说明越相似。
             *      c、说明：处理无打分的偏好数据。
             *  7、对数似然相似度      LogLikelihoodSimilarity
             *      a、原理：重叠的个数，不重叠的个数，都没有的个数
             *      b、范围：具体可去百度文库中查找论文《Accurate Methods for the Statistics of Surprise and Coincidence》
             *      c、说明：处理无打分的偏好数据，比Tanimoto系数的计算方法更为智能。
             */
            ItemSimilarity itemSimilarity = new EuclideanDistanceSimilarity(dataModel);
            /**
             *  利用Similarity找到待推荐item集合后的各种推荐策略的推荐器实现类
             *  1、GenericUserBasedRecommender               user-based模式的推荐器实现类
             *      a、使用UserNeighborhood获取跟指定用户Ui最相似的K个用户{U1…Uk}
             *      b、{U1…Uk}喜欢的item集合中排除掉Ui喜欢的item, 得到一个item集合 {Item0...Itemm}
             *      c、对{Item0...Itemm}每个itemj计算 Ui可能喜欢的程度值perf(Ui , Itemj) ，并把item按这个数值从高到低排序，把前N个item推荐给Ui
             *  2、GenericBooleanPerfUserBasedRecommender    user-based模式的推荐器实现类
             *      a、处理逻辑跟GenericUserBasedRecommender一样,计算公式不同
             *  3、GenericItemBasedRecommender               item-based模式的推荐器实现类
             *      a、获取用户Ui喜好的item集合{It1…Itm}
             *      b、获取跟用户喜好集合里每个item最相似的其他Item构成集合 {Item1…Itemk}
             *      c、对{Item1...Itemk}里的每个itemj计算 Ui可能喜欢的程度值perf(Ui , Itemj) ，并把item按这个数值从高到低排序，把前N个Item推荐给Ui
             *  4、GenericBooleanPrefItemBasedRecommender    item-based模式的推荐器实现类
             *      a、处理逻辑跟GenericItemBasedRecommender一样,计算公式不同
             *  5、KnnItemBasedRecommender                   item-based模式的推荐器实现类
             *      a、处理逻辑跟GenericItemBasedRecommender一样,计算公式不同
             *  6、ItemAverageRecommender                    预测一个用户对一个未知item的喜好值是所有用户对这个item喜好值的平均值
             *      a、提供给实验用的推荐类，简单但计算快速，推荐结果可能会不够好
             *  7、ItemUserAverageRecommender
             *      a、在ItemAverageRecommender的基础上，考虑了用户喜好的平均值和全局所有喜好的平均值进行调整
             *  8、RandomRecommender
             *      a、随机推荐item,  测试性能用。
             *  9、SlopeOneRecommender
             *      a、基于Slopeone算法的推荐器
             *  10、SVDRecommender
             *  11、TreeClusteringRecommender
             */
            Recommender recommender = new GenericItemBasedRecommender(dataModel,  itemSimilarity);
            // 输出推荐结果
            LongPrimitiveIterator iter = dataModel.getUserIDs();
            while (iter.hasNext()) {
                long userId = iter.nextLong();
                List<RecommendedItem> recommendList = recommender.recommend(userId, RECOMMENDER_NUM);
                for (RecommendedItem item : recommendList) {
                    log.info("##### userId : {}   ({},{})", userId,item.getItemID(),item.getValue());
                }
            }
        } catch (IOException e) {
            log.error("##### 输入文件{}不存在", inputPath);
            e.printStackTrace();
            System.exit(1);
        } catch (TasteException e) {
            e.printStackTrace();
        }
    }
}
