package org.example.service;

import lombok.Getter;
import lombok.Setter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

/**
 * 混合检索服务
 * 结合向量搜索（语义匹配）和 BM25 搜索（精确匹配），通过 RRF 融合排序
 *
 * 流程：
 * 1. 接收多个查询（Multi-Query）
 * 2. 每个查询同时走向量搜索和 BM25 搜索（双路召回）
 * 3. 用 RRF（Reciprocal Rank Fusion）融合所有结果的排名
 * 4. 返回最终排序后的结果
 */
@Service
public class HybridSearchService {

    private static final Logger logger = LoggerFactory.getLogger(HybridSearchService.class);

    // RRF 平滑参数（经验值，越大排名靠后的影响越小）
    private static final int RRF_K = 60;

    @Autowired
    private VectorSearchService vectorSearchService;

    @Autowired
    private BM25SearchService bm25SearchService;

    @Value("${rag.top-k:3}")
    private int topK;

    @Value("${rag.hybrid-enabled:true}")
    private boolean hybridEnabled;

    /**
     * 混合检索入口
     * 对多个查询（Multi-Query）执行双路召回 + RRF 融合
     *
     * @param queries 查询列表（来自 QueryRewriteService）
     * @param topK    返回结果数
     * @return 融合后的搜索结果
     */
    public List<HybridResult> hybridSearch(List<String> queries, int topK) {
        if (queries == null || queries.isEmpty()) {
            return Collections.emptyList();
        }

        int effectiveTopK = topK > 0 ? topK : this.topK;
        if (!hybridEnabled) {
            return vectorSearchService.searchSimilarDocuments(queries.get(0), effectiveTopK).stream()
                    .map(this::toHybridResult)
                    .collect(Collectors.toList());
        }

        int recallTopK = effectiveTopK * 3;  // 召回阶段多取一些，供 RRF 融合排序

        // 存储每个查询的双路召回结果
        // key: docId, value: 该文档在各路检索中的排名信息
        Map<String, List<RankInfo>> docRanks = new HashMap<>();
        // 用于回填 content 和 metadata
        Map<String, HybridResult> docInfo = new HashMap<>();

        for (String query : queries) {
            // 路径 1：向量搜索（语义匹配）
            try {
                List<VectorSearchService.SearchResult> vectorResults =
                        vectorSearchService.searchSimilarDocuments(query, recallTopK);

                for (int rank = 0; rank < vectorResults.size(); rank++) {
                    VectorSearchService.SearchResult r = vectorResults.get(rank);
                    docRanks.computeIfAbsent(r.getId(), k -> new ArrayList<>())
                            .add(new RankInfo("vector", rank + 1, r.getScore()));
                    docInfo.putIfAbsent(r.getId(), toHybridResult(r));
                }
                logger.debug("向量搜索 [query='{}']: 找到 {} 个结果", query, vectorResults.size());
            } catch (Exception e) {
                logger.warn("向量搜索失败 [query='{}']: {}", query, e.getMessage());
            }

            // 路径 2：BM25 搜索（精确匹配）
            try {
                List<BM25SearchService.BM25Result> bm25Results =
                        bm25SearchService.search(query, recallTopK);

                for (int rank = 0; rank < bm25Results.size(); rank++) {
                    BM25SearchService.BM25Result r = bm25Results.get(rank);
                    docRanks.computeIfAbsent(r.getId(), k -> new ArrayList<>())
                            .add(new RankInfo("bm25", rank + 1, r.getScore()));
                    docInfo.putIfAbsent(r.getId(), toHybridResult(r));
                }
                logger.debug("BM25搜索 [query='{}']: 找到 {} 个结果", query, bm25Results.size());
            } catch (Exception e) {
                logger.warn("BM25搜索失败 [query='{}']: {}", query, e.getMessage());
            }
        }

        // RRF 融合排序
        List<HybridResult> fusedResults = rrfFusion(docRanks, docInfo);

        // 阈值过滤
        List<HybridResult> filtered = fusedResults.stream()
                .filter(r -> r.getRrfScore() > 0)
                .collect(Collectors.toList());

        logger.info("混合检索完成: {} 个查询, 融合后 {} 个结果, 返回 Top{}",
                queries.size(), filtered.size(), effectiveTopK);

        return filtered.stream()
                .limit(effectiveTopK)
                .collect(Collectors.toList());
    }

    /**
     * 单查询混合检索（简化接口）
     */
    public List<HybridResult> hybridSearch(String query, int topK) {
        return hybridSearch(List.of(query), topK);
    }

    /**
     * RRF（Reciprocal Rank Fusion）融合算法
     * 不看原始分数，只看排名，将各路检索的排名融合为统一分数
     *
     * 公式: score(doc) = Σ 1 / (k + rank_i)
     */
    private List<HybridResult> rrfFusion(
            Map<String, List<RankInfo>> docRanks,
            Map<String, HybridResult> docInfo) {

        List<HybridResult> results = new ArrayList<>();

        for (Map.Entry<String, List<RankInfo>> entry : docRanks.entrySet()) {
            String docId = entry.getKey();
            List<RankInfo> ranks = entry.getValue();

            // 计算 RRF 分数：每条检索路径的 1/(k+rank) 之和
            double rrfScore = 0;
            double maxVectorScore = 0;

            for (RankInfo rankInfo : ranks) {
                rrfScore += 1.0 / (RRF_K + rankInfo.rank);
                if ("vector".equals(rankInfo.path)) {
                    maxVectorScore = Math.max(maxVectorScore, rankInfo.originalScore);
                }
            }

            HybridResult result = docInfo.get(docId);
            if (result != null) {
                result.setRrfScore(rrfScore);
                result.setVectorScore((float) maxVectorScore);

                // 统计被多少条路径命中（命中路径越多越可靠）
                long pathCount = ranks.stream().map(r -> r.path).distinct().count();
                result.setHitCount((int) pathCount);

                results.add(result);
            }
        }

        // 按 RRF 分数降序排列
        results.sort((a, b) -> Double.compare(b.getRrfScore(), a.getRrfScore()));

        return results;
    }

    private HybridResult toHybridResult(VectorSearchService.SearchResult r) {
        HybridResult result = new HybridResult();
        result.setId(r.getId());
        result.setContent(r.getContent());
        result.setMetadata(r.getMetadata());
        result.setVectorScore(r.getScore());
        return result;
    }

    private HybridResult toHybridResult(BM25SearchService.BM25Result r) {
        HybridResult result = new HybridResult();
        result.setId(r.getId());
        result.setContent(r.getContent());
        result.setMetadata(r.getMetadata());
        return result;
    }

    public boolean isEnabled() {
        return hybridEnabled;
    }

    // ========== 内部数据结构 ==========

    private static class RankInfo {
        String path;      // 检索路径："vector" 或 "bm25"
        int rank;         // 排名（从 1 开始）
        float originalScore;  // 原始分数

        RankInfo(String path, int rank, float originalScore) {
            this.path = path;
            this.rank = rank;
            this.originalScore = originalScore;
        }
    }

    @Setter
    @Getter
    public static class HybridResult {
        private String id;
        private String content;
        private float vectorScore;   // 向量搜索原始分数（余弦相似度）
        private double rrfScore;     // RRF 融合分数
        private int hitCount;        // 被几条路径命中
        private String metadata;
    }
}
