package org.example.service;

import lombok.Getter;
import lombok.Setter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * BM25 关键词搜索服务
 * 基于内存倒排索引实现轻量级 BM25 搜索，与向量搜索互补
 *
 * BM25 优势：擅长精确关键词匹配（如产品型号、错误码、专有名词）
 * BM25 劣势：不理解语义（"退货" 和 "申请售后" 无法匹配）
 */
@Service
public class BM25SearchService {

    private static final Logger logger = LoggerFactory.getLogger(BM25SearchService.class);

    // BM25 参数
    private static final float K1 = 1.5f;  // 词频饱和参数
    private static final float B = 0.75f;  // 文档长度归一化参数

    // 倒排索引：term -> [(docId, termFrequency)]
    private final Map<String, List<PostingEntry>> invertedIndex = new ConcurrentHashMap<>();

    // 文档存储：docId -> document
    private final Map<String, Document> documents = new ConcurrentHashMap<>();

    // 文档长度缓存
    private final Map<String, Integer> docLengths = new ConcurrentHashMap<>();

    // 平均文档长度
    private double avgDocLength = 0;

    @Value("${rag.bm25.enabled:true}")
    private boolean bm25Enabled;

    /**
     * 添加文档到 BM25 索引
     *
     * @param docId  文档唯一标识
     * @param content 文档内容
     * @param metadata 文档元数据（JSON 字符串）
     */
    public void addDocument(String docId, String content, String metadata) {
        if (!bm25Enabled) return;

        removeDocument(docId);

        // 分词
        List<String> tokens = tokenize(content);

        // 计算词频
        Map<String, Integer> termFreq = new HashMap<>();
        for (String token : tokens) {
            termFreq.merge(token, 1, Integer::sum);
        }

        // 存储文档
        Document doc = new Document();
        doc.id = docId;
        doc.content = content;
        doc.metadata = metadata;
        doc.termFreq = termFreq;
        doc.tokenCount = tokens.size();

        documents.put(docId, doc);
        docLengths.put(docId, tokens.size());

        // 更新倒排索引
        for (Map.Entry<String, Integer> entry : termFreq.entrySet()) {
            String term = entry.getKey();
            int freq = entry.getValue();

            invertedIndex.computeIfAbsent(term, k -> new ArrayList<>())
                    .add(new PostingEntry(docId, freq));
        }

        // 重新计算平均文档长度
        recalculateAvgDocLength();

        logger.debug("BM25 索引添加文档: {}, tokens: {}", docId, tokens.size());
    }

    /**
     * 从 BM25 索引中删除文档
     */
    public void removeDocument(String docId) {
        if (!bm25Enabled) return;

        Document doc = documents.remove(docId);
        if (doc == null) return;

        docLengths.remove(docId);

        // 从倒排索引中移除
        for (String term : doc.termFreq.keySet()) {
            List<PostingEntry> postings = invertedIndex.get(term);
            if (postings != null) {
                postings.removeIf(e -> e.docId.equals(docId));
                if (postings.isEmpty()) {
                    invertedIndex.remove(term);
                }
            }
        }

        recalculateAvgDocLength();
        logger.debug("BM25 索引删除文档: {}", docId);
    }

    /**
     * 删除同一个源文件下的所有 BM25 文档。
     */
    public int removeDocumentsBySource(String sourcePath) {
        if (!bm25Enabled || sourcePath == null || sourcePath.isBlank()) return 0;

        List<String> docIds = documents.values().stream()
                .filter(doc -> sourcePath.equals(extractSource(doc.metadata)))
                .map(doc -> doc.id)
                .toList();

        for (String docId : docIds) {
            removeDocument(docId);
        }

        logger.debug("BM25 索引按 source 删除文档: {}, count: {}", sourcePath, docIds.size());
        return docIds.size();
    }

    private String extractSource(String metadata) {
        try {
            if (metadata == null || metadata.isBlank()) return null;
            com.google.gson.JsonObject json = com.google.gson.JsonParser.parseString(metadata).getAsJsonObject();
            return json.has("_source") ? json.get("_source").getAsString() : null;
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * BM25 搜索
     *
     * @param query 查询文本
     * @param topK  返回结果数
     * @return 搜索结果列表，按分数降序排列
     */
    public List<BM25Result> search(String query, int topK) {
        if (!bm25Enabled || documents.isEmpty()) {
            return Collections.emptyList();
        }

        List<String> queryTokens = tokenize(query);
        if (queryTokens.isEmpty()) {
            return Collections.emptyList();
        }

        int totalDocs = documents.size();
        double avgDl = avgDocLength;

        // 对每个文档计算 BM25 分数
        Map<String, Double> scores = new HashMap<>();

        for (String token : queryTokens) {
            List<PostingEntry> postings = invertedIndex.get(token);
            if (postings == null) continue;

            // IDF 部分：log((N - n + 0.5) / (n + 0.5) + 1)
            double idf = Math.log((totalDocs - postings.size() + 0.5) / (postings.size() + 0.5) + 1);

            for (PostingEntry entry : postings) {
                int docLen = docLengths.getOrDefault(entry.docId, 1);
                double tf = entry.frequency;

                // BM25 分数公式
                double score = idf * (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * docLen / avgDl));

                scores.merge(entry.docId, score, Double::sum);
            }
        }

        // 按分数降序排列
        return scores.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .limit(topK)
                .map(entry -> {
                    Document doc = documents.get(entry.getKey());
                    BM25Result result = new BM25Result();
                    result.setId(entry.getKey());
                    result.setContent(doc.content);
                    result.setMetadata(doc.metadata);
                    result.setScore(entry.getValue().floatValue());
                    return result;
                })
                .toList();
    }

    /**
     * 简单中文分词：按标点、空格、换行分割，转小写
     * 对于生产环境建议使用 HanLP 或 jieba，这里用轻量方案
     */
    private List<String> tokenize(String text) {
        if (text == null || text.isEmpty()) return List.of();

        // 转小写，按非中英文数字字符分割
        String[] parts = text.toLowerCase()
                .replaceAll("[^\\u4e00-\\u9fa5a-z0-9]+", " ")
                .trim()
                .split("\\s+");

        List<String> tokens = new ArrayList<>();
        for (String part : parts) {
            if (part.isEmpty()) continue;

            // 对英文做 n-gram（bigram）增强匹配
            if (part.matches(".*[a-z].*")) {
                tokens.add(part);  // 完整英文词
                // bigram
                for (int i = 0; i < part.length() - 1; i++) {
                    tokens.add(part.substring(i, i + 2));
                }
            }

            // 对中文做单字和双字切分
            String[] chars = part.split("");
            for (String c : chars) {
                if (c.matches("[\\u4e00-\\u9fa5]")) {
                    tokens.add(c);  // 单字
                }
            }
            // 中文双字组合
            String noAlpha = part.replaceAll("[a-z0-9]", "");
            for (int i = 0; i < noAlpha.length() - 1; i++) {
                String bigram = noAlpha.substring(i, i + 2);
                if (bigram.matches("[\\u4e00-\\u9fa5]{2}")) {
                    tokens.add(bigram);
                }
            }
        }

        return tokens;
    }

    private void recalculateAvgDocLength() {
        if (documents.isEmpty()) {
            avgDocLength = 0;
            return;
        }
        avgDocLength = docLengths.values().stream()
                .mapToInt(Integer::intValue)
                .average()
                .orElse(0);
    }

    public int getDocumentCount() {
        return documents.size();
    }

    public boolean isEnabled() {
        return bm25Enabled;
    }

    // ========== 内部数据结构 ==========

    private static class PostingEntry {
        String docId;
        int frequency;

        PostingEntry(String docId, int frequency) {
            this.docId = docId;
            this.frequency = frequency;
        }
    }

    private static class Document {
        String id;
        String content;
        String metadata;
        Map<String, Integer> termFreq;
        int tokenCount;
    }

    @Setter
    @Getter
    public static class BM25Result {
        private String id;
        private String content;
        private float score;
        private String metadata;
    }
}
