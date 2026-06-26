package org.example.service;

import com.alibaba.dashscope.aigc.generation.Generation;
import com.alibaba.dashscope.aigc.generation.GenerationParam;
import com.alibaba.dashscope.aigc.generation.GenerationResult;
import com.alibaba.dashscope.common.Message;
import com.alibaba.dashscope.common.Role;
import com.alibaba.dashscope.utils.Constants;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * 查询改写服务
 * 在检索前对用户原始问题进行改写和扩展，提升检索召回率
 *
 * 核心能力：
 * 1. Query Rewriting：将口语化、模糊的 query 改写为更精准的书面表达
 * 2. Multi-Query：将一个问题扩展为多个不同角度的查询，撒网式检索
 */
@Service
public class QueryRewriteService {

    private static final Logger logger = LoggerFactory.getLogger(QueryRewriteService.class);

    @Value("${dashscope.api.key}")
    private String apiKey;

    @Value("${rag.query-rewrite-enabled:true}")
    private boolean queryRewriteEnabled;

    @Value("${rag.multi-query-count:3}")
    private int multiQueryCount;

    @Value("${rag.model:qwen3-max}")
    private String model;

    @Autowired
    private VectorEmbeddingService embeddingService;

    private Generation generation;

    @PostConstruct
    public void init() {
        Constants.apiKey = apiKey;
        generation = new Generation();
        logger.info("查询改写服务初始化完成, enabled: {}, multiQueryCount: {}",
                queryRewriteEnabled, multiQueryCount);
    }

    /**
     * 对用户问题进行改写，返回多个查询版本
     * 始终保留原始问题在结果中
     *
     * @param query   用户原始问题
     * @param history 对话历史（可为空列表）
     * @return 改写后的查询列表（包含原始查询）
     */
    public List<String> rewriteQuery(String query, List<Message> history) {
        if (!queryRewriteEnabled) {
            return List.of(query);
        }

        try {
            // 1. 查询改写：将口语化表达转为书面精准表达
            String rewrittenQuery = rewriteWithLLM(query, history);
            logger.info("查询改写: '{}' -> '{}'", query, rewrittenQuery);

            // 2. Multi-Query 扩展：生成多个不同角度的查询
            List<String> expandedQueries = expandQueries(rewrittenQuery);

            // 3. 始终保留原始问题（防止改写丢失关键细节）
            List<String> allQueries = new ArrayList<>();
            allQueries.add(query);               // 原始问题（最高优先）
            allQueries.add(rewrittenQuery);       // 改写版本
            allQueries.addAll(expandedQueries);   // 扩展版本

            // 去重
            List<String> uniqueQueries = new ArrayList<>();
            for (String q : allQueries) {
                String trimmed = q.trim();
                if (!trimmed.isEmpty() && !uniqueQueries.contains(trimmed)) {
                    uniqueQueries.add(trimmed);
                }
            }

            logger.info("Multi-Query 生成完毕，共 {} 个查询: {}", uniqueQueries.size(), uniqueQueries);
            return uniqueQueries;

        } catch (Exception e) {
            logger.error("查询改写失败，回退使用原始查询", e);
            return List.of(query);
        }
    }

    /**
     * 用 LLM 改写用户查询
     */
    private String rewriteWithLLM(String query, List<Message> history) {
        try {
            StringBuilder historyContext = new StringBuilder();
            if (history != null && !history.isEmpty()) {
                // 取最近 4 条历史作为上下文
                int start = Math.max(0, history.size() - 4);
                for (int i = start; i < history.size(); i++) {
                    Message msg = history.get(i);
                    historyContext.append(msg.getRole()).append(": ").append(msg.getContent()).append("\n");
                }
            }

            String systemPrompt = "你是一个查询改写助手。你的任务是将用户的口语化、模糊的查询改写为更精准的书面表达，以便在知识库中更准确地检索相关信息。\n" +
                    "改写规则：\n" +
                    "1. 保持用户的核心意图不变\n" +
                    "2. 将口语化表达转为书面表达\n" +
                    "3. 补充可能缺失的关键信息\n" +
                    "4. 如果有对话历史，利用上下文消解指代（如'它'、'那个'）\n" +
                    "5. 只输出改写后的查询，不要解释";

            List<Message> messages = new ArrayList<>();
            messages.add(Message.builder().role(Role.SYSTEM.getValue()).content(systemPrompt).build());

            // 添加对话历史
            if (history != null) {
                for (Message msg : history) {
                    messages.add(msg);
                }
            }

            messages.add(Message.builder()
                    .role(Role.USER.getValue())
                    .content("请改写以下查询：" + query)
                    .build());

            GenerationParam param = GenerationParam.builder()
                    .apiKey(apiKey)
                    .model(model)
                    .resultFormat("message")
                    .messages(messages)
                    .maxTokens(100)
                    .build();

            GenerationResult result = generation.call(param);
            String rewritten = result.getOutput().getChoices().get(0).getMessage().getContent().trim();

            // 去掉可能的引号包裹
            if (rewritten.startsWith("\"") && rewritten.endsWith("\"")) {
                rewritten = rewritten.substring(1, rewritten.length() - 1);
            }
            if (rewritten.startsWith("'") && rewritten.endsWith("'")) {
                rewritten = rewritten.substring(1, rewritten.length() - 1);
            }

            return rewritten.isEmpty() ? query : rewritten;

        } catch (Exception e) {
            logger.warn("LLM 改写失败，返回原始查询: {}", e.getMessage());
            return query;
        }
    }

    /**
     * 将改写后的查询扩展为多个不同角度的查询
     */
    private List<String> expandQueries(String rewrittenQuery) {
        try {
            String systemPrompt = "你是一个查询扩展助手。给定一个查询，生成 " + multiQueryCount + " 个不同角度但语义相关的查询。\n" +
                    "每个查询占一行，不要编号，不要解释，只输出查询文本。\n" +
                    "规则：\n" +
                    "1. 每个查询从不同角度描述同一问题\n" +
                    "2. 使用不同的关键词和表达方式\n" +
                    "3. 保持与原始查询相同的领域\n" +
                    "4. 只输出 " + multiQueryCount + " 行查询文本";

            List<Message> messages = new ArrayList<>();
            messages.add(Message.builder().role(Role.SYSTEM.getValue()).content(systemPrompt).build());
            messages.add(Message.builder()
                    .role(Role.USER.getValue())
                    .content("请为以下查询生成 " + multiQueryCount + " 个不同角度的查询：" + rewrittenQuery)
                    .build());

            GenerationParam param = GenerationParam.builder()
                    .apiKey(apiKey)
                    .model(model)
                    .resultFormat("message")
                    .messages(messages)
                    .maxTokens(200)
                    .build();

            GenerationResult result = generation.call(param);
            String content = result.getOutput().getChoices().get(0).getMessage().getContent().trim();

            // 按行分割，过滤空行
            List<String> expanded = new ArrayList<>();
            for (String line : content.split("\n")) {
                String cleaned = line.replaceAll("^\\d+[.、)\\]\\s]+", "").trim();
                if (!cleaned.isEmpty() && !cleaned.equals(rewrittenQuery)) {
                    expanded.add(cleaned);
                }
            }

            return expanded;

        } catch (Exception e) {
            logger.warn("Multi-Query 扩展失败: {}", e.getMessage());
            return List.of();
        }
    }

    /**
     * 是否启用查询改写
     */
    public boolean isEnabled() {
        return queryRewriteEnabled;
    }
}
