package org.example.service;

import com.alibaba.dashscope.aigc.generation.Generation;
import com.alibaba.dashscope.aigc.generation.GenerationParam;
import com.alibaba.dashscope.aigc.generation.GenerationResult;
import com.alibaba.dashscope.common.Message;
import com.alibaba.dashscope.common.Role;
import com.alibaba.dashscope.exception.ApiException;
import com.alibaba.dashscope.exception.InputRequiredException;
import com.alibaba.dashscope.exception.NoApiKeyException;
import com.alibaba.dashscope.utils.Constants;
import io.reactivex.Flowable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * RAG (Retrieval-Augmented Generation) 服务
 * 结合向量检索和大语言模型生成答案
 */
@Service
public class RagService {

    private static final Logger logger = LoggerFactory.getLogger(RagService.class);

    @Autowired
    private VectorSearchService vectorSearchService;

    @Autowired
    private QueryRewriteService queryRewriteService;

    @Autowired
    private HybridSearchService hybridSearchService;

    @Value("${dashscope.api.key}")
    private String apiKey;

    @Value("${rag.top-k:3}")
    private int topK;

    @Value("${rag.model:qwen3-30b-a3b-thinking-2507}")
    private String model;

    private Generation generation;

    @PostConstruct
    public void init() {
        // 设置 API Key 和 Base URL
        Constants.apiKey = apiKey;
        Constants.baseHttpApiUrl = "https://dashscope.aliyuncs.com/api/v1";
        
        // 创建 Generation 实例
        generation = new Generation();
        
        logger.info("RAG 服务初始化完成，model: {}, topK: {}", model, topK);
    }

    /**
     * 流式处理用户问题（不带历史消息）
     * 
     * @param question 用户问题
     * @param callback 流式回调接口
     */
    public void queryStream(String question, StreamCallback callback) {
        queryStream(question, new ArrayList<>(), callback);
    }

    /**
     * 流式处理用户问题（带历史消息）
     * 
     * @param question 用户问题
     * @param history 历史消息列表，格式：[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
     * @param callback 流式回调接口
     */
    public void queryStream(String question, List<Map<String, String>> history, StreamCallback callback) {
        try {
            logger.info("收到 RAG 流式查询: {}", question);

            // ===== 查询优化层 =====
            // 将历史消息转为 DashScope Message 格式
            List<Message> dashscopeHistory = convertToDashScopeMessages(history);
            List<String> queries = queryRewriteService.rewriteQuery(question, dashscopeHistory);
            logger.info("查询改写完成，共 {} 个查询: {}", queries.size(), queries);

            // ===== 召回优化层 =====
            // 多查询 + 向量/BM25 双路召回 + RRF 融合
            List<HybridSearchService.HybridResult> hybridResults =
                    hybridSearchService.hybridSearch(queries, topK);

            // 发送检索结果（转换为旧格式兼容 callback）
            List<VectorSearchService.SearchResult> searchResults = new ArrayList<>();
            for (HybridSearchService.HybridResult hr : hybridResults) {
                VectorSearchService.SearchResult sr = new VectorSearchService.SearchResult();
                sr.setId(hr.getId());
                sr.setContent(hr.getContent());
                sr.setScore(hr.getVectorScore());
                sr.setMetadata(hr.getMetadata());
                searchResults.add(sr);
            }
            callback.onSearchResults(searchResults);

            if (hybridResults.isEmpty()) {
                logger.warn("未找到相关文档");
                callback.onComplete("抱歉，我在知识库中没有找到相关信息来回答您的问题。", "");
                return;
            }

            // ===== 生成优化层 =====
            // 构建增强的上下文和提示词
            String context = buildEnhancedContext(hybridResults);
            String prompt = buildEnhancedPrompt(question, context, queries);

            // 流式调用大语言模型（传入历史消息）
            generateAnswerStream(prompt, history, callback);

        } catch (Exception e) {
            logger.error("RAG 流式查询失败", e);
            callback.onError(e);
        }
    }

    /**
     * 构建上下文
     */
    private String buildContext(List<VectorSearchService.SearchResult> searchResults) {
        StringBuilder context = new StringBuilder();
        
        for (int i = 0; i < searchResults.size(); i++) {
            VectorSearchService.SearchResult result = searchResults.get(i);
            context.append("【参考资料 ").append(i + 1).append("】\n");
            context.append(result.getContent()).append("\n\n");
        }
        
        return context.toString();
    }

    /**
     * 构建提示词
     */
    private String buildPrompt(String question, String context) {
        return String.format(
            "你是一个专业的AI助手。请根据以下参考资料回答用户的问题。\n\n" +
            "参考资料：\n%s\n" +
            "用户问题：%s\n\n" +
            "请基于上述参考资料给出准确、详细的回答。如果参考资料中没有相关信息，请明确说明。",
            context, question
        );
    }

    /**
     * 构建增强的上下文（支持混合检索结果）
     * 包含匹配分数和命中路径信息
     */
    private String buildEnhancedContext(List<HybridSearchService.HybridResult> results) {
        StringBuilder context = new StringBuilder();
        for (int i = 0; i < results.size(); i++) {
            HybridSearchService.HybridResult result = results.get(i);
            context.append("【参考资料 ").append(i + 1).append("】");
            context.append("(匹配度: ").append(String.format("%.2f", result.getVectorScore()));
            context.append(", 命中路径: ").append(result.getHitCount()).append("条)\n");
            context.append(result.getContent()).append("\n\n");
        }
        return context.toString();
    }

    /**
     * 构建增强的提示词（CoT + 引用标注 + 查询改写上下文）
     */
    private String buildEnhancedPrompt(String question, String context, List<String> queries) {
        StringBuilder queryInfo = new StringBuilder();
        if (queries.size() > 1) {
            queryInfo.append("系统识别了以下相关查询角度：\n");
            for (int i = 0; i < queries.size(); i++) {
                queryInfo.append("- ").append(queries.get(i)).append("\n");
            }
            queryInfo.append("\n");
        }

        return "你是一个专业的AI助手。请严格根据以下参考资料回答用户的问题。\n\n" +
                "回答要求：\n" +
                "1. 只基于提供的参考资料回答，不要编造信息\n" +
                "2. 如果参考资料中没有相关信息，明确说明「根据现有知识库未找到相关信息」\n" +
                "3. 回答时标注引用来源，格式：[参考X]（X为参考资料编号）\n" +
                "4. 如果多个参考资料相互矛盾，优先使用匹配度更高的来源\n" +
                "5. 逐步分析问题，给出清晰、结构化的回答\n\n" +
                queryInfo +
                "参考资料：\n" + context + "\n" +
                "用户问题：" + question + "\n\n" +
                "请逐步分析并给出回答：";
    }

    /**
     * 将 Map 格式的历史消息转为 DashScope Message 格式
     */
    private List<Message> convertToDashScopeMessages(List<Map<String, String>> history) {
        List<Message> messages = new ArrayList<>();
        if (history == null) return messages;
        for (Map<String, String> msg : history) {
            String role = msg.get("role");
            String content = msg.get("content");
            if ("user".equals(role)) {
                messages.add(Message.builder().role(Role.USER.getValue()).content(content).build());
            } else if ("assistant".equals(role)) {
                messages.add(Message.builder().role(Role.ASSISTANT.getValue()).content(content).build());
            }
        }
        return messages;
    }

    /**
     * 生成答案（流式）
     * 
     * @param prompt 当前问题的提示词
     * @param history 历史消息列表
     * @param callback 流式回调接口
     */
    private void generateAnswerStream(String prompt, List<Map<String, String>> history, StreamCallback callback) 
            throws NoApiKeyException, ApiException, InputRequiredException {
        
        // 构建消息列表：历史消息 + 当前问题
        List<Message> messages = new ArrayList<>();
        
        // 添加历史消息
        for (Map<String, String> historyMsg : history) {
            String role = historyMsg.get("role");
            String content = historyMsg.get("content");
            
            if ("user".equals(role)) {
                messages.add(Message.builder()
                        .role(Role.USER.getValue())
                        .content(content)
                        .build());
            } else if ("assistant".equals(role)) {
                messages.add(Message.builder()
                        .role(Role.ASSISTANT.getValue())
                        .content(content)
                        .build());
            }
        }
        
        // 添加当前用户问题
        Message userMsg = Message.builder()
                .role(Role.USER.getValue())
                .content(prompt)
                .build();
        messages.add(userMsg);
        
        logger.debug("发送给AI模型的消息数量: {}（包含 {} 条历史消息）", 
            messages.size(), history.size());

        GenerationParam param = GenerationParam.builder()
                .apiKey(apiKey)
                .model(model)
                .incrementalOutput(true)
                .resultFormat("message")
                .messages(messages)
                .build();

        logger.info("开始调用AI模型流式接口...");
        
        Flowable<GenerationResult> result = generation.streamCall(param);
        
        StringBuilder reasoningContent = new StringBuilder();
        StringBuilder finalContent = new StringBuilder();
        
        logger.info("开始接收AI模型流式响应...");

        result.blockingForEach(message -> {
            if (message.getOutput() != null && 
                message.getOutput().getChoices() != null && 
                !message.getOutput().getChoices().isEmpty()) {
                
                // 获取消息内容
                // 注意：qwen3-30b-a3b-thinking-2507 模型会在 content 中返回完整内容
                // reasoning 部分可能需要通过特殊方式提取或者直接包含在 content 中
                String content = message.getOutput().getChoices().get(0).getMessage().getContent();

                if (content != null && !content.isEmpty()) {
                    logger.debug("收到AI模型内容块: {}", content);
                    
                    // 对于thinking 模型，content 可能包含思考过程和最终答案
                    // 这里我们将所有内容都作为答案返回
                    finalContent.append(content);
                    callback.onContentChunk(content);
                    
                    logger.debug("已调用 onContentChunk 回调");
                } else {
                    logger.debug("收到空内容块，跳过");
                }
            }
        });
        
        logger.info("AI模型流式响应完成，总内容长度: {}", finalContent.length());

        callback.onComplete(finalContent.toString(), reasoningContent.toString());
        logger.info("已调用 onComplete 回调");
    }

    /**
     * 流式回调接口
     */
    public interface StreamCallback {
        void onSearchResults(List<VectorSearchService.SearchResult> results);
        void onReasoningChunk(String chunk);
        void onContentChunk(String chunk);
        void onComplete(String fullContent, String fullReasoning);
        void onError(Exception e);
    }
}
