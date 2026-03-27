# -*- coding: utf-8 -*-
"""
工程管理期刊文献分析系统（Streamlit 版本）
可部署到 Streamlit Cloud 生成永久公网链接
"""

import streamlit as st
import pandas as pd
import os
import re
import json
import time
from datetime import datetime
from pathlib import Path
import warnings
import zipfile
import shutil
import io
warnings.filterwarnings('ignore')

# 导入必要的库
try:
    from openai import OpenAI
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    import hdbscan
    import jieba
    import jieba.posseg as pseg
    import string
except ImportError as e:
    st.error(f"缺少必要的库：{e}")
    st.stop()

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="工程管理期刊文献分析系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 全局变量 ====================
OUTPUT_DIR = "./analysis_output"

# ==================== 侧边栏配置 ====================
with st.sidebar:
    st.title("⚙️ 配置面板")
    
    st.markdown("### 🔑 API 配置")
    api_key = st.text_input(
        "DeepSeek API Key",
        type="password",
        placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxx",
        help="从 https://platform.deepseek.com 获取"
    )
    
    st.markdown("### 📊 参数设置")
    num_topics = st.slider(
        "主题聚类数量",
        min_value=5,
        max_value=50,
        value=20,
        step=1
    )
    
    batch_size = st.slider(
        "API 批量大小 (篇/批)",
        min_value=1,
        max_value=20,
        value=10,
        step=1
    )
    
    enable_expert_summary = st.checkbox(
        "🎓 启用主题专家总结",
        value=True,
        help="启用后将调用 DeepSeek API 对每个主题进行专业总结"
    )
    
    st.markdown("---")
    
    # 清空按钮
    if st.button("🗑️ 清空所有数据", type="secondary"):
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        if 'df_result' in st.session_state:
            del st.session_state.df_result
        if 'topic_model' in st.session_state:
            del st.session_state.topic_model
        st.rerun()

# ==================== 工具函数 ====================
def try_read_excel(file):
    """尝试多种方式读取 Excel 文件"""
    try:
        return pd.read_excel(file, engine='openpyxl')
    except:
        try:
            return pd.read_excel(file, engine='xlrd')
        except:
            try:
                return pd.read_excel(file)
            except:
                return None

def find_column(df, target_name):
    """模糊匹配列名"""
    df_columns = [str(col).strip() for col in df.columns]
    target_lower = target_name.lower()
    
    if target_name in df_columns:
        return target_name
    
    for col in df_columns:
        col_lower = col.lower()
        if (target_lower in col_lower or 
            target_lower.replace(' ', '') in col_lower.replace(' ', '') or
            col_lower.endswith(target_lower)):
            return col
    
    return None

def process_uploaded_files(uploaded_files):
    """处理上传的 Excel 文件"""
    if not uploaded_files:
        return None, "❌ 请上传至少一个 Excel 文件"
    
    all_articles = []
    log_messages = []
    
    progress_bar = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        try:
            df = try_read_excel(file)
            
            if df is None:
                log_messages.append(f"⚠️ 无法读取文件：{file.name}")
                continue
            
            title_col = find_column(df, 'Article Title') or find_column(df, 'Title') or find_column(df, 'TI')
            year_col = find_column(df, 'Publication Year') or find_column(df, 'Year') or find_column(df, 'PY')
            abstract_col = find_column(df, 'Abstract') or find_column(df, 'AB')
            
            if not all([title_col, year_col, abstract_col]):
                log_messages.append(f"⚠️ 文件 {file.name} 缺少必要列")
                continue
            
            temp_df = df[[title_col, year_col, abstract_col]].copy()
            temp_df.columns = ['Article Title', 'Publication Year', 'Abstract']
            temp_df['Source File'] = file.name
            
            temp_df['Article Title'] = temp_df['Article Title'].astype(str).str.strip()
            temp_df['Abstract'] = temp_df['Abstract'].astype(str).str.strip()
            temp_df = temp_df.dropna(subset=['Article Title', 'Abstract'])
            temp_df['Publication Year'] = pd.to_numeric(temp_df['Publication Year'], errors='coerce')
            temp_df = temp_df.dropna(subset=['Publication Year'])
            
            temp_df = temp_df[
                (temp_df['Article Title'].str.len() > 10) & 
                (temp_df['Abstract'].str.len() > 50)
            ]
            
            all_articles.append(temp_df)
            log_messages.append(f"✅ 成功读取 {file.name}: {len(temp_df)} 篇文章")
            
        except Exception as e:
            log_messages.append(f"❌ 读取 {file.name} 时出错：{str(e)[:100]}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    progress_bar.empty()
    
    if not all_articles:
        return None, "❌ 没有成功读取任何有效数据"
    
    combined_df = pd.concat(all_articles, ignore_index=True)
    
    log_messages.append(f"\n📊 总计：{len(combined_df)} 篇文章")
    log_messages.append(f"📁 文件数量：{combined_df['Source File'].nunique()}")
    log_messages.append(f"📅 年份范围：{combined_df['Publication Year'].min():.0f} - {combined_df['Publication Year'].max():.0f}")
    
    return combined_df, "\n".join(log_messages)

# ==================== DeepSeek API 分析函数 ====================
def get_deepseek_client(api_key):
    """创建 DeepSeek 客户端"""
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

def analyze_batch_with_deepseek(batch_data, client, max_retries=3):
    """批量分析多篇文章"""
    system_prompt = """你是一位工程管理领域的资深教授。请批量分析工程管理领域的学术论文。
对于每篇论文，根据标题和摘要识别：
1. 研究方法：使用学术专有名词
2. 研究内容：建筑工程管理中针对的具体问题

请严格按照 JSON 格式返回。"""

    articles_text = ""
    for i, article in enumerate(batch_data):
        articles_text += f"\n文章 {i+1}:\n标题：{article['title']}\n摘要：{article['abstract'][:500]}...\n"
    
    user_prompt = f"请分析以下{len(batch_data)}篇论文：\n{articles_text}\n请以 JSON 数组格式返回结果。"

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=8000,
                temperature=0.1,
                stream=False
            )
            
            result = response.choices[0].message.content
            
            json_match = re.search(r'\[\s*\{.*\}\s*\]', result, re.DOTALL)
            if json_match:
                try:
                    results = json.loads(json_match.group(0))
                    if isinstance(results, list) and len(results) == len(batch_data):
                        return results
                except:
                    pass
            
            parsed_results = []
            for i in range(len(batch_data)):
                parsed_results.append({
                    "article_index": i+1,
                    "research_method": "未识别",
                    "research_content": "未识别"
                })
            return parsed_results
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return [{"article_index": i+1, "research_method": "API 失败", "research_content": "API 失败"} 
                        for i in range(len(batch_data))]

def run_deepseek_analysis(df, client, batch_size=10):
    """执行 DeepSeek 分析"""
    if df is None or len(df) == 0:
        return None, "❌ 没有数据可供分析"
    
    log_messages = []
    log_messages.append(f"🚀 开始分析 {len(df)} 篇文章")
    log_messages.append(f"📦 批量大小：{batch_size} 篇/批")
    
    df = df.copy()
    df['研究方法'] = ""
    df['研究内容'] = ""
    
    total_articles = len(df)
    num_batches = (total_articles + batch_size - 1) // batch_size
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, total_articles)
        
        batch_data = []
        for i in range(start_idx, end_idx):
            row = df.iloc[i]
            batch_data.append({
                "index": i,
                "title": row['Article Title'],
                "abstract": row['Abstract']
            })
        
        status_text.text(f"分析进度：{batch_num + 1}/{num_batches} 批次")
        results = analyze_batch_with_deepseek(batch_data, client)
        
        for result in results:
            article_idx = result.get("article_index", 1) - 1
            actual_idx = start_idx + article_idx
            if actual_idx < len(df):
                df.at[actual_idx, '研究方法'] = result.get("research_method", "未识别")
                df.at[actual_idx, '研究内容'] = result.get("research_content", "未识别")
        
        progress_bar.progress((batch_num + 1) / num_batches)
        
        if batch_num < num_batches - 1:
            time.sleep(2)
    
    progress_bar.empty()
    status_text.empty()
    
    valid_methods = df[df['研究方法'] != '未识别'].shape[0]
    valid_content = df[df['研究内容'] != '未识别'].shape[0]
    
    log_messages.append(f"\n✅ 分析完成！")
    log_messages.append(f"📊 成功识别研究方法：{valid_methods}/{len(df)} ({valid_methods/len(df)*100:.1f}%)")
    log_messages.append(f"📊 成功识别研究内容：{valid_content}/{len(df)} ({valid_content/len(df)*100:.1f}%)")
    
    return df, "\n".join(log_messages)

# ==================== BERTopic 主题挖掘函数 ====================
def preprocess_text_keep_nouns(text):
    """中文文本预处理，保留名词"""
    punctuation = set(string.punctuation)
    chinese_punctuation = "，。！？；：、""''（）《》〈〉【】[]—…·,.!?;:\"'()<>/\\|-_=+*&^%$#@~`"
    punctuation.update(set(chinese_punctuation))
    
    noun_flags = {"n", "nr", "ns", "nt", "nz", "nl", "ng"}
    
    basic_stopwords = {
        "的", "了", "和", "是", "在", "与", "及", "或", "而", "并", "等", "中", "对", "将", "把",
        "被", "为", "以", "于", "上", "下", "内", "外", "后", "前", "时", "其", "其中", "一个",
        "一种", "一些", "这", "那", "这些", "那些", "该", "各", "每", "多", "很多", "较", "进一步",
        "通过", "根据", "对于", "由于", "因此", "所以", "但是", "如果", "然后", "以及", "此外", "同时",
        "并且", "进行", "采用", "使用", "基于", "具有", "存在", "可以", "可能", "能够", "已经", "主要",
        "相关", "不同", "分别", "非常", "比较", "一定", "目前", "本文", "本研究", "本论文", "本章", "本节",
        "作者", "研究", "结果", "问题", "分析", "影响", "意义", "方面", "情况", "因素", "过程", "工作",
        "数据", "实验", "模型", "系统", "设计", "实现", "结论", "发现", "提出", "说明", "表明", "讨论",
        "总结", "指出", "应用", "特征", "机制", "水平", "能力", "功能", "部分", "基础", "目标", "背景",
        "现状", "策略", "路径", "效果", "建议", "价值", "理论", "实践", "视角", "视域", "维度"
    }
    
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    
    words = pseg.cut(text)
    filtered_words = []
    
    for word, flag in words:
        word = word.strip()
        if not word:
            continue
        if all(char in punctuation for char in word):
            continue
        if re.fullmatch(r"\d+(\.\d+)?", word):
            continue
        if len(word) == 1 and word not in {"法", "学", "史", "论"}:
            continue
        if flag not in noun_flags:
            continue
        if word in basic_stopwords:
            continue
        filtered_words.append(word)
    
    return " ".join(filtered_words)

def run_bertopic_analysis(df, num_topics=20):
    """执行 BERTopic 主题挖掘"""
    if df is None or len(df) == 0:
        return None, "❌ 没有数据可供分析", None
    
    if '研究内容' not in df.columns:
        return None, "❌ 请先执行 DeepSeek 分析", None
    
    log_messages = []
    log_messages.append("🔍 开始 BERTopic 主题挖掘")
    
    raw_docs = df['研究内容'].dropna().astype(str).tolist()
    log_messages.append(f"📄 共读取 {len(raw_docs)} 条文本")
    
    docs = [preprocess_text_keep_nouns(doc) for doc in raw_docs]
    processed_pairs = [(raw, doc) for raw, doc in zip(raw_docs, docs) if doc.strip()]
    docs = [x[1] for x in processed_pairs]
    raw_docs_cleaned = [x[0] for x in processed_pairs]
    
    log_messages.append(f"🧹 清洗后保留 {len(docs)} 条文本")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    log_messages.append("⚙️ 初始化 BERTopic 模型...")
    
    try:
        with st.spinner("正在加载 BERTopic 模型..."):
            embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        
        def whitespace_tokenizer(text):
            return text.split()
        
        vectorizer_model = CountVectorizer(
            tokenizer=whitespace_tokenizer,
            token_pattern=None,
            min_df=2
        )
        
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42
        )
        
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=10,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True
        )
        
        topic_model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=vectorizer_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            nr_topics=num_topics,
            language="multilingual",
            calculate_probabilities=True,
            verbose=True
        )
        
        log_messages.append("🚀 开始训练模型...")
        
        with st.spinner("正在训练主题模型..."):
            topics, probs = topic_model.fit_transform(docs)
        
        topic_info = topic_model.get_topic_info()
        doc_info = topic_model.get_document_info(docs)
        
        log_messages.append("💾 保存 BERTopic 结果...")
        
        topic_model.save(os.path.join(OUTPUT_DIR, "bertopic_model"))
        
        log_messages.append("📊 生成可视化...")
        
        try:
            fig_topics = topic_model.visualize_topics()
            fig_topics.write_html(os.path.join(OUTPUT_DIR, "bertopic_topics.html"))
        except:
            pass
        
        try:
            fig_barchart = topic_model.visualize_barchart(top_n_topics=num_topics)
            fig_barchart.write_html(os.path.join(OUTPUT_DIR, "bertopic_barchart.html"))
        except:
            pass
        
        log_messages.append(f"\n✅ 主题挖掘完成！")
        log_messages.append(f"📊 发现 {len(topic_info)} 个主题")
        
        log_messages.append(f"\n📈 前 10 个主题:")
        for idx, row in topic_info.head(10).iterrows():
            log_messages.append(f"   主题{row['Topic']}: {row['Name']} ({row['Count']}篇)")
        
        df_with_topics = df.copy()
        df_with_topics['主题编号'] = doc_info["Topic"].values
        df_with_topics['主题名称'] = doc_info["Name"].values
        df_with_topics['主题概率'] = doc_info["Probability"].values
        
        st.session_state.topic_model = topic_model
        st.session_state.topic_info = topic_info
        
        return df_with_topics, "\n".join(log_messages), topic_info
        
    except Exception as e:
        log_messages.append(f"❌ BERTopic 分析出错：{str(e)}")
        return None, "\n".join(log_messages), None

# ==================== 主题专家总结函数 ====================
def get_topic_keywords(topic_model, topic_id, top_n=10):
    """获取某个主题的代表性关键词"""
    try:
        topic_words, _ = topic_model.get_topic(topic_id)
        keywords = [word for word, _ in topic_words[:top_n]]
        return ", ".join(keywords)
    except:
        return "无法获取关键词"

def summarize_topic_with_deepseek(topic_id, topic_keywords, topic_name, client, max_retries=3):
    """使用 DeepSeek 对单个主题进行专家总结"""
    
    system_prompt = """你是一位工程管理领域的资深教授和学术专家。
请根据 BERTopic 主题模型生成的主题关键词，对该研究主题进行专业、深入的学术总结。

请从以下角度进行分析：
1. 主题定位
2. 研究焦点
3. 理论价值
4. 实践意义
5. 发展趋势

请用专业、严谨的学术语言进行总结，字数控制在 200-300 字之间。"""

    user_prompt = f"""请对以下工程管理研究主题进行专家总结：

主题编号：{topic_id}
主题名称：{topic_name}
主题关键词：{topic_keywords}

请作为工程管理专家，对该主题进行深入分析和总结。"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=600,
                temperature=0.3,
                stream=False
            )
            
            result = response.choices[0].message.content
            return result.strip()
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"API 调用失败：{str(e)[:50]}"

def run_topic_expert_summary(df, topic_info, client):
    """执行主题专家总结"""
    if 'topic_model' not in st.session_state:
        return None, "❌ BERTopic 模型未初始化", None
    
    if df is None or len(df) == 0:
        return None, "❌ 没有数据可供分析", None
    
    topic_model = st.session_state.topic_model
    
    log_messages = []
    log_messages.append("=" * 60)
    log_messages.append("🎓 步骤 4: 主题专家总结 (DeepSeek API)")
    log_messages.append("=" * 60)
    
    topics_to_summarize = topic_info[topic_info['Topic'] != -1].copy()
    num_topics = len(topics_to_summarize)
    
    log_messages.append(f"📊 共有 {num_topics} 个主题需要总结")
    log_messages.append(f"🤖 调用 DeepSeek API 进行专家总结...\n")
    
    topic_summaries = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in enumerate(topics_to_summarize.iterrows()):
        topic_id = row[1]['Topic']
        topic_name = row[1]['Name']
        
        status_text.text(f"主题总结进度：{idx + 1}/{num_topics}")
        
        topic_keywords = get_topic_keywords(topic_model, topic_id, top_n=15)
        
        summary = summarize_topic_with_deepseek(topic_id, topic_keywords, topic_name, client)
        
        topic_summaries.append({
            '主题编号': topic_id,
            '主题名称': topic_name,
            '主题关键词': topic_keywords,
            '文章数量': row[1]['Count'],
            '专家总结': summary
        })
        
        progress_bar.progress((idx + 1) / num_topics)
        time.sleep(1)
    
    progress_bar.empty()
    status_text.empty()
    
    topic_summary_df = pd.DataFrame(topic_summaries)
    
    log_messages.append(f"\n✅ 主题专家总结完成！")
    log_messages.append(f"📊 成功总结 {len(topic_summary_df)} 个主题")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary_path = os.path.join(OUTPUT_DIR, f"主题专家总结_{timestamp}.xlsx")
    topic_summary_df.to_excel(summary_path, index=False, encoding='utf-8')
    
    df_with_summary = df.copy()
    
    summary_map = topic_summary_df.set_index('主题编号')['专家总结'].to_dict()
    keywords_map = topic_summary_df.set_index('主题编号')['主题关键词'].to_dict()
    
    df_with_summary['主题关键词'] = df_with_summary['主题编号'].map(keywords_map)
    df_with_summary['主题专家总结'] = df_with_summary['主题编号'].map(summary_map)
    
    st.session_state.topic_summary_df = topic_summary_df
    
    return df_with_summary, "\n".join(log_messages), topic_summary_df

# ==================== 保存和打包结果函数 ====================
def create_download_package(df, topic_summary_df):
    """创建可下载的 ZIP 包"""
    if df is None or len(df) == 0:
        return None, "❌ 没有数据可保存"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    final_path = os.path.join(OUTPUT_DIR, f"工程管理期刊分析结果_完整版_{timestamp}.xlsx")
    df.to_excel(final_path, index=False)
    
    csv_path = os.path.join(OUTPUT_DIR, f"工程管理期刊分析结果_完整版_{timestamp}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    if topic_summary_df is not None and len(topic_summary_df) > 0:
        topic_summary_path = os.path.join(OUTPUT_DIR, f"主题专家总结_{timestamp}.xlsx")
        topic_summary_df.to_excel(topic_summary_path, index=False, encoding='utf-8')
    
    stats_path = os.path.join(OUTPUT_DIR, f"分析统计报告_{timestamp}.txt")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("工程管理期刊文献分析统计报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("1. 数据概况\n")
        f.write(f"   总文章数：{len(df)}\n")
        f.write(f"   来源文件数：{df['Source File'].nunique()}\n")
        f.write(f"   年份范围：{df['Publication Year'].min():.0f} - {df['Publication Year'].max():.0f}\n\n")
        
        if '研究方法' in df.columns:
            f.write("2. 研究方法分布 (前 15)\n")
            method_counts = df['研究方法'].value_counts().head(15)
            for method, count in method_counts.items():
                f.write(f"   {method}: {count} 次 ({count/len(df)*100:.1f}%)\n")
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                if file.endswith(('.xlsx', '.csv', '.txt', '.html')):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, OUTPUT_DIR)
                    zipf.write(file_path, arcname)
    
    zip_buffer.seek(0)
    
    return zip_buffer, f"工程管理期刊分析结果_完整包_{timestamp}.zip"

# ==================== 主界面 ====================
def main():
    st.title("📚 工程管理期刊文献分析系统")
    
    st.markdown("""
    ### ✨ 核心功能
    - ✅ 批量上传 WoS 导出的 Excel 文件
    - ✅ DeepSeek API 自动识别研究方法和研究内容
    - ✅ BERTopic 智能主题聚类与可视化
    - ✅ 🎓 DeepSeek 专家对每个主题进行专业总结
    - ✅ 📥 一键下载所有分析结果（ZIP 打包）
    """)
    
    st.markdown("### 📁 上传文件")
    uploaded_files = st.file_uploader(
        "上传从 WoS 下载的 Excel 文件",
        type=['xlsx', 'xls'],
        accept_multiple_files=True,
        help="支持 WoS 导出的标准格式，需包含标题、年份、摘要列"
    )
    
    if 'df_result' not in st.session_state:
        st.session_state.df_result = None
    if 'topic_model' not in st.session_state:
        st.session_state.topic_model = None
    if 'topic_summary_df' not in st.session_state:
        st.session_state.topic_summary_df = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        start_analysis = st.button("🚀 开始分析", type="primary", use_container_width=True)
    
    if start_analysis and uploaded_files and api_key:
        try:
            st.markdown("### 📊 分析进度")
            with st.expander("📁 步骤 1: 处理上传文件", expanded=True):
                df, file_log = process_uploaded_files(uploaded_files)
                st.text(file_log)
                
                if df is None:
                    st.error("文件处理失败，请检查文件格式")
                    st.stop()
                
                st.session_state.df_result = df
                st.success(f"✅ 成功加载 {len(df)} 篇文章")
            
            with st.expander("🤖 步骤 2: DeepSeek API 分析", expanded=True):
                client = get_deepseek_client(api_key)
                df_result, api_log = run_deepseek_analysis(st.session_state.df_result, client, batch_size)
                st.text(api_log)
                st.session_state.df_result = df_result
                st.success("✅ 研究方法与分析内容识别完成")
            
            with st.expander("🔍 步骤 3: BERTopic 主题挖掘", expanded=True):
                df_with_topics, topic_log, topic_info = run_bertopic_analysis(st.session_state.df_result, num_topics)
                st.text(topic_log)
                if df_with_topics is not None:
                    st.session_state.df_result = df_with_topics
                    st.success("✅ 主题聚类完成")
            
            if enable_expert_summary:
                with st.expander("🎓 步骤 4: 主题专家总结", expanded=True):
                    client = get_deepseek_client(api_key)
                    df_result, summary_log, topic_summary_df = run_topic_expert_summary(
                        st.session_state.df_result, 
                        st.session_state.topic_info, 
                        client
                    )
                    st.text(summary_log)
                    if df_result is not None:
                        st.session_state.df_result = df_result
                        st.success("✅ 主题专家总结完成")
            
            with st.expander("💾 步骤 5: 创建下载包", expanded=True):
                zip_buffer, zip_filename = create_download_package(
                    st.session_state.df_result,
                    st.session_state.topic_summary_df
                )
                
                if zip_buffer:
                    st.download_button(
                        label="📥 点击下载完整结果包 (ZIP)",
                        data=zip_buffer,
                        file_name=zip_filename,
                        mime="application/zip",
                        use_container_width=True,
                        type="primary"
                    )
                    st.session_state.analysis_complete = True
                    st.success("🎉 分析完成！请点击上方按钮下载结果")
            
        except Exception as e:
            st.error(f"❌ 分析过程中出错：{str(e)}")
            st.exception(e)
    
    elif start_analysis and not api_key:
        st.error("❌ 请先在侧边栏输入 DeepSeek API Key")
    
    elif start_analysis and not uploaded_files:
        st.error("❌ 请先上传 Excel 文件")
    
    if st.session_state.df_result is not None:
        st.markdown("### 📈 结果预览")
        
        tab1, tab2, tab3 = st.tabs(["📊 数据预览", "🎯 主题信息", "🎓 专家总结"])
        
        with tab1:
            st.dataframe(
                st.session_state.df_result[['Article Title', '研究方法', '研究内容', '主题编号', '主题名称']].head(10),
                use_container_width=True
            )
        
        with tab2:
            if 'topic_info' in st.session_state:
                st.dataframe(
                    st.session_state.topic_info[['Topic', 'Name', 'Count']].head(20),
                    use_container_width=True
                )
        
        with tab3:
            if st.session_state.topic_summary_df is not None:
                st.dataframe(
                    st.session_state.topic_summary_df[['主题编号', '主题名称', '主题关键词', '文章数量']].head(20),
                    use_container_width=True
                )
    
    with st.expander("💡 使用说明", expanded=False):
        st.markdown("""
        ### 📋 使用流程
        1. 在侧边栏输入 DeepSeek API Key
        2. 上传从 WoS 下载的 Excel 文件
        3. 设置主题聚类数量（建议 15-25）
        4. 点击"开始分析"按钮
        5. 等待分析完成
        6. 点击"下载完整结果包"按钮
        
        ### ⚠️ 注意事项
        - 请确保 DeepSeek API 密钥有效且有足够额度
        - 大量文章分析时可能需要较长时间
        - 启用主题专家总结会增加 API 调用次数（每个主题 1 次调用）
        - 首次运行需要下载 BERTopic 模型（约 500MB）
        """)

if __name__ == "__main__":
    main()
