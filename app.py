# -*- coding: utf-8 -*-
"""
工程管理期刊文献分析系统（Streamlit 版本）- DeepSeek 纯 API 版
移除 BERTopic，全部使用 DeepSeek API 进行分析
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
from collections import Counter
warnings.filterwarnings('ignore')

# 导入必要的库
try:
    from openai import OpenAI
    import jieba
    import plotly.express as px
    import plotly.graph_objects as go
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

# ==================== 初始化 session_state ====================
def init_session_state():
    """初始化所有 session_state 变量"""
    if 'df_result' not in st.session_state:
        st.session_state.df_result = None
    if 'method_summary_df' not in st.session_state:
        st.session_state.method_summary_df = None
    if 'content_summary_df' not in st.session_state:
        st.session_state.content_summary_df = None
    if 'method_evolution_df' not in st.session_state:
        st.session_state.method_evolution_df = None
    if 'content_evolution_df' not in st.session_state:
        st.session_state.content_evolution_df = None
    if 'method_expert_summary' not in st.session_state:
        st.session_state.method_expert_summary = None
    if 'content_expert_summary' not in st.session_state:
        st.session_state.content_expert_summary = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

# ==================== 侧边栏配置 ====================
with st.sidebar:
    st.title("⚙️ 配置面板")
    
    st.markdown("### 🔑 API 配置")
    api_key_input = st.text_input(
        "DeepSeek API Key",
        type="password",
        placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxx",
        help="从 https://platform.deepseek.com 获取",
        key="api_key_sidebar"
    )
    
    st.markdown("### 📊 参数设置")
    batch_size_input = st.slider(
        "API 批量大小 (篇/批)",
        min_value=1,
        max_value=20,
        value=10,
        step=1,
        key="batch_size_sidebar"
    )
    
    enable_content_summary = st.checkbox(
        "📝 启用研究内容总结",
        value=True,
        help="启用后将调用 DeepSeek API 对研究内容进行分类总结",
        key="enable_content_sidebar"
    )
    
    enable_evolution_analysis = st.checkbox(
        "📈 启用时间演变分析",
        value=True,
        help="启用后将分析研究方法和研究内容的时间演变趋势",
        key="enable_evolution_sidebar"
    )
    
    st.markdown("---")
    
    # 清空按钮
    if st.button("🗑️ 清空所有数据", type="secondary"):
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        init_session_state()
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

# ==================== DeepSeek API 函数 ====================
def get_deepseek_client(api_key):
    """创建 DeepSeek 客户端"""
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

def analyze_articles_batch(batch_data, client, max_retries=3):
    """批量分析文章的研究方法和研究内容"""
    system_prompt = """你是一位工程管理领域的资深教授。请批量分析工程管理领域的学术论文。
对于每篇论文，根据标题和摘要识别：
1. 研究方法：使用学术专有名词（如：案例分析、问卷调查、文献综述、实证研究、建模与仿真、实验研究、结构方程模型、回归分析等）
2. 研究内容：建筑工程管理中针对的具体问题（如：成本控制、进度管理、风险管理、质量管理、供应链管理等）

请严格按照 JSON 格式返回：
[
  {"article_index": 1, "research_method": "方法", "research_content": "内容"},
  ...
]"""

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

def run_article_analysis(df, client, batch_size=10):
    """执行文章分析"""
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
        results = analyze_articles_batch(batch_data, client)
        
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

# ==================== ⭐ 研究内容分类总结函数 ====================
def summarize_content_categories(df, client):
    """使用 DeepSeek 对研究内容进行分类总结"""
    if df is None or len(df) == 0:
        return None, "❌ 没有数据可供分析", None
    
    if '研究内容' not in df.columns:
        return None, "❌ 请先完成文章分析", None
    
    log_messages = []
    log_messages.append("=" * 60)
    log_messages.append("📝 步骤 2: 研究内容分类总结 (DeepSeek API)")
    log_messages.append("=" * 60)
    
    # 收集所有研究内容
    all_contents = df['研究内容'].dropna().unique().tolist()
    all_contents = [c for c in all_contents if c and c != '未识别' and c != 'API 失败']
    
    if len(all_contents) == 0:
        return None, "❌ 没有有效的研究内容数据", None
    
    log_messages.append(f"📊 共有 {len(all_contents)} 条独立的研究内容")
    log_messages.append(f"🤖 调用 DeepSeek API 进行分类总结...\n")
    
    # 分批处理（每批 20 条内容）
    batch_size = 20
    num_batches = (len(all_contents) + batch_size - 1) // batch_size
    
    all_categories = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(all_contents))
        batch_contents = all_contents[start_idx:end_idx]
        
        status_text.text(f"分类进度：{batch_num + 1}/{num_batches} 批次")
        
        # 构建提示词
        system_prompt = """你是一位工程管理领域的资深教授和学术专家。
请对以下研究内容进行学术分类和总结。

请将研究内容归类到以下主要类别中（可补充）：
1. 成本管理
2. 进度管理
3. 质量管理
4. 风险管理
5. 安全管理
6. 供应链管理
7. 可持续发展/绿色建筑
8. 数字化/BIM/智能建造
9. 合同与法律
10. 组织与人力资源
11. 创新管理
12. 其他

请按照以下 JSON 格式返回：
[
  {"content": "原始研究内容", "category": "分类", "keywords": "关键词 1, 关键词 2, ..."},
  ...
]"""
        
        contents_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(batch_contents)])
        user_prompt = f"请对以下{len(batch_contents)}条研究内容进行分类：\n\n{contents_text}"
        
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4000,
                temperature=0.1,
                stream=False
            )
            
            result = response.choices[0].message.content
            
            # 解析 JSON
            json_match = re.search(r'\[\s*\{.*\}\s*\]', result, re.DOTALL)
            if json_match:
                try:
                    batch_results = json.loads(json_match.group(0))
                    all_categories.extend(batch_results)
                except:
                    # 解析失败，手动处理
                    for content in batch_contents:
                        all_categories.append({
                            "content": content,
                            "category": "其他",
                            "keywords": ""
                        })
            else:
                for content in batch_contents:
                    all_categories.append({
                        "content": content,
                        "category": "其他",
                        "keywords": ""
                    })
        
        except Exception as e:
            for content in batch_contents:
                all_categories.append({
                    "content": content,
                    "category": "其他",
                    "keywords": ""
                })
        
        progress_bar.progress((batch_num + 1) / num_batches)
        time.sleep(2)
    
    progress_bar.empty()
    status_text.empty()
    
    # 创建分类总结 DataFrame
    category_df = pd.DataFrame(all_categories)
    
    # 统计各分类的文章数量
    category_stats = []
    for category in category_df['category'].unique():
        cat_contents = category_df[category_df['category'] == category]['content'].tolist()
        article_count = df[df['研究内容'].isin(cat_contents)].shape[0]
        category_stats.append({
            '分类': category,
            '研究内容数量': len(cat_contents),
            '文章数量': article_count,
            '占比': f"{article_count/len(df)*100:.1f}%"
        })
    
    category_stats_df = pd.DataFrame(category_stats)
    category_stats_df = category_stats_df.sort_values('文章数量', ascending=False)
    
    log_messages.append(f"\n✅ 研究内容分类完成！")
    log_messages.append(f"📊 共识别 {len(category_stats_df)} 个研究类别")
    
    log_messages.append(f"\n📈 主要研究类别分布:")
    for idx, row in category_stats_df.head(10).iterrows():
        log_messages.append(f"   {row['分类']}: {row['文章数量']}篇 ({row['占比']})")
    
    # 保存到 session_state
    st.session_state.content_summary_df = category_stats_df
    st.session_state.content_detail_df = category_df
    
    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    category_stats_df.to_excel(os.path.join(OUTPUT_DIR, f"研究内容分类统计_{timestamp}.xlsx"), index=False)
    category_df.to_excel(os.path.join(OUTPUT_DIR, f"研究内容分类详情_{timestamp}.xlsx"), index=False)
    
    return category_stats_df, "\n".join(log_messages), category_df

# ==================== ⭐ 研究方法总结函数 ====================
def summarize_method_categories(df, client):
    """使用 DeepSeek 对研究方法进行分类总结"""
    if df is None or len(df) == 0:
        return None, "❌ 没有数据可供分析", None
    
    if '研究方法' not in df.columns:
        return None, "❌ 请先完成文章分析", None
    
    log_messages = []
    log_messages.append("=" * 60)
    log_messages.append("🎯 步骤 3: 研究方法分类总结 (DeepSeek API)")
    log_messages.append("=" * 60)
    
    # 收集所有研究方法
    all_methods = []
    for methods in df['研究方法'].dropna():
        if methods and methods != '未识别' and methods != 'API 失败':
            method_list = re.split(r'[,,;,]', methods)
            all_methods.extend([m.strip() for m in method_list if m.strip()])
    
    if len(all_methods) == 0:
        return None, "❌ 没有有效的研究方法数据", None
    
    unique_methods = list(set(all_methods))
    log_messages.append(f"📊 共有 {len(unique_methods)} 种独立的研究方法")
    log_messages.append(f"📊 总计 {len(all_methods)} 次方法使用")
    log_messages.append(f"🤖 调用 DeepSeek API 进行分类总结...\n")
    
    # 统计每种方法的使用频率
    method_counts = Counter(all_methods)
    
    # 构建方法分类提示词
    system_prompt = """你是一位工程管理领域的资深教授和学术专家。
请对以下研究方法进行学术分类。

请将研究方法归类到以下主要类别中：
1. 定性研究（案例分析、访谈、观察法、扎根理论等）
2. 定量研究（问卷调查、回归分析、结构方程模型等）
3. 混合方法（定性 + 定量结合）
4. 文献研究（文献综述、元分析、系统综述等）
5. 建模与仿真（系统动力学、Agent-based、离散事件仿真等）
6. 实验研究（实验室实验、现场实验、准实验等）
7. 其他

请按照以下 JSON 格式返回：
[
  {"method": "方法名称", "category": "分类", "description": "方法简述"},
  ...
]"""
    
    methods_text = "\n".join([f"{i+1}. {m}" for i, m in enumerate(unique_methods[:50])])  # 限制 50 种
    user_prompt = f"请对以下研究方法进行分类：\n\n{methods_text}"
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=4000,
            temperature=0.1,
            stream=False
        )
        
        result = response.choices[0].message.content
        
        # 解析 JSON
        json_match = re.search(r'\[\s*\{.*\}\s*\]', result, re.DOTALL)
        if json_match:
            try:
                method_categories = json.loads(json_match.group(0))
            except:
                method_categories = []
        else:
            method_categories = []
        
        # 创建方法分类 DataFrame
        method_summary_data = []
        for method, count in method_counts.most_common(30):  # 前 30 种方法
            category = "其他"
            description = ""
            for mc in method_categories:
                if mc.get('method') == method:
                    category = mc.get('category', '其他')
                    description = mc.get('description', '')
                    break
            
            method_summary_data.append({
                '研究方法': method,
                '使用次数': count,
                '占比': f"{count/len(all_methods)*100:.1f}%",
                '方法类别': category,
                '方法描述': description
            })
        
        method_summary_df = pd.DataFrame(method_summary_data)
        
        # 按类别统计
        category_summary = method_summary_df.groupby('方法类别').agg({
            '使用次数': 'sum',
            '研究方法': 'count'
        }).reset_index()
        category_summary.columns = ['方法类别', '总使用次数', '方法数量']
        category_summary = category_summary.sort_values('总使用次数', ascending=False)
        
        log_messages.append(f"\n✅ 研究方法分类完成！")
        log_messages.append(f"📊 共识别 {len(category_summary)} 个方法类别")
        
        log_messages.append(f"\n📈 主要方法类别分布:")
        for idx, row in category_summary.iterrows():
            log_messages.append(f"   {row['方法类别']}: {row['总使用次数']}次 ({row['方法数量']}种方法)")
        
        # 保存到 session_state
        st.session_state.method_summary_df = method_summary_df
        st.session_state.method_category_df = category_summary
        
        # 保存结果
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        method_summary_df.to_excel(os.path.join(OUTPUT_DIR, f"研究方法分类统计_{timestamp}.xlsx"), index=False)
        category_summary.to_excel(os.path.join(OUTPUT_DIR, f"研究方法类别汇总_{timestamp}.xlsx"), index=False)
        
        return method_summary_df, "\n".join(log_messages), category_summary
        
    except Exception as e:
        log_messages.append(f"❌ 分类失败：{str(e)}")
        
        # 返回基础统计
        method_summary_data = []
        for method, count in method_counts.most_common(30):
            method_summary_data.append({
                '研究方法': method,
                '使用次数': count,
                '占比': f"{count/len(all_methods)*100:.1f}%",
                '方法类别': '未分类',
                '方法描述': ''
            })
        
        method_summary_df = pd.DataFrame(method_summary_data)
        st.session_state.method_summary_df = method_summary_df
        
        return method_summary_df, "\n".join(log_messages), None

# ==================== ⭐ 时间演变分析函数 ====================
def extract_top_methods(df, top_n=10):
    """提取出现频率最高的研究方法"""
    if '研究方法' not in df.columns:
        return []
    
    all_methods = []
    for methods in df['研究方法'].dropna():
        if methods and methods != '未识别' and methods != 'API 失败':
            method_list = re.split(r'[,,;,]', methods)
            all_methods.extend([m.strip() for m in method_list if m.strip()])
    
    method_counts = Counter(all_methods)
    top_methods = [method for method, count in method_counts.most_common(top_n)]
    
    return top_methods

def extract_keywords_from_content(df, top_n=20):
    """从研究内容中提取高频关键词"""
    if '研究内容' not in df.columns:
        return []
    
    all_keywords = []
    for content in df['研究内容'].dropna():
        if content and content != '未识别':
            words = jieba.cut(content)
            for word in words:
                word = word.strip()
                if len(word) >= 2:
                    all_keywords.append(word)
    
    keyword_counts = Counter(all_keywords)
    top_keywords = [keyword for keyword, count in keyword_counts.most_common(top_n)]
    
    return top_keywords

def summarize_method_evolution(df, client, top_methods):
    """使用 DeepSeek 对研究方法演变进行专家总结"""
    
    evolution_data = []
    for method in top_methods:
        yearly_counts = df[df['研究方法'].str.contains(method, na=False)].groupby('Publication Year').size()
        for year, count in yearly_counts.items():
            evolution_data.append({
                '年份': int(year),
                '研究方法': method,
                '数量': count
            })
    
    if not evolution_data:
        return "数据不足，无法生成总结", None
    
    system_prompt = """你是一位工程管理领域的资深教授和学术专家。
请根据研究方法的时间分布数据，分析该领域研究方法的演变趋势。

请从以下角度进行分析：
1. 总体演变趋势：哪些方法在兴起，哪些在衰退
2. 关键转折点：哪一年出现了明显的方法论转变
3. 原因分析：为什么会出现这样的演变
4. 未来预测：未来 3-5 年可能的主流研究方法
5. 建议：对研究者的方法论选择建议

请用专业、严谨的学术语言进行总结，字数控制在 300-400 字之间。"""

    data_summary = "研究方法时间分布数据：\n"
    for method in top_methods[:5]:
        method_data = [d for d in evolution_data if d['研究方法'] == method]
        data_summary += f"\n{method}:\n"
        for d in sorted(method_data, key=lambda x: x['年份']):
            data_summary += f"  {d['年份']}年：{d['数量']}篇\n"
    
    user_prompt = f"""请分析以下工程管理领域研究方法的演变数据：

{data_summary}

请作为工程管理专家，对研究方法的演变趋势进行深入分析和总结。"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=800,
            temperature=0.3,
            stream=False
        )
        
        result = response.choices[0].message.content
        return result.strip(), pd.DataFrame(evolution_data)
        
    except Exception as e:
        return f"API 调用失败：{str(e)[:100]}", pd.DataFrame(evolution_data) if evolution_data else None

def summarize_content_evolution(df, client, top_keywords):
    """使用 DeepSeek 对研究内容演变进行专家总结"""
    
    evolution_data = []
    for keyword in top_keywords:
        yearly_counts = df[df['研究内容'].str.contains(keyword, na=False)].groupby('Publication Year').size()
        for year, count in yearly_counts.items():
            evolution_data.append({
                '年份': int(year),
                '关键词': keyword,
                '数量': count
            })
    
    if not evolution_data:
        return "数据不足，无法生成总结", None
    
    system_prompt = """你是一位工程管理领域的资深教授和学术专家。
请根据研究内容关键词的时间分布数据，分析该领域研究热点的演变趋势。

请从以下角度进行分析：
1. 研究热点演变：哪些主题在兴起，哪些在衰退
2. 关键转折点：哪一年出现了明显的研究焦点转变
3. 驱动因素：政策、技术、社会需求等如何影响研究热点
4. 新兴方向：正在兴起的新研究主题
5. 未来展望：未来 3-5 年的研究趋势预测

请用专业、严谨的学术语言进行总结，字数控制在 300-400 字之间。"""

    data_summary = "研究内容关键词时间分布数据：\n"
    for keyword in top_keywords[:5]:
        keyword_data = [d for d in evolution_data if d['关键词'] == keyword]
        data_summary += f"\n{keyword}:\n"
        for d in sorted(keyword_data, key=lambda x: x['年份']):
            data_summary += f"  {d['年份']}年：{d['数量']}篇\n"
    
    user_prompt = f"""请分析以下工程管理领域研究内容的演变数据：

{data_summary}

请作为工程管理专家，对研究内容的演变趋势进行深入分析和总结。"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=800,
            temperature=0.3,
            stream=False
        )
        
        result = response.choices[0].message.content
        return result.strip(), pd.DataFrame(evolution_data)
        
    except Exception as e:
        return f"API 调用失败：{str(e)[:100]}", pd.DataFrame(evolution_data) if evolution_data else None

def create_method_evolution_chart(evolution_df, top_methods):
    """创建研究方法演变可视化图表"""
    if evolution_df is None or len(evolution_df) == 0:
        return None
    
    pivot_df = evolution_df.pivot_table(
        index='年份',
        columns='研究方法',
        values='数量',
        aggfunc='sum',
        fill_value=0
    )
    
    if len(top_methods) > 8:
        top_methods = top_methods[:8]
        pivot_df = pivot_df[top_methods]
    
    fig = go.Figure()
    
    for method in pivot_df.columns:
        fig.add_trace(go.Scatter(
            x=pivot_df.index,
            y=pivot_df[method],
            mode='lines+markers',
            name=method,
            line=dict(width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title='📈 工程管理研究方法时间演变趋势',
        xaxis_title='年份',
        yaxis_title='文章数量',
        hovermode='x unified',
        legend_title='研究方法',
        template='plotly_white',
        height=600,
        font=dict(size=12)
    )
    
    return fig

def create_content_evolution_chart(evolution_df, top_keywords):
    """创建研究内容演变可视化图表"""
    if evolution_df is None or len(evolution_df) == 0:
        return None
    
    pivot_df = evolution_df.pivot_table(
        index='年份',
        columns='关键词',
        values='数量',
        aggfunc='sum',
        fill_value=0
    )
    
    if len(top_keywords) > 8:
        top_keywords = top_keywords[:8]
        pivot_df = pivot_df[top_keywords]
    
    fig = go.Figure()
    
    for keyword in pivot_df.columns:
        fig.add_trace(go.Scatter(
            x=pivot_df.index,
            y=pivot_df[keyword],
            mode='lines+markers',
            name=keyword,
            line=dict(width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title='📝 工程管理研究内容热点演变趋势',
        xaxis_title='年份',
        yaxis_title='文章数量',
        hovermode='x unified',
        legend_title='研究热点',
        template='plotly_white',
        height=600,
        font=dict(size=12)
    )
    
    return fig

def run_evolution_analysis(df, client):
    """执行时间演变分析"""
    if df is None or len(df) == 0:
        return None, None, "❌ 没有数据可供分析", None, None
    
    if 'Publication Year' not in df.columns:
        return None, None, "❌ 缺少年份数据", None, None
    
    log_messages = []
    log_messages.append("=" * 60)
    log_messages.append("📈 步骤 4: 时间维度演变分析 (DeepSeek API)")
    log_messages.append("=" * 60)
    
    log_messages.append("🔍 提取高频研究方法和关键词...")
    top_methods = extract_top_methods(df, top_n=10)
    top_keywords = extract_keywords_from_content(df, top_n=20)
    
    log_messages.append(f"   识别到 {len(top_methods)} 个主要研究方法")
    log_messages.append(f"   识别到 {len(top_keywords)} 个研究热点关键词")
    
    log_messages.append("\n🎯 分析研究方法演变趋势...")
    method_summary, method_evolution_df = summarize_method_evolution(df, client, top_methods)
    
    if method_evolution_df is not None and len(method_evolution_df) > 0:
        log_messages.append(f"   ✅ 生成 {len(method_evolution_df)} 条研究方法演变数据")
        st.session_state.method_evolution_df = method_evolution_df
        st.session_state.method_expert_summary = method_summary
    
    log_messages.append("\n📝 分析研究内容演变趋势...")
    content_summary, content_evolution_df = summarize_content_evolution(df, client, top_keywords)
    
    if content_evolution_df is not None and len(content_evolution_df) > 0:
        log_messages.append(f"   ✅ 生成 {len(content_evolution_df)} 条研究内容演变数据")
        st.session_state.content_evolution_df = content_evolution_df
        st.session_state.content_expert_summary = content_summary
    
    log_messages.append(f"\n✅ 时间演变分析完成！")
    
    log_messages.append("\n📊 生成可视化图表...")
    method_chart = create_method_evolution_chart(method_evolution_df, top_methods)
    content_chart = create_content_evolution_chart(content_evolution_df, top_keywords)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if method_chart:
        method_chart.write_html(os.path.join(OUTPUT_DIR, f"研究方法演变趋势_{timestamp}.html"))
    if content_chart:
        content_chart.write_html(os.path.join(OUTPUT_DIR, f"研究内容演变趋势_{timestamp}.html"))
    
    return method_summary, content_summary, "\n".join(log_messages), method_chart, content_chart

# ==================== 保存和打包结果函数 ====================
def create_download_package(df, method_summary_df, content_summary_df, method_evolution_df, content_evolution_df):
    """创建可下载的 ZIP 包"""
    if df is None or len(df) == 0:
        return None, "❌ 没有数据可保存"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存完整结果
    final_path = os.path.join(OUTPUT_DIR, f"工程管理期刊分析结果_完整版_{timestamp}.xlsx")
    df.to_excel(final_path, index=False)
    
    csv_path = os.path.join(OUTPUT_DIR, f"工程管理期刊分析结果_完整版_{timestamp}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    # 保存方法总结
    if method_summary_df is not None and len(method_summary_df) > 0:
        method_summary_df.to_excel(os.path.join(OUTPUT_DIR, f"研究方法分类统计_{timestamp}.xlsx"), index=False)
    
    # 保存内容总结
    if content_summary_df is not None and len(content_summary_df) > 0:
        content_summary_df.to_excel(os.path.join(OUTPUT_DIR, f"研究内容分类统计_{timestamp}.xlsx"), index=False)
    
    # 保存演变数据
    if method_evolution_df is not None and len(method_evolution_df) > 0:
        method_evolution_df.to_excel(os.path.join(OUTPUT_DIR, f"研究方法演变数据_{timestamp}.xlsx"), index=False)
    
    if content_evolution_df is not None and len(content_evolution_df) > 0:
        content_evolution_df.to_excel(os.path.join(OUTPUT_DIR, f"研究内容演变数据_{timestamp}.xlsx"), index=False)
    
    # 保存统计报告
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
            f.write("\n")
        
        if method_summary_df is not None and len(method_summary_df) > 0:
            f.write("3. 研究方法类别\n")
            if '方法类别' in method_summary_df.columns:
                category_counts = method_summary_df.groupby('方法类别')['使用次数'].sum()
                for category, count in category_counts.items():
                    f.write(f"   {category}: {count} 次\n")
            f.write("\n")
        
        if content_summary_df is not None and len(content_summary_df) > 0:
            f.write("4. 研究内容类别\n")
            for idx, row in content_summary_df.head(10).iterrows():
                f.write(f"   {row['分类']}: {row['文章数量']}篇 ({row['占比']})\n")
            f.write("\n")
        
        if method_evolution_df is not None and len(method_evolution_df) > 0:
            f.write("5. 研究方法演变趋势\n")
            years = sorted(method_evolution_df['年份'].unique())
            f.write(f"   时间跨度：{min(years)} - {max(years)}\n")
            f.write(f"   数据点数：{len(method_evolution_df)}\n\n")
        
        if content_evolution_df is not None and len(content_evolution_df) > 0:
            f.write("6. 研究内容演变趋势\n")
            years = sorted(content_evolution_df['年份'].unique())
            f.write(f"   时间跨度：{min(years)} - {max(years)}\n")
            f.write(f"   数据点数：{len(content_evolution_df)}\n\n")
    
    # 打包为 ZIP
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
    # 初始化 session_state
    init_session_state()
    
    st.title("📚 工程管理期刊文献分析系统")
    
    st.markdown("""
    ### ✨ 核心功能
    - ✅ 批量上传 WoS 导出的 Excel 文件
    - ✅ DeepSeek API 自动识别研究方法和研究内容
    - ✅ 📝 DeepSeek 专家对研究内容进行分类总结
    - ✅ 🎯 DeepSeek 专家对研究方法进行分类总结
    - ✅ 📈 研究方法和研究内容时间演变分析
    - ✅ 📥 一键下载所有分析结果（ZIP 打包）
    
    ### 🚀 优势
    - ⚡ **更快速**：无需下载大型模型，启动即用
    - 💾 **更轻量**：移除 BERTopic，减少依赖
    - 🎯 **更智能**：全部使用 DeepSeek API 进行专家级分析
    """)
    
    # 从侧边栏获取配置
    api_key = st.session_state.api_key_sidebar if hasattr(st.session_state, 'api_key_sidebar') else None
    batch_size = st.session_state.batch_size_sidebar if hasattr(st.session_state, 'batch_size_sidebar') else 10
    enable_content_summary = st.session_state.enable_content_sidebar if hasattr(st.session_state, 'enable_content_sidebar') else True
    enable_evolution_analysis = st.session_state.enable_evolution_sidebar if hasattr(st.session_state, 'enable_evolution_sidebar') else True
    
    st.markdown("### 📁 上传文件")
    uploaded_files = st.file_uploader(
        "上传从 WoS 下载的 Excel 文件",
        type=['xlsx', 'xls'],
        accept_multiple_files=True,
        help="支持 WoS 导出的标准格式，需包含标题、年份、摘要列",
        key="file_uploader_main"
    )
    
    # 开始分析按钮
    col1, col2 = st.columns([3, 1])
    with col1:
        start_analysis = st.button("🚀 开始分析", type="primary", use_container_width=True, key="start_btn")
    
    # 执行分析
    if start_analysis and uploaded_files and api_key:
        try:
            st.markdown("### 📊 分析进度")
            
            # 步骤 1: 处理上传文件
            with st.expander("📁 步骤 1: 处理上传文件", expanded=True):
                df, file_log = process_uploaded_files(uploaded_files)
                st.text(file_log)
                
                if df is None:
                    st.error("文件处理失败，请检查文件格式")
                    st.stop()
                
                st.session_state.df_result = df
                st.success(f"✅ 成功加载 {len(df)} 篇文章")
            
            # 步骤 2: DeepSeek API 分析文章
            with st.expander("🤖 步骤 2: DeepSeek API 分析文章", expanded=True):
                client = get_deepseek_client(api_key)
                df_result, api_log = run_article_analysis(st.session_state.df_result, client, batch_size)
                st.text(api_log)
                st.session_state.df_result = df_result
                st.success("✅ 研究方法与分析内容识别完成")
            
            # 步骤 3: 研究方法分类总结
            with st.expander("🎯 步骤 3: 研究方法分类总结", expanded=True):
                client = get_deepseek_client(api_key)
                method_summary, method_log, method_category = summarize_method_categories(
                    st.session_state.df_result, 
                    client
                )
                st.text(method_log)
                if method_summary is not None:
                    st.success("✅ 研究方法分类完成")
            
            # 步骤 4: 研究内容分类总结
            if enable_content_summary:
                with st.expander("📝 步骤 4: 研究内容分类总结", expanded=True):
                    client = get_deepseek_client(api_key)
                    content_summary, content_log, content_detail = summarize_content_categories(
                        st.session_state.df_result, 
                        client
                    )
                    st.text(content_log)
                    if content_summary is not None:
                        # 显示分类统计
                        st.markdown("#### 📊 研究内容分类统计")
                        st.dataframe(content_summary.head(10), use_container_width=True)
                        st.success("✅ 研究内容分类完成")
            
            # 步骤 5: 时间演变分析
            if enable_evolution_analysis:
                with st.expander("📈 步骤 5: 时间维度演变分析", expanded=True):
                    client = get_deepseek_client(api_key)
                    method_evo_summary, content_evo_summary, evo_log, method_chart, content_chart = run_evolution_analysis(
                        st.session_state.df_result,
                        client
                    )
                    st.text(evo_log)
                    
                    if method_evo_summary and content_evo_summary:
                        st.markdown("#### 🎯 研究方法演变专家总结")
                        st.info(method_evo_summary)
                        
                        st.markdown("#### 📝 研究内容演变专家总结")
                        st.info(content_evo_summary)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if method_chart:
                                st.plotly_chart(method_chart, use_container_width=True)
                        with col2:
                            if content_chart:
                                st.plotly_chart(content_chart, use_container_width=True)
                        
                        st.success("✅ 时间演变分析完成")
            
            # 步骤 6: 创建下载包
            with st.expander("💾 步骤 6: 创建下载包", expanded=True):
                zip_buffer, zip_filename = create_download_package(
                    st.session_state.df_result,
                    st.session_state.method_summary_df,
                    st.session_state.content_summary_df,
                    st.session_state.method_evolution_df,
                    st.session_state.content_evolution_df
                )
                
                if zip_buffer:
                    st.download_button(
                        label="📥 点击下载完整结果包 (ZIP)",
                        data=zip_buffer,
                        file_name=zip_filename,
                        mime="application/zip",
                        use_container_width=True,
                        type="primary",
                        key="download_btn"
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
    
    # 结果显示
    if st.session_state.df_result is not None and len(st.session_state.df_result) > 0:
        st.markdown("### 📈 结果预览")
        
        required_columns = ['Article Title', '研究方法', '研究内容']
        existing_columns = [col for col in required_columns if col in st.session_state.df_result.columns]
        
        # 选项卡展示
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 数据预览",
            "🎯 方法统计",
            "📝 内容分类",
            "📈 方法演变",
            "📝 内容演变",
            "📥 下载"
        ])
        
        with tab1:
            if len(existing_columns) > 0:
                st.dataframe(
                    st.session_state.df_result[existing_columns].head(10),
                    use_container_width=True
                )
        
        with tab2:
            if st.session_state.method_summary_df is not None:
                st.dataframe(
                    st.session_state.method_summary_df.head(20),
                    use_container_width=True
                )
            else:
                st.warning("研究方法统计尚未生成")
        
        with tab3:
            if st.session_state.content_summary_df is not None:
                st.dataframe(
                    st.session_state.content_summary_df.head(20),
                    use_container_width=True
                )
            else:
                st.warning("研究内容分类尚未生成")
        
        with tab4:
            if st.session_state.method_evolution_df is not None:
                st.dataframe(
                    st.session_state.method_evolution_df.head(20),
                    use_container_width=True
                )
            else:
                st.warning("研究方法演变数据尚未生成")
        
        with tab5:
            if st.session_state.content_evolution_df is not None:
                st.dataframe(
                    st.session_state.content_evolution_df.head(20),
                    use_container_width=True
                )
            else:
                st.warning("研究内容演变数据尚未生成")
        
        with tab6:
            if st.session_state.analysis_complete:
                st.success("✅ 分析已完成，请在上方下载按钮处下载结果")
            else:
                st.info("请先完成分析流程")
    
    # 使用说明
    with st.expander("💡 使用说明", expanded=False):
        st.markdown("""
        ### 📋 使用流程
        1. 在侧边栏输入 DeepSeek API Key
        2. 上传从 WoS 下载的 Excel 文件
        3. 选择是否启用研究内容总结和时间演变分析
        4. 点击"开始分析"按钮
        5. 等待分析完成
        6. 点击"下载完整结果包"按钮
        
        ### ⚠️ 注意事项
        - 请确保 DeepSeek API 密钥有效且有足够额度
        - 大量文章分析时可能需要较长时间
        - 启用研究内容总结会增加 API 调用次数
        - 启用时间演变分析会增加 2 次 API 调用
        
        ### 📁 输出文件说明
        | 文件名 | 说明 |
        |--------|------|
        | 工程管理期刊分析结果_完整版.xlsx | 完整分析结果 |
        | 研究方法分类统计.xlsx | 研究方法分类统计 |
        | 研究内容分类统计.xlsx | 研究内容分类统计 |
        | 研究方法演变数据.xlsx | 研究方法时间序列 |
        | 研究内容演变数据.xlsx | 研究内容时间序列 |
        | 研究方法演变趋势.html | 交互式演变图表 |
        | 研究内容演变趋势.html | 交互式演变图表 |
        | 分析统计报告.txt | 完整统计报告 |
        """)

if __name__ == "__main__":
    main()
