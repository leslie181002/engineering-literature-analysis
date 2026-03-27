#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工程文献智能分析系统
功能：上传文献数据 -> 数据分析 -> 主题提取 -> 专家总结
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Optional, Dict, List, Any

# ============ 页面配置（必须在最前面）============
st.set_page_config(
    page_title="工程文献智能分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ 自定义 CSS 样式 ============
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #17a2b8;
    }
    .stProgress > div > div > div > div {
        background-color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)


# ============ Session State 初始化 ============
def initialize_session_state():
    """
    初始化所有 session_state 变量
    这是防止 AttributeError 的关键步骤
    """
    defaults = {
        # 数据相关
        'uploaded_file': None,
        'df_original': None,
        'df_result': None,
        'df_processed': None,
        
        # 分析相关
        'topic_info': None,
        'analysis_results': None,
        'expert_summary': None,
        'analysis_complete': False,
        'step1_complete': False,
        'step2_complete': False,
        'step3_complete': False,
        'step4_complete': False,
        
        # 流程控制
        'current_step': 0,
        'total_steps': 4,
        
        # 错误处理
        'error_message': None,
        'warning_message': None,
        
        # 配置
        'config': {
            'num_topics': 5,
            'min_samples': 10,
            'language': 'zh'
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============ 工具函数 ============
def check_and_normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    检查并标准化列名
    解决列名不匹配的问题
    """
    # 原始列名
    original_columns = df.columns.tolist()
    
    # 列名映射字典（支持多种命名方式）
    column_mapping = {
        # 英文列名
        'article title': 'Article Title',
        'title': 'Article Title',
        'paper title': 'Article Title',
        'document title': 'Article Title',
        
        # 研究方法
        'research method': '研究方法',
        'method': '研究方法',
        'methodology': '研究方法',
        '研究方法': '研究方法',
        
        # 研究内容
        'research content': '研究内容',
        'content': '研究内容',
        'abstract': '研究内容',
        '摘要': '研究内容',
        '研究内容': '研究内容',
        
        # 主题编号
        'topic_id': '主题编号',
        'topic id': '主题编号',
        'cluster_id': '主题编号',
        'cluster id': '主题编号',
        '主题编号': '主题编号',
        
        # 主题名称
        'topic_name': '主题名称',
        'topic name': '主题名称',
        'cluster_name': '主题名称',
        'cluster name': '主题名称',
        '主题名称': '主题名称',
        '主题': '主题名称',
    }
    
    # 创建列名映射
    rename_dict = {}
    for col in original_columns:
        col_lower = col.lower().strip()
        if col_lower in column_mapping:
            rename_dict[col] = column_mapping[col_lower]
    
    # 重命名列
    if rename_dict:
        df = df.rename(columns=rename_dict)
        st.info(f"📝 已标准化列名：{list(rename_dict.values())}")
    
    return df


def safe_get_session_state(key: str, default: Any = None) -> Any:
    """
    安全获取 session_state 值
    避免 AttributeError
    """
    if key in st.session_state:
        return st.session_state[key]
    return default


def clear_session_state():
    """清除所有 session_state"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()


# ============ 分析功能函数 ============
def load_data(file) -> pd.DataFrame:
    """
    加载上传的文件
    """
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8')
        elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            df = pd.read_excel(file)
        else:
            raise ValueError("不支持的文件格式，请上传 CSV 或 Excel 文件")
        
        return df
    except Exception as e:
        st.error(f"❌ 文件读取失败：{str(e)}")
        return None


def analyze_topics(df: pd.DataFrame, num_topics: int = 5) -> Dict:
    """
    分析主题信息
    """
    try:
        # 检查必要的列
        required_columns = ['Article Title', '研究内容']
        available_columns = df.columns.tolist()
        
        # 统计基本信息
        topic_stats = {
            'total_articles': len(df),
            'num_topics': num_topics,
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'columns': available_columns,
        }
        
        # 如果有主题编号列，统计每个主题的文章数
        if '主题编号' in df.columns:
            topic_counts = df['主题编号'].value_counts().to_dict()
            topic_stats['topic_distribution'] = topic_counts
        
        # 如果有主题名称列，获取主题列表
        if '主题名称' in df.columns:
            unique_topics = df['主题名称'].unique().tolist()
            topic_stats['topic_list'] = unique_topics[:num_topics]
        
        return topic_stats
        
    except Exception as e:
        st.error(f"❌ 主题分析失败：{str(e)}")
        return {'error': str(e)}


def generate_expert_summary(df: pd.DataFrame, topic_info: Dict) -> str:
    """
    生成专家总结
    """
    try:
        summary_lines = []
        summary_lines.append("## 📊 工程文献分析报告")
        summary_lines.append("")
        summary_lines.append(f"**分析时间**: {topic_info.get('analysis_time', 'N/A')}")
        summary_lines.append(f"**文献总数**: {topic_info.get('total_articles', 0)} 篇")
        summary_lines.append(f"**主题数量**: {topic_info.get('num_topics', 0)} 个")
        summary_lines.append("")
        
        # 主题分布
        if 'topic_distribution' in topic_info:
            summary_lines.append("### 🎯 主题分布")
            for topic_id, count in topic_info['topic_distribution'].items():
                percentage = (count / topic_info['total_articles']) * 100
                summary_lines.append(f"- 主题 {topic_id}: {count} 篇 ({percentage:.1f}%)")
            summary_lines.append("")
        
        # 主题列表
        if 'topic_list' in topic_info:
            summary_lines.append("### 📋 主要主题")
            for i, topic in enumerate(topic_info['topic_list'], 1):
                summary_lines.append(f"{i}. {topic}")
            summary_lines.append("")
        
        # 研究建议
        summary_lines.append("### 💡 研究建议")
        summary_lines.append("1. 关注高频率主题的研究趋势")
        summary_lines.append("2. 分析不同主题间的关联性")
        summary_lines.append("3. 识别新兴研究方向")
        summary_lines.append("")
        
        summary_lines.append("---")
        summary_lines.append("*本报告由工程文献智能分析系统自动生成*")
        
        return '\n'.join(summary_lines)
        
    except Exception as e:
        return f"❌ 生成总结失败：{str(e)}"


# ============ 界面组件 ============
def render_header():
    """渲染页面头部"""
    st.markdown('<h1 class="main-header">📚 工程文献智能分析系统</h1>', unsafe_allow_html=True)
    st.markdown("---")


def render_progress_bar(current: int, total: int):
    """渲染进度条"""
    progress = current / total
    st.progress(progress)
    st.caption(f"进度：{current}/{total} 步骤")


def render_step_card(step_num: int, title: str, content: Any, 
                     is_complete: bool = False, is_active: bool = False):
    """渲染步骤卡片"""
    status_icon = "✅" if is_complete else ("🔄" if is_active else "⏳")
    
    with st.container():
        st.markdown(f"### {status_icon} 步骤 {step_num}: {title}")
        if content:
            if isinstance(content, str):
                st.markdown(content)
            elif isinstance(content, pd.DataFrame):
                st.dataframe(content, use_container_width=True)
            else:
                st.write(content)


def render_debug_panel():
    """渲染调试面板（侧边栏）"""
    with st.sidebar:
        st.subheader("🔍 调试信息")
        
        # Session State 状态
        st.write("**Session State Keys:**")
        st.json(list(st.session_state.keys()))
        
        # 数据状态
        st.write("**数据状态:**")
        st.write(f"- 文件已上传：{st.session_state.uploaded_file is not None}")
        st.write(f"- 原始数据：{st.session_state.df_original is not None}")
        st.write(f"- 处理数据：{st.session_state.df_result is not None}")
        
        # 分析状态
        st.write("**分析状态:**")
        st.write(f"- 步骤 1 完成：{st.session_state.step1_complete}")
        st.write(f"- 步骤 2 完成：{st.session_state.step2_complete}")
        st.write(f"- 步骤 3 完成：{st.session_state.step3_complete}")
        st.write(f"- 步骤 4 完成：{st.session_state.step4_complete}")
        
        # DataFrame 列信息
        if st.session_state.df_result is not None:
            st.write("**DataFrame 列名:**")
            st.write(st.session_state.df_result.columns.tolist())
            st.write(f"**数据形状:** {st.session_state.df_result.shape}")
        
        # 错误信息
        if st.session_state.error_message:
            st.error(f"❌ 错误：{st.session_state.error_message}")
        
        # 重置按钮
        st.markdown("---")
        if st.button("🔄 重置所有状态", use_container_width=True):
            clear_session_state()
            st.rerun()


# ============ 主流程函数 ============
def step1_upload_file():
    """
    步骤 1: 上传文件
    """
    st.session_state.current_step = 1
    render_step_card(1, "上传文献数据", None, 
                    is_complete=st.session_state.step1_complete,
                    is_active=True)
    
    uploaded_file = st.file_uploader(
        "📁 上传 Excel/CSV 文件",
        type=['xlsx', 'xls', 'csv'],
        help="支持的文件格式：.xlsx, .xls, .csv"
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        
        # 显示文件信息
        st.success(f"✅ 文件上传成功：{uploaded_file.name}")
        st.info(f"📊 文件大小：{uploaded_file.size / 1024:.2f} KB")
        
        # 读取数据
        df = load_data(uploaded_file)
        
        if df is not None:
            st.session_state.df_original = df
            st.session_state.df_result = df
            
            # 标准化列名
            df = check_and_normalize_columns(df)
            st.session_state.df_result = df
            st.session_state.df_processed = df
            
            # 显示数据预览
            st.subheader("📋 数据预览")
            st.dataframe(df.head(10), use_container_width=True)
            
            # 显示数据统计
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总行数", len(df))
            with col2:
                st.metric("总列数", len(df.columns))
            with col3:
                st.metric("内存使用", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            
            # 标记步骤完成
            st.session_state.step1_complete = True
            
            # 自动进入下一步
            st.success("✅ 步骤 1 完成！请继续下一步")
    
    return st.session_state.step1_complete


def step2_data_analysis():
    """
    步骤 2: 数据分析
    """
    st.session_state.current_step = 2
    render_step_card(2, "数据分析", None,
                    is_complete=st.session_state.step2_complete,
                    is_active=st.session_state.current_step == 2 and st.session_state.step1_complete)
    
    if not st.session_state.step1_complete:
        st.warning("⚠️ 请先完成步骤 1：上传文件")
        return False
    
    df = st.session_state.df_result
    
    if df is not None:
        # 显示可用列
        st.info(f"📋 可用列名：{df.columns.tolist()}")
        
        # 分析配置
        with st.expander("⚙️ 分析配置"):
            num_topics = st.slider("主题数量", 2, 20, 5)
            st.session_state.config['num_topics'] = num_topics
        
        # 开始分析按钮
        if st.button("🔍 开始分析", type="primary", disabled=st.session_state.step2_complete):
            with st.spinner("正在分析数据..."):
                try:
                    # 执行主题分析
                    topic_info = analyze_topics(df, num_topics)
                    st.session_state.topic_info = topic_info
                    st.session_state.analysis_results = topic_info
                    
                    # 标记完成
                    st.session_state.step2_complete = True
                    st.session_state.analysis_complete = True
                    
                    st.success("✅ 分析完成！")
                    
                    # 显示分析结果预览
                    if topic_info:
                        st.json(topic_info)
                    
                except Exception as e:
                    st.error(f"❌ 分析失败：{str(e)}")
                    st.session_state.error_message = str(e)
    else:
        st.warning("⚠️ 数据未加载，请返回步骤 1 重新上传文件")
    
    return st.session_state.step2_complete


def step3_topic_info():
    """
    步骤 3: 主题信息
    """
    st.session_state.current_step = 3
    render_step_card(3, "主题信息", None,
                    is_complete=st.session_state.step3_complete,
                    is_active=st.session_state.current_step == 3 and st.session_state.step2_complete)
    
    if not st.session_state.step2_complete:
        st.warning("⚠️ 请先完成步骤 2：数据分析")
        return False
    
    # 安全访问 topic_info
    topic_info = safe_get_session_state('topic_info', None)
    
    if topic_info is not None:
        # 显示主题信息
        st.subheader("🎯 主题分析结果")
        
        # 基本信息
        col1, col2 = st.columns(2)
        with col1:
            st.metric("文献总数", topic_info.get('total_articles', 0))
        with col2:
            st.metric("主题数量", topic_info.get('num_topics', 0))
        
        # 主题分布
        if 'topic_distribution' in topic_info:
            st.subheader("📊 主题分布")
            dist_df = pd.DataFrame({
                '主题编号': list(topic_info['topic_distribution'].keys()),
                '文章数量': list(topic_info['topic_distribution'].values())
            })
            st.bar_chart(dist_df.set_index('主题编号'))
        
        # 主题列表
        if 'topic_list' in topic_info:
            st.subheader("📋 主题列表")
            for i, topic in enumerate(topic_info['topic_list'], 1):
                st.write(f"{i}. {topic}")
        
        # 标记完成
        st.session_state.step3_complete = True
        
    else:
        st.info("ℹ️ 主题信息将在分析后自动生成")
        st.session_state.step3_complete = False
    
    return st.session_state.step3_complete


def step4_expert_summary():
    """
    步骤 4: 专家总结
    """
    st.session_state.current_step = 4
    render_step_card(4, "专家总结", None,
                    is_complete=st.session_state.step4_complete,
                    is_active=st.session_state.current_step == 4 and st.session_state.step3_complete)
    
    if not st.session_state.step3_complete:
        st.warning("⚠️ 请先完成步骤 3：主题信息")
        return False
    
    # 安全获取数据
    df = safe_get_session_state('df_result', None)
    topic_info = safe_get_session_state('topic_info', None)
    
    if df is not None and topic_info is not None:
        st.subheader("📈 结果预览")
        
        # 安全选择列
        available_columns = df.columns.tolist()
        target_columns = ['Article Title', '研究方法', '研究内容', '主题编号', '主题名称']
        existing_columns = [col for col in target_columns if col in available_columns]
        
        if existing_columns:
            st.dataframe(df[existing_columns].head(10), use_container_width=True)
        else:
            st.warning(f"⚠️ 未找到预期列名，显示所有可用列")
            st.dataframe(df.head(10), use_container_width=True)
        
        # 数据预览
        st.subheader("📊 数据预览")
        st.dataframe(df.head(20), use_container_width=True)
        
        # 主题信息
        st.subheader("🎯 主题信息")
        if topic_info:
            st.json(topic_info)
        
        # 专家总结
        st.subheader("🎓 专家总结")
        
        # 生成或获取专家总结
        if st.session_state.expert_summary is None:
            expert_summary = generate_expert_summary(df, topic_info)
            st.session_state.expert_summary = expert_summary
        
        st.markdown(st.session_state.expert_summary)
        
        # 下载按钮
        st.subheader("💾 导出报告")
        col1, col2 = st.columns(2)
        with col1:
            csv_data = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下载 CSV",
                data=csv_data,
                file_name=f"analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            st.download_button(
                label="📥 下载总结报告",
                data=st.session_state.expert_summary,
                file_name=f"expert_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        # 标记完成
        st.session_state.step4_complete = True
        st.success("✅ 所有步骤完成！")
        
    else:
        st.warning("⚠️ 数据或主题信息缺失，请检查前面的步骤")
        st.session_state.step4_complete = False
    
    return st.session_state.step4_complete


# ============ 主函数 ============
def main():
    """
    主函数 - 应用入口
    """
    # 第一步：初始化 session_state（必须放在最前面）
    initialize_session_state()
    
    # 渲染头部
    render_header()
    
    # 渲染进度条
    completed_steps = sum([
        st.session_state.step1_complete,
        st.session_state.step2_complete,
        st.session_state.step3_complete,
        st.session_state.step4_complete
    ])
    render_progress_bar(completed_steps, st.session_state.total_steps)
    
    # 主内容区域
    st.markdown("---")
    
    # 步骤 1: 上传文件
    step1_upload_file()
    
    st.markdown("---")
    
    # 步骤 2: 数据分析
    step2_data_analysis()
    
    st.markdown("---")
    
    # 步骤 3: 主题信息
    step3_topic_info()
    
    st.markdown("---")
    
    # 步骤 4: 专家总结
    step4_expert_summary()
    
    # 侧边栏 - 调试信息
    render_debug_panel()
    
    # 页脚
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>工程文献智能分析系统 v1.0 | 
            最后更新：""" + datetime.now().strftime('%Y-%m-%d') + """</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============ 程序入口 ============
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ 系统错误：{str(e)}")
        st.exception(e)
