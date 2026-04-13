import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class StockVisualizer:
    """股票分析结果可视化工具"""
    
    def __init__(self, save_path='./visualization_results'):
        self.save_path = save_path
        import os
        os.makedirs(save_path, exist_ok=True)
        self.colors = {
            'buy': '#4CAF50',  # 绿色
            'sell': '#F44336',  # 红色
            'neutral': '#FF9800',  # 橙色
            'primary': '#2196F3',  # 蓝色
            'secondary': '#9C27B0'  # 紫色
        }
    
    def plot_top_stocks(self, top_df, n=30, title="Top 30 股票预测结果"):
        """可视化Top N股票"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. 得分柱状图
        ax1 = axes[0, 0]
        top_df = top_df.head(n).copy()
        bars = ax1.barh(range(len(top_df)), top_df['加权得分'].values, 
                       color=self.colors['primary'])
        ax1.set_yticks(range(len(top_df)))
        ax1.set_yticklabels(top_df['股票代码'] + '\n' + top_df['股票名称'], fontsize=9)
        ax1.set_xlabel('加权得分')
        ax1.set_title('Top 30 股票得分排名')
        ax1.invert_yaxis()  # 最高分在顶部
        
        # 在柱子上添加数值
        for i, (bar, score) in enumerate(zip(bars, top_df['加权得分'].values)):
            ax1.text(score + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{score:.4f}', va='center', fontsize=8)
        
        # 2. 得分分布直方图
        ax2 = axes[0, 1]
        scores = top_df['加权得分'].values
        ax2.hist(scores, bins=20, alpha=0.7, color=self.colors['secondary'], edgecolor='black')
        ax2.axvline(np.mean(scores), color='red', linestyle='--', label=f'平均分: {np.mean(scores):.4f}')
        ax2.axvline(np.median(scores), color='green', linestyle='--', label=f'中位数: {np.median(scores):.4f}')
        ax2.set_xlabel('得分')
        ax2.set_ylabel('数量')
        ax2.set_title('得分分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 得分散点图（按代码排序）
        ax3 = axes[1, 0]
        indices = range(len(top_df))
        sc = ax3.scatter(indices, scores, c=scores, cmap='RdYlGn', s=100, alpha=0.7)
        ax3.plot(indices, scores, 'b-', alpha=0.3)
        ax3.set_xlabel('排名')
        ax3.set_ylabel('得分')
        ax3.set_title('得分随排名变化')
        ax3.grid(True, alpha=0.3)
        
        # 添加颜色条
        plt.colorbar(sc, ax=ax3)
        
        # 4. 箱线图
        ax4 = axes[1, 1]
        box_data = []
        box_labels = []
        
        # 将数据分为5组
        chunk_size = 6
        for i in range(0, len(scores), chunk_size):
            chunk = scores[i:i+chunk_size]
            if len(chunk) > 0:
                box_data.append(chunk)
                box_labels.append(f'第{i//chunk_size+1}组')
        
        bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
        
        # 设置箱体颜色
        for patch in bp['boxes']:
            patch.set_facecolor(self.colors['primary'])
            patch.set_alpha(0.7)
        
        ax4.set_ylabel('得分')
        ax4.set_title('得分箱线图（分组）')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/top_stocks.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_bottom_stocks(self, bottom_df, n=10, title="倒数 Top 10 股票"):
        """可视化底部股票"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        bottom_df = bottom_df.head(n).copy()
        
        # 1. 水平柱状图
        bars = ax1.barh(range(len(bottom_df)), bottom_df['加权得分'].values, 
                       color=self.colors['sell'])
        ax1.set_yticks(range(len(bottom_df)))
        ax1.set_yticklabels(bottom_df['股票代码'] + '\n' + bottom_df['股票名称'], fontsize=9)
        ax1.set_xlabel('加权得分')
        ax1.set_title('倒数 Top 10 股票得分')
        ax1.invert_yaxis()
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, bottom_df['加权得分'].values)):
            ax1.text(score - 0.002, bar.get_y() + bar.get_height()/2, 
                    f'{score:.4f}', va='center', fontsize=8, ha='right', color='white')
        
        # 2. 得分雷达图
        ax2 = plt.subplot(122, projection='polar')
        angles = np.linspace(0, 2*np.pi, len(bottom_df), endpoint=False).tolist()
        scores = bottom_df['加权得分'].values
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        # 闭合图形
        scores_normalized = np.concatenate([scores_normalized, [scores_normalized[0]]])
        angles = np.concatenate([angles, [angles[0]]])
        
        ax2.plot(angles, scores_normalized, 'o-', linewidth=2, color=self.colors['sell'])
        ax2.fill(angles, scores_normalized, alpha=0.25, color=self.colors['sell'])
        
        # 设置角度标签
        ax2.set_xticks(angles[:-1])
        labels = [f"{code}\n{name[:3]}" for code, name in 
                 zip(bottom_df['股票代码'], bottom_df['股票名称'])]
        ax2.set_xticklabels(labels, fontsize=8)
        
        ax2.set_title('倒数股票得分雷达图', pad=20)
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/bottom_stocks.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_feature_importance(self, feature_importance_dict, n=20, title="特征重要性 Top 20"):
        """可视化特征重要性"""
        if not feature_importance_dict:
            print("特征重要性数据为空")
            return None
        
        # 转换为DataFrame
        importance_df = pd.DataFrame(list(feature_importance_dict.items()), 
                                     columns=['Feature', 'Importance'])
        importance_df = importance_df.sort_values('Importance', ascending=False).head(n)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. 水平条形图
        ax1 = axes[0]
        bars = ax1.barh(range(len(importance_df)), importance_df['Importance'].values, 
                       color=plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_df))))
        ax1.set_yticks(range(len(importance_df)))
        ax1.set_yticklabels(importance_df['Feature'].values, fontsize=9)
        ax1.set_xlabel('重要性分数')
        ax1.set_title('特征重要性排名')
        ax1.invert_yaxis()  # 最重要在顶部
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, importance_df['Importance'].values)):
            ax1.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.0f}', va='center', fontsize=8)
        
        # 2. 饼图（前10个特征）
        ax2 = axes[1]
        if len(importance_df) >= 10:
            top10 = importance_df.head(10)
            explode = [0.1] + [0]*(len(top10)-1)  # 突出最重要的特征
            wedges, texts, autotexts = ax2.pie(top10['Importance'], 
                                              labels=top10['Feature'],
                                              autopct='%1.1f%%',
                                              startangle=90,
                                              explode=explode,
                                              colors=plt.cm.Set3(np.linspace(0, 1, len(top10))))
            
            # 美化标签
            for text in texts:
                text.set_fontsize(8)
            for autotext in autotexts:
                autotext.set_fontsize(8)
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax2.set_title('Top 10 特征占比', pad=20)
        else:
            ax2.text(0.5, 0.5, '数据不足', ha='center', va='center', fontsize=12)
            ax2.set_axis_off()
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_llm_analysis_summary(self, analysis_text, title="大模型分析结果摘要"):
        """可视化大模型分析结果"""
        fig = plt.figure(figsize=(12, 8))
        
        # 清理文本
        lines = analysis_text.split('\n')
        clean_lines = []
        for line in lines:
            if line.strip() and len(line.strip()) > 3:
                clean_lines.append(line.strip())
        
        # 创建文本显示区域
        ax = plt.subplot(111, frameon=False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # 设置标题
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
        
        # 显示文本
        y_pos = 0.9
        line_height = 0.03
        
        for i, line in enumerate(clean_lines[:30]):  # 最多显示30行
            fontsize = 9
            fontweight = 'normal'
            color = 'black'
            
            # 根据内容设置格式
            if line.startswith(('🚀', '📈', '🎯', '✅', '⚠️', '🏆')):
                fontsize = 10
                fontweight = 'bold'
                color = self.colors['buy'] if '买入' in line else self.colors['neutral']
            elif '评分' in line or '目标价' in line or '仓位' in line:
                fontweight = 'bold'
                color = self.colors['primary']
            elif '风险' in line or '注意' in line:
                color = self.colors['sell']
            
            plt.text(0.05, y_pos, line, fontsize=fontsize, fontweight=fontweight,
                    color=color, verticalalignment='top',
                    transform=ax.transAxes, wrap=True)
            y_pos -= line_height
        
        if len(clean_lines) > 30:
            plt.text(0.05, y_pos, f"... 还有 {len(clean_lines)-30} 行未显示 ...", 
                    fontsize=8, color='gray', style='italic',
                    verticalalignment='top', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/llm_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 同时保存为文本文件
        with open(f'{self.save_path}/llm_analysis_full.txt', 'w', encoding='utf-8') as f:
            f.write(analysis_text)
        
        return fig
    
    def plot_comprehensive_dashboard(self, top_df, bottom_df, feature_importance_dict, llm_analysis, 
                                    use_index="hs300"):
        """创建综合仪表板 - 修复版本"""
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'股票分析综合仪表板 ({use_index.upper()})', fontsize=20, fontweight='bold')
        
        # 创建网格布局
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Top 5 股票得分
        ax1 = fig.add_subplot(gs[0, 0])
        if not top_df.empty:
            top5 = top_df.head(5)
            bars1 = ax1.bar(range(len(top5)), top5['加权得分'], color=self.colors['buy'])
            ax1.set_xticks(range(len(top5)))
            ax1.set_xticklabels(top5['股票代码'], rotation=45, fontsize=9)
            ax1.set_ylabel('得分')
            ax1.set_title('Top 5 股票')
            ax1.grid(True, alpha=0.3, axis='y')
        else:
            ax1.text(0.5, 0.5, '无Top股票数据', ha='center', va='center', fontsize=12)
            ax1.set_title('Top 5 股票（无数据）')
            ax1.set_axis_off()
        
        # 2. Bottom 5 股票得分
        ax2 = fig.add_subplot(gs[0, 1])
        if not bottom_df.empty:
            bottom5 = bottom_df.head(5)
            bars2 = ax2.bar(range(len(bottom5)), bottom5['加权得分'], color=self.colors['sell'])
            ax2.set_xticks(range(len(bottom5)))
            ax2.set_xticklabels(bottom5['股票代码'], rotation=45, fontsize=9)
            ax2.set_ylabel('得分')
            ax2.set_title('Bottom 5 股票')
            ax2.grid(True, alpha=0.3, axis='y')
        else:
            ax2.text(0.5, 0.5, '无Bottom股票数据', ha='center', va='center', fontsize=12)
            ax2.set_title('Bottom 5 股票（无数据）')
            ax2.set_axis_off()
        
        # 3. 特征重要性前10
        ax3 = fig.add_subplot(gs[0, 2])
        if feature_importance_dict:
            # 创建DataFrame并排序
            importance_df = pd.DataFrame(list(feature_importance_dict.items()), 
                                        columns=['Feature', 'Importance'])
            importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
            
            if not importance_df.empty:
                ax3.barh(range(len(importance_df)), importance_df['Importance'], 
                        color=plt.cm.tab20c(np.linspace(0, 1, len(importance_df))))
                ax3.set_yticks(range(len(importance_df)))
                ax3.set_yticklabels(importance_df['Feature'], fontsize=8)
                ax3.set_xlabel('重要性')
                ax3.set_title('特征重要性 Top 10')
                ax3.invert_yaxis()
            else:
                ax3.text(0.5, 0.5, '无特征重要性数据', ha='center', va='center', fontsize=12)
                ax3.set_title('特征重要性（无数据）')
                ax3.set_axis_off()
        else:
            ax3.text(0.5, 0.5, '无特征重要性数据', ha='center', va='center', fontsize=12)
            ax3.set_title('特征重要性（无数据）')
            ax3.set_axis_off()
        
        # 4. 得分分布直方图
        ax4 = fig.add_subplot(gs[1, 0])
        all_scores = pd.concat([top_df['加权得分'], bottom_df['加权得分']])
        if not all_scores.empty:
            ax4.hist(all_scores, bins=30, alpha=0.7, color=self.colors['primary'], 
                    edgecolor='black', density=True)
            ax4.set_xlabel('得分')
            ax4.set_ylabel('密度')
            ax4.set_title('所有股票得分分布')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, '无得分数据', ha='center', va='center', fontsize=12)
            ax4.set_title('得分分布（无数据）')
            ax4.set_axis_off()
        
        # 5. 得分箱线图
        ax5 = fig.add_subplot(gs[1, 1])
        if not top_df.empty and not bottom_df.empty:
            data_to_plot = [top_df['加权得分'].values, bottom_df['加权得分'].values]
            bp = ax5.boxplot(data_to_plot, labels=['Top', 'Bottom'], patch_artist=True)
            colors = [self.colors['buy'], self.colors['sell']]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax5.set_ylabel('得分')
            ax5.set_title('Top vs Bottom 得分对比')
            ax5.grid(True, alpha=0.3, axis='y')
        else:
            ax5.text(0.5, 0.5, '数据不足，无法比较', ha='center', va='center', fontsize=12)
            ax5.set_title('得分对比（无数据）')
            ax5.set_axis_off()
        
        # 6. 大模型分析摘要
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        if llm_analysis:
            summary_lines = []
            for line in llm_analysis.split('\n'):
                if any(keyword in line for keyword in ['建议', '评级', '目标价', '仓位', '风险']):
                    summary_lines.append(line.strip())
                    if len(summary_lines) >= 8:
                        break
            
            if summary_lines:
                summary_text = '\n'.join(summary_lines[:8])
                ax6.text(0.1, 0.9, '大模型分析摘要:', fontsize=10, fontweight='bold', 
                        transform=ax6.transAxes)
                ax6.text(0.1, 0.7, summary_text, fontsize=8, transform=ax6.transAxes,
                        verticalalignment='top')
            else:
                ax6.text(0.5, 0.5, '无有效摘要', ha='center', va='center', fontsize=12)
        else:
            ax6.text(0.5, 0.5, '无大模型分析结果', ha='center', va='center', fontsize=12)
        
        # 7. 得分热力图（Top 20）
        ax7 = fig.add_subplot(gs[2, :])
        if not top_df.empty:
            n_display = min(20, len(top_df))
            top_display = top_df.head(n_display)
            
            # 创建矩阵，行数固定为3，列数为实际显示的数量
            score_matrix = np.zeros((3, n_display))
            
            # 使用enumerate获取连续索引
            for idx, (_, row) in enumerate(top_display.iterrows()):
                if idx >= n_display:  # 安全保护
                    break
                score_matrix[0, idx] = row['加权得分']
                # 尝试获取每日得分
                for d in [1, 2]:
                    col_name = f'D{d}得分'
                    if col_name in row:
                        score_matrix[d, idx] = row[col_name]
            
            im = ax7.imshow(score_matrix, aspect='auto', cmap='RdYlGn')
            ax7.set_xticks(range(n_display))
            ax7.set_xticklabels(top_display['股票代码'], rotation=45, fontsize=8)
            ax7.set_yticks(range(3))
            ax7.set_yticklabels(['加权得分', 'D1得分', 'D2得分'])
            ax7.set_title(f'Top {n_display} 股票得分热力图')
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax7, shrink=0.8)
            cbar.set_label('得分')
        else:
            ax7.text(0.5, 0.5, '无Top股票数据', ha='center', va='center', fontsize=12)
            ax7.set_title('Top 股票得分热力图（无数据）')
            ax7.set_axis_off()
        
        # 调整布局
        plt.tight_layout()
        
        # 保存
        plt.savefig(f'{self.save_path}/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_html_report(self, top_df, bottom_df, feature_importance_dict, llm_analysis, 
                            use_index="hs300", timestamp=None):
        """使用字符串拼接生成HTML报告 - 最安全的方法"""
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 开始构建HTML
        html_parts = []
        
        html_parts.append("""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>股票分析报告 - """)
        html_parts.append(str(use_index.upper()))
        html_parts.append("""</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            h2 { color: #666; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .high-score { color: green; font-weight: bold; }
            .low-score { color: red; font-weight: bold; }
            .timestamp { color: #888; font-size: 12px; text-align: right; }
        </style>
    </head>
    <body>
        <h1>📈 股票分析报告</h1>
        <div class="timestamp">生成时间: """)
        html_parts.append(str(timestamp))
        html_parts.append("""</div>
        
        <h2>📊 分析概要</h2>
        <p>• 指数类型: """)
        html_parts.append(str(use_index.upper()))
        html_parts.append("""</p>
        <p>• Top 30 股票数量: """)
        html_parts.append(str(len(top_df)))
        html_parts.append("""</p>""")
        
        if not top_df.empty:
            html_parts.append("""<p>• 平均得分: """)
            html_parts.append(f"{top_df['加权得分'].mean():.4f}")
            html_parts.append("""</p>
            <p>• 最高得分: """)
            html_parts.append(f"{top_df['加权得分'].max():.4f}")
            html_parts.append("""</p>""")
        
        if not bottom_df.empty:
            html_parts.append("""<p>• 最低得分: """)
            html_parts.append(f"{bottom_df['加权得分'].min():.4f}")
            html_parts.append("""</p>""")
        
        html_parts.append("""
        <h2>🏆 Top 10 推荐股票</h2>
        <table>
            <tr>
                <th>排名</th><th>股票代码</th><th>股票名称</th>
                <th>加权得分</th>
            </tr>""")
        
        # 添加Top 30行
        for i, (_, row) in enumerate(top_df.head(30).iterrows(), 1):
            html_parts.append(f"""
            <tr>
                <td>{i}</td>
                <td>{row['股票代码']}</td>
                <td>{row['股票名称']}</td>
                <td class="high-score">{row['加权得分']:.4f}</td>
            </tr>""")
        
        html_parts.append("""
        </table>
        
        <h2>📉 倒数 Top 10 股票</h2>
        <table>
            <tr><th>排名</th><th>股票代码</th><th>股票名称</th><th>加权得分</th></tr>""")
        
        # 添加Bottom 5行
        for i, (_, row) in enumerate(bottom_df.head(10).iterrows(), 1):
            html_parts.append(f"""
            <tr>
                <td>{i}</td>
                <td>{row['股票代码']}</td>
                <td>{row['股票名称']}</td>
                <td class="low-score">{row['加权得分']:.4f}</td>
            </tr>""")
        
        html_parts.append("""
        </table>
        
        <h2>🤖 大模型分析结果</h2>
        <div style="white-space: pre-wrap; background: #f5f5f5; padding: 15px; border-radius: 5px; font-family: monospace;">
        """)
        html_parts.append(str(llm_analysis))
        html_parts.append("""
        </div>
        
        <h2>📁 生成文件</h2>
        <p>以下图表已保存到 """)
        html_parts.append(str(self.save_path))
        html_parts.append(""" 目录：</p>
        <ul>
            <li>top_stocks.png - Top 30 股票得分图</li>
            <li>bottom_stocks.png - 倒数 Top 10 股票图</li>
            <li>feature_importance.png - 特征重要性图</li>
            <li>llm_analysis.png - 大模型分析摘要图</li>
            <li>comprehensive_dashboard.png - 综合仪表板</li>
        </ul>
        
        <div class="timestamp">报告结束</div>
    </body>
    </html>""")
        
        # 合并所有部分
        html_content = ''.join(html_parts)
        
        # 保存HTML文件
        import os
        html_path = os.path.join(self.save_path, f"stock_analysis_report_{timestamp}.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML报告已保存到: {html_path}")
        
        return html_content