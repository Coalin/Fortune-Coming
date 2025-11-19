import akshare as ak
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (roc_auc_score, precision_score, 
                            recall_score, confusion_matrix)
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from utils import *
from sklearn.preprocessing import RobustScaler


# 动态标签生成函数
def generate_dynamic_label(stock_return, index_return, lookback_window=20):
    """
    复合标签生成逻辑（满足任一条件即为1）：
    1. 动态阈值：超额收益 >= 1.5倍指数波动率（限制在1.5%-5%）
    2. 绝对超额：比沪深300好5%
    3. 绝对收益：上涨15%
    """
    # 计算指数波动率
    index_vol = index_return.rolling(lookback_window).std().fillna(0.02)
    
    # 动态阈值（1.5倍波动率，限制在1.5%-5%区间）
    dynamic_threshold = np.clip(1.5 * index_vol, 0.015, 0.05)
    
    # 计算超额收益
    excess_return = stock_return - index_return

    # 条件1：动态阈值条件
    cond_dynamic = (excess_return >= dynamic_threshold)
    
    # 条件2：比沪深300好3%（绝对超额）
    cond_outperform = (excess_return >= 0.03)
    
    # 条件3：上涨10%（绝对收益）
    cond_abs_gain = (stock_return >= 0.088)
    
    # 生成标签
    # labels = (cond_abs_gain).astype(int)

    labels = np.where(
        stock_return > 0.1, 1,  # 正样本
        np.where(stock_return <= 0.05, 0, np.nan)  # 负样本或排除
    )
    
    # 打印各条件触发比例（调试用）
    if len(labels) > 0:
        print("\n标签生成统计:")
        print(f"动态阈值触发: {cond_dynamic.mean():.2%}")
        print(f"跑赢3%触发: {cond_outperform.mean():.2%}")
        print(f"上涨5%触发: {cond_abs_gain.mean():.2%}")
        print(f"最终正样本比例: {labels.mean():.2%}")
    
    return labels

# 回测评估函数
def backtest_evaluation(results_df, pred_proba, true_return, index_return, top_pct=0.1, holding_period=15):
    """
    综合回测评估：包含夏普比率、特雷诺比率
    """
    results_df['pred_proba'] = pred_proba
    results_df = results_df.sort_values('pred_proba', ascending=False)
    
    # 选取前10%股票
    top_stocks = results_df.iloc[:int(len(results_df)*top_pct)]
    
    if len(top_stocks) == 0:
        return {
            'excess_return': 0, 'win_rate': 0, 'sharpe_ratio': 0, 'treynor_ratio': 0,
            'annualized_excess': 0, 'num_trades': 0, 'hit_rate': 0
        }
    
    # 计算实际收益（考虑交易成本）
    net_return = top_stocks[true_return] - 0.0001
    benchmark_return = top_stocks[index_return]
    
    # 计算超额收益
    excess_returns = net_return - benchmark_return
    
    # 基础指标
    avg_excess_return = excess_returns.mean()
    win_rate = (net_return > benchmark_return).mean()
    hit_rate = (excess_returns > 0).mean()
    
    # 年化计算（关键修正）
    periods_per_year = 252 / holding_period  # 一年有多少个持有期
    
    # 年化超额收益 = 平均每期超额收益 × 每年期数
    annualized_excess = avg_excess_return * periods_per_year
    
    # 年化波动率 = 每期超额收益标准差 × √每年期数
    if len(excess_returns) > 1:
        annual_volatility = excess_returns.std() * np.sqrt(periods_per_year)
    else:
        annual_volatility = 0
    
    # 1. 夏普比率（总风险调整）
    sharpe_ratio = annualized_excess / annual_volatility if annual_volatility > 0 else 0
    
    # 2. 特雷诺比率（系统性风险调整）- 需要计算Beta
    # 简化版：假设与市场高度相关，用市场波动率代替
    market_volatility = benchmark_return.std() * np.sqrt(periods_per_year) if len(benchmark_return) > 1 else 0.15
    treynor_ratio = annualized_excess / market_volatility if market_volatility > 0 else 0
    
    return {
        'excess_return': avg_excess_return,      # 平均每期超额收益
        'annualized_excess': annualized_excess,   # 年化超额收益
        'win_rate': win_rate,                     # 胜率（绝对收益胜率）
        'hit_rate': hit_rate,                     # 超额收益胜率
        'sharpe_ratio': sharpe_ratio,             # 夏普比率（总风险）
        'treynor_ratio': treynor_ratio,           # 特雷诺比率（系统风险）
        'annual_volatility': annual_volatility,   # 年化波动率
        'num_trades': len(top_stocks),
        'periods_per_year': periods_per_year      # 年化系数（用于验证）
    }

def main():
    # 设置时间窗口
    end_date = get_valid_date()
    train_end_date = end_date - timedelta(days=20)
    train_start_date = end_date - timedelta(days=2000)
    predict_start_date = end_date - timedelta(days=60)
    predict_end_date = end_date
    
    print(f"训练数据时间窗口: {train_start_date.strftime('%Y%m%d')} - {train_end_date.strftime('%Y%m%d')}")
    print(f"推理数据时间窗口: {predict_start_date.strftime('%Y%m%d')} - {predict_end_date.strftime('%Y%m%d')}")

    # 获取沪深300成分股
    hs300 = ak.index_stock_cons_csindex(symbol="000300")
    symbols = hs300['成分券代码'].tolist()
    name_map = dict(zip(hs300['成分券代码'], hs300['成分券名称']))


    def process_data(df, index_data, is_training=True):
        # 确保列名是中文的（新浪财经可能返回英文列名）
        if 'open' in df.columns:
            # 英文列名转中文
            df = df.rename(columns={
                'open': '开盘', 'high': '最高', 'low': '最低', 
                'close': '收盘', 'volume': '成交量', 'date': '日期'
            })
        
        # 验证必要列是否存在
        required_cols = ['开盘', '最高', '最低', '收盘', '成交量', '日期']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺失必要列: {missing_cols}")
        
        # 确保数据类型正确
        numeric_cols = ['开盘', '最高', '最低', '收盘', '成交量']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 确保日期列正确
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
        df = df.dropna(subset=['日期'])
        
        # 后续处理保持不变...
        df = df.copy()
        df = df.sort_values('日期')
        
        # 基础价格数据
        close = df['收盘'].ffill()
        high = df['最高'].ffill()
        low = df['最低'].ffill()
        volume = df['成交量'].ffill()
        
        # 计算技术指标（核心）
        df['MA5'] = calculate_ma(close, 5)
        df['MA10'] = calculate_ma(close, 10)
        df['MA20'] = calculate_ma(close, 20)
        df['MA30'] = calculate_ma(close, 30)
        df['RSI'] = calculate_rsi(close)
        df['MACD'], df['MACD_Signal'] = calculate_macd(close)
        
        # 布林带指标
        bollinger_df = calculate_bollinger(close)
        df['BOLL_upper'] = bollinger_df['BOLL_upper']
        df['BOLL_mid'] = bollinger_df['BOLL_mid']
        df['BOLL_lower'] = bollinger_df['BOLL_lower']
        
        # 其他技术指标
        df['ATR'] = calculate_atr(high, low, close)
        df['OBV'] = calculate_obv(close, volume)
        df['CCI'] = calculate_cci(high, low, close)
        df['VOLATILITY'] = close.pct_change().rolling(20).std().ffill()
        
        # 新增特征
        df['INDUSTRY_RS'] = industry_relative_strength(close, index_data['close'])
        df['PV_DIVERGENCE'] = price_volume_divergence(close, volume)
        df['VOL_RATIO'] = volatility_ratio(close)

        # 新增特征1：短期动量加速（3日收益率的一阶差分）
        df['Momentum_Accel'] = df['收盘'].pct_change(3).diff().fillna(0)
        
        # 新增特征2：量价共振（成交量扩大且价格上涨）
        vol_ma = df['成交量'].rolling(5).mean().shift(1)
        df['Volume_Price_Resonance'] = (
            (df['成交量'] > vol_ma * 1.2) & 
            (df['收盘'] > df['收盘'].shift(1))
        ).astype(float).fillna(0)
        
        # 新增特征3：波动收缩突破（布林带宽度/ATR比值）
        boll_width = df['BOLL_upper'] - df['BOLL_lower']
        atr_ma = df['ATR'].rolling(10).mean()
        df['Volatility_Squeeze_Break'] = (
            (boll_width / atr_ma < 0.8) &  # 波动收缩
            (df['收盘'] > df['BOLL_upper'])  # 突破上轨
        ).astype(float).fillna(0)

        # 1. 聪明资金流
        df['SmartMoneyFlow'] = smart_money_flow(df['最高'], df['最低'], df['收盘'], df['成交量'])
        
        # 2. 趋势持续性
        df['TrendPersistence'] = trend_persistence(df['收盘'])
        
        # 4. 异常成交量
        df['VolumeSpike'] = volume_spike(df['成交量'])
        
        # 5. 斐波那契位
        fib_levels = fibonacci_levels(df['收盘'])
        df = pd.concat([df, fib_levels], axis=1)

        # 添加神奇九转特征
        nt_basic = smart_nine_turn(df['收盘'], df['成交量'])
        nt_div = nine_turn_divergence(df['收盘'], df['最低'], df['最高'])
        df['NT_Threshold'] = dynamic_nine_turn_threshold(df['收盘'])
        
        df = pd.concat([df, nt_basic, nt_div], axis=1)
        
        # 添加九转组合特征
        df['NT_Strength'] = df['NT_NetSignal'] * df['NT_Threshold']
        df['NT_VolumeRatio'] = df['成交量'] / df['成交量'].rolling(20).mean().shift(1)

        df['SmartMoneyDiv'] = calculate_smart_money_divergence(close, volume)
        df['VolCluster'] = calculate_volatility_clustering(close)
        df['LiquidityShock'] = calculate_liquidity_shock(close, volume)
        df['OrderFlow'] = calculate_order_flow(high, low, close, volume)
        df['SentimentExtreme'] = calculate_sentiment_extremes(df['RSI'], df['CCI'])

        df = add_return_features(df, window=30)  # 添加10天收益率序列
        df = add_volume_features(df, window=30)  # 添加10天成交量变化率

        if is_training:
            lookahead_days = 15
            # 计算未来15天内的最高收盘价（使用rolling+shift技巧）
            df['未来15日收益'] = df['收盘'].rolling(window=lookahead_days, min_periods=1).max().shift(-lookahead_days)
            # 保留原始数据长度
            df = df.iloc[:-lookahead_days]
            # 计算最大潜在收益率 = (未来15日最高价 - 当前价格)/当前价格
            df['未来15日收益'] = df['未来15日收益'] / df['收盘'] - 1
            # 删除包含NaN的行
            df = df.dropna(subset=['未来15日收益'])
        
        return df

    # 获取大盘数据
    hs300_data = ak.stock_zh_index_daily(symbol="sh000300")
    hs300_data['date'] = pd.to_datetime(hs300_data['date'])
    hs300_data = hs300_data.set_index('date').sort_index()
    hs300_data['未来收盘价'] = hs300_data['close'].shift(-10)
    hs300_data = hs300_data.dropna(subset=['未来收盘价'])
    hs300_data['未来15日收益'] = hs300_data['未来收盘价'] / hs300_data['close'] - 1

    # 获取训练数据
    print("\n获取训练数据...")
    train_features = []
    for symbol in symbols[:300]: 
        try:
            df = safe_get_stock_data(
                symbol=symbol,
                start_date=train_start_date.strftime('%Y%m%d'),
                end_date=train_end_date.strftime('%Y%m%d')
            )

            processed = process_data(df, hs300_data, is_training=True)
            # processed = check_data_quality(processed)
            if not processed.empty:
                processed['symbol'] = symbol
                train_features.append(processed)
                print(f"\r已处理 {len(train_features)}/{len(symbols)}", end="")
                
        except Exception as e:
            print(f"训练数据跳过 {symbol} ({name_map.get(symbol, '未知')}): {str(e)}")

    if not train_features:
        raise ValueError("没有获取到任何有效训练数据")

    # 合并训练数据
    train_df = pd.concat(train_features).set_index('日期')
    print(f"\n成功处理 {len(train_df)} 条训练数据")

    # 合并股票和大盘数据
    merged_train = pd.merge(
        train_df.reset_index(),
        hs300_data[['未来15日收益']].reset_index(),
        left_on='日期',
        right_on='date',
        how='inner'
    )

    if merged_train.empty:
        raise ValueError("训练数据日期对齐失败")

    # 动态生成标签
    merged_train['label'] = generate_dynamic_label(
        merged_train['未来15日收益_x'],
        merged_train['未来15日收益_y']
    )

    # 关键过滤步骤：删除中间地带的样本
    merged_train = merged_train.dropna(subset=['label']).copy()
    merged_train['label'] = merged_train['label'].astype(int)

    # 更新特征列定义
    feature_cols = ['MA5', 'MA10', 'MA20', 'MA30', 'RSI', 'MACD', 'MACD_Signal', 
                'BOLL_upper', 'BOLL_mid', 'BOLL_lower',
                'ATR', 'OBV', 'CCI', 'VOLATILITY',
                'INDUSTRY_RS', 'PV_DIVERGENCE', 'VOL_RATIO',
                'Momentum_Accel', 'Volume_Price_Resonance', 'Volatility_Squeeze_Break', 'SmartMoneyFlow',
                'TrendPersistence', 'VolumeSpike', 'FIB_0.236', 'FIB_0.382', 'FIB_0.618',
                'NT_BuyCount', 'NT_SellCount', 'NT_NetSignal',
                'NT_TopDiv', 'NT_BottomDiv', 'NT_Threshold',
                'NT_Strength', 'NT_VolumeRatio']
    feature_cols += ['SmartMoneyDiv', 'VolCluster', 'LiquidityShock', 
                'OrderFlow', 'SentimentExtreme']
    feature_cols += [f'Ret_{i}day' for i in range(1,31)] + \
                    [f'VolChg_{i}day' for i in range(1,31)]

    # 时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=10)
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(merged_train)):
        print(f"\n正在训练第 {fold+1} 折...")
        
        # 划分训练集和验证集
        X_train = merged_train.iloc[train_idx][feature_cols]
        y_train = merged_train.iloc[train_idx]['label']
        X_val = merged_train.iloc[val_idx][feature_cols]
        y_val = merged_train.iloc[val_idx]['label']

        # 中位数填充
        train_medians = X_train.median()

        # 处理训练集
        X_train = X_train.replace([np.inf, -np.inf], np.nan)  # 转换无限值为NaN
        X_train = X_train.fillna(train_medians)
        X_train = X_train.clip(-1e10, 1e10)  # 限制数值范围

        # 处理验证集
        X_val = X_val.replace([np.inf, -np.inf], np.nan)
        X_val = X_val.fillna(train_medians)
        X_val = X_val.clip(-1e10, 1e10)

        # 打印缺失值统计（在填充前）
        def print_nan_stats(df, dataset_name):
            na_percent = df.isna().mean() * 100
            total_na = df.isna().sum().sum()
            print(f"\n[{dataset_name}] 缺失值统计:")
            print(f"总缺失值数量: {total_na} ({total_na/df.size:.2%})")
            # print("各特征缺失比例：")
            # print(na_percent.sort_values(ascending=False).apply(lambda x: f"{x:.2f}%").to_string())
        
        print_nan_stats(X_train, "训练集")
        print_nan_stats(X_val, "验证集")

        X_train = X_train.fillna(train_medians)
        X_val = X_val.fillna(train_medians)

        # 再次验证填充后数据
        print("\n填充后缺失值检查:")
        print(f"训练集剩余缺失值: {X_train.isna().sum().sum()}")
        print(f"验证集剩余缺失值: {X_val.isna().sum().sum()}")

        # 在交叉验证循环中添加
        scaler = RobustScaler(quantile_range=(5, 95))  # 使用鲁棒缩放
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 样本权重（近期数据权重更高）
        train_dates = merged_train.iloc[train_idx]['日期']
        weights = np.linspace(0.9, 1.1, num=len(train_dates))
        
        # 训练模型
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=8000,
            max_depth=3,  # 减小深度防止过拟合
            learning_rate=0.2,
            subsample=0.7,
            colsample_bytree=0.5,
            reg_alpha=10,  # 增强L1正则化
            reg_lambda=10,  # 增强L2正则化
            min_child_weight=10,  # 防止过小的叶节点
            max_bin=16,  # 限制分箱数
            tree_method='hist',  # 使用直方图算法
            random_state=2025,
            early_stopping_rounds=500,
            eval_metric=['logloss', 'auc'],
            enable_categorical=False,
            missing=np.nan  # 显式处理缺失值
        )
        
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            sample_weight=weights,
            verbose=50
        )

        # ✅ 修正：计算训练集和验证集得分
        train_score = model.score(X_train_scaled, y_train)
        val_score = model.score(X_val_scaled, y_val)
        print(f"训练集得分: {train_score:.4f}, 验证集得分: {val_score:.4f}")
        
        # ✅ 修正：使用缩放后的数据进行预测
        y_pred = model.predict(X_val_scaled)
        y_proba = model.predict_proba(X_val_scaled)[:,1]

        from sklearn.metrics import roc_auc_score, classification_report

        # 计算AUC和其他指标
        train_proba = model.predict_proba(X_train_scaled)[:,1]
        val_proba = model.predict_proba(X_val_scaled)[:,1]

        train_auc = roc_auc_score(y_train, train_proba)
        val_auc = roc_auc_score(y_val, val_proba)

        print(f"训练集AUC: {train_auc:.4f}, 验证集AUC: {val_auc:.4f}")
        
        # 存储当前折的结果
        val_results = X_val.copy()
        val_results['实际标签'] = y_val
        val_results['预测概率'] = y_proba
        val_results['预测标签'] = y_pred
        val_results['未来15日收益_x'] = merged_train.iloc[val_idx]['未来15日收益_x']
        val_results['未来15日收益_y'] = merged_train.iloc[val_idx]['未来15日收益_y']
        
        # 执行回测
        bt_result = backtest_evaluation(
            val_results, y_proba, 
            '未来15日收益_x', '未来15日收益_y'
        )
        cv_results.append(bt_result)
        
        # 打印当前折结果
        print(f"第 {fold+1} 折回测结果:")
        print(f"超额收益: {bt_result['excess_return']:.2%}")
        print(f"胜率: {bt_result['win_rate']:.2%}")
        print(f"夏普比率: {bt_result['sharpe_ratio']:.2f}")

        # 新增指标（如果存在）
        if 'treynor_ratio' in bt_result:
            print(f"特雷诺比率: {bt_result['treynor_ratio']:.2f}")
        if 'annualized_excess' in bt_result:
            print(f"年化超额收益: {bt_result['annualized_excess']:.2%}")

    # 计算交叉验证平均结果
    avg_excess_return = np.mean([r['excess_return'] for r in cv_results])
    avg_win_rate = np.mean([r['win_rate'] for r in cv_results])
    
    print("\n交叉验证平均结果:")
    print(f"平均超额收益: {avg_excess_return:.2%}")
    print(f"平均胜率: {avg_win_rate:.2%}")

    # 使用全部数据训练最终模型
    print("\n准备最终模型训练数据...")
    
    # 数据预处理
    X_final = merged_train[feature_cols].copy()
    y_final = merged_train['label'].copy()

    # 处理无限大值和缺失值
    X_final = X_final.replace([np.inf, -np.inf], np.nan)
    medians = X_final.median()
    X_final = X_final.fillna(medians)
    X_final = X_final.clip(-1e10, 1e10)

    # 验证数据
    print("\n最终模型训练数据验证:")
    print(f"剩余缺失值数量: {X_final.isna().sum().sum()}")
    print(f"无限大值数量: {np.isinf(X_final).sum().sum()}")

    # 训练最终模型
    print("\n训练最终模型...")
    final_model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=8000,
            max_depth=3,  # 减小深度防止过拟合
            learning_rate=0.2,
            subsample=0.7,
            colsample_bytree=0.5,
            reg_alpha=10,  # 增强L1正则化
            reg_lambda=10,  # 增强L2正则化
            min_child_weight=10,  # 防止过小的叶节点
            max_bin=16,  # 限制分箱数
            tree_method='hist',  # 使用直方图算法
            random_state=2025,
            enable_categorical=False,
            missing=np.nan  # 显式处理缺失值
    )

    final_model.fit(
        X_final, 
        y_final,
        verbose=50
    )

    # 获取最新数据用于推理
    print("\n获取最新数据用于推理...")
    predict_features = []
    count = 0  # 添加计数器变量
    for symbol in symbols[:300]:
        try:
            count += 1
            if count > 1 and count % 10 == 0:  # 修复这里：使用count代替i
                time.sleep(2)
            df = safe_get_stock_data(
                symbol=symbol,
                start_date=train_start_date.strftime('%Y%m%d'),
                end_date=train_end_date.strftime('%Y%m%d')
            )
            if len(df) < 30:
                print(f"推理数据跳过 {symbol} ({name_map.get(symbol, '未知')}): 数据不足 ({len(df)}天)")
                continue
                
            processed = process_data(df, hs300_data, is_training=False)
            if not processed.empty:
                processed = processed.tail(3)
                processed['symbol'] = symbol
                predict_features.append(processed)
                print(f"\r已处理 {len(predict_features)}/{len(symbols)}", end="")
                
        except Exception as e:
            print(f"推理数据跳过 {symbol} ({name_map.get(symbol, '未知')}): {str(e)}")

    if not predict_features:
        raise ValueError("没有获取到任何有效推理数据")

    # 合并推理数据
    predict_df = pd.concat(predict_features).set_index('日期')
    print(f"\n成功处理 {len(predict_df)} 只股票的最新数据")

    # 预测前处理推理数据
    X_predict = predict_df[feature_cols].copy()
    X_predict = X_predict.replace([np.inf, -np.inf], np.nan)
    X_predict = X_predict.fillna(medians)  # 使用训练集中位数填充
    X_predict = X_predict.clip(-1e10, 1e10)

    # 使用3天加权平均进行预测
    print("\n使用3天加权平均进行预测...")
    symbol_probas = {}
    symbol_daily_scores = {}  # 新增：存储每天得分

    # 按股票代码分组处理
    for symbol, group in predict_df.groupby('symbol'):
        # 取最后3天数据
        recent_data = group.tail(3)
        
        if len(recent_data) == 0:
            continue
            
        # 设置权重（最近的一天权重最高）
        if len(recent_data) == 3:
            weights = [0.2, 0.3, 0.5]  # 3天权重
        elif len(recent_data) == 2:
            weights = [0.3, 0.7]  # 2天权重
        else:
            weights = [1.0]  # 1天权重
        
        # 准备特征数据
        X_recent = recent_data[feature_cols].copy()
        X_recent = X_recent.replace([np.inf, -np.inf], np.nan)
        X_recent = X_recent.fillna(medians)
        X_recent = X_recent.clip(-1e10, 1e10)
        
        # 预测每条数据的概率
        daily_probas = final_model.predict_proba(X_recent)[:, 1]
        
        # 计算加权平均
        weighted_proba = np.average(daily_probas, weights=weights)
        symbol_probas[symbol] = weighted_proba
        
        # 新增：存储每天得分和日期
        dates = recent_data.index.strftime('%m-%d').tolist()  # 只显示月-日
        symbol_daily_scores[symbol] = {
            'weighted': weighted_proba,
            'daily': list(zip(dates, daily_probas))[-3:]  # 只保留最后3天
        }

    # 创建结果DataFrame
    result_data = []
    for symbol, proba in symbol_probas.items():
        # 获取该股票的每日得分
        daily_info = symbol_daily_scores.get(symbol, {})
        daily_scores = daily_info.get('daily', [])
        
        # 提取最近3天的得分
        score_info = {}
        for i, (date, score) in enumerate(daily_scores[-3:], 1):  # 只取最近3天
            score_info[f'D{i}得分'] = score
            score_info[f'D{i}日期'] = date
        
        result_data.append({
            '股票代码': symbol,
            '股票名称': name_map.get(symbol, '未知'),
            '加权得分': proba,
            **score_info  # 加入每日得分
        })

    result_df = pd.DataFrame(result_data).sort_values('加权得分', ascending=False).head(30)
    result_df_lose = pd.DataFrame(result_data).sort_values('加权得分', ascending=True).head(10)

    print(f"加权平均预测完成，共处理 {len(symbol_probas)} 只股票")

    # 输出结果
    print("\n特征重要性: ")
    importance = final_model.get_booster().get_score(importance_type='weight')
    for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v}")

    print("\nTop 30预测结果（基于最新数据）: ")
    print(result_df.to_string(index=False, float_format="%.4f"))

    print("\n倒数Top 10预测结果（基于最新数据）: ")
    print(result_df_lose.to_string(index=False, float_format="%.4f"))

    # 最健壮的合并方案
    detailed_df = predict_df.reset_index().merge(
        result_df,
        left_on='symbol',
        right_on='股票代码',
        how='right'
    )

    # 确保股票代码列存在
    if '股票代码' not in detailed_df.columns:
        detailed_df['股票代码'] = detailed_df['symbol'] if 'symbol' in detailed_df.columns else detailed_df.index

    # 打印函数最终版
    def print_stock_features(stock_code, features_df, feature_cols, proba, daily_scores):
        try:
            print("\n" + "="*80)
            print(f"股票代码: {stock_code} | 股票名称: {name_map.get(stock_code, '未知')}")
            print(f"加权得分: {proba:.4f}")
            
            # 打印每日得分
            if daily_scores:
                print("-"*40 + " 每日得分趋势 " + "-"*40)
                sorted_dates = sorted(daily_scores.keys())
                for date in sorted_dates[-3:]:  # 只显示最近3天
                    print(f"{date}: {daily_scores[date]:.4f}")
            
            print("-"*40 + " 特征详情 " + "-"*40)
            
            importance = final_model.get_booster().get_score(importance_type='weight')
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            # 获取该股票数据
            stock_data = features_df[features_df['股票代码'] == stock_code].iloc[0]
            
            for i in range(0, len(sorted_features), 5):
                batch = sorted_features[i:i+5]
                for feat, imp in batch:
                    if feat in stock_data:
                        print(f"{feat:>25}: {stock_data[feat]:>10.4f} (重要性: {imp:.0f})", end=" | ")
                    else:
                        print(f"{feat:>25}: {'N/A':>10} (重要性: {imp:.0f})", end=" | ")
                print()
        except Exception as e:
            print(f"\n打印股票 {stock_code} 特征时出错: {str(e)}")
        finally:
            print("="*80)
    
    # # 对每只股票调用打印函数 
    # for _, row in result_df.iterrows():
    #     print_stock_features(
    #         stock_code=row['股票代码'],
    #         features_df=detailed_df,
    #         feature_cols=feature_cols,
    #         proba=row['跑赢概率']
    #     )

    # 添加历史时间点截面分析
    historical_dates = ['20240624', '20241224', '20230624', '20231224', '20220624', '20221224']  # 示例日期

    # 获取训练集中的所有日期
    all_dates = merged_train['日期'].unique()
    all_dates = sorted(all_dates)

    # 每隔7天选取一个日期
    selected_dates = []
    for i in range(0, len(all_dates), 7):
        if i < len(all_dates):
            selected_dates.append(all_dates[i].strftime('%Y%m%d'))
        if len(selected_dates) >= 300:  # 只取100个日期
            break
    
    print("\n历史时间点截面分析:")
    for date_str in selected_dates:
        try:
            # 转换为datetime对象
            target_date = datetime.strptime(date_str, '%Y%m%d')
            
            # 从训练数据中提取该日期的截面数据
            date_data = merged_train[merged_train['日期'] == target_date]
            
            if len(date_data) == 0:
                print(f"未找到 {date_str} 的数据")
                continue
                
            # 准备特征数据
            X_date = date_data[feature_cols].copy()
            X_date = X_date.replace([np.inf, -np.inf], np.nan)
            X_date = X_date.fillna(medians)  # 使用训练集中位数填充
            X_date = X_date.clip(-1e10, 1e10)
            
            # 预测
            probas = final_model.predict_proba(X_date)[:,1]
            
            # 找到TOP1股票
            top_idx = np.argmax(probas)
            top_stock = date_data.iloc[top_idx]
            
            # 输出结果
            print(f"{date_str}: {top_stock['symbol']} ({name_map.get(top_stock['symbol'], '未知')}), "
                  f"预测概率: {probas[top_idx]:.4f}, "
                  f"实际15日收益: {top_stock['未来15日收益_x']:.2%}")
        except Exception as e:
            print(f"处理 {date_str} 时出错: {str(e)}")

if __name__ == "__main__":
    main()
