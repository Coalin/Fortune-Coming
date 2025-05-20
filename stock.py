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


# 技术指标计算函数（增强版）
def calculate_ma(series, window):
    return series.rolling(window, min_periods=1).mean().ffill()

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 添加平滑处理
    avg_gain = gain.rolling(window, min_periods=1).mean().ffill().clip(lower=1e-8)
    avg_loss = loss.rolling(window, min_periods=1).mean().ffill().clip(lower=1e-8)
    
    rs = avg_gain / (avg_loss + 1e-8)  # 防止除以0
    return (100 - (100 / (1 + rs))).fillna(50).clip(0, 100)

def calculate_cci(high, low, close, window=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(window, min_periods=1).mean().ffill()
    
    # 使用更稳健的MAD计算
    def robust_mad(x):
        med = np.median(x)
        return np.median(np.abs(x - med))
    
    mad = tp.rolling(window).apply(robust_mad, raw=True).ffill().clip(lower=1e-8)
    return ((tp - sma) / (0.015 * mad)).fillna(0).clip(-200, 200)

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, min_periods=1).mean().ffill()
    ema_slow = series.ewm(span=slow, min_periods=1).mean().ffill()
    macd = (ema_fast - ema_slow).ffill()
    signal_line = macd.ewm(span=signal, min_periods=1).mean().ffill()
    return macd, signal_line

def calculate_bollinger(series, window=20, num_std=2):
    rolling_mean = series.rolling(window, min_periods=1).mean().ffill()
    rolling_std = series.rolling(window, min_periods=1).std().ffill()
    upper = (rolling_mean + (rolling_std * num_std)).ffill()
    lower = (rolling_mean - (rolling_std * num_std)).ffill()
    return pd.DataFrame({
        'BOLL_upper': upper,
        'BOLL_mid': rolling_mean,
        'BOLL_lower': lower
    })

def calculate_atr(high, low, close, window=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=1).mean().ffill()

def calculate_obv(close, volume):
    return (np.sign(close.diff()) * volume).ffill().cumsum().fillna(0)

# 新增特征计算函数
def industry_relative_strength(close, index_close, window=10):
    stock_ma = close.rolling(window).mean()
    index_ma = index_close.rolling(window).mean()
    return (stock_ma / index_ma).fillna(1)

def price_volume_divergence(close, volume, window=5):
    """确保输入为数值类型"""
    close = pd.to_numeric(close, errors='coerce')
    volume = pd.to_numeric(volume, errors='coerce')
    
    price_change = close.astype(float).pct_change(window)
    volume_change = volume.astype(float).pct_change(window)
    return (price_change - volume_change).fillna(0)

def volatility_ratio(close, short_window=5, long_window=20):
    short_vol = close.pct_change().rolling(short_window).std()
    long_vol = close.pct_change().rolling(long_window).std()
    return (short_vol / long_vol).fillna(1)

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
    cond_abs_gain = (stock_return >= 0.1)
    
    # 生成标签
    labels = (cond_outperform & cond_abs_gain).astype(int)
    
    # 打印各条件触发比例（调试用）
    if len(labels) > 0:
        print("\n标签生成统计:")
        print(f"动态阈值触发: {cond_dynamic.mean():.2%}")
        print(f"跑赢3%触发: {cond_outperform.mean():.2%}")
        print(f"上涨5%触发: {cond_abs_gain.mean():.2%}")
        print(f"最终正样本比例: {labels.mean():.2%}")
    
    return labels

# 回测评估函数
def backtest_evaluation(results_df, pred_proba, true_return, index_return, top_pct=0.1):
    """
    完整回测评估逻辑（考虑交易成本）
    """
    results_df['pred_proba'] = pred_proba
    results_df = results_df.sort_values('pred_proba', ascending=False)
    
    # 选取前10%股票
    top_stocks = results_df.iloc[:int(len(results_df)*top_pct)]
    
    # 计算实际收益（考虑0.01%交易成本）
    net_return = top_stocks[true_return] - 0.0001
    benchmark_return = top_stocks[index_return]
    
    # 计算关键指标
    excess_return = (net_return - benchmark_return).mean()
    win_rate = (net_return > benchmark_return).mean()
    sharpe_ratio = excess_return / (net_return.std() + 1e-8)
    
    return {
        'excess_return': excess_return,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'num_trades': len(top_stocks)
    }

def main():
    # 获取最新有效日期
    def get_valid_date():
        hs300 = ak.stock_zh_index_daily(symbol="sh000300")
        return pd.to_datetime(hs300['date'].iloc[-1])
    
    # 设置时间窗口
    end_date = get_valid_date()
    train_end_date = end_date - timedelta(days=60)
    train_start_date = end_date - timedelta(days=2000)
    predict_start_date = end_date - timedelta(days=60)
    predict_end_date = end_date
    
    print(f"训练数据时间窗口: {train_start_date.strftime('%Y%m%d')} - {train_end_date.strftime('%Y%m%d')}")
    print(f"推理数据时间窗口: {predict_start_date.strftime('%Y%m%d')} - {predict_end_date.strftime('%Y%m%d')}")

    # 获取沪深300成分股
    hs300 = ak.index_stock_cons_csindex(symbol="000300")
    symbols = hs300['成分券代码'].tolist()
    name_map = dict(zip(hs300['成分券代码'], hs300['成分券名称']))

    # 数据处理函数（增强版）
    def process_data(df, index_data, is_training=True):
        # 确保所有数值列都是float类型
        numeric_cols = ['开盘', '最高', '最低', '收盘', '成交量']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        df = df.copy()
        df['日期'] = pd.to_datetime(df['日期'])
        
        # 基础价格数据
        close = df['收盘'].ffill()
        high = df['最高'].ffill()
        low = df['最低'].ffill()
        volume = df['成交量'].ffill()
        
        # 计算技术指标（核心）
        df['MA5'] = calculate_ma(close, 5)
        df['MA20'] = calculate_ma(close, 20)
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
        
        if is_training:
            lookahead_days = 15
            df['未来收盘价'] = df['收盘'].shift(-lookahead_days)
            df = df.iloc[:-lookahead_days]
            df['未来15日收益'] = df['未来收盘价'] / df['收盘'] - 1
            df = df.dropna(subset=['未来15日收益'])
        
        return df

    # 获取大盘数据
    hs300_data = ak.stock_zh_index_daily(symbol="sh000300")
    hs300_data['date'] = pd.to_datetime(hs300_data['date'])
    hs300_data = hs300_data.set_index('date').sort_index()
    hs300_data['未来收盘价'] = hs300_data['close'].shift(-15)
    hs300_data = hs300_data.dropna(subset=['未来收盘价'])
    hs300_data['未来15日收益'] = hs300_data['未来收盘价'] / hs300_data['close'] - 1

    # 获取训练数据
    print("\n获取训练数据...")
    train_features = []
    for symbol in symbols[:300]: 
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                start_date=train_start_date.strftime('%Y%m%d'),
                end_date=train_end_date.strftime('%Y%m%d'),
                adjust="qfq"
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

    # 更新特征列定义
    feature_cols = ['MA5', 'MA20', 'RSI', 'MACD', 'MACD_Signal', 
                'BOLL_upper', 'BOLL_mid', 'BOLL_lower',
                'ATR', 'OBV', 'CCI', 'VOLATILITY',
                'INDUSTRY_RS', 'PV_DIVERGENCE', 'VOL_RATIO',
                'Momentum_Accel', 'Volume_Price_Resonance', 'Volatility_Squeeze_Break', 'SmartMoneyFlow',
                'TrendPersistence', 'VolumeSpike', 'FIB_0.236', 'FIB_0.382', 'FIB_0.618',
                'NT_BuyCount', 'NT_SellCount', 'NT_NetSignal',
                'NT_TopDiv', 'NT_BottomDiv', 'NT_Threshold',
                'NT_Strength', 'NT_VolumeRatio']

    # 时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=5)
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
            print("各特征缺失比例：")
            print(na_percent.sort_values(ascending=False).apply(lambda x: f"{x:.2f}%").to_string())
        
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
            n_estimators=5000,
            max_depth=5,  # 减小深度防止过拟合
            learning_rate=0.01,
            subsample=0.7,
            colsample_bytree=0.6,
            reg_alpha=0.1,  # 增强L1正则化
            reg_lambda=0.1,  # 增强L2正则化
            min_child_weight=2,  # 防止过小的叶节点
            max_bin=64,  # 限制分箱数
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
        
        # 验证集预测
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:,1]
        
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
        n_estimators=int(2000 * 1.2),
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=2025,
        missing=np.nan
    )

    final_model.fit(
        X_final, 
        y_final,
        verbose=50
    )

    # 获取最新数据用于推理
    print("\n获取最新数据用于推理...")
    predict_features = []
    for symbol in symbols[:300]:
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                start_date=predict_start_date.strftime('%Y%m%d'),
                end_date=predict_end_date.strftime('%Y%m%d'),
                adjust="qfq"
            )
            if len(df) < 30:
                print(f"推理数据跳过 {symbol} ({name_map.get(symbol, '未知')}): 数据不足 ({len(df)}天)")
                continue
                
            processed = process_data(df, hs300_data, is_training=False).tail(1)
            if not processed.empty:
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

    # 预测
    probas = final_model.predict_proba(X_predict)[:,1]
    result_df = pd.DataFrame({
        '股票代码': predict_df['symbol'],
        '股票名称': predict_df['symbol'].map(name_map),
        '跑赢概率': probas
    }).sort_values('跑赢概率', ascending=False).head(20)

    # 输出结果
    print("\n特征重要性: ")
    importance = final_model.get_booster().get_score(importance_type='weight')
    for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v}")

    print("\nTop 20预测结果（基于最新数据）: ")
    print(result_df.to_string(index=False, float_format="%.4f"))

if __name__ == "__main__":
    main()