import akshare as ak
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# 技术指标计算函数（保持不变）
def calculate_ma(series, window)
    return series.rolling(window, min_periods=1).mean().ffill()

def calculate_rsi(series, window=14)
    delta = series.diff()
    gain = delta.where(delta  0, 0)
    loss = -delta.where(delta  0, 0)
    avg_gain = gain.rolling(window, min_periods=1).mean().ffill()
    avg_loss = loss.rolling(window, min_periods=1).mean().ffill()
    rs = avg_gain  (avg_loss + 1e-8)
    return (100 - (100  (1 + rs))).fillna(50)

def calculate_macd(series, fast=12, slow=26, signal=9)
    ema_fast = series.ewm(span=fast, min_periods=1).mean().ffill()
    ema_slow = series.ewm(span=slow, min_periods=1).mean().ffill()
    macd = (ema_fast - ema_slow).ffill()
    signal_line = macd.ewm(span=signal, min_periods=1).mean().ffill()
    return macd, signal_line

def calculate_bollinger(series, window=20, num_std=2)
    rolling_mean = series.rolling(window, min_periods=1).mean().ffill()
    rolling_std = series.rolling(window, min_periods=1).std().ffill()
    upper = (rolling_mean + (rolling_std  num_std)).ffill()
    lower = (rolling_mean - (rolling_std  num_std)).ffill()
    return pd.DataFrame({
        'BOLL_upper' upper,
        'BOLL_mid' rolling_mean,
        'BOLL_lower' lower
    })


def calculate_atr(high, low, close, window=14)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=1).mean().ffill()

def calculate_obv(close, volume)
    return (np.sign(close.diff())  volume).ffill().cumsum().fillna(0)

def calculate_cci(high, low, close, window=20)
    tp = (high + low + close)  3
    sma = tp.rolling(window, min_periods=1).mean().ffill()
    mad = tp.rolling(window, min_periods=1).apply(
        lambda x np.mean(np.abs(x - np.mean(x))), 
        raw=False
    ).ffill()
    return ((tp - sma)  (0.015  (mad + 1e-8))).fillna(0)


def main()
    # 获取最新有效日期
    def get_valid_date()
        hs300 = ak.stock_zh_index_daily(symbol=sh000300)
        return pd.to_datetime(hs300['date'].iloc[-1])
    
    # 设置时间窗口
    end_date = get_valid_date()
    train_end_date = end_date - timedelta(days=30)
    train_start_date = end_date - timedelta(days=2000)
    predict_start_date = end_date - timedelta(days=60)
    predict_end_date = end_date
    
    print(f训练数据时间窗口 {train_start_date.strftime('%Y%m%d')} - {train_end_date.strftime('%Y%m%d')})
    print(f推理数据时间窗口 {predict_start_date.strftime('%Y%m%d')} - {predict_end_date.strftime('%Y%m%d')})

    # 获取沪深300成分股
    hs300 = ak.index_stock_cons_csindex(symbol=000300)
    symbols = hs300['成分券代码'].tolist()
    name_map = dict(zip(hs300['成分券代码'], hs300['成分券名称']))

    # 数据处理函数（训练和推理通用）
    def process_data(df, is_training=True)
        df = df.copy()
        df['日期'] = pd.to_datetime(df['日期'])
        
        # 基础价格数据
        close = df['收盘'].ffill()
        high = df['最高'].ffill()
        low = df['最低'].ffill()
        volume = df['成交量'].ffill()

        # 计算技术指标
        df['MA5'] = calculate_ma(close, 5)
        df['MA20'] = calculate_ma(close, 20)
        df['RSI'] = calculate_rsi(close)
        df['MACD'], df['MACD_Signal'] = calculate_macd(close)
        
        # 修改布林带计算方式
        bollinger_df = calculate_bollinger(close)
        df['BOLL_upper'] = bollinger_df['BOLL_upper']
        df['BOLL_mid'] = bollinger_df['BOLL_mid']
        df['BOLL_lower'] = bollinger_df['BOLL_lower']
        
        df['ATR'] = calculate_atr(high, low, close)
        df['OBV'] = calculate_obv(close, volume)
        df['CCI'] = calculate_cci(high, low, close)
        df['VOLATILITY'] = close.pct_change().rolling(20).std().ffill()
        
        if is_training
            lookahead_days = 5
            df['未来收盘价'] = df['收盘'].shift(-lookahead_days)
            df = df.iloc[-lookahead_days]
            df['未来5日收益'] = df['未来收盘价']  df['收盘'] - 1
            df = df.dropna(subset=['未来5日收益'])
        
        return df

    # 获取训练数据（单线程）
    print(n获取训练数据...)
    train_features = []
    for symbol in symbols
        try
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                start_date=train_start_date.strftime('%Y%m%d'),
                end_date=train_end_date.strftime('%Y%m%d'),
                adjust=qfq
            )
            if len(df)  100
                print(f训练数据跳过 {symbol} ({name_map.get(symbol, '未知')}) 数据不足 ({len(df)}天))
                continue
                
            processed = process_data(df, is_training=True)
            if not processed.empty
                processed['symbol'] = symbol
                train_features.append(processed)
                print(fr已处理 {len(train_features)}{len(symbols)}, end=)
                
        except Exception as e
            print(f训练数据跳过 {symbol} ({name_map.get(symbol, '未知')}) {str(e)})

    if not train_features
        raise ValueError(没有获取到任何有效训练数据)

    # 合并训练数据
    train_df = pd.concat(train_features).set_index('日期')
    print(fn成功处理 {len(train_df)} 条训练数据)

    # 获取并处理大盘数据
    hs300_train = ak.stock_zh_index_daily(symbol=sh000300)
    hs300_train['date'] = pd.to_datetime(hs300_train['date'])
    hs300_train = hs300_train.set_index('date').sort_index()
    hs300_train = hs300_train.loc[train_start_datetrain_end_date]
    hs300_train['未来收盘价'] = hs300_train['close'].shift(-5)
    hs300_train = hs300_train.dropna(subset=['未来收盘价'])
    hs300_train['未来5日收益'] = hs300_train['未来收盘价']  hs300_train['close'] - 1

    # 合并股票和大盘数据
    merged_train = pd.merge(
        train_df.reset_index(),
        hs300_train[['未来5日收益']].reset_index(),
        left_on='日期',
        right_on='date',
        how='inner'
    )

    if merged_train.empty
        raise ValueError(训练数据日期对齐失败)

    # 特征工程
    merged_train['超额收益'] = merged_train['未来5日收益_x'] - merged_train['未来5日收益_y']
    merged_train['label'] = (merged_train['超额收益'] = 0.03).astype(int)
    feature_cols = ['MA5', 'MA20', 'RSI', 'MACD', 'MACD_Signal', 
                   'BOLL_upper', 'BOLL_mid', 'BOLL_lower',
                   'ATR', 'OBV', 'CCI', 'VOLATILITY']

    # 拆分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        merged_train[feature_cols], 
        merged_train['label'],
        test_size=0.2,
        stratify=merged_train['label'],
        random_state=2025
    )

    # 训练模型
    print(n训练XGBoost模型...)
    model = xgb.XGBClassifier(
        objective='binarylogistic',
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=2025,
        early_stopping_rounds=30,
        eval_metric='logloss'  # 添加评估指标
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=20
    )

    # 获取最新数据用于推理（单线程）
    print(n获取最新数据用于推理...)
    predict_features = []
    for symbol in symbols
        try
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                start_date=predict_start_date.strftime('%Y%m%d'),
                end_date=predict_end_date.strftime('%Y%m%d'),
                adjust=qfq
            )
            if len(df)  30
                print(f推理数据跳过 {symbol} ({name_map.get(symbol, '未知')}) 数据不足 ({len(df)}天))
                continue
                
            processed = process_data(df, is_training=False).tail(1)
            if not processed.empty
                processed['symbol'] = symbol
                predict_features.append(processed)
                print(fr已处理 {len(predict_features)}{len(symbols)}, end=)
                
        except Exception as e
            print(f推理数据跳过 {symbol} ({name_map.get(symbol, '未知')}) {str(e)})

    if not predict_features
        raise ValueError(没有获取到任何有效推理数据)

    # 合并推理数据
    predict_df = pd.concat(predict_features).set_index('日期')
    print(fn成功处理 {len(predict_df)} 只股票的最新数据)

    # 预测
    probas = model.predict_proba(predict_df[feature_cols])[,1]
    result_df = pd.DataFrame({
        '股票代码' predict_df['symbol'],
        '股票名称' predict_df['symbol'].map(name_map),
        '跑赢概率' probas
    }).sort_values('跑赢概率', ascending=False).head(20)

    # 输出结果
    print(n特征重要性：)
    importance = model.get_booster().get_score(importance_type='weight')
    for k, v in sorted(importance.items(), key=lambda x x[1], reverse=True)
        print(f{k} {v})

    print(nTop 20预测结果（基于最新数据）：)
    print(result_df.to_string(index=False, float_format=%.4f))

if __name__ == __main__
    main()