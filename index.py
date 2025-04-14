import akshare as ak
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

# 技术指标计算函数（保持不变）
def calculate_ma(series, window):
    return series.rolling(window, min_periods=1).mean().ffill()

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window, min_periods=1).mean().ffill()
    avg_loss = loss.rolling(window, min_periods=1).mean().ffill()
    rs = avg_gain / (avg_loss + 1e-8)
    return (100 - (100 / (1 + rs))).fillna(50)

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

def calculate_cci(high, low, close, window=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(window, min_periods=1).mean().ffill()
    mad = tp.rolling(window, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), 
        raw=False
    ).ffill()
    return ((tp - sma) / (0.015 * (mad + 1e-8))).fillna(0)

def main():
    # 获取最新有效日期
    def get_valid_date():
        hs300 = ak.stock_zh_index_daily(symbol="sh000300")
        return pd.to_datetime(hs300['date'].iloc[-1])
    
    # 设置时间窗口
    end_date = get_valid_date()
    train_end_date = end_date - timedelta(days=30)  # 1个月前
    train_start_date = end_date - timedelta(days=1000)  # 1000天前
    predict_start_date = end_date - timedelta(days=60)  # 获取足够数据计算指标
    predict_end_date = end_date
    
    print(f"训练数据时间窗口: {train_start_date.strftime('%Y%m%d')} - {train_end_date.strftime('%Y%m%d')}")
    print(f"推理数据时间窗口: {predict_start_date.strftime('%Y%m%d')} - {predict_end_date.strftime('%Y%m%d')}")

    # 获取沪深300指数数据
    print("\n获取沪深300指数数据...")
    hs300_data = ak.stock_zh_index_daily(symbol="sh000300")
    hs300_data['date'] = pd.to_datetime(hs300_data['date'])
    hs300_data = hs300_data.set_index('date').sort_index()
    
    # 数据处理函数
    def process_index_data(df, is_training=True):
        df = df.copy()
        
        # 基础价格数据
        close = df['close'].ffill()
        high = df['high'].ffill()
        low = df['low'].ffill()
        volume = df['volume'].ffill()

        # 计算技术指标
        df['MA5'] = calculate_ma(close, 5)
        df['MA20'] = calculate_ma(close, 20)
        df['RSI'] = calculate_rsi(close)
        df['MACD'], df['MACD_Signal'] = calculate_macd(close)
        
        # 布林带计算
        bollinger_df = calculate_bollinger(close)
        df['BOLL_upper'] = bollinger_df['BOLL_upper']
        df['BOLL_mid'] = bollinger_df['BOLL_mid']
        df['BOLL_lower'] = bollinger_df['BOLL_lower']
        
        df['ATR'] = calculate_atr(high, low, close)
        df['OBV'] = calculate_obv(close, volume)
        df['CCI'] = calculate_cci(high, low, close)
        df['VOLATILITY'] = close.pct_change().rolling(20).std().ffill()
        
        if is_training:
            lookahead_days = 5  # 预测未来5个交易日(1周)
            df['未来收盘价'] = df['close'].shift(-lookahead_days)
            df = df.iloc[:-lookahead_days]  # 移除最后几天没有未来数据
            df['未来5日收益'] = df['未来收盘价'] / df['close'] - 1
            df['label'] = (df['未来5日收益'] > 0).astype(int)  # 1表示上涨，0表示下跌
            df = df.dropna(subset=['label'])
        
        return df

    # 准备训练数据
    print("\n准备训练数据...")
    train_data = hs300_data.loc[train_start_date:train_end_date]
    processed_train = process_index_data(train_data, is_training=True)
    
    if processed_train.empty:
        raise ValueError("没有获取到任何有效训练数据")
    
    print(f"成功处理 {len(processed_train)} 条训练数据")

    # 特征工程
    feature_cols = ['MA5', 'MA20', 'RSI', 'MACD', 'MACD_Signal', 
                   'BOLL_upper', 'BOLL_mid', 'BOLL_lower',
                   'ATR', 'OBV', 'CCI', 'VOLATILITY']

    # 拆分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        processed_train[feature_cols], 
        processed_train['label'],
        test_size=0.2,
        stratify=processed_train['label'],
        random_state=2025
    )

    # 训练模型
    print("\n训练XGBoost模型...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=2025
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        # early_stopping_rounds=30,
        verbose=20
    )

    # 准备预测数据
    print("\n准备预测数据...")
    predict_data = hs300_data.loc[predict_start_date:predict_end_date]
    processed_predict = process_index_data(predict_data, is_training=False).tail(1)
    
    if processed_predict.empty:
        raise ValueError("没有获取到任何有效预测数据")

    # 预测
    proba = model.predict_proba(processed_predict[feature_cols])[:,1][0]
    prediction = model.predict(processed_predict[feature_cols])[0]
    
    # 输出结果
    print("\n特征重要性：")
    importance = model.get_booster().get_score(importance_type='weight')
    for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v}")

    print("\n沪深300指数预测结果：")
    print(f"最新日期: {processed_predict.index[0].strftime('%Y-%m-%d')}")
    print(f"预测未来一周上涨概率: {proba:.2%}")
    print(f"预测方向: {'上涨' if prediction == 1 else '下跌'}")

if __name__ == "__main__":
    main()