import akshare as ak
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from datetime import datetime, timedelta
from logger import init_logger, close_logger, get_logger

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

def calculate_kdj(high, low, close, window=9):
    low_n = low.rolling(window, min_periods=1).min()
    high_n = high.rolling(window, min_periods=1).max()
    rsv = (close - low_n) / (high_n - low_n + 1e-8) * 100
    k = rsv.ewm(com=2, adjust=False).mean().fillna(50)
    d = k.ewm(com=2, adjust=False).mean().fillna(50)
    j = 3 * k - 2 * d
    return k, d, j

def calculate_williams_r(high, low, close, window=14):
    high_n = high.rolling(window, min_periods=1).max()
    low_n = low.rolling(window, min_periods=1).min()
    return ((high_n - close) / (high_n - low_n + 1e-8) * -100).fillna(-50)

def calculate_roc(close, window=12):
    return (close / close.shift(window) - 1).fillna(0) * 100

def calculate_adx(high, low, close, window=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    plus_di = 100 * (plus_dm.rolling(window).mean() / tr.rolling(window).mean())
    minus_di = 100 * (minus_dm.rolling(window).mean() / tr.rolling(window).mean())

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    adx = dx.rolling(window).mean()

    return adx.fillna(0)

def calculate_aroon(high, low, window=25):
    aroon_up = high.rolling(window + 1).apply(lambda x: x.argmax(), raw=True)
    aroon_down = low.rolling(window + 1).apply(lambda x: x.argmin(), raw=True)
    aroon_up = (window - aroon_up) / window * 100
    aroon_down = (window - aroon_down) / window * 100
    return aroon_up.fillna(0), aroon_down.fillna(0)

def calculate_stochastic(high, low, close, window=14):
    low_n = low.rolling(window, min_periods=1).min()
    high_n = high.rolling(window, min_periods=1).max()
    k = 100 * (close - low_n) / (high_n - low_n + 1e-8)
    d = k.rolling(3).mean()
    return k.fillna(50), d.fillna(50)

def process_index_data(df, lookahead_days=10, is_training=True):
    df = df.copy()

    close = df['close'].ffill()
    high = df['high'].ffill()
    low = df['low'].ffill()
    volume = df['volume'].ffill()

    df['MA5'] = calculate_ma(close, 5)
    df['MA10'] = calculate_ma(close, 10)
    df['MA20'] = calculate_ma(close, 20)
    df['MA30'] = calculate_ma(close, 30)
    df['MA60'] = calculate_ma(close, 60)

    df['RSI'] = calculate_rsi(close)
    df['RSI_7'] = calculate_rsi(close, window=7)
    df['RSI_21'] = calculate_rsi(close, window=21)

    df['MACD'], df['MACD_Signal'] = calculate_macd(close)
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    bollinger_df = calculate_bollinger(close)
    df['BOLL_upper'] = bollinger_df['BOLL_upper']
    df['BOLL_mid'] = bollinger_df['BOLL_mid']
    df['BOLL_lower'] = bollinger_df['BOLL_lower']
    df['BOLL_Width'] = (df['BOLL_upper'] - df['BOLL_lower']) / df['BOLL_mid']

    df['ATR'] = calculate_atr(high, low, close)
    df['ATR_Ratio'] = df['ATR'] / close * 100

    df['OBV'] = calculate_obv(close, volume)
    df['OBV_Change'] = df['OBV'].pct_change()

    df['CCI'] = calculate_cci(high, low, close)
    df['CCI_10'] = calculate_cci(high, low, close, window=10)
    df['CCI_30'] = calculate_cci(high, low, close, window=30)

    df['VOLATILITY'] = close.pct_change().rolling(20).std().ffill()
    df['VOLATILITY_10'] = close.pct_change().rolling(10).std().ffill()
    df['VOLATILITY_60'] = close.pct_change().rolling(60).std().ffill()

    df['K'], df['D'], df['J'] = calculate_kdj(high, low, close)
    df['Williams_R'] = calculate_williams_r(high, low, close)
    df['ROC'] = calculate_roc(close)
    df['ROC_6'] = calculate_roc(close, window=6)
    df['ADX'] = calculate_adx(high, low, close)

    df['Aroon_Up'], df['Aroon_Down'] = calculate_aroon(high, low)
    df['Aroon_Osc'] = df['Aroon_Up'] - df['Aroon_Down']

    df['Stoch_K'], df['Stoch_D'] = calculate_stochastic(high, low, close)

    df['Volume_MA5'] = calculate_ma(volume, 5)
    df['Volume_MA20'] = calculate_ma(volume, 20)
    df['Volume_Ratio'] = volume / df['Volume_MA20']

    df['Price_Momentum_5'] = close.pct_change(5)
    df['Price_Momentum_10'] = close.pct_change(10)
    df['Price_Momentum_20'] = close.pct_change(20)

    df['MA5_Trend'] = (df['MA5'] > df['MA5'].shift(1)).astype(int)
    df['MA10_Trend'] = (df['MA10'] > df['MA10'].shift(1)).astype(int)
    df['MA20_Trend'] = (df['MA20'] > df['MA20'].shift(1)).astype(int)

    df['Close_Position'] = (close - df['BOLL_lower']) / (df['BOLL_upper'] - df['BOLL_lower'] + 1e-8)

    if is_training:
        df['未来收盘价'] = df['close'].shift(-lookahead_days)
        df = df.iloc[:-lookahead_days]
        df['未来收益'] = df['未来收盘价'] / df['close'] - 1
        df['label'] = (df['未来收益'] > 0).astype(int)
        df = df.dropna(subset=['label'])

    return df

def threshold_analysis(y_true, y_proba, thresholds=None):
    if thresholds is None:
        thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    results = []
    for thresh in thresholds:
        mask = y_proba >= thresh
        if mask.sum() > 0:
            actual_up_rate = y_true[mask].mean()
            count = mask.sum()
        else:
            actual_up_rate = 0
            count = 0
        results.append({
            'threshold': thresh,
            'sample_count': count,
            'actual_up_rate': actual_up_rate,
            'total_samples': len(y_true)
        })
    return pd.DataFrame(results)

def main():
    log = init_logger(log_dir='./logs')
    log.section("大盘指数预测系统 - 预测未来2周涨跌")

    def get_valid_date():
        hs300 = ak.stock_zh_index_daily(symbol="sh000300")
        return pd.to_datetime(hs300['date'].iloc[-1])

    end_date = get_valid_date()
    train_end_date = end_date - timedelta(days=30)
    train_start_date = end_date - timedelta(days=1500)
    predict_start_date = end_date - timedelta(days=80)
    predict_end_date = end_date

    print(f"训练数据时间窗口: {train_start_date.strftime('%Y%m%d')} - {train_end_date.strftime('%Y%m%d')}")
    print(f"推理数据时间窗口: {predict_start_date.strftime('%Y%m%d')} - {predict_end_date.strftime('%Y%m%d')}")

    print("获取沪深300指数数据...")
    hs300_data = ak.stock_zh_index_daily(symbol="sh000300")
    hs300_data['date'] = pd.to_datetime(hs300_data['date'])
    hs300_data = hs300_data.set_index('date').sort_index()

    LOOKAHEAD_DAYS = 10

    print(f"处理训练数据 (预测周期: {LOOKAHEAD_DAYS}个交易日 ≈ 2周)...")
    train_data = hs300_data.loc[train_start_date:train_end_date]
    processed_train = process_index_data(train_data, lookahead_days=LOOKAHEAD_DAYS, is_training=True)

    if processed_train.empty:
        raise ValueError("没有获取到任何有效训练数据")

    print(f"成功处理 {len(processed_train)} 条训练数据")

    feature_cols = [
        'MA5', 'MA10', 'MA20', 'MA30', 'MA60',
        'RSI', 'RSI_7', 'RSI_21',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'BOLL_upper', 'BOLL_mid', 'BOLL_lower', 'BOLL_Width',
        'ATR', 'ATR_Ratio',
        'OBV', 'OBV_Change',
        'CCI', 'CCI_10', 'CCI_30',
        'VOLATILITY', 'VOLATILITY_10', 'VOLATILITY_60',
        'K', 'D', 'J', 'Williams_R',
        'ROC', 'ROC_6', 'ADX',
        'Aroon_Up', 'Aroon_Down', 'Aroon_Osc',
        'Stoch_K', 'Stoch_D',
        'Volume_MA5', 'Volume_MA20', 'Volume_Ratio',
        'Price_Momentum_5', 'Price_Momentum_10', 'Price_Momentum_20',
        'MA5_Trend', 'MA10_Trend', 'MA20_Trend',
        'Close_Position'
    ]

    X = processed_train[feature_cols].replace([np.inf, -np.inf], np.nan)
    medians = X.median()
    X = X.fillna(medians).clip(-1e10, 1e10)
    y = processed_train['label']

    print(f"原始特征数量: {len(feature_cols)}")
    print(f"正样本比例: {y.mean():.2%}")

    print("\n特征选择：训练临时模型获取特征重要性...")
    temp_model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=2000,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=0.5,
        reg_alpha=20,
        reg_lambda=20,
        min_child_weight=10,
        tree_method='hist',
        random_state=2025,
        enable_categorical=False,
        missing=np.nan
    )
    temp_model.fit(X, y, verbose=False)
    importance = temp_model.get_booster().get_score(importance_type='weight')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    TOP_N = 20
    selected_features = [feat for feat, _ in sorted_importance[:TOP_N]]
    print(f"选择 Top {TOP_N} 特征: {selected_features[:5]}...")

    X = X[selected_features]
    print(f"特征选择完成: {len(selected_features)} 个特征")

    log.section("模型交叉验证训练")
    tscv = TimeSeriesSplit(n_splits=10)
    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train = X_train.fillna(medians).clip(-1e10, 1e10)
        X_val = X_val.fillna(medians).clip(-1e10, 1e10)

        model = LogisticRegression(
            C=0.05,
            penalty='l2',
            solver='lbfgs',
            max_iter=5000,
            class_weight='balanced',
            random_state=2025
        )

        model.fit(X_train, y_train)

        val_proba = model.predict_proba(X_val)[:, 1]
        train_proba = model.predict_proba(X_train)[:, 1]

        val_auc = roc_auc_score(y_val, val_proba)
        train_auc = roc_auc_score(y_train, train_proba)

        cv_results.append({
            'fold': fold + 1,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'val_proba_mean': val_proba.mean(),
            'val_proba_std': val_proba.std()
        })

        print(f"第 {fold+1} 折: Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}, Gap={train_auc-val_auc:.4f}, 预测均值={val_proba.mean():.4f}")

    avg_auc = np.mean([r['val_auc'] for r in cv_results])
    print(f"\n交叉验证平均 AUC: {avg_auc:.4f}")

    log.section("训练最终模型")
    X_final = X.fillna(medians).clip(-1e10, 1e10)

    final_model = LogisticRegression(
        C=0.05,
        penalty='l2',
        solver='lbfgs',
        max_iter=5000,
        class_weight='balanced',
        random_state=2025
    )
    final_model.fit(X_final, y)

    print("最终模型训练完成")

    log.section("预测结果")
    predict_data = hs300_data.loc[predict_start_date:predict_end_date]
    processed_predict = process_index_data(predict_data, lookahead_days=LOOKAHEAD_DAYS, is_training=False)

    if processed_predict.empty:
        raise ValueError("没有获取到任何有效预测数据")

    latest = processed_predict.tail(1)
    X_latest = latest[selected_features].replace([np.inf, -np.inf], np.nan)
    X_latest = X_latest.fillna(medians).clip(-1e10, 1e10)

    proba = final_model.predict_proba(X_latest)[0, 1]
    prediction = "上涨" if proba >= 0.5 else "下跌"

    print(f"最新日期: {latest.index[0].strftime('%Y-%m-%d')}")
    print(f"当前收盘价: {latest['close'].values[0]:.2f}")
    print(f"预测未来 {LOOKAHEAD_DAYS} 个交易日 (约2周) 上涨概率: {proba:.2%}")
    print(f"预测方向: {prediction}")

    log.section("阈值敏感性分析")
    print("在不同预测得分阈值下，验证集实际上涨比例:")

    val_results = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_val = X.iloc[val_idx].fillna(medians).clip(-1e10, 1e10)
        y_val = y.iloc[val_idx]
        model_temp = LogisticRegression(
            C=0.05,
            penalty='l2',
            solver='lbfgs',
            max_iter=5000,
            class_weight='balanced',
            random_state=2025
        )
        model_temp.fit(X.iloc[train_idx].fillna(medians).clip(-1e10, 1e10), y.iloc[train_idx])
        val_proba = model_temp.predict_proba(X_val)[:, 1]
        val_results.append({
            'y_true': y_val.values,
            'y_proba': val_proba
        })

    all_y_true = np.concatenate([r['y_true'] for r in val_results])
    all_y_proba = np.concatenate([r['y_proba'] for r in val_results])

    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    thresh_analysis = threshold_analysis(all_y_true, all_y_proba, thresholds)

    print("\n" + "=" * 70)
    print(f"{'阈值':<10} {'样本数':<12} {'实际上涨比例':<15} {'累计覆盖比例':<15}")
    print("=" * 70)

    for _, row in thresh_analysis.iterrows():
        coverage = row['sample_count'] / row['total_samples'] * 100
        marker = " ★" if row['threshold'] == 0.5 else ""
        print(f"{row['threshold']:<10.1f} {int(row['sample_count']):<12} {row['actual_up_rate']:<15.2%} {coverage:<15.2f}{marker}")

    print("=" * 70)
    print("★ 标记表示默认分类阈值 (0.5)")

    best_thresh = thresh_analysis.loc[thresh_analysis['actual_up_rate'].idxmax()]
    print(f"\n最优阈值: {best_thresh['threshold']:.1f} (实际上涨比例: {best_thresh['actual_up_rate']:.2%})")

    log.section("综合结论")

    print("\n" + "=" * 70)
    print("  大盘指数预测报告")
    print("=" * 70)
    print(f"  预测目标: 沪深300指数未来2周 (10个交易日)")
    print(f"  最新日期: {latest.index[0].strftime('%Y-%m-%d')}")
    print(f"  当前收盘: {latest['close'].values[0]:.2f}")
    print(f"  模型置信度: {proba:.2%}")
    print("-" * 70)
    print(f"  预测结论: 未来2周 {'大概率上涨 ↑' if proba >= 0.5 else '大概率下跌 ↓'}")
    print("-" * 70)
    print(f"  历史10折CV平均AUC: {avg_auc:.4f}")
    print(f"  阈值0.5时历史上涨命中率: {thresh_analysis[thresh_analysis['threshold']==0.5]['actual_up_rate'].values[0]:.2%}")
    print("=" * 70)

    log.finish()

if __name__ == "__main__":
    main()