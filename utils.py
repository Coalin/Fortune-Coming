import numpy as np
import pandas as pd
import akshare as ak
import time
import random
from datetime import datetime, timedelta
import requests
import json
import warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


def check_data_quality(df):
    """检查数据中的异常值和缺失值"""
    print("\n数据质量检查报告:")
    # 检查无穷值
    inf_cols = df.columns[df.isin([np.inf, -np.inf]).any()]
    print(f"包含inf的列: {list(inf_cols)}")

    # 检查过大值（超过1e6）
    large_val_cols = df.columns[(df.abs() > 1e6).any()]
    print(f"包含过大值的列: {list(large_val_cols)}")

    # 检查缺失值
    na_cols = df.columns[df.isna().any()]
    print(f"包含缺失值的列: {list(na_cols)}")

    return df.replace([np.inf, -np.inf], np.nan)


# =============================================
# 原有特征函数
# =============================================

def smart_money_flow(high, low, close, volume, window=5):
    typical_price = (high + low + close)/3
    money_flow = typical_price * volume
    return (money_flow.rolling(window).mean() / volume.rolling(window).mean()).fillna(1)


def trend_persistence(close, short=3, long=10):
    short_trend = close.rolling(short).mean() > close.rolling(short*2).mean().shift(short)
    long_trend = close.rolling(long).mean() > close.rolling(long*2).mean().shift(long)
    return (short_trend & long_trend).astype(float)


def market_regime(index_close, window=20):
    ma = index_close.rolling(window).mean()
    std = index_close.rolling(window).std()
    regime = pd.cut(index_close,
                    bins=[-np.inf, ma-1.5*std, ma+1.5*std, np.inf],
                    labels=[0, 1, 2])  # 0=熊市 1=震荡 2=牛市
    return regime.astype(float).fillna(1)


def volume_spike(volume, window=20):
    median = volume.rolling(window).median()
    mad = (volume - median).abs().rolling(window).median()
    return ((volume - median) / (mad + 1e-8)).fillna(0)


def fibonacci_levels(close, window=20):
    max_price = close.rolling(window).max()
    min_price = close.rolling(window).min()
    range_ = max_price - min_price
    return pd.DataFrame({
        'FIB_0.236': max_price - 0.236*range_,
        'FIB_0.382': max_price - 0.382*range_,
        'FIB_0.618': max_price - 0.618*range_
    })


def smart_nine_turn(close, volume, threshold=0.995):
    """改进版九转序列，加入成交量验证"""
    cond = close > close.shift(1)
    up_seq = cond.rolling(4).sum()  # 近期上涨天数
    down_seq = (~cond).rolling(4).sum()

    # 量价双重验证
    vol_cond = volume > volume.rolling(20).mean().shift(1) * 1.2
    price_cond = close > close.rolling(13).mean() * threshold

    # 生成序列
    buy_signal = (up_seq >= 4) & vol_cond & price_cond
    sell_signal = (down_seq >= 4) & vol_cond & (~price_cond)

    return pd.DataFrame({
        'NT_BuyCount': buy_signal.rolling(9).sum().fillna(0),
        'NT_SellCount': sell_signal.rolling(9).sum().fillna(0),
        'NT_NetSignal': (buy_signal.rolling(9).sum() - sell_signal.rolling(9).sum()).fillna(0)
    })


def nine_turn_divergence(close, low, high, window=9):
    """检测价格新高但九转信号减弱的情况"""
    max_close = close.rolling(window).max()
    new_high = close == max_close

    buy_signal = (close > close.shift(1)).rolling(4).sum() >= 3
    sell_signal = (close < close.shift(1)).rolling(4).sum() >= 3

    top_divergence = new_high & (buy_signal.rolling(window).sum() < 4)
    bottom_divergence = (close == close.rolling(window).min()) & (sell_signal.rolling(window).sum() < 4)

    return pd.DataFrame({
        'NT_TopDiv': top_divergence.astype(int),
        'NT_BottomDiv': bottom_divergence.astype(int)
    })


def dynamic_nine_turn_threshold(close, volatility_window=20):
    """根据波动率动态调整九转触发阈值"""
    volatility = close.pct_change().rolling(volatility_window).std()
    dynamic_threshold = 1 - (volatility * 1.5).clip(0.002, 0.02)
    return dynamic_threshold.fillna(0.995)


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


def calculate_smart_money_divergence(close, volume, window=10):
    """
    识别主力资金与价格背离：
    1. 计算价格趋势（20日收益率）
    2. 计算资金流趋势（OBV的20日变化率）
    3. 当价格创新高但资金流下降时发出信号
    """
    price_trend = close.pct_change(window)
    obv_trend = (calculate_obv(close, volume).pct_change(window))
    divergence = (price_trend > 0) & (obv_trend < 0)
    return divergence.astype(float).fillna(0)


def calculate_volatility_clustering(close, short_window=5, long_window=20):
    """
    捕捉波动率聚集现象（高风险时段延续性）：
    1. 计算短期波动率与长期波动率比值
    2. 当比值突破2倍标准差时标记
    """
    ret = close.pct_change()
    short_vol = ret.rolling(short_window).std()
    long_vol = ret.rolling(long_window).std()
    z_score = (short_vol/long_vol - 1) / (long_vol + 1e-8)
    return (z_score > 2).astype(float).fillna(0)


def calculate_liquidity_shock(close, volume, window=10):
    """
    识别流动性突变：
    1. 计算成交量Z-Score
    2. 结合价格波动率放大效应
    """
    volume_z = (volume - volume.rolling(window).mean()) / (volume.rolling(window).std() + 1e-8)
    vol_ratio = close.pct_change().abs().rolling(window).sum()
    return (volume_z * vol_ratio).fillna(0)


def calculate_order_flow(high, low, close, volume, window=5):
    """
    估算机构订单流：
    1. 使用Tick规则判断买卖方向
    2. 累计净订单流（成交量加权）
    """
    tick_rule = np.where(close > (high + low)/2, 1,
                        np.where(close < (high + low)/2, -1, 0))
    net_flow = (tick_rule * volume).rolling(window).sum()
    return (net_flow / volume.rolling(window).sum()).fillna(0)


def calculate_sentiment_extremes(rsi, cci, window=14):
    """
    综合RSI和CCI识别情绪极端点：
    1. 标准化两个指标到[0,1]区间
    2. 计算加权情绪得分
    3. 识别超买超卖区域
    """
    norm_rsi = (rsi - 30) / (70 - 30)  # 30-70标准化
    norm_cci = (cci + 100) / (100 - (-100))  # -100到100标准化
    combined = (norm_rsi * 0.6 + norm_cci * 0.4).clip(0, 1)
    return ((combined > 0.8) | (combined < 0.2)).astype(float).fillna(0)


def add_return_features(df, window=10):
    """添加过去N天的日收益率序列"""
    close = df['收盘'].ffill()
    for i in range(1, window+1):
        df[f'Ret_{i}day'] = close.pct_change(i).fillna(0)
    return df


def add_volume_features(df, window=10):
    """添加过去N天的成交量变化率"""
    volume = df['成交量'].ffill()
    for i in range(1, window+1):
        df[f'VolChg_{i}day'] = (volume / volume.shift(i) - 1).fillna(0)
    return df


# =============================================
# 新增动量特征函数
# =============================================

def add_momentum_features(df, index_close=None):
    """
    添加综合动量特征

    核心理念：动量效应(Momentum Effect)——"强者恒强，弱者恒弱"
    过去表现好的资产未来倾向于继续表现好

    参数:
        df: 包含OHLCV数据的DataFrame
        index_close: 大盘/指数收盘价序列（用于计算相对动量）
    """
    close = df['收盘'].ffill()
    returns = close.pct_change()

    # =============================================
    # 1. 多周期动量（时间维度）
    # =============================================
    window_list = [5, 10, 20, 60, 120]
    for w in window_list:
        df[f'Mom_{w}d'] = close.pct_change(w)

    # =============================================
    # 2. 相对动量（相对基准）
    # =============================================
    if index_close is not None:
        # 个股 vs 大盘/行业的相对强弱
        for w in [20, 60]:
            try:
                index_reindexed = index_close.reindex(close.index, method='ffill')
                df[f'Mom_Relative_vs_HS300_{w}d'] = close.pct_change(w) - index_reindexed.pct_change(w)
            except:
                df[f'Mom_Relative_vs_HS300_{w}d'] = 0.0

    # =============================================
    # 3. 动量斜率（趋势质量）
    # =============================================
    def calc_momentum_slope(x):
        """计算动量斜率"""
        if len(x) < 3:
            return 0
        y = x.values
        x_vals = np.arange(len(y))
        # 使用线性回归计算斜率
        if np.std(y) < 1e-8:
            return 0
        try:
            slope = np.polyfit(x_vals, y, 1)[0]
            return slope
        except:
            return 0

    df['Mom_Slope_5d'] = returns.rolling(5).apply(calc_momentum_slope)
    df['Mom_Slope_10d'] = returns.rolling(10).apply(calc_momentum_slope)
    df['Mom_Slope_20d'] = returns.rolling(20).apply(calc_momentum_slope)

    # =============================================
    # 4. 动量加速/减速（动量质量）
    # =============================================
    # 短期动量 vs 长期动量：判断趋势是在加速还是减速
    df['Mom_Accel_5_20'] = close.pct_change(5) - close.pct_change(20)
    df['Mom_Accel_10_60'] = close.pct_change(10) - close.pct_change(60)
    df['Mom_Accel_20_60'] = close.pct_change(20) - close.pct_change(60)

    # 二阶导数（加速度）
    mom_5 = close.pct_change(5)
    df['Mom_2nd_Derivative'] = mom_5.diff().fillna(0)

    # =============================================
    # 5. 动量稳定性（趋势一致性）
    # =============================================
    for w in [10, 20]:
        # 上涨天数占比（趋势一致性）
        df[f'Mom_Stability_{w}d'] = (returns > 0).rolling(w).mean()

    # =============================================
    # 6. 风险调整动量（动量效率）
    # =============================================
    for w in [10, 20, 60]:
        volatility = returns.rolling(w).std()
        df[f'Mom_RiskAdjusted_{w}d'] = df[f'Mom_{w}d'] / (volatility + 1e-8)

    # =============================================
    # 7. 动量衰减率（反转预警）
    # =============================================
    # 短期太强可能反转
    df['Mom_Decay_5_60'] = close.pct_change(5) - close.pct_change(60)
    df['Mom_Decay_10_120'] = close.pct_change(10) - close.pct_change(120)

    # 动量反转信号
    df['Mom_Reversal_Signal'] = (df['Mom_Decay_5_60'] > 0).astype(float)

    # =============================================
    # 8. 动量集中度（分布特征）
    # =============================================
    for w in [20, 60]:
        # 收益离散度（越低表示趋势越有序）
        df[f'Mom_Consistency_{w}d'] = 1 / (returns.rolling(w).std() + 1e-8)

        # 偏度（正偏表示右尾更长，动量可能持续）
        df[f'Mom_Skewness_{w}d'] = returns.rolling(w).apply(lambda x: pd.Series(x).skew() if len(x) >= 3 else 0)

    # =============================================
    # 9. 动量突破/跌破
    # =============================================
    for w in [20, 60]:
        # 价格相对均线的动量
        ma_w = close.rolling(w).mean()
        df[f'Mom_vs_MA_{w}d'] = (close - ma_w) / (ma_w + 1e-8)

        # 创N日新高/新低
        rolling_max = close.rolling(w).max()
        rolling_min = close.rolling(w).min()
        df[f'Mom_NewHigh_{w}d'] = (close == rolling_max).astype(float)
        df[f'Mom_NewLow_{w}d'] = (close == rolling_min).astype(float)

    # =============================================
    # 10. 动量与成交量结合
    # =============================================
    for w in [10, 20]:
        # 量增价涨（确认趋势）
        vol_ratio = df['成交量'] / df['成交量'].rolling(w).mean()
        mom = close.pct_change(w)
        df[f'Mom_VolumeConfirm_{w}d'] = ((vol_ratio > 1) & (mom > 0)).astype(float)

        # 价涨量缩（可能见顶）
        df[f'Mom_PriceUp_VolDown_{w}d'] = ((vol_ratio < 1) & (mom > 0)).astype(float)

    # =============================================
    # 11. 动量周期分析
    # =============================================
    for w in [20, 60]:
        # 动量强度（收益/最大回撤）
        # 先计算累计最大值
        cummax_series = close.cummax()
        rolling_max = cummax_series.rolling(w, min_periods=1).max()
        rolling_dd = (close - rolling_max) / (rolling_max + 1e-8)
        max_dd = rolling_dd.rolling(w).min()
        df[f'Mom_Strength_{w}d'] = df[f'Mom_{w}d'] / (abs(max_dd) + 0.01)

    # =============================================
    # 12. 动量Z-Score（标准化动量）
    # =============================================
    for w in [20, 60]:
        rolling_mean = df[f'Mom_{w}d'].rolling(w).mean()
        rolling_std = df[f'Mom_{w}d'].rolling(w).std()
        df[f'Mom_ZScore_{w}d'] = (df[f'Mom_{w}d'] - rolling_mean) / (rolling_std + 1e-8)

    # =============================================
    # 13. 行业动量排名（需要额外数据）
    # =============================================
    # 注意：这里使用自身历史分位数
    for w in [20, 60]:
        df[f'Mom_Percentile_{w}d'] = df[f'Mom_{w}d'].rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 20 else 0.5
        )

    # =============================================
    # 14. 动量交叉信号
    # =============================================
    # 短期均线上穿/下穿长期均线
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    # 金叉/死叉
    df['Mom_GoldenCross'] = ((ma5 > ma20) & (ma5.shift(1) <= ma20.shift(1))).astype(float)
    df['Mom_DeathCross'] = ((ma5 < ma20) & (ma5.shift(1) >= ma20.shift(1))).astype(float)

    # 多头排列
    df['Mom_BullishAlignment'] = ((ma5 > ma20) & (ma20 > ma60)).astype(float)
    df['Mom_BearishAlignment'] = ((ma5 < ma20) & (ma20 < ma60)).astype(float)

    # =============================================
    # 15. 动量持续性指标
    # =============================================
    for w in [20, 60]:
        # 使用自相关判断动量持续性
        df[f'Mom_Persistence_{w}d'] = returns.rolling(w).apply(
            lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 3 else 0
        )

    # =============================================
    # 16. 复合动量指标
    # =============================================
    # 综合动量得分（标准化后加权平均）
    mom_5d = df['Mom_5d'].fillna(0)
    mom_20d = df['Mom_20d'].fillna(0)
    mom_60d = df['Mom_60d'].fillna(0)

    # 标准化
    mom_5d_norm = (mom_5d - mom_5d.rolling(60).mean()) / (mom_5d.rolling(60).std() + 1e-8)
    mom_20d_norm = (mom_20d - mom_20d.rolling(60).mean()) / (mom_20d.rolling(60).std() + 1e-8)
    mom_60d_norm = (mom_60d - mom_60d.rolling(60).mean()) / (mom_60d.rolling(60).std() + 1e-8)

    # 加权综合（短期权重更高）
    df['Mom_Composite'] = mom_5d_norm * 0.5 + mom_20d_norm * 0.3 + mom_60d_norm * 0.2

    # 动量方向综合判断
    df['Mom_Direction'] = np.where(df['Mom_Composite'] > 0.5, 1,
                                   np.where(df['Mom_Composite'] < -0.5, -1, 0))

    return df


def robust_stock_data(symbol, start_date, end_date, max_retries=3):
    """
    健壮的股票数据获取函数 - 主要使用新浪财经API
    返回字段对齐：['日期', '开盘', '最高', '最低', '收盘', '成交量']
    """
    # 1. 优先使用新浪财经API（测试证明最稳定）
    for attempt in range(max_retries):
        try:
            market = "sh" if symbol.startswith("6") else "sz"
            url = "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
            params = {
                "symbol": f"{market}{symbol}",
                "scale": "240",  # 日线
                "datalen": "1000",
                "ma": "no",
                "begin_date": start_date,
                "end_date": end_date
            }

            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if data and len(data) > 0:
                df = pd.DataFrame(data)

                # 统一列名为中文（与主函数对齐）
                column_mapping = {
                    'day': '日期',
                    'open': '开盘',
                    'high': '最高',
                    'low': '最低',
                    'close': '收盘',
                    'volume': '成交量'
                }

                # 只重命名存在的列
                existing_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
                df = df.rename(columns=existing_cols)

                # 确保日期格式正确
                if '日期' in df.columns:
                    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
                    df = df.dropna(subset=['日期'])

                # 转换数值列为浮点数
                numeric_cols = ['开盘', '最高', '最低', '收盘', '成交量']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                print(f"✅ 新浪财经API成功获取 {symbol} ({len(df)}条数据)")
                return df

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️ 新浪API第{attempt+1}次尝试失败: {str(e)[:50]}...")
                time.sleep(random.uniform(1, 2))
                continue
            print(f"❌ 新浪API最终失败 {symbol}: {str(e)[:50]}...")

    # 2. 备用方案：东方财富直接API
    for attempt in range(max_retries):
        try:
            market = "1" if symbol.startswith("6") else "0"
            url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
            params = {
                "secid": f"{market}.{symbol}",
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                "klt": "101",
                "fqt": "1",
                "beg": start_date,
                "end": end_date,
            }

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://quote.eastmoney.com/",
            }

            response = requests.get(url, params=params, headers=headers, timeout=10)
            data = response.json()

            if data.get("data") and data["data"].get("klines"):
                klines = data["data"]["klines"]
                df = pd.DataFrame([kline.split(",") for kline in klines])
                df.columns = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]

                # 转换数值列
                numeric_cols = ["开盘", "收盘", "最高", "最低", "成交量"]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # 转换日期
                df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
                df = df.dropna(subset=['日期'])

                print(f"✅ 东方财富API成功获取 {symbol} ({len(df)}条数据)")
                return df[['日期', '开盘', '最高', '最低', '收盘', '成交量']]  # 只返回需要的列

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(random.uniform(1, 2))
                continue

    print(f"💥 所有数据源均失败 {symbol}")
    return None


def safe_get_stock_data(symbol, start_date, end_date, max_retries=3):
    """
    安全包装函数，添加延迟和验证
    """
    # 添加随机延迟避免请求过快
    time.sleep(random.uniform(0.5, 1.5))

    return robust_stock_data(symbol, start_date, end_date, max_retries)


def get_valid_date():
    """
    获取有效日期 - 简化版本
    """
    try:
        # 获取沪深300指数最新日期
        hs300_data = ak.stock_zh_index_daily(symbol="sh000300")
        if not hs300_data.empty:
            latest_date = pd.to_datetime(hs300_data['date'].iloc[-1])
            print(f"找到有效日期: {latest_date.strftime('%Y-%m-%d')}")
            return latest_date
    except:
        pass

    # 备用：使用当前日期减1天
    fallback_date = datetime.now() - timedelta(days=1)
    print(f"使用备用日期: {fallback_date.strftime('%Y-%m-%d')}")
    return fallback_date


from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

def call_workflow_api(user_message, system_prompt='', stream=False):
    """调用 MiniMax API"""
    from openai import OpenAI
    from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
    
    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    )
    
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        stream=False
    )
    
    return response.choices[0].message.content


def get_llm_analysis(top_stocks_df, n=3):
    """
    用大模型分析股票买卖建议
    """
    # 选取TOP3股票
    top_3 = top_stocks_df.head(n)

    # 准备数据
    stock_info = []
    for _, row in top_3.iterrows():
        stock_info.append(f"股票代码: {row['股票代码']}, 股票名称: {row['股票名称']}")

    stocks_str = "\n".join(stock_info)

    # 构建prompt
    prompt = f"""
    当前日期：{datetime.now().strftime('%Y-%m-%d')}
    分析师：Dr. Fan Zhang, Goldman Sachs Equity Research
    任务：对以下{len(top_stocks_df)}只股票进行深度分析，提供投资建议

    股票列表：
    {stocks_str}

    ## 高盛机构股票分析框架

    ### 1. 定性分析（Qualitative Analysis）

    **1.1 商业模式深度解构**
    - 价值链定位：上游/中游/下游，议价能力分析
    - 盈利驱动因素：单位经济模型、客户生命周期价值
    - 可扩展性：边际成本曲线、网络效应强度
    - 护城河评估：技术专利、品牌价值、转换成本、规模经济

    **1.2 竞争格局与行业结构**
    - 行业集中度：赫芬达尔指数（HHI）计算
    - 波特五力模型定量化评分
    - 竞争优势可持续性：Greenwald框架评估
    - 监管环境与政策风险敞口

    **1.3 管理层与公司治理**
    - 管理层过往资本配置记录（ROIC趋势）
    - 股权激励与股东利益一致性
    - ESG整合评分（环境、社会、治理）

    ### 2. 定量分析（Quantitative Analysis）

    **2.1 财务质量三维评估**

    **盈利能力质量**
    - 核心盈利能力：调整后EBIT利润率（剔除非经常性）
    - 资本回报率：ROIC vs WACC差值（经济利润）
    - 盈利可持续性：经营性现金流/净利润比率
    - 杜邦分析分解：杠杆、周转、利润率贡献度

    **财务健康度**
    - 资产负债表韧性：净债务/EBITDA比率
    - 流动性压力测试：现金转换周期、速动比率
    - 或有负债评估：表外负债、养老金缺口

    **增长质量**
    - 有机增长vs并购增长分解
    - 增量资本回报率（ICR）趋势
    - 增长投资效率：Capex/收入比率

    **2.2 估值模型矩阵（Valuation Matrix）**

    **绝对估值模型**
    - 三阶段DCF模型：详细假设透明化
    - 第一阶段（5年）：详细业务预测
    - 第二阶段（5年）：收敛期，ROIC回归WACC
    - 第三阶段：永续期，永续增长率2.5%
    - WACC计算：无风险利率、Beta、股权风险溢价、债务成本
    - 情景分析：牛市/基本/熊市概率加权
    - 敏感性分析：关键变量对估值影响

    **相对估值框架**
    - 跨周期估值：当前vs历史估值百分位（10年）
    - 跨国比较：与全球同业估值差距分析
    - 增长调整估值：PEG、EV/EBITDA-to-growth
    - 质量调整估值：P/E-to-ROE、P/B-to-ROE散点图位置

    **2.3 风险调整收益分析**
    - 预期回报分解：股息+盈利增长+估值变化
    - 下行风险量化：在险价值（VaR）、最大回撤
    - 夏普比率、索提诺比率计算
    - 相关性分析：与市场、行业Beta

    ### 3. 催化剂与时间线（Catalysts & Timeline）

    **3.1 近期催化剂（0-6个月）**
    - 季度业绩预期vs共识
    - 产品发布/监管审批时间线
    - 行业供需格局变化

    **3.2 中期驱动（6-18个月）**
    - 产能扩张投产进度
    - 市场份额获取验证点
    - 成本结构优化效果

    **3.3 长期主题（18个月+）**
    - 结构性增长趋势契合度
    - 竞争优势演进路径
    - 潜在市场总规模（TAM）扩张

    ### 4. 投资建议框架

    **4.1 评级标准（基于风险调整后预期回报）**

    买入（Buy）：预期回报 > 15%，风险调整后Alpha显著
    - 确信度：高（>70%概率达到目标价）
    - 仓位建议：3-5%（投资组合）

    中性（Neutral）：预期回报 5-15%，无明显Alpha
    - 确信度：中
    - 仓位建议：基准权重

    卖出（Sell）：预期回报 < 5%或下行风险 > 20%
    - 确信度：中高
    - 行动建议：减持或对冲

    **4.2 目标价设定方法**
    - 基于DCF估值：50%权重
    - 相对估值区间：30%权重
    - 技术面支撑：20%权重
    - 给出三种情景目标价（牛市/基本/熊市）

    ## 输出要求

    ### 📊 分析摘要表

    | 股票 | 评级 | 目标价 | 上行空间 | 下行风险 | 预期回报 | 夏普比率 | 建议仓位 |
    |------|------|--------|----------|----------|----------|----------|----------|

    ### 📈 详细分析（每只股票）

    **股票1代码 - 股票1名称**

    **投资论文（Investment Thesis）**
    - 核心论点：1-2句话总结投资逻辑
    - 关键假设：3-4个必须验证的假设
    - 差异化认知：与市场共识的不同点

    **估值摘要**
    - DCF内在价值：XX元（范围：XX-XX）
    - 相对估值：当前交易于历史X%分位，同业X%分位
    - 目标价：XX元（上涨空间X%）

    **风险回报分析**
    - 上行情景（30%概率）：XX元（+X%）
    - 基本情景（50%概率）：XX元（+X%）
    - 下行情景（20%概率）：XX元（-X%）

    **关键监控指标**
    1. 季度营收增长 > X%
    2. 毛利率维持在 X%以上
    3. 经营现金流 > 净利润X%

    **投资建议**
    - 评级：【买入/中性/卖出】
    - 建议仓位：X%
    - 时间框架：X个月
    - 替代方案：如无法买入，可考虑[相关标的]

    ### 🎯 投资组合建议

    **相对吸引力排序**
    1. [最佳股票代码] - 风险调整后预期回报最高
    2. [次选股票代码] - 风险回报比合理
    3. [最次股票代码] - 存在结构性担忧

    **组合构建建议**
    - 核心持仓：[股票代码]，权重X%
    - 战术配置：[股票代码]，权重X%
    - 风险对冲：建议[对冲工具]对冲[特定风险]

    ### ⚠️ 风险披露

    **系统性风险**
    - 宏观：利率路径、经济增长、通胀
    - 地缘政治：贸易政策、区域冲突

    **个股特定风险**
    - 估值风险：关键假设不成立的可能性
    - 执行风险：管理层执行能力
    - 行业风险：技术颠覆、监管变化

    ---

    *本报告基于公开信息，已遵循高盛研究标准流程。估值模型假设可应要求提供。投资建议基于12个月时间框架。*

    请严格按照此框架进行分析，确保分析的严谨性、可追溯性和透明度。
    """

    # 调用大模型
    system_prompt = ""

    try:
        analysis = call_workflow_api(
            system_prompt=system_prompt,
            user_message=prompt
        )
        return analysis
    except Exception as e:
        return f"大模型分析失败: {str(e)}"


# =============================================
# 获取新增动量特征列表
# =============================================

def get_momentum_feature_cols():
    """
    返回新增的动量特征列名列表
    用于在主文件中添加到 feature_cols
    """
    feature_cols = []

    # 1. 多周期动量
    feature_cols += [f'Mom_{w}d' for w in [5, 10, 20, 60, 120]]

    # 2. 相对动量
    feature_cols += [f'Mom_Relative_vs_HS300_{w}d' for w in [20, 60]]

    # 3. 动量斜率
    feature_cols += ['Mom_Slope_5d', 'Mom_Slope_10d', 'Mom_Slope_20d']

    # 4. 动量加速/减速
    feature_cols += ['Mom_Accel_5_20', 'Mom_Accel_10_60', 'Mom_Accel_20_60', 'Mom_2nd_Derivative']

    # 5. 动量稳定性
    feature_cols += [f'Mom_Stability_{w}d' for w in [10, 20]]

    # 6. 风险调整动量
    feature_cols += [f'Mom_RiskAdjusted_{w}d' for w in [10, 20, 60]]

    # 7. 动量衰减/反转
    feature_cols += ['Mom_Decay_5_60', 'Mom_Decay_10_120', 'Mom_Reversal_Signal']

    # 8. 动量一致性
    feature_cols += [f'Mom_Consistency_{w}d' for w in [20, 60]]
    feature_cols += [f'Mom_Skewness_{w}d' for w in [20, 60]]

    # 9. 动量突破/跌破
    feature_cols += [f'Mom_vs_MA_{w}d' for w in [20, 60]]
    feature_cols += [f'Mom_NewHigh_{w}d' for w in [20, 60]]
    feature_cols += [f'Mom_NewLow_{w}d' for w in [20, 60]]

    # 10. 动量成交量确认
    feature_cols += [f'Mom_VolumeConfirm_{w}d' for w in [10, 20]]
    feature_cols += [f'Mom_PriceUp_VolDown_{w}d' for w in [10, 20]]

    # 11. 动量强度
    feature_cols += [f'Mom_Strength_{w}d' for w in [20, 60]]

    # 12. 动量Z-Score
    feature_cols += [f'Mom_ZScore_{w}d' for w in [20, 60]]

    # 13. 动量分位数
    feature_cols += [f'Mom_Percentile_{w}d' for w in [20, 60]]

    # 14. 动量交叉信号
    feature_cols += ['Mom_GoldenCross', 'Mom_DeathCross', 'Mom_BullishAlignment', 'Mom_BearishAlignment']

    # 15. 动量持续性
    feature_cols += [f'Mom_Persistence_{w}d' for w in [20, 60]]

    # 16. 复合动量
    feature_cols += ['Mom_Composite', 'Mom_Direction']

    return feature_cols
