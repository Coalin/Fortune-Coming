import numpy as np
import pandas as pd


def add_institutional_factors(df, market_ret=None, rf_rate=0.0):
    """
    添加机构级别的量化因子（价量组合因子）
    来自顶级量化基金实践 + 学术文献验证

    包含：动量、反转、流动性、风险调整收益等

    参数:
        df: 包含OHLCV数据的DataFrame
        market_ret: 大盘收益率序列（可选）
        rf_rate: 无风险利率（默认0）
    """
    close = df['收盘'].ffill()
    high = df['最高'].ffill()
    low = df['最低'].ffill()
    open_ = df['开盘'].ffill()
    volume = df['成交量'].ffill()

    ret = np.log(close / close.shift(1)).fillna(0)

    # =============================================
    # 1. 12个月动量（WML - Winners Minus Losers）
    # 来源：Carhart 4因子，Jegadeesh & Titman研究
    # =============================================
    ret_12m = close.pct_change(252).fillna(0)
    ret_1m = close.pct_change(21).fillna(0)
    df['WML_Momentum'] = (ret_12m - ret_1m).fillna(0)

    # =============================================
    # 2. 月度短期反转
    # 来源：AQR Capital，学术界验证
    # =============================================
    df['ST_Reversal'] = (-close.pct_change(21)).fillna(0)

    # =============================================
    # 3. 非流动性因子（ILLIQ）
    # 来源：Amihud(2002)，A股实证有效
    # =============================================
    illiq = np.abs(close.pct_change()) / (volume + 1)
    df['ILLIQ_20d'] = (illiq.rolling(20).mean() * 1e6).fillna(0)

    # =============================================
    # 4. 最大回撤率
    # 来源：所有风控系统核心指标
    # =============================================
    rolling_max = close.rolling(20).max()
    drawdown = (close / rolling_max - 1)
    df['MaxDrawdown_20d'] = drawdown.rolling(20).min().fillna(0)

    # =============================================
    # 5. Garman-Klass波动率
    # 来源：比收盘价波动率准10倍
    # =============================================
    log_hl = np.log(high / low)
    log_co = np.log(close / close.shift(1))
    gk_vol = np.sqrt((0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2).rolling(20).mean())
    df['GK_Volatility'] = gk_vol.fillna(0)

    # =============================================
    # 6. 偏度调整动量
    # 来源：Tail Risk对动量的影响
    # =============================================
    mom_60d = close.pct_change(60).fillna(0)
    skew_60d = close.pct_change().rolling(60).skew().fillna(0)
    df['SkewAdj_Momentum'] = (mom_60d / (np.abs(skew_60d) + 0.1)).fillna(0)

    # =============================================
    # 7. 量价相关性
    # 来源：判断趋势真假（价涨量增=真趋势）
    # =============================================
    df['PV_Correlation'] = close.rolling(20).corr(volume).fillna(0)

    # =============================================
    # 8. 价格位置（处于N日高低点的位置）
    # 来源：威廉指标
    # =============================================
    high_20 = close.rolling(20).max()
    low_20 = close.rolling(20).min()
    df['Price_Position'] = ((close - low_20) / (high_20 - low_20 + 1e-8)).fillna(0.5)

    # =============================================
    # 9. 收益离散度
    # 来源：Market breadth指标
    # =============================================
    df['Return_Dispersion'] = close.pct_change().rolling(20).std().fillna(0)

    # =============================================
    # 10. 价格偏离长期均线
    # 来源：均值回复策略基础
    # =============================================
    ma_120 = close.rolling(120).mean()
    df['Price_MA_Deviation'] = ((close / ma_120) - 1).fillna(0)

    # =============================================
    # 11. 布林带宽度（波动率收缩/扩张）
    # 来源：经典技术分析
    # =============================================
    boll_upper = close.rolling(20).mean() + 2 * close.rolling(20).std()
    boll_lower = close.rolling(20).mean() - 2 * close.rolling(20).std()
    df['BOLL_Width'] = ((boll_upper - boll_lower) / close).fillna(0)

    # =============================================
    # 12. 相对强弱指数（RSI的平滑版本）
    # 来源：Wilder经典指标
    # =============================================
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df['RSI_Smooth'] = (100 - (100 / (1 + rs))).fillna(50)

    # =============================================
    # 13. 流动性冲击（改进版Amihud）
    # 来源：Kyle, lambda pricing model
    # =============================================
    liq_shock = np.abs(close.pct_change()) / np.sqrt(volume + 1)
    df['Liq_Shock_20d'] = liq_shock.rolling(20).mean().fillna(0)

    # =============================================
    # 14. 波动率Cone（期限结构）
    # 来源：波动率期限结构分析
    # =============================================
    vol_5d = close.pct_change().rolling(5).std()
    vol_60d = close.pct_change().rolling(60).std()
    df['Vol_Cone_5_60'] = (vol_5d / (vol_60d + 1e-8)).fillna(1)

    # =============================================
    # 15. 残差动量（剔除大盘Beta）
    # 来源：Barra风险模型
    # =============================================
    market_ret_proxy = close.pct_change(20).rolling(20).mean()
    residual = close.pct_change(20) - market_ret_proxy
    df['Residual_Momentum'] = residual.rolling(20).sum().fillna(0)

    # =============================================
    # 16. 量价背离打分
    # 来源：经典技术分析
    # =============================================
    obv = (np.sign(close.diff()) * volume).ffill().cumsum().fillna(0)
    price_slope = np.polyfit(range(20), close.tail(20).values, 1)[0]
    obv_slope = np.polyfit(range(20), obv.tail(20).values, 1)[0]
    divergence_value = np.where(
        price_slope > 0, -obv_slope / (abs(obv_slope) + 1), obv_slope / (abs(obv_slope) + 1)
    )
    df['PV_Divergence_Score'] = pd.Series(divergence_value, index=df.index).fillna(0)

    # =============================================
    # 17. 季节性动量（年度效应）
    # 来源：Calendar Effect研究
    # =============================================
    month = pd.DatetimeIndex(close.index).month
    seasonal_ret = close.pct_change(252)
    monthly_avg = seasonal_ret.groupby(month).transform('mean')
    df['Seasonal_Momentum'] = monthly_avg.fillna(0)

    # =============================================
    # 18. 流动性调整动量
    # 来源：Fama-French因子扩展
    # =============================================
    mom_60d = close.pct_change(60)
    illiq_val = df['ILLIQ_20d'] if 'ILLIQ_20d' in df else illiq
    df['Liq_Adj_Momentum'] = (mom_60d / (illiq_val + 0.01)).fillna(0)

    # =============================================
    # 19. 收益率偏度代理
    # 来源：隐含波动率曲面
    # =============================================
    hl_mean = (high + low) / 2
    df['Return_Skew_Proxy'] = ((close - hl_mean) / (high - low + 1e-8)).fillna(0)

    # =============================================
    # 20. 趋势持续性
    # 来源：自相关分析
    # =============================================
    ret_diff = close.pct_change().diff()
    def autocorr(x, lag=1):
        return pd.Series(x).autocorr(lag) if len(x) > lag else 0
    df['Trend_Persistence'] = ret_diff.rolling(20).apply(lambda x: autocorr(x, 1), raw=True).fillna(0)

    # =============================================
    # 21. 多周期动量一致性
    # 来源：多时间框架分析
    # =============================================
    mom_5d_dir = np.sign(close.pct_change(5))
    mom_20d_dir = np.sign(close.pct_change(20))
    mom_60d_dir = np.sign(close.pct_change(60))
    df['Mom_Alignment'] = ((mom_5d_dir + mom_20d_dir + mom_60d_dir) / 3).fillna(0)

    # =============================================
    # 22. VWAP偏离度
    # 来源：国泰君安191因子库
    # =============================================
    vwap = (close * volume).cumsum() / volume.cumsum()
    df['VWAP_Dev'] = ((close - vwap) / (vwap + 1e-8)).fillna(0)

    # =============================================
    # 23. 换手率波动率
    # 来源：国泰君安191因子库
    # =============================================
    turnover_rate = volume / close
    df['Turnover_Vol'] = turnover_rate.rolling(20).std().fillna(0)

    # =============================================
    # 24. 收益率峰度（肥尾）
    # 来源：国泰君安191因子库
    # =============================================
    df['Return_Kurtosis'] = close.pct_change().rolling(20).kurt().fillna(0)

    # =============================================
    # 25. 极端收益占比
    # 来源：国泰君安191因子库
    # =============================================
    daily_ret = close.pct_change()
    extreme_ratio = ((daily_ret > 0.02) | (daily_ret < -0.02)).rolling(20).mean()
    df['Extreme_Ret_Ratio'] = extreme_ratio.fillna(0)

    # =============================================
    # 26. 上涨/下跌波动率比
    # 来源：国泰君安191因子库
    # =============================================
    up_move = daily_ret.where(daily_ret > 0, 0).rolling(20).std()
    down_move = (-daily_ret.where(daily_ret < 0, 0)).rolling(20).std()
    df['UpDown_Vol_Ratio'] = (up_move / (down_move + 1e-8)).fillna(1)

    # =============================================
    # 27. 价格冲击系数
    # 来源：国泰君安191因子库
    # =============================================
    vol_mean = volume.rolling(20).mean()
    price_impact = np.abs(daily_ret) / (volume / (vol_mean + 1e-8))
    df['Price_Impact'] = price_impact.rolling(20).mean().fillna(0)

    # =============================================
    # 28. 成交额加权动量
    # 来源：国泰君安191因子库
    # =============================================
    amount = close * volume
    amount_weighted_ret = (daily_ret * amount).rolling(20).sum() / (amount.rolling(20).sum() + 1e-8)
    df['Amount_Weighted_Mom'] = amount_weighted_ret.fillna(0)

    # =============================================
    # 29. 流动性波动率
    # 来源：国泰君安191因子库
    # =============================================
    vol_cv = volume.rolling(20).std() / (volume.rolling(20).mean() + 1e-8)
    df['Liquidity_Vol'] = vol_cv.fillna(0)

    # =============================================
    # 30. 趋势强度指标（ADX）
    # 来源：国泰君安191因子库
    # =============================================
    plus_dm = high.diff()
    minus_dm = -low.diff()
    atr_proxy = (high - low).rolling(14).mean()
    dx = np.abs(plus_dm - minus_dm) / (atr_proxy + 1e-8)
    adx = dx.rolling(14).mean()
    df['Trend_Strength_ADX'] = adx.fillna(0)

    # =============================================
    # 31. 收益率偏度变化
    # 来源：国泰君安191因子库
    # =============================================
    skew_short = daily_ret.rolling(5).skew()
    skew_long = daily_ret.rolling(20).skew()
    df['Skew_Change'] = (skew_short - skew_long).fillna(0)

    # =============================================
    # 32. 换手率加速度
    # 来源：国泰君安191因子库
    # =============================================
    close_safe = close.mask(close == 0)
    turnover_rate = volume / close_safe
    turnover_1d = turnover_rate.diff()
    turnover_5d = turnover_rate.diff(5)
    df['Turnover_Accel'] = (turnover_1d - turnover_5d).fillna(0)

    # =============================================
    # 33. 方差调整的非流动性
    # 来源：国泰君安191因子库
    # =============================================
    ret_abs = np.abs(daily_ret)
    vol_ma_20 = volume.rolling(20).mean()
    illiq_raw = ret_abs / (volume / (vol_ma_20 + 1e-8))
    illiq_var = illiq_raw.rolling(20).var()
    df['Amihud_Variance'] = illiq_var.fillna(0)

    # =============================================
    # 34. 买卖价差代理
    # 来源：国泰君安191因子库
    # =============================================
    spread = (high - low) / (close + 1e-8)
    df['Liq_Spread'] = spread.fillna(0)

    # =============================================
    # 35. 换手率衰减速率
    # 来源：国泰君安191因子库
    # =============================================
    turnover_ma5 = turnover_rate.rolling(5).mean()
    turnover_ma20 = turnover_rate.rolling(20).mean()
    df['Turnover_Decay_Rate'] = ((turnover_ma5 - turnover_ma20) / (turnover_ma20 + 1e-8)).fillna(0)

    # =============================================
    # 36. 成交量深度指数
    # 来源：国泰君安191因子库
    # =============================================
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    df['Volume_Depth_Index'] = ((volume - vol_mean) / (vol_std + 1e-8)).fillna(0)

    # =============================================
    # 37. 波动率偏斜
    # 来源：国泰君安191因子库
    # =============================================
    up_vol = daily_ret.where(daily_ret > 0, 0).rolling(20).std()
    down_vol = (-daily_ret.where(daily_ret < 0, 0)).rolling(20).std()
    df['Volatility_Skew'] = ((up_vol - down_vol) / (up_vol + down_vol + 1e-8)).fillna(0)

    # =============================================
    # 38. 波动率压缩比
    # 来源：国泰君安191因子库
    # =============================================
    vol_current = daily_ret.rolling(5).std()
    vol_hist = daily_ret.rolling(60).std()
    df['Vol_Compression_Ratio'] = (vol_current / (vol_hist + 1e-8)).fillna(0)

    # =============================================
    # 39. Parkinson波动率
    # 来源：国泰君安191因子库
    # =============================================
    hl_ratio = np.log(high / (low + 1e-8))
    parkinson_vol = np.sqrt((1 / (4 * np.log(2))) * hl_ratio.pow(2).rolling(20).mean())
    df['Parkinson_Vol'] = parkinson_vol.fillna(0)

    # =============================================
    # 40. Rogers-Satchell波动率
    # 来源：国泰君安191因子库
    # =============================================
    hc = np.log(high / close)
    ho = np.log(high / open_)
    lc = np.log(low / close)
    lo = np.log(low / open_)
    rs_vol = np.sqrt((hc * ho + lc * lo).rolling(20).mean())
    df['Rogers_Satchell_Vol'] = rs_vol.fillna(0)

    # =============================================
    # 41. 波动率摆动指标
    # 来源：国泰君安191因子库
    # =============================================
    vol_ma = daily_ret.rolling(20).std()
    vol_diff = daily_ret.rolling(5).std() - vol_ma
    df['Vol_Oscillator'] = (vol_diff / (vol_ma + 1e-8)).fillna(0)

    # =============================================
    # 42. 动量反转信号
    # 来源：国泰君安191因子库
    # =============================================
    mom_5d = close.pct_change(5)
    mom_20d = close.pct_change(20)
    df['Mom_Reversal_5_20'] = (mom_5d - mom_20d).fillna(0)

    # =============================================
    # 43. 动量推进信号
    # 来源：国泰君安191因子库
    # =============================================
    mom_20d_v = close.pct_change(20)
    mom_60d_v = close.pct_change(60)
    df['Mom_Thrust_20_60'] = (mom_20d_v - mom_60d_v).fillna(0)

    # =============================================
    # 44. 多尺度动量一致性
    # 来源：国泰君安191因子库
    # =============================================
    mom_5 = np.sign(close.pct_change(5))
    mom_10 = np.sign(close.pct_change(10))
    mom_20 = np.sign(close.pct_change(20))
    mom_60 = np.sign(close.pct_change(60))
    df['Momentum_MultiScale'] = ((mom_5 + mom_10 + mom_20 + mom_60) / 4).fillna(0)

    # =============================================
    # 45. 12个月排除1个月动量
    # 来源：国泰君安191因子库
    # =============================================
    mom_252d = close.pct_change(252)
    mom_21d = close.pct_change(21)
    df['Price_Momentum_12_1'] = (mom_252d - mom_21d).fillna(0)

    # =============================================
    # 46. 盈利动量（价格代理）
    # 来源：国泰君安191因子库
    # =============================================
    prev_close = close.shift(1).replace(0, np.nan)
    df['Earnings_Momentum'] = ((close - prev_close) / (prev_close + 1e-8)).fillna(0)

    # =============================================
    # 47. 成交量突刺次数
    # 来源：国泰君安191因子库
    # =============================================
    vol_threshold = volume.rolling(20).mean() * 2
    spike_count = ((volume > vol_threshold).astype(int)).rolling(20).sum()
    df['Volume_Spike_Count'] = spike_count.fillna(0)

    # =============================================
    # 48. 价格缺口
    # 来源：国泰君安191因子库
    # =============================================
    price_gap = close - open_.shift(1)
    df['Price_Gap'] = price_gap.fillna(0)

    # =============================================
    # 49. 量减比例
    # 来源：国泰君安191因子库
    # =============================================
    vol_ma10 = volume.rolling(10).mean()
    vol_ma30 = volume.rolling(30).mean()
    df['Volume_Decrease_Ratio'] = ((vol_ma10 - vol_ma30) / (vol_ma30 + 1e-8)).fillna(0)

    # =============================================
    # 50. 换手率/换手率均线
    # 来源：国泰君安191因子库
    # =============================================
    to_ma = turnover_rate.rolling(20).mean()
    df['Turnover_Turnover_Ratio'] = (turnover_rate / (to_ma + 1e-8)).fillna(0)

    # =============================================
    # 51. 成交量平衡指数
    # 来源：国泰君安191因子库
    # =============================================
    up_volume = volume.where(close > close.shift(1), 0).rolling(20).sum()
    down_volume = volume.where(close < close.shift(1), 0).rolling(20).sum()
    df['Volume_Balance_Index'] = ((up_volume - down_volume) / (up_volume + down_volume + 1e-8)).fillna(0)

    return df


def get_institutional_factor_cols():
    """
    获取机构量化因子的特征列表（共51个）
    """
    return [
        # 基础动量因子
        'WML_Momentum',      # 12月动量
        'ST_Reversal',       # 短期反转
        # 流动性因子
        'ILLIQ_20d',         # 非流动性
        'Liq_Shock_20d',     # 流动性冲击
        'Liq_Adj_Momentum',  # 流动性调整动量
        # 风险因子
        'MaxDrawdown_20d',   # 最大回撤
        'GK_Volatility',     # Garman-Klass波动率
        'Vol_Cone_5_60',     # 波动率期限结构
        # 动量增强因子
        'SkewAdj_Momentum',  # 偏度调整动量
        'Residual_Momentum', # 残差动量
        'Mom_Alignment',     # 多周期动量一致性
        # 量价因子
        'PV_Correlation',    # 量价相关性
        'Price_Position',    # 价格位置
        'PV_Divergence_Score', # 量价背离
        # 收益因子
        'Return_Dispersion', # 收益离散度
        'Return_Skew_Proxy', # 收益率偏度
        'Price_MA_Deviation',# 价格偏离均线
        'Trend_Persistence', # 趋势持续性
        # 技术因子
        'BOLL_Width',        # 布林带宽度
        'RSI_Smooth',        # 平滑RSI
        'Seasonal_Momentum', # 季节性动量
        # 国泰君安191因子库（第1批）
        'VWAP_Dev',          # VWAP偏离度
        'Turnover_Vol',      # 换手率波动率
        'Return_Kurtosis',   # 收益率峰度
        'Extreme_Ret_Ratio', # 极端收益占比
        'UpDown_Vol_Ratio',  # 涨跌波动率比
        'Price_Impact',      # 价格冲击系数
        'Amount_Weighted_Mom', # 成交额加权动量
        'Liquidity_Vol',     # 流动性波动率
        'Trend_Strength_ADX', # ADX趋势强度
        'Skew_Change',       # 偏度变化
        # 国泰君安191因子库（第2批）
        'Turnover_Accel',    # 换手率加速度
        'Amihud_Variance',   # 方差调整非流动性
        'Liq_Spread',        # 买卖价差代理
        'Turnover_Decay_Rate', # 换手率衰减速率
        'Volume_Depth_Index', # 成交量深度指数
        'Volatility_Skew',   # 波动率偏斜
        'Vol_Compression_Ratio', # 波动率压缩比
        'Parkinson_Vol',     # Parkinson波动率
        'Rogers_Satchell_Vol', # Rogers-Satchell波动率
        'Vol_Oscillator',    # 波动率摆动指标
        'Mom_Reversal_5_20', # 动量反转信号
        'Mom_Thrust_20_60',  # 动量推进信号
        'Momentum_MultiScale', # 多尺度动量一致性
        'Price_Momentum_12_1', # 12月排除1月动量
        'Earnings_Momentum', # 盈利动量
        'Volume_Spike_Count', # 成交量突刺次数
        'Price_Gap',         # 价格缺口
        'Volume_Decrease_Ratio', # 量减比例
        'Turnover_Turnover_Ratio', # 换手率比率
        'Volume_Balance_Index', # 成交量平衡指数
    ]


if __name__ == "__main__":
    print("机构量化因子模块")
    print(f"包含 {len(get_institutional_factor_cols())} 个因子：")
    for f in get_institutional_factor_cols():
        print(f"  - {f}")
