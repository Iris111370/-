import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from WindPy import w

# 确保wind打开
if not w.isconnected():
    w.start()

'''数据'''
file_path = "/Users/yangxinyi/Desktop/GMARS.WI-行情统计-20250729.csv"
df = pd.read_csv(file_path)
# 只保留日期和收盘价
df = df[['交易日期', '收盘价']].dropna()
# 日期处理
df['交易日期'] = pd.to_datetime(df['交易日期'])
df = df.sort_values('交易日期').reset_index(drop=True)

# 重命名：交易日期--date；收盘价--close
df.rename(columns={'交易日期': 'date', '收盘价': 'close'}, inplace=True)

# 以估值时间点的价格作为S0
try:
    S0 = df.loc[df['date'] == '2025-03-31', 'close'].values[0]
except IndexError:
    raise ValueError("找不到2025-03-31对应的收盘价")

# print(df.head(10))

'''波动率计算，在这里暂不需要，固定为0.03'''
# 估值日期为2025-03-31
valuation_date = pd.to_datetime("2025-03-31")
'''
# 筛选估值日前一年内的数据，计算波动率
start_date = valuation_date - pd.Timedelta(days=252)
df_one_year = df[(df['date'] > start_date) & (df['date'] <= valuation_date)].copy()

# 计算日收益率，即每日变动
df_one_year['return'] = df_one_year['close'].pct_change()
# 计算年化波动率
vol = df_one_year['return'].std() * np.sqrt(252)
'''
vol = 0.035  # 在这里文档中固定是0.035
print(f"年化波动率为：{vol:.4%}")

print('*' * 100)

para = {
    "S0": S0,
    "sigma": vol,
    "r": 0.02,
    "q": 0.02,
    "T_days": 365,
    "I": 250000,
    "seed": 42,
    "interest_barrier_up": 0.8,
    "interest_barrier_down": 1.2,
    "knock_out_barrier_up": 1.0,
    "knock_out_barrier_down": 1.0,
    "final_knock_out_barrier_up": 0.8,
    "final_knock_out_barrier_down": 1.2,
    "interest_rate_up": 0.02,
    "interest_rate_down": 0.02,
    "direction": "up",
    "principal": 100000000,
}


'''生成路径'''
def generate_mc_paths(**para):
    '''
    params:
        S0: 初始价格
        sigma: 年化波动率
        r: 无风险利率
        q: 分红率
        T_days: 到期总天数
        I: 模拟路径数

    output:
        DataFrame: 模拟价格路径 (I 条路径 × T_days + 1 列，含第 0 天)
    '''

    S0 = para["S0"]
    print(f'开始价格为{S0}')
    sigma = para["sigma"]
    T_days = para.get("T_days")
    I = para.get("I")
    seed = para.get("seed")

    np.random.seed(seed)
    days = np.arange(0, T_days + 1)
    delta_days = np.diff(days, prepend=0)

    # 标准正态随机数矩阵
    z = np.random.standard_normal(size=(T_days + 1, I))
    z[0, :] = 0  # 起始日不变

    exponent = (-0.5 * sigma ** 2) * (delta_days[:, np.newaxis] / 365) + \
               sigma * np.sqrt(delta_days[:, np.newaxis] / 365) * z

    S_ratio = np.exp(exponent)

    S_T = S0 * S_ratio.cumprod(axis=0)  # 按照行累乘
    S_T = pd.DataFrame(S_T.T, columns=np.arange(T_days + 1))
    return S_T


S_T = generate_mc_paths(**para)


def plot_mc_paths(S_df: pd.DataFrame, num_paths: int = 50, title: str = "Monte Carlo Simulated Paths"):
    """
    绘制前n条蒙特卡洛路径轨迹
    参数:
        S_df (pd.DataFrame): 模拟路径数据，行数为路径数，列为时间步
        num_paths (int): 要绘制的路径数量（默认绘制前50条）
    """
    for i in range(min(num_paths, S_df.shape[0])):
        plt.plot(S_df.columns, S_df.iloc[i])

    plt.xlabel("Time Steps (Days)")
    plt.ylabel("Simulated Price")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

'''提取观察日价格'''
def extract_month_end_observation_days(S_df: pd.DataFrame, start_date: pd.Timestamp):
    """
    提取模拟路径中每月最后一个沪深交易日的观察点（使用Wind api获取交易日）。
    """

    # 1. 构造路径日期序列
    all_dates = pd.date_range(start=start_date, periods=S_df.shape[1], freq="D")

    # 2. 获取沪深交易日历（w.tdays）
    end_date = all_dates[-1]
    trade_days_data = w.tdays(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    if trade_days_data.ErrorCode != 0:
        raise ValueError("Wind API 获取交易日历失败")

    trade_days = pd.to_datetime(trade_days_data.Data[0])

    # 3. 保留 all_dates 中是沪深交易日的部分
    trading_dates = [d for d in all_dates if d in set(trade_days)]

    # 4. 每月最后一个交易日
    trading_df = pd.DataFrame({"date": trading_dates})
    trading_df["month"] = trading_df["date"].dt.to_period("M")
    month_end_trading_days = trading_df.groupby("month")["date"].max()

    # 5. 将以上日期转换为在模拟路径中的索引
    obs_day_dates = month_end_trading_days.tolist()
    obs_day_indices = [(d - all_dates[0]).days for d in obs_day_dates]

    # 6. 提取对应路径列
    S_obs = S_df.iloc[:, obs_day_indices].copy()
    S_obs.columns = [d.strftime("%Y-%m-%d") for d in obs_day_dates]

    return S_obs

S_obs = extract_month_end_observation_days(S_T, start_date = valuation_date)
# print(S_obs)
S_obs.columns = [
    (pd.to_datetime(col) - pd.to_datetime("2025-03-31")).days
    for col in S_obs.columns
]
# print(S_obs)


'''计算payoff'''
def evaluate_payoff(S_obs: pd.DataFrame, **para):
    """
    逐天判断是否触发计息或者敲出

    参数：
        S_obs: DataFrame，每行是路径，每列是观察日价格
        S0: 初始价格
        interest_barrier: 触发计息的门槛（比例）
        knock_out_barrier: 敲出门槛（比例）
        interest_rate: 每次计息事件固定收益
        rf: 贴现利率
        dt: 每个观察日间隔
        direction: "up" or "down"

    返回：
        Series: 每条路径贴现后的总收益
    """
    S0 = para["S0"]
    r = para.get("r", 0.02)
    direction = para.get("direction", "up")
    principal = para.get("principal", 100000000)

    interest_barrier_up = para.get("interest_barrier_up", 0.8)
    interest_barrier_down = para.get("interest_barrier_down", 1.2)
    knock_out_barrier_up = para.get("knock_out_barrier_up", 1.0)
    knock_out_barrier_down = para.get("knock_out_barrier_down", 1.0)
    final_knock_out_barrier_up = para.get("final_knock_out_barrier_up", 0.8)
    final_knock_out_barrier_down = para.get("final_knock_out_barrier_down", 1.2)

    interest_rate_up = para.get("interest_rate_up", 0.02)
    interest_rate_down = para.get("interest_rate_down", 0.02)

    payoff = pd.Series(0.0, index=S_obs.index)
    final_day = S_obs.columns[-1]  # 期末观察日

    for i in S_obs.index:  # 遍历每条路径
        knocked_out = False
        # for j, day in enumerate(S_obs.columns):  # j代表第几个观察日，day目前是21,42...
        for j, day in enumerate(S_obs.columns[:-1]):  # 遍历除最后一个观察日外的日期
            price = S_obs.loc[i, day]

            if direction == "up":
                if price > knock_out_barrier_up * S0:
                    knocked_out = True
                    break
                elif price > interest_barrier_up * S0:
                    payoff[i] += principal * interest_rate_up
                    # payoff[i] += principal * interest_rate_up * discount
                    # payoff[i] += principal * (interest_rate_up * day / 252) * discount
            elif direction == "down":
                if price < knock_out_barrier_down * S0:
                    knocked_out = True
                    break
                elif price < interest_barrier_down * S0:
                    payoff[i] += principal * interest_rate_down
                    # payoff[i] += principal * interest_rate_down * discount
                    # payoff[i] += principal * (interest_rate_up * day / 252) * discount

        # 期末判断
        if not knocked_out:
            final_price = S_obs.loc[i, final_day]
            if direction == "up" and final_price < final_knock_out_barrier_up * S0:
                loss = (1 - final_price/S0) * principal
                payoff[i] -= loss
            elif direction == "down" and final_price > final_knock_out_barrier_down * S0:
                loss = (final_price/S0 - 1) * principal
                payoff[i] -= loss
    discount = np.exp(-r * 1)
    # discount = 1 / (1 + r)
    payoff = payoff * discount
    return payoff

payoff = evaluate_payoff(S_obs, **para)
price = payoff.mean()
print(f"期权估值结果：{price:,.2f}")

