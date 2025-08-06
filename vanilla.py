import numpy as np
from scipy.stats import norm

# 参数字典
params = {
    "S": 100,      # 标的价格
    "K": 100,      # 行权价格
    "T": 1,        # 到期时间（单位：年）
    "r": 0.05,     # 无风险利率
    "sigma": 0.2,   # 波动率
    'n_sim': 100000,  # 模拟次数
    'seed': 1,
    "q": 0,      # 分红率
}

params_bond = {
    "B_t": 101.5,        # 当前全价
    "I": 1.5,            # 计算日到期权定盘日期间标的债券产生的利息现金流的贴现值
    "D": 0.975,       # 贴现因子
    "K": 100,          # 执行价格（全价形式(T+0)）
    "sigma": 0.05,     # 年化波动率（债券价格）
    "T": 0.75,           # 剩余期限（以年为单位）
    "option_type": "call"  # 可选："call" 或 "put"
}


'''欧式看涨看跌'''
def european_call_bs(params):
    """
    欧式看涨期权 Black-Scholes 定价（含分红率）
    """
    S = params["S"]       # 标的初始价格
    K = params["K"]        # 执行价
    T = params["T"]        # 到期时间
    r = params["r"]       # 无风险利率
    sigma = params["sigma"]  # 波动率
    q = params["q"]  # 分红率，若无则默认为0

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    c = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return c


# 仅供参考，文档中使用的是bs模型估值
def european_call_mc(params):
    """
    使用蒙特卡洛方法估值欧式看涨期权

    参数：
        n_sim: 模拟路径数量，默认10万条
    """
    S = params["S"]
    K = params["K"]
    T = params["T"]
    r = params["r"]
    sigma = params["sigma"]
    n_sim = params['n_sim']
    seed = params['seed']

    np.random.seed(seed)

    # 生成标准正态随机变量
    z = np.random.randn(n_sim)

    # 计算到期资产价格 S_T
    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)

    # 计算贴现后的 payoff
    payoff = np.maximum(ST - K, 0)
    c = np.exp(-r * T) * np.mean(payoff)

    return c


def european_put_bs(params):
    """
    欧式看跌期权 Black-Scholes 定价（含分红率）
    """
    S = params["S"]          # 标的价格
    K = params["K"]          # 行权价格
    T = params["T"]          # 到期时间（年）
    r = params["r"]         # 无风险利率
    sigma = params["sigma"]  # 波动率
    q = params.get("q", 0)   # 分红率，默认为0

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    p = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return p


# 仅供参考，文档中使用的是bs模型估值
def european_put_mc(params):
    """
    使用蒙特卡洛方法估值欧式看跌期权

    参数：
        n_sim: 模拟路径数量，默认10万条
    """
    S = params["S"]
    K = params["K"]
    T = params["T"]
    r = params["r"]
    sigma = params["sigma"]
    n_sim = params["n_sim"]
    seed = params["seed"]

    np.random.seed(seed)

    # 生成标准正态随机变量
    z = np.random.randn(n_sim)

    # 模拟到期资产价格 S_T
    ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)

    # 计算贴现后的 payoff（Put 的 payoff）
    payoff = np.maximum(K - ST, 0)
    p = np.exp(-r * T) * np.mean(payoff)

    return p


'''美式看涨看跌'''
def american_call_binomial(params, N=100):
    """
    使用CRR二叉树法定价美式看涨期权，支持分红率q
    """
    S = params["S"]
    K = params["K"]
    T = params["T"]
    r = params["r"]
    sigma = params["sigma"]
    q = params.get("q", 0)
    phi = 1  # 美式看涨时为1，美式看跌时为-1

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)

    # 构建价格树
    ST = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            ST[j, i] = S * (u ** (i - j)) * (d ** j)

    # 构建价值树,（最后一层是intrinsic value, 不再有时间价值）
    option = np.zeros_like(ST)
    option[:, N] = np.maximum(phi * (ST[:, N] - K), 0)  # φ = 1

    # 回溯每个节点
    discount = np.exp(-r * dt)
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            hold = discount * (p * option[j, i + 1] + (1 - p) * option[j + 1, i + 1])
            exercise = max(phi * (ST[j, i] - K), 0)
            option[j, i] = max(hold, exercise)

    return option[0, 0]


def american_put_binomial(params, N=100):
    """
    美式看跌期权 CRR 二叉树定价
    """
    S = params["S"]
    K = params["K"]
    T = params["T"]
    r = params["r"]
    sigma = params["sigma"]
    q = params.get("q", 0)
    phi = -1

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)

    ST = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            ST[j, i] = S * (u ** (i - j)) * (d ** j)

    option = np.zeros_like(ST)
    option[:, N] = np.maximum(phi * (ST[:, N] - K), 0)

    discount = np.exp(-r * dt)
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            hold = discount * (p * option[j, i + 1] + (1 - p) * option[j + 1, i + 1])
            exercise = max(phi * (ST[j, i] - K), 0)
            option[j, i] = max(hold, exercise)

    return option[0, 0]


'''挂钩债券价格的欧式香草期权 --- 标的为债券价格'''
def european_bond_bs(params_bond: dict):
    """
    bs估值挂钩债券价格的欧式期权
    """
    B_t = params_bond["B_t"]
    I = params_bond["I"]
    D = params_bond["D"]
    K = params_bond["K"]
    sigma = params_bond["sigma"]
    T = params_bond["T"]
    option_type = params_bond["option_type"]
    phi = 1 if option_type == "call" else -1

    # 远期债券价格
    F_B = (B_t - I) / D

    # d1, d2
    d1 = (np.log(F_B / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # 价格计算公式
    price = phi * D * (F_B * norm.cdf(phi * d1) - K * norm.cdf(phi * d2))
    return price


'''欧式香草期权 -- 标的为恒定期限利率'''




if __name__ == "__main__":
    '''
    euro_bs_c = european_call_bs(params)
    euro_bs_p = european_put_bs(params)
    euro_mc_c = european_call_mc(params)
    euro_mc_p = european_put_mc(params)

    print(f"欧式看涨期权价格(B-S): {euro_bs_c:.4f}")
    print(f"欧式看跌期权价格(B-S): {euro_bs_p:.4f}")
    print(f"欧式看涨期权价格（MC）: {euro_mc_c:.4f}")
    print(f"欧式看跌期权价格（MC）: {euro_mc_p:.4f}")

    print('*' * 100)

    american_c = american_call_binomial(params, N=500)
    american_p = american_put_binomial(params, N=500)

    print(f"美式看涨期权价格（Binomial）: {american_c:.4f}")
    print(f"美式看跌期权价格（Binomial）: {american_p:.4f}")
    '''

    euro_bond_bs = european_bond_bs(params_bond)
    option_type = params_bond["option_type"]
    print(f"欧式{option_type}期权价格: {euro_bond_bs:.4f}")


