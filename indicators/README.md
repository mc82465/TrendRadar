# 股票技术指标策略库

## 目录结构

```
indicators/
├── golden_cross.py   # MACD 金叉死叉扫描（纯显示）
├── save_signals.py   # MACD 指标写入 SQLite 数据库
data/
└── signals.db        # SQLite 指标数据库
```

## 数据源

- 股票数据：`/home/ubuntu/stock_data/stock-trading-data-pro/`
- CSV 格式：gbk 编码，第一行是表头

## 使用方法

### 1. MACD 金叉死叉扫描（只显示，不存库）

```bash
cd /home/ubuntu/TrendRadar
python3 indicators/golden_cross.py
```

输出示例：
```
======================================================================
MACD 金叉死叉策略扫描: MACD(12, 26, 9)
======================================================================
【MACD 金叉信号】(144只)
  贵州茅台: 2026-03-17 | 收盘:1485.0 | DIF:-1.12 | Signal:-3.31 | 柱:2.19 | 强势空头
  港迪技术: 2026-03-17 | 收盘:69.85 | DIF:-0.51 | Signal:-0.73 | 柱:0.22 | 中性
...
```

### 2. MACD 指标写入 SQLite

```bash
python3 indicators/save_signals.py
```

写入数据库 `data/signals.db`，包含：
- `indicator_signals` 表：每只股票每天的指标信号
- `daily_summary` 表：每天的信号汇总

### 3. 查看数据库

```bash
sqlite3 data/signals.db

-- 查看今日金叉（前10）
SELECT stock_code, stock_name, close_price, dif, histogram, strength
FROM indicator_signals 
WHERE trade_date = '2026-03-17' AND signal_type = 'golden_cross'
ORDER BY histogram DESC LIMIT 10;

-- 查看每日汇总
SELECT * FROM daily_summary ORDER BY trade_date DESC;
```

## 数据库表结构

### indicator_signals

| 字段 | 说明 |
|------|------|
| trade_date | 交易日期 |
| stock_code | 股票代码 |
| stock_name | 股票名称 |
| indicator_type | 指标类型（MACD） |
| signal_type | 信号类型（golden_cross/death_cross） |
| dif | DIF 值 |
| signal_line | Signal 线值 |
| histogram | MACD 柱值 |
| strength | 强弱（强势多头/略有多头/中性/略有空头/强势空头） |
| close_price | 当日收盘价 |

### daily_summary

| 字段 | 说明 |
|------|------|
| trade_date | 交易日期 |
| golden_count | 金叉数量 |
| death_count | 死叉数量 |
| indicator_type | 指标类型 |

## MACD 参数说明

- 快线周期：12
- 慢线周期：26
- Signal 周期：9

## 后续扩展

后续加入 KDJ、RSI、布林带等指标时：
1. 新建 `indicators/kdj_signals.py`、`indicators/rsi_signals.py`
2. 写入同一个 `data/signals.db`，通过 `indicator_type` 区分
3. uniapp 小程序通过 API 读取展示

## Navicat 连接

使用 SQLite 3 连接 `data/signals.db`，或通过 SSH 隧道连接服务器。
