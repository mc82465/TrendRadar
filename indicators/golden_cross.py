#!/usr/bin/env python3
"""
MACD 金叉死叉策略 (12, 26, 9) - 仅限最新交易日
"""

import os
import pandas as pd
import pandas_ta as ta
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/home/ubuntu/stock_data/stock-trading-data-pro/stock-trading-data-pro'
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


def get_stock_files():
    if not os.path.exists(DATA_DIR):
        print(f"数据目录不存在: {DATA_DIR}")
        return []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    return [os.path.join(DATA_DIR, f) for f in files]


def check_macd_cross_on_date(df, target_date):
    """检查指定日期是否有 MACD 金叉/死叉"""
    if df is None or len(df) < 35:
        return None

    macd_result = ta.macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    if macd_result is None or len(macd_result) == 0:
        return None

    dif_col = macd_result.columns[0]
    hist_col = macd_result.columns[1]
    signal_col = macd_result.columns[2]

    macd_last = macd_result.tail(30).reset_index(drop=True)
    df_last = df.tail(30).reset_index(drop=True)

    # 只检查 target_date 当天和前一天
    for i in range(len(macd_last) - 1, 0, -1):
        curr_date = df_last.iloc[i]['date']
        prev_date = df_last.iloc[i - 1]['date']
        
        # 格式化为 date 对象比较
        curr_date_str = pd.to_datetime(curr_date).date() if hasattr(curr_date, 'date') else pd.to_datetime(curr_date).date()
        target_date_obj = pd.to_datetime(target_date).date()
        
        if curr_date_str != target_date_obj:
            continue

        prev_dif = macd_last.iloc[i - 1][dif_col]
        curr_dif = macd_last.iloc[i][dif_col]
        prev_signal = macd_last.iloc[i - 1][signal_col]
        curr_signal = macd_last.iloc[i][signal_col]

        if (pd.isna(prev_dif) or pd.isna(prev_signal) or
            pd.isna(curr_dif) or pd.isna(curr_signal)):
            continue

        curr_hist = macd_last.iloc[i][hist_col]
        curr_close = df_last.iloc[i]['close']

        # 金叉
        if prev_dif <= prev_signal and curr_dif > curr_signal:
            hist_avg = macd_last[hist_col].tail(5).mean()
            if hist_avg > 0.5:
                strength = '强势多头'
            elif hist_avg > 0.1:
                strength = '略有多头'
            elif hist_avg > -0.1:
                strength = '中性'
            elif hist_avg > -0.5:
                strength = '略有空头'
            else:
                strength = '强势空头'
            return 'golden_cross', curr_date, curr_close, curr_dif, curr_signal, curr_hist, strength

        # 死叉
        if prev_dif >= prev_signal and curr_dif < curr_signal:
            hist_avg = macd_last[hist_col].tail(5).mean()
            if hist_avg > 0.5:
                strength = '强势多头'
            elif hist_avg > 0.1:
                strength = '略有多头'
            elif hist_avg > -0.1:
                strength = '中性'
            elif hist_avg > -0.5:
                strength = '略有空头'
            else:
                strength = '强势空头'
            return 'death_cross', curr_date, curr_close, curr_dif, curr_signal, curr_hist, strength

    return None


def find_latest_date():
    """找最新交易日"""
    files = get_stock_files()
    latest = None
    for f in files[:100]:
        df = pd.read_csv(f, encoding='gbk', skiprows=1)
        for col in df.columns:
            if col == '交易日期':
                df = df.rename(columns={col: 'date'})
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                d = df['date'].max()
                if d and (latest is None or d > latest):
                    latest = d
                break
    return latest


def scan_stocks():
    latest_date = find_latest_date()
    if latest_date is None:
        print("无法确定最新日期")
        return
    
    print(f"{'='*70}")
    print(f"MACD 金叉死叉策略扫描: MACD({MACD_FAST}, {MACD_SLOW}, {MACD_SIGNAL})")
    print(f"限定交易日: {latest_date.strftime('%Y-%m-%d')}")
    print(f"{'='*70}")
    print()

    files = get_stock_files()
    print(f"股票数量: {len(files)}")

    golden_crosses = []
    death_crosses = []

    for i, f in enumerate(files):
        try:
            stock_code = os.path.basename(f).replace('.csv', '')
            df = pd.read_csv(f, encoding='gbk', skiprows=1)
            if df is None or len(df) == 0:
                continue

            cols = df.columns.tolist()
            rename_map = {}
            for col in cols:
                if col == '交易日期':
                    rename_map[col] = 'date'
                elif col == '收盘价':
                    rename_map[col] = 'close'

            if 'date' not in rename_map.values() or 'close' not in rename_map.values():
                continue

            df = df.rename(columns=rename_map)

            if isinstance(df['close'], pd.DataFrame):
                df['close'] = df['close'].iloc[:, 0]

            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date', 'close'])
            df = df.sort_values('date').reset_index(drop=True)
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df = df.dropna(subset=['close'])

            if len(df) < 35:
                continue

            result = check_macd_cross_on_date(df, latest_date)

            if result:
                signal, date, close, dif, signal_val, hist, strength = result
                info = {
                    'code': stock_code,
                    'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                    'close': round(close, 2),
                    'dif': round(dif, 4),
                    'signal': round(signal_val, 4),
                    'histogram': round(hist, 4),
                    'strength': strength
                }

                if signal == 'golden_cross':
                    golden_crosses.append(info)
                else:
                    death_crosses.append(info)

            if (i + 1) % 1000 == 0:
                print(f"已处理: {i+1}/{len(files)}")

        except Exception as e:
            continue

    # 输出结果
    print(f"\n{'='*70}")
    print(f"【{latest_date.strftime('%Y-%m-%d')} MACD 金叉信号】({len(golden_crosses)}只)")
    print(f"{'='*70}")
    if golden_crosses:
        golden_crosses.sort(key=lambda x: x['histogram'], reverse=True)
        for s in golden_crosses:
            print(f"  {s['code']}: 收盘:{s['close']} | DIF:{s['dif']} | Signal:{s['signal']} | 柱:{s['histogram']} | {s['strength']}")
    else:
        print("  无")

    print(f"\n{'='*70}")
    print(f"【{latest_date.strftime('%Y-%m-%d')} MACD 死叉信号】({len(death_crosses)}只)")
    print(f"{'='*70}")
    if death_crosses:
        for s in death_crosses:
            print(f"  {s['code']}: 收盘:{s['close']} | DIF:{s['dif']} | Signal:{s['signal']} | 柱:{s['histogram']} | {s['strength']}")
    else:
        print("  无")

    return golden_crosses, death_crosses


if __name__ == '__main__':
    scan_stocks()
