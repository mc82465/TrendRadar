#!/usr/bin/env python3
"""
MACD 指标写入 SQLite 数据库
每天自动记录金叉/死叉信号到 signals.db
"""

import os
import sqlite3
import pandas as pd
import pandas_ta as ta
import warnings
warnings.filterwarnings('ignore')

# 配置
DATA_DIR = '/home/ubuntu/stock_data/stock-trading-data-pro/stock-trading-data-pro'
DB_PATH = '/home/ubuntu/strategies/data/signals.db'
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
INDICATOR_TYPE = 'MACD'


def get_stock_files():
    if not os.path.exists(DATA_DIR):
        print(f"数据目录不存在: {DATA_DIR}")
        return []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    return [os.path.join(DATA_DIR, f) for f in files]


def find_latest_date(files):
    """找最新交易日"""
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

    for i in range(len(macd_last) - 1, 0, -1):
        curr_date = df_last.iloc[i]['date']
        curr_date_obj = pd.to_datetime(curr_date).date()
        target_date_obj = pd.to_datetime(target_date).date()

        if curr_date_obj != target_date_obj:
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
            return 'golden_cross', curr_dif, curr_signal, curr_hist, strength, curr_close

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
            return 'death_cross', curr_dif, curr_signal, curr_hist, strength, curr_close

    return None


def get_stock_name(f):
    """从CSV获取股票名称（读取最后一行的名称，去掉N前缀）"""
    try:
        # 读全部数据，取最后一行的名称
        df = pd.read_csv(f, encoding='gbk', skiprows=1)
        cols = df.columns.tolist()
        for col in cols:
            if '名称' in col:
                name = str(df.iloc[-1][col])  # 取最后一行（最新）
                # 去掉N前缀
                if name.startswith('N'):
                    name = name[1:]
                return name
    except:
        pass
    return ''


def save_to_db(records, trade_date):
    """写入 SQLite"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    golden_count = 0
    death_count = 0

    for rec in records:
        cursor.execute('''
            INSERT OR REPLACE INTO indicator_signals
            (trade_date, stock_code, stock_name, indicator_type, signal_type,
             signal_value, strength, dif, signal_line, histogram, close_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_date,
            rec['stock_code'],
            rec.get('stock_name', ''),
            INDICATOR_TYPE,
            rec['signal_type'],
            rec['histogram'],
            rec['strength'],
            rec['dif'],
            rec['signal_line'],
            rec['histogram'],
            rec['close']
        ))
        if rec['signal_type'] == 'golden_cross':
            golden_count += 1
        else:
            death_count += 1

    # 写入汇总
    cursor.execute('''
        INSERT OR REPLACE INTO daily_summary
        (trade_date, golden_count, death_count, indicator_type)
        VALUES (?, ?, ?, ?)
    ''', (trade_date, golden_count, death_count, INDICATOR_TYPE))

    conn.commit()
    conn.close()
    return golden_count, death_count


def scan_and_save():
    """扫描并保存"""
    print(f"{'='*60}")
    print(f"MACD 指标写入 SQLite")
    print(f"{'='*60}")
    print(f"数据库: {DB_PATH}")

    files = get_stock_files()
    print(f"股票数量: {len(files)}")

    latest_date = find_latest_date(files)
    if latest_date is None:
        print("无法确定最新日期")
        return

    trade_date = latest_date.strftime('%Y-%m-%d')
    print(f"交易日: {trade_date}")
    print()

    records = []

    for i, f in enumerate(files):
        try:
            stock_code = os.path.basename(f).replace('.csv', '')
            stock_name = get_stock_name(f)

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
                signal_type, dif, signal_line, hist, strength, close = result
                records.append({
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'signal_type': signal_type,
                    'dif': round(dif, 4),
                    'signal_line': round(signal_line, 4),
                    'histogram': round(hist, 4),
                    'strength': strength,
                    'close': round(close, 2)
                })

            if (i + 1) % 1000 == 0:
                print(f"已处理: {i+1}/{len(files)}")

        except Exception as e:
            continue

    # 保存到数据库
    if records:
        golden_count, death_count = save_to_db(records, trade_date)
        print(f"\n已写入数据库:")
        print(f"  金叉: {golden_count} 只")
        print(f"  死叉: {death_count} 只")
    else:
        print("\n无信号记录")

    return records


if __name__ == '__main__':
    scan_and_save()
