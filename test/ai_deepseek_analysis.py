# coding=utf-8
"""
DeepSeek 财经分析脚本
需求：不再使用 frequency_words.txt。直接调用 DeepSeek API，分析指定文本内容。

功能：
- 从项目根 `.env` 读取 `deepseek_API_KEY`（或环境变量）
- 支持从目标文件的指定行读取文本，或手动覆盖分析文本
- 将中文分析任务（提示词）单独提成字符串，便于后期修改
- 调用 DeepSeek Chat API，输出结构化、可执行的结论
- 将分析结果打印并写入输出文件
"""

import os
import json
import requests
from pathlib import Path
from typing import Optional, Tuple


# ===== 可修改：分析目标文本 =====
# 1) 若设置为非空字符串，则直接使用该文本作为分析对象
# 2) 若留空，将从目标文件的指定行读取文本（默认：本文件第 142 行）
TEXT_TO_ANALYZE_OVERRIDE = ""

# ===== 可修改：目标文件与行号（用于示例指向） =====
# 目标文件改为指定的新闻文本，默认读取全文
TARGET_FILE_PATH = r"d:\study\github\TrendRadar\output\2025年11月10日\txt\09时08分.txt"
# 若为 None 表示读取全文；设置为整数则读取该行（1-based）
TARGET_LINE_NUMBER = None

# ===== 可修改：DeepSeek 模型与 API 端点 =====
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# ===== 可修改：中文提示词（任务说明） =====
INSTRUCTION_PROMPT = (
    "请执行以下任务并给出结构化、可执行的中文结论：\n"
    "1) 自动筛选优先级：从输入文本中提炼最重要事项，给出标题、原因、紧迫度(高/中/低)、置信度(0-100)、具体行动建议。\n"
    "2) 关键信息汇总与AI解读：简洁归纳要点，并解释其在宏观/行业/个股层面的实际影响。\n"
    "3) 大盘复盘：概述近期市场趋势、投资者情绪（偏乐观/中性/偏悲观）、风格倾向（成长/价值、大盘/小盘、权重/题材等）。\n"
    "4) 未来14天重大事件前瞻：列出可能发生的重要事件（如CPI/PPI数据、议息/降息会议、失业率、PMI、财报季节点、地缘风险等），给出预计日期或时间窗口、前瞻观点、可能的市场影响、受益/受损板块与代表性标的（标的请给名称或代码）、提前布局建议与风险对冲。\n"
    "请分段清晰，避免空话，突出可执行建议与风险提示。"
)


def load_deepseek_api_key(env_path: Optional[Path] = None) -> str:
    """从 .env 或环境变量读取 DeepSeek API Key。
    优先读取项目根目录 `.env` 中的 `deepseek_API_KEY`，否则读取环境变量。
    """
    candidates = ["deepseek_API_KEY", "DEEPSEEK_API_KEY"]
    api_key = None

    if env_path is None:
        # 本文件在 test/ 下，项目根在父级目录
        env_path = Path(__file__).resolve().parents[1] / ".env"

    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() in candidates:
                            api_key = v.strip()
                            break
        except Exception:
            pass

    if not api_key:
        for name in candidates:
            api_key = os.environ.get(name)
            if api_key:
                break

    if not api_key:
        raise RuntimeError(
            "未找到 DeepSeek API Key。请在项目根 .env 设置 'deepseek_API_KEY=...' 或配置环境变量。"
        )
    return api_key


def read_line_text(file_path: str, line_number: int) -> str:
    """读取目标文件的指定行文本内容（1-based）。"""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"目标文件不存在: {file_path}")
    with open(p, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if line_number < 1 or line_number > len(lines):
        raise ValueError(f"行号超出范围: {line_number} (文件总行数: {len(lines)})")
    return lines[line_number - 1].strip()


def read_full_text(file_path: str) -> str:
    """读取目标文件全文。"""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"目标文件不存在: {file_path}")
    return p.read_text(encoding="utf-8")


def build_messages(text_to_analyze: str, instruction_prompt: str):
    """构建 DeepSeek Chat 消息。"""
    system_msg = (
        "你是一名专业的金融与宏观分析助手，输出要结构化、可执行，避免空话。"
    )
    user_msg = (
        f"待分析文本：\n{text_to_analyze}\n\n"
        f"任务说明：\n{instruction_prompt}\n\n"
        "请按以下结构输出：\n"
        "1) 优先级筛选（标题/原因/紧迫度/置信度/执行建议）\n"
        "2) 关键信息汇总与AI解读\n"
        "3) 大盘复盘（趋势/情绪/风格）\n"
        "4) 未来14天事件前瞻（事件/日期/观点/影响/板块与标的/建议）\n"
        "5) 风险提示\n"
        "务必使用简洁中文、有序分段、强调可执行建议。"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def call_deepseek(api_key: str, messages, model: str = DEEPSEEK_MODEL, temperature: float = 0.2) -> str:
    """调用 DeepSeek Chat API 并返回文本结果。"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    resp = requests.post(DEEPSEEK_API_BASE, headers=headers, data=json.dumps(payload), timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"DeepSeek API 调用失败：HTTP {resp.status_code} - {resp.text}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return json.dumps(data, ensure_ascii=False, indent=2)


def save_output(text: str, out_path: Optional[str] = None) -> Path:
    """保存分析结果到文件。"""
    if out_path is None:
        out_path = Path(__file__).resolve().parent / "ai_deepseek_analysis_output.md"
    else:
        out_path = Path(out_path)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def main():
    print("=" * 80)
    print("🔍 DeepSeek 财经分析（基于指定文本）")
    print("=" * 80)

    # 1) 读取 API Key
    api_key = load_deepseek_api_key()
    print("✅ 已读取 DeepSeek API Key")

    # 2) 准备分析文本
    if TEXT_TO_ANALYZE_OVERRIDE.strip():
        text_to_analyze = TEXT_TO_ANALYZE_OVERRIDE.strip()
        src_desc = "来自 TEXT_TO_ANALYZE_OVERRIDE"
    else:
        if TARGET_LINE_NUMBER is None:
            text_to_analyze = read_full_text(TARGET_FILE_PATH)
            src_desc = f"来自 {TARGET_FILE_PATH} 全文"
        else:
            text_to_analyze = read_line_text(TARGET_FILE_PATH, TARGET_LINE_NUMBER)
            src_desc = f"来自 {TARGET_FILE_PATH} 第 {TARGET_LINE_NUMBER} 行"
    print(f"📄 分析文本来源：{src_desc}")
    print("-" * 80)
    print(text_to_analyze)
    print("-" * 80)

    # 3) 构建消息并调用 API
    messages = build_messages(text_to_analyze, INSTRUCTION_PROMPT)
    print("⏳ 正在调用 DeepSeek API……")
    result = call_deepseek(api_key, messages)

    # 4) 输出与保存
    print("\n" + "=" * 80)
    print("📊 分析结果（来自 DeepSeek）")
    print("=" * 80)
    print(result)

    out_file = save_output(result)
    print("\n💾 已保存分析结果：", out_file)


if __name__ == "__main__":
    main()