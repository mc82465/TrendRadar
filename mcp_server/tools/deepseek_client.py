# coding=utf-8
import os
import json
from pathlib import Path
import re
from typing import Optional, List

import requests


DEEPSEEK_API_BASE = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"


def load_deepseek_api_key(env_path: Optional[Path] = None) -> str:
    """从项目根 .env 或环境变量读取 DeepSeek API Key。"""
    candidates = ["deepseek_API_KEY", "DEEPSEEK_API_KEY"]
    api_key = None

    if env_path is None:
        # 当前文件位于 mcp_server/tools/ 下，项目根为上上级目录
        env_path = Path(__file__).resolve().parents[2] / ".env"

    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
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


def build_messages(text_to_analyze: str, instruction_prompt: str) -> List[dict]:
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


def call_deepseek(api_key: str, messages: List[dict], model: str = DEEPSEEK_MODEL, temperature: float = 0.2) -> str:
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
    """保存分析结果到文件。默认保存到 test/ai_deepseek_analysis_output.md。"""
    if out_path is None:
        out_path = Path(__file__).resolve().parents[2] / "test" / "ai_deepseek_analysis_output.md"
    else:
        out_path = Path(out_path)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def generate_ai_html_from_md(md_text: str, output_html_path: Path) -> Path:
    """根据 MD 文本生成静态 HTML（样式对齐 test/ai_deepseek_analysis_output_11.10.html）。"""
    css = """
    * { box-sizing: border-box; }
    body { 
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        margin: 0; 
        padding: 16px; 
        background: #fafafa;
        color: #333;
        line-height: 1.5;
    }
    .container {
        max-width: 600px;
        margin: 0 auto;
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    }
    .header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 32px 24px;
        text-align: center;
        position: relative;
    }
    .save-buttons { position: absolute; top: 16px; right: 16px; display: flex; gap: 8px; }
    .save-btn {
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: white; padding: 8px 16px; border-radius: 6px; cursor: pointer;
        font-size: 13px; font-weight: 500; transition: all 0.2s ease; backdrop-filter: blur(10px);
        white-space: nowrap;
    }
    .save-btn:hover { background: rgba(255,255,255,0.3); border-color: rgba(255,255,255,0.5); transform: translateY(-1px); }
    .header-title { font-size: 22px; font-weight: 700; margin: 0 0 20px 0; }
    .header-info { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; font-size: 14px; opacity: 0.95; }
    .info-item { text-align: center; }
    .info-label { display: block; font-size: 12px; opacity: 0.8; margin-bottom: 4px; }
    .info-value { font-weight: 600; font-size: 16px; }
    .content { padding: 24px; }
    .markdown { font-size: 14px; color: #1a1a1a; }
    .markdown h3 { font-size: 18px; font-weight: 700; margin: 18px 0 10px; }
    .markdown h4 { font-size: 16px; font-weight: 600; margin: 14px 0 8px; }
    .markdown p { margin: 8px 0; }
    .markdown ul { margin: 8px 0 8px 18px; }
    .markdown table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .markdown th, .markdown td { border: 1px solid #e5e7eb; padding: 8px; text-align: left; }
    /* 卡片样式（参考 11.10 页面） */
    .section { margin-bottom: 24px; }
    .card { border: 1px solid #f0f0f0; border-radius: 10px; padding: 16px; margin: 12px 0; background:#fff; }
    .card-title { font-size: 15px; font-weight: 600; margin: 0 0 10px 0; color: #111827; }
    .meta { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 13px; color: #555; margin-bottom: 8px; }
    .meta-item { background: #f8f9fa; border-radius: 6px; padding: 8px; }
    .meta-label { display:block; font-size:12px; opacity:.7; }
    .meta-value { font-weight:600; }
    .sub-title { font-size: 14px; font-weight: 600; color:#374151; margin: 10px 0 6px; }
    .list { margin: 0; padding-left: 18px; }
    .list li { margin: 4px 0; }
    .footer { margin-top: 32px; padding: 20px 24px; background: #f8f9fa; border-top: 1px solid #e5e7eb; text-align: center; }
    .footer-content { font-size: 13px; color: #6b7280; line-height: 1.6; }
    .footer-link { color: #4f46e5; text-decoration: none; font-weight: 500; transition: color 0.2s ease; }
    .footer-link:hover { color: #7c3aed; text-decoration: underline; }
    .project-name { font-weight: 600; color: #374151; }
    @media (max-width: 480px) {
        body { padding: 12px; }
        .header { padding: 24px 20px; }
        .content { padding: 20px; }
        .footer { padding: 16px 20px; }
        .header-info { grid-template-columns: 1fr; gap: 12px; }
    }
    """

    # 更稳健的 Markdown 渲染：支持 h3/h4、连续列表、表格块、分割线
    def md_to_html(md: str) -> str:
        lines = md.splitlines()
        out = []
        in_list = False
        list_items = []
        in_table = False
        table_lines = []

        def flush_list():
            nonlocal in_list, list_items, out
            if in_list and list_items:
                out.append("<ul>")
                for item in list_items:
                    out.append(f"<li>{item}</li>")
                out.append("</ul>")
            in_list = False
            list_items = []

        def flush_table():
            nonlocal in_table, table_lines, out
            if in_table and table_lines:
                # 解析表格：首行为表头，第二行若为分隔线则跳过
                rows = []
                for tline in table_lines:
                    row = [c.strip() for c in tline.strip().strip("|").split("|")]
                    rows.append(row)
                header = rows[0] if rows else []
                body_rows = rows[1:]
                # 移除分隔线行（由 - 组成）
                if body_rows and all(re.fullmatch(r"-+", c) or c == "" for c in body_rows[0]):
                    body_rows = body_rows[1:]
                out.append("<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in header) + "</tr></thead><tbody>")
                for row in body_rows:
                    out.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
                out.append("</tbody></table>")
            in_table = False
            table_lines = []

        for raw in lines:
            line = raw.rstrip()
            stripped = line.strip()

            # 标题
            if stripped.startswith("### "):
                flush_list(); flush_table()
                out.append(f"<h3>{stripped[4:].strip()}</h3>")
                continue
            if stripped.startswith("#### "):
                flush_list(); flush_table()
                out.append(f"<h4>{stripped[5:].strip()}</h4>")
                continue

            # 表格块：以 | 开头的连续行
            if stripped.startswith("|"):
                flush_list()
                in_table = True
                table_lines.append(stripped)
                continue
            else:
                if in_table:
                    flush_table()

            # 分隔线（---）忽略或转为水平线
            if stripped == "---":
                flush_list()
                out.append("<hr style=\"border:none;border-top:1px solid #eee; margin:12px 0;\" />")
                continue

            # 列表项：连续的 - 开头行聚合为一个 ul
            if stripped.startswith("- "):
                if not in_list:
                    in_list = True
                item = stripped[2:].replace("**", "").strip()
                list_items.append(item)
                continue
            else:
                flush_list()

            # 其他普通段落
            if stripped:
                content = stripped.replace("**", "")
                out.append(f"<p>{content}</p>")

        flush_list(); flush_table()
        return "\n".join(out)

    html_body = md_to_html(md_text)
    html = (
        "<!DOCTYPE html>\n"
        "<html lang=\"zh-CN\">\n"
        "<head>\n"
        "  <meta charset=\"UTF-8\" />\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n"
        "  <title>AI 深度分析报告 · TrendRadar</title>\n"
        "  <script src=\"https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js\" crossorigin=\"anonymous\"></script>\n"
        "  <style>" + css + "</style>\n"
        "</head>\n"
        "<body>\n"
        "  <div class=\"container\">\n"
        "    <div class=\"header\">\n"
        "      <div class=\"save-buttons\"><button class=\"save-btn\" onclick=\"saveAsImage()\">保存为图片</button></div>\n"
        "      <div class=\"header-title\">AI 深度分析报告</div>\n"
        "      <div class=\"header-info\">\n"
        "        <div class=\"info-item\">\n"
        "          <span class=\"info-label\">报告类型</span>\n"
        "          <span class=\"info-value\">当日深度分析</span>\n"
        "        </div>\n"
        "        <div class=\"info-item\">\n"
        "          <span class=\"info-label\">数据来源</span>\n"
        "          <span class=\"info-value\">deepseek分析的 Markdown</span>\n"
        "        </div>\n"
        "        <div class=\"info-item\">\n"
        "          <span class=\"info-label\">生成时间</span>\n"
        "          <span class=\"info-value\" id=\"gen-time\">--:--</span>\n"
        "        </div>\n"
        "        <div class=\"info-item\">\n"
        "          <span class=\"info-label\">版本</span>\n"
        "          <span class=\"info-value\">v1.0</span>\n"
        "        </div>\n"
        "      </div>\n"
        "    </div>\n"
        "    <div class=\"content\"><div class=\"markdown\">" + html_body + "</div></div>\n"
        "    <div class=\"footer\">\n"
        "      <div class=\"footer-content\">\n"
        "        由 <span class=\"project-name\">TrendRadar</span> 生成 ·\n"
        "        <a href=\"https://github.com/sansan0/TrendRadar\" target=\"_blank\" class=\"footer-link\">GitHub 开源项目</a>\n"
        "      </div>\n"
        "    </div>\n"
        "  </div>\n"
        "  <script>\n"
        "  (function(){\n"
        "    const el = document.getElementById('gen-time');\n"
        "    const d = new Date();\n"
        "    const v = (\n"
        "      String(d.getMonth()+1).padStart(2,'0') + '-' +\n"
        "      String(d.getDate()).padStart(2,'0') + ' ' +\n"
        "      String(d.getHours()).padStart(2,'0') + ':' +\n"
        "      String(d.getMinutes()).padStart(2,'0')\n"
        "    );\n"
        "    if (el) el.textContent = v;\n"
        "  })();\n"
        "  async function saveAsImage(){\n"
        "    const button = event.target; const originalText = button.textContent;\n"
        "    try {\n"
        "      button.textContent = '生成中...'; button.disabled = true; window.scrollTo(0,0);\n"
        "      await new Promise(r=>setTimeout(r,150)); const buttons = document.querySelector('.save-buttons'); buttons.style.visibility='hidden';\n"
        "      await new Promise(r=>setTimeout(r,100)); const container = document.querySelector('.container');\n"
        "      const canvas = await html2canvas(container, { backgroundColor: '#ffffff', scale: 1.5 });\n"
        "      buttons.style.visibility='visible'; const link = document.createElement('a'); const now = new Date();\n"
        "      const filename = `TrendRadar_AI深度分析_${now.getFullYear()}${String(now.getMonth()+1).padStart(2,'0')}${String(now.getDate()).padStart(2,'0')}_${String(now.getHours()).padStart(2,'0')}${String(now.getMinutes()).padStart(2,'0')}.png`;\n"
        "      link.download = filename; link.href = canvas.toDataURL('image/png', 1.0); document.body.appendChild(link); link.click(); document.body.removeChild(link);\n"
        "      button.textContent = '保存成功!'; setTimeout(()=>{ button.textContent = originalText; button.disabled=false; }, 2000);\n"
        "    } catch(e) { const buttons = document.querySelector('.save-buttons'); buttons.style.visibility='visible'; button.textContent = '保存失败'; setTimeout(()=>{ button.textContent = originalText; button.disabled=false; }, 2000); }\n"
        "  }\n"
        "  // Markdown → 卡片化：优先级事项与宏观/行业/个股等\n"
        "  document.addEventListener('DOMContentLoaded', function(){\n"
        "    try {\n"
        "      const md = document.querySelector('.markdown');\n"
        "      if (!md) return;\n"
        "      const priorityLabels = ['高优先级事项：','中优先级事项：','低优先级事项：'];\n"
        "      Array.from(md.querySelectorAll('p')).forEach(p => {\n"
        "        const text = p.textContent.trim();\n"
        "        if (priorityLabels.includes(text)) {\n"
        "          const ul = p.nextElementSibling;\n"
        "          if (ul && ul.tagName === 'UL') {\n"
        "            const items = Array.from(ul.querySelectorAll('li')).map(li => li.textContent.trim());\n"
        "            let title='', reason='', urgency='', confidence='', suggest='';\n"
        "            items.forEach(t => {\n"
        "              if (t.startsWith('标题')) title = t.split('：')[1]?.trim() || t.replace(/^标题\s*：?/, '').trim();\n"
        "              else if (t.startsWith('原因')) reason = t.split('：')[1]?.trim() || t.replace(/^原因\s*：?/, '').trim();\n"
        "              else if (t.startsWith('紧迫度')) urgency = t.split('：')[1]?.trim() || t.replace(/^紧迫度\s*：?/, '').trim();\n"
        "              else if (t.startsWith('置信度')) confidence = t.split('：')[1]?.trim() || t.replace(/^置信度\s*：?/, '').trim();\n"
        "              else if (t.startsWith('执行建议')) suggest = t.split('：')[1]?.trim() || t.replace(/^执行建议\s*：?/, '').trim();\n"
        "            });\n"
        "            const card = document.createElement('div'); card.className = 'card';\n"
        "            card.innerHTML = `\n"
        "              <div class=\"card-title\">${title || '重点事项'}</div>\n"
        "              <div class=\"meta\">\n"
        "                <div class=\"meta-item\"><span class=\"meta-label\">紧迫度</span><span class=\"meta-value\">${urgency || '-'}</span></div>\n"
        "                <div class=\"meta-item\"><span class=\"meta-label\">置信度</span><span class=\"meta-value\">${confidence || '-'}</span></div>\n"
        "              </div>\n"
        "              ${reason ? `<div><span class=\\\"sub-title\\\">原因</span>：${reason}</div>` : ''}\n"
        "              ${suggest ? `<div class=\\\"sub-title\\\">执行建议</div><ul class=\\\"list\\\"><li>${suggest}</li></ul>` : ''}\n"
        "            `;\n"
        "            ul.replaceWith(card); p.remove();\n"
        "          }\n"
        "        }\n"
        "      });\n"
        "      const cardSectionTitles = ['宏观层面：','行业层面：','个股层面：','布局建议：','主要风险：','对冲建议：','宏观影响：','行业影响：','个股影响：'];\n"
        "      Array.from(md.querySelectorAll('p')).forEach(p => {\n"
        "        const text = p.textContent.trim();\n"
        "        if (cardSectionTitles.includes(text)) {\n"
        "          const ul = p.nextElementSibling;\n"
        "          if (ul && ul.tagName === 'UL') {\n"
        "            const card = document.createElement('div'); card.className = 'card';\n"
        "            const title = text.replace(/：$/, '');\n"
        "            const listHtml = ul.outerHTML.replace('<ul', '<ul class=\\\"list\\\"');\n"
        "            card.innerHTML = `<div class=\\\"card-title\\\">${title}</div>${listHtml}`;\n"
        "            ul.replaceWith(card); p.remove();\n"
        "          }\n"
        "        }\n"
        "      });\n"
        "      // 事件表格 → 卡片化\n"
        "      const table = md.querySelector('table');\n"
        "      if (table) {\n"
        "        try {\n"
        "          const headers = Array.from(table.querySelectorAll('thead th')).map(th => th.textContent.trim());\n"
        "          const rows = Array.from(table.querySelectorAll('tbody tr'));\n"
        "          const container = document.createElement('div'); container.className = 'section';\n"
        "          rows.forEach(tr => {\n"
        "            const cells = Array.from(tr.querySelectorAll('td')).map(td => td.textContent.trim());\n"
        "            const map = {}; headers.forEach((h,i)=> map[h] = cells[i] || '');\n"
        "            const card = document.createElement('div'); card.className = 'card';\n"
        "            const evt = map['事件'] || map['事件名称'] || cells[0] || '事件';\n"
        "            const date = map['日期'] || map['日期/窗口'] || map['时间窗口'] || cells[1] || '';\n"
        "            const view = map['观点'] || map['前瞻观点'] || cells[2] || '';\n"
        "            const effect = map['影响'] || map['市场影响'] || cells[3] || '';\n"
        "            const sector = map['板块与标的'] || cells[4] || '';\n"
        "            const adv = map['建议'] || map['建议与对冲'] || cells[5] || '';\n"
        "            card.innerHTML = `\n"
        "              <div class=\\\"card-title\\\">${evt}</div>\n"
        "              <div class=\\\"meta\\\">\n"
        "                <div class=\\\"meta-item\\\"><span class=\\\"meta-label\\\">日期</span><span class=\\\"meta-value\\\">${date || '-'}</span></div>\n"
        "                <div class=\\\"meta-item\\\"><span class=\\\"meta-label\\\">影响</span><span class=\\\"meta-value\\\">${effect || '-'}</span></div>\n"
        "              </div>\n"
        "              ${view ? `<div><span class=\\\"sub-title\\\">前瞻观点</span>：${view}</div>` : ''}\n"
        "              ${sector ? `<div class=\\\"sub-title\\\">板块与标的</div><ul class=\\\"list\\\"><li>${sector}</li></ul>` : ''}\n"
        "              ${adv ? `<div class=\\\"sub-title\\\">建议</div><ul class=\\\"list\\\"><li>${adv}</li></ul>` : ''}\n"
        "            `;\n"
        "            container.appendChild(card);\n"
        "          });\n"
        "          table.replaceWith(container);\n"
        "        } catch(e) { console.warn('事件表格卡片化失败', e); }\n"
        "      }\n"
        "      // 优先级筛选 → 卡片化（支持 标题/原因/紧迫度/置信度/执行建议 的序列）\n"
        "      const priH3 = Array.from(md.querySelectorAll('h3')).find(h => /优先级/.test(h.textContent));\n"
        "      if (priH3) {\n"
        "        const cardsContainer = document.createElement('div'); cardsContainer.className = 'section';\n"
        "        const toRemove = [];\n"
        "        let node = priH3.nextElementSibling;\n"
        "        while (node && node.tagName !== 'H3') {\n"
        "          if (node.tagName === 'UL') {\n"
        "            const li = node.querySelector('li');\n"
        "            const titleText = li ? li.textContent.trim() : '';\n"
        "            // 仅处理以“标题”开头的块\n"
        "            if (/^标题/.test(titleText)) {\n"
        "              let reason='', urgency='', confidence='', suggest='';\n"
        "              let cursor = node.nextElementSibling;\n"
        "              while (cursor && cursor.tagName !== 'UL' && cursor.tagName !== 'H3') {\n"
        "                if (cursor.tagName === 'P') {\n"
        "                  const t = cursor.textContent.trim();\n"
        "                  if (/^原因/.test(t)) reason = t.replace(/^原因\s*：?/, '').trim();\n"
        "                  else if (/^紧迫度/.test(t)) urgency = t.replace(/^紧迫度\s*：?/, '').trim();\n"
        "                  else if (/^置信度/.test(t)) confidence = t.replace(/^置信度\s*：?/, '').trim();\n"
        "                  else if (/^执行建议/.test(t)) suggest = t.replace(/^执行建议\s*：?/, '').trim();\n"
        "                }\n"
        "                toRemove.push(cursor);\n"
        "                cursor = cursor.nextElementSibling;\n"
        "              }\n"
        "              const card = document.createElement('div'); card.className = 'card';\n"
        "              const pureTitle = titleText.split('：')[1]?.trim() || titleText.replace(/^标题\s*：?/, '').trim();\n"
        "              card.innerHTML = `\n"
        "                <div class=\\\"card-title\\\">${pureTitle || '重点事项'}</div>\n"
        "                <div class=\\\"meta\\\">\n"
        "                  <div class=\\\"meta-item\\\"><span class=\\\"meta-label\\\">紧迫度</span><span class=\\\"meta-value\\\">${urgency || '-'}</span></div>\n"
        "                  <div class=\\\"meta-item\\\"><span class=\\\"meta-label\\\">置信度</span><span class=\\\"meta-value\\\">${confidence || '-'}</span></div>\n"
        "                </div>\n"
        "                ${reason ? `<div><span class=\\\"sub-title\\\">原因</span>：${reason}</div>` : ''}\n"
        "                ${suggest ? `<div class=\\\"sub-title\\\">执行建议</div><ul class=\\\"list\\\"><li>${suggest}</li></ul>` : ''}\n"
        "              `;\n"
        "              cardsContainer.appendChild(card);\n"
        "              toRemove.push(node);\n"
        "              node = cursor;\n"
        "              continue;\n"
        "            }\n"
        "          }\n"
        "          toRemove.push(node);\n"
        "          node = node.nextElementSibling;\n"
        "        }\n"
        "        if (toRemove.length) {\n"
        "          priH3.parentNode.insertBefore(cardsContainer, toRemove[0]);\n"
        "          toRemove.forEach(n => { try{ n.remove(); } catch(_){} });\n"
        "        }\n"
        "      }\n"
        "      // 大盘复盘 → 卡片化（趋势/情绪/风格）\n"
        "      const revH3 = Array.from(md.querySelectorAll('h3')).find(h => /大盘复盘/.test(h.textContent));\n"
        "      if (revH3) {\n"
        "        const sec = document.createElement('div'); sec.className = 'section';\n"
        "        const removeList = [];\n"
        "        let node = revH3.nextElementSibling;\n"
        "        const keys = ['趋势','情绪','风格'];\n"
        "        while (node && node.tagName !== 'H3') {\n"
        "          if (node.tagName === 'P') {\n"
        "            const t = node.textContent.trim();\n"
        "            const key = keys.find(k => new RegExp('^' + k).test(t));\n"
        "            if (key) {\n"
        "              let contentHtml = '';\n"
        "              let cursor = node.nextElementSibling;\n"
        "              if (cursor && cursor.tagName === 'UL') {\n"
        "                contentHtml = cursor.outerHTML.replace('<ul', '<ul class=\\\"list\\\"');\n"
        "                removeList.push(cursor);\n"
        "                cursor = cursor.nextElementSibling;\n"
        "              } else if (cursor && cursor.tagName === 'P') {\n"
        "                contentHtml = `<p>${cursor.textContent.trim()}</p>`;\n"
        "                removeList.push(cursor);\n"
        "                cursor = cursor.nextElementSibling;\n"
        "              }\n"
        "              const card = document.createElement('div'); card.className = 'card';\n"
        "              card.innerHTML = `<div class=\\\"card-title\\\">${key}</div>` + (contentHtml || '');\n"
        "              sec.appendChild(card);\n"
        "              removeList.push(node);\n"
        "              node = cursor;\n"
        "              continue;\n"
        "            }\n"
        "          }\n"
        "          removeList.push(node);\n"
        "          node = node.nextElementSibling;\n"
        "        }\n"
        "        if (removeList.length) {\n"
        "          revH3.parentNode.insertBefore(sec, removeList[0]);\n"
        "          removeList.forEach(n => { try{ n.remove(); } catch(_){} });\n"
        "        }\n"
        "      }\n"
        "    } catch(e) { console.error('Markdown卡片化失败:', e); }\n"
        "  });\n"
        "  </script>\n"
        "</body>\n"
        "</html>\n"
    )

    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    output_html_path.write_text(html, encoding="utf-8")
    return output_html_path


def send_md_to_notifications(md_text: str, config: dict, proxy_url: Optional[str] = None) -> dict:
    """将 MD 文本通过已配置的通知渠道发送（飞书/钉钉/企业微信/Telegram）。"""
    results = {}
    proxies = None
    if proxy_url:
        proxies = {"http": proxy_url, "https": proxy_url}

    # 飞书
    if config.get("FEISHU_WEBHOOK_URL"):
        url = config["FEISHU_WEBHOOK_URL"]
        payload = {"msg_type": "text", "content": {"text": md_text}}
        try:
            resp = requests.post(url, json=payload, timeout=30, proxies=proxies)
            results["feishu"] = resp.status_code
        except Exception as e:
            results["feishu_error"] = str(e)

    # 钉钉
    if config.get("DINGTALK_WEBHOOK_URL"):
        url = config["DINGTALK_WEBHOOK_URL"]
        payload = {"msgtype": "markdown", "markdown": {"title": "AI深度分析", "text": md_text}}
        headers = {"Content-Type": "application/json"}
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30, proxies=proxies)
            results["dingtalk"] = resp.status_code
        except Exception as e:
            results["dingtalk_error"] = str(e)

    # 企业微信
    if config.get("WEWORK_WEBHOOK_URL"):
        url = config["WEWORK_WEBHOOK_URL"]
        payload = {"msgtype": "markdown", "markdown": {"content": md_text}}
        try:
            resp = requests.post(url, json=payload, timeout=30, proxies=proxies)
            results["wework"] = resp.status_code
        except Exception as e:
            results["wework_error"] = str(e)

    # Telegram
    if config.get("TELEGRAM_BOT_TOKEN") and config.get("TELEGRAM_CHAT_ID"):
        url = f"https://api.telegram.org/bot{config['TELEGRAM_BOT_TOKEN']}/sendMessage"
        payload = {"chat_id": config["TELEGRAM_CHAT_ID"], "text": md_text, "parse_mode": "Markdown"}
        try:
            resp = requests.post(url, data=payload, timeout=30, proxies=proxies)
            results["telegram"] = resp.status_code
        except Exception as e:
            results["telegram_error"] = str(e)

    return results