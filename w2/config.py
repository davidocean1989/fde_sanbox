"""集中配置：敏感值一律从环境变量读取，本文件禁止出现明文密钥。

变量来源优先级：真实环境变量 > 项目根目录 .env 文件。
缺 key 时立刻抛错，不要等到调 API 收到 401 才发现。
"""

import os
from pathlib import Path


def _load_env_file() -> None:
    """读取项目根目录的 .env（简化格式：KEY=VALUE，# 开头为注释）。

    已存在的环境变量不会被覆盖，保证 CI / 服务器注入的值优先。
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()


def _require(name: str) -> str:
    """必填项：缺失直接失败，错误信息指明怎么修。"""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"环境变量 {name} 未设置。请在项目根目录创建 .env "
            f"（可参考 .env.example），或通过 shell 导出该变量。"
        )
    return value


SILICONFLOW_API_KEY = _require("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
SILICONFLOW_BASE_URL_ALL = f"{SILICONFLOW_BASE_URL}/chat/completions"
SILICONFLOW_MODEL_DEEPSEEK = os.getenv("SILICONFLOW_MODEL_DEEPSEEK", "deepseek-ai/DeepSeek-V3")
SILICONFLOW_MODEL_QWEN = os.getenv("SILICONFLOW_MODEL_QWEN", "Qwen/Qwen2.5-7B-Instruct")

PROMPT_CHINESE = "我是一个地理学研究者。请计算北京天安门（北纬 39.9°、东经 116.4°）的对跖点。答三件事：1. 该对跖点的经纬度坐标（保留两位小数）2. 该点大致位于哪个海域或陆地，靠近哪个国家3. 写出你的计算过程：经度、纬度分别如何变换。顺序是：先给坐标结论，再给计算过程，最后用一到两句话，说明你判断该点地理位置的依据。"
PROMPT_CHINESE_B = "我是一个地理学研究者。请计算北京天安门（北纬 39.9°、东经 116.4°）的对跖点。答三件事：1. 该对跖点的经纬度坐标（保留两位小数）2. 该点大致位于哪个海域或陆地，靠近哪个国家3. 写出你的计算过程：经度、纬度分别如何变换。顺序是：先给坐标结论，再给计算过程，最后用一到两句话，说明你判断该点地理位置的依据。\n用 50 字以内回答"

PROMPT_ENGLISH = "As a geographer, I'm curious about the antipodal point of Tiananmen Square in Beijing, China — if you dig a tunnel from Tiananmen straight through the Earth's core, where would you emerge on the other side of the globe?"
PROMPT_EASY = "how to check if a Python object is an instance of a class?"


PROMPT_TRAP = """标题：红烧肉家常做法
摘要：五花肉 500 克切块，冷水下锅焯水，冰糖小火炒出糖色，下肉翻炒上色，加料酒、生抽，小火焖 40 分钟收汁。"""
PROMPT_TEST1 = """题：自建 RAG 知识库：完整方案
摘要：制定日期 2026-09-18，适用对象为重度 ChatGPT 用户、域名托管在 Cloudflare、长期运维。RAG 框架采用 Dify（知识库管理）+ Qdrant（向量库）+ 自写 FastMCP Server（给 ChatGPT 用）三件套。通过 Cloudflare Tunnel 出网建链，服务器 0 开放入站端口、源站 IP 永不暴露，数据安全采用 3-2-1 备份策略。"""
PROMPT_TEST2= """标题：FDE-执行02-12周日程总表
摘要：全职压缩版：12 周 / 每周 50–55 小时 / 累计约 620 小时 / 每天约 9 小时 × 每周 6 天（周日强制休息）。W2 主题为 LLM API 核心，交付六槽位 prompt 对比表与结构化输出修复率 ≥95%，W5 为评测与可靠性，交付评测报告 v1。"""
PROMPT_TEST3 ="""标题：烙画 / 针刺双路线定位与产品结构
摘要：烙画做 IP 人物有一个天然死穴：烙铁头太粗，五官容易糊。葫芦表面是弧面，烙铁笔头接触面积大、温度又不好压到极精细，市面上烙画葫芦基本都是山水、花鸟、写意人物，几乎没人敢做精细的现代 IP 角色，不是不想，是工艺做不到。"""
PROMPT_TEST4 = """标题：ZFC足球队财务报表系统架构与运维手册
摘要：ZFC足球队财务报表系统由 Excel 数据源 + 静态 HTML 展示页 + Cloudflare Pages 部署三部分组成。日常更新流程为：openpyxl 修改 Excel → Python 读取 Excel 生成 HTML → wrangler 部署到 football-finance 项目 → finance.hululugarden.com 生效。"""

SYSTEM_PROMPT= """1.角色与边界，是一个知识库入库的分类员，只管理分类情况，不评论内容的质量的好坏。 
2.任务定义：输入标题和摘要，输出主题+置信度+理由。
3.事实来源：只能依据给定标题和摘要判断，理由必须引用摘要原句。证据不足就返回 unknown，reason 说明无法匹配的原因。
4.约束与红线：主题能够从fde/hulu/live/public/rag/wechat/zfc这七个主题中选取，当无法满足这七个主题的时候，可放置于 unknown。不能编造主题名。
5.输出格式：明确 JSON 字段，为topic/confidence/reason。confidence的取值范围是 0-1，topic 的合法集合是七个主题（fde/hulu/live/public/rag/wechat/zfc）和 unknown。reason 要求引用摘要。只输出 JSON 本体，不加任何代码块围栏或解释文字。
6.示例：
示例 1：
输入： Hululu Garden官网：www.hululugarden.com记录的是 Hululu Garden 的主页内容，给于所有的 葫芦售卖信息进行整体展示的地方。 
好的示例： 
{
"topic":"hulu",
"confidence":0.8,
"reason":"记录的是 Hululu Garden 的主页内容"
}
坏的示例：
{
"topic":"hululu", 
"confidence": "Hululu Garden 的主页内容，给于所有", 
"reason":"包含对应的 hulu 内容，可被信赖。" 
} --（编造主题）--（置信度应该是0-1 的数字）--（空话+主观评论。应该是引用）

示例 2：
输入：八段锦学习的奥义：学习八段锦需要具有十足的信念才能将其完整的摸索清晰，自身要有极强的信心。
好的示例： 
{
"topic":"unknown",
"confidence":0,
"reason":"摘要内容为八段锦学习方法，无法归类到给定的七个主题中。"
}"""