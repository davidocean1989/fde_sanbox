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
SILICONFLOW_BASE_URL = os.getenv(
    "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
)
SILICONFLOW_BASE_URL_ALL = f"{SILICONFLOW_BASE_URL}/chat/completions"
SILICONFLOW_MODEL_DEEPSEEK = os.getenv(
    "SILICONFLOW_MODEL_DEEPSEEK", "deepseek-ai/DeepSeek-V3"
)  # input: 0.002000/ K Tokens    output:0.008000/ K Tokens. 属于 671B 激活37B
SILICONFLOW_MODEL_QWEN = os.getenv(
    "SILICONFLOW_MODEL_QWEN", "Qwen/Qwen2.5-72B-Instruct"
)  # input:0.004130/ K Tokens output:0.004130/ K Tokens  属于 72B
SILICONFLOW_MODEL_QWEN7B = os.getenv(
    "SILICONFLOW_MODEL_QWEN7B", "Qwen/Qwen2.5-7B-Instruct"
)  # 官方标记 (Free)：input 0.000000 / output 0.000000 元 / K Tokens（2026-11-20 控制台核对）

# 替 Hy4 的新档：智谱 GLM-4.5-Air，106B MoE / 128K 上下文，标签含「快速响应」。
# 价目来自控制台「在线推理」(2026-11-20)：input 0.001000 / output 0.006000 元 / K Tokens。
# ⚠ 它是混合推理模型，默认会思考（实测 440 reasoning tokens / 7.5s）。T9 必须统一关闭
#   推理模式（client.py 的 extra_body），否则会重演 Hy4 的推理税，四档不可比。
SILICONFLOW_MODEL_GLM45AIR = os.getenv(
    "SILICONFLOW_MODEL_GLM45AIR", "zai-org/GLM-4.5-Air"
)

# 2026-11-20 剔除 tencent/Hy4-preview（770B MoE 推理旗舰）：
#   理由不是「数据难看」，是任务错配 —— 20 条短分类用不到 1M 上下文与复杂推理。
#   实测 23.7× 成本 / P95 75.6s / 正确率反而 80%。旧 80 行留档在
#   probes_t9_20261120_with_hy4.jsonl，供 W5 报告引用，不删除、不静默替换。
PROMPT_CHINESE = "我是一个地理学研究者。请计算北京天安门（北纬 39.9°、东经 116.4°）的对跖点。答三件事：1. 该对跖点的经纬度坐标（保留两位小数）2. 该点大致位于哪个海域或陆地，靠近哪个国家3. 写出你的计算过程：经度、纬度分别如何变换。顺序是：先给坐标结论，再给计算过程，最后用一到两句话，说明你判断该点地理位置的依据。"
PROMPT_CHINESE_B = "我是一个地理学研究者。请计算北京天安门（北纬 39.9°、东经 116.4°）的对跖点。答三件事：1. 该对跖点的经纬度坐标（保留两位小数）2. 该点大致位于哪个海域或陆地，靠近哪个国家3. 写出你的计算过程：经度、纬度分别如何变换。顺序是：先给坐标结论，再给计算过程，最后用一到两句话，说明你判断该点地理位置的依据。\n用 50 字以内回答"

PROMPT_ENGLISH = "As a geographer, I'm curious about the antipodal point of Tiananmen Square in Beijing, China — if you dig a tunnel from Tiananmen straight through the Earth's core, where would you emerge on the other side of the globe?"
PROMPT_EASY = "how to check if a Python object is an instance of a class?"


PROMPT_TRAP = (
    "红烧肉家常做法：五花肉 500 克切块，冷水下锅焯水，冰糖小火炒出糖色，下肉翻炒上色，加料酒、生抽，小火焖 40 分钟收汁。"
    ""
)
PROMPT_TEST1 = "自建 RAG 知识库：完整方案：制定日期 2026-09-18，适用对象为重度 ChatGPT 用户、域名托管在 Cloudflare、长期运维。RAG 框架采用 Dify（知识库管理）+ Qdrant（向量库）+ 自写 FastMCP Server（给 ChatGPT 用）三件套。通过 Cloudflare Tunnel 出网建链，服务器 0 开放入站端口、源站 IP 永不暴露，数据安全采用 3-2-1 备份策略。"
PROMPT_TEST2 = "FDE-执行02-12周日程总表：全职压缩版：12 周 / 每周 50–55 小时 / 累计约 620 小时 / 每天约 9 小时 × 每周 6 天（周日强制休息）。W2 主题为 LLM API 核心，交付六槽位 prompt 对比表与结构化输出修复率 ≥95%，W5 为评测与可靠性，交付评测报告 v1。"
PROMPT_TEST3 = "烙画 / 针刺双路线定位与产品结构：烙画做 IP 人物有一个天然死穴：烙铁头太粗，五官容易糊。葫芦表面是弧面，烙铁笔头接触面积大、温度又不好压到极精细，市面上烙画葫芦基本都是山水、花鸟、写意人物，几乎没人敢做精细的现代 IP 角色，不是不想，是工艺做不到。"
PROMPT_TEST4 = "ZFC足球队财务报表系统架构与运维手册：ZFC足球队财务报表系统由 Excel 数据源 + 静态 HTML 展示页 + Cloudflare Pages 部署三部分组成。日常更新流程为：openpyxl 修改 Excel → Python 读取 Excel 生成 HTML → wrangler 部署到 football-finance 项目 → finance.hululugarden.com 生效。"
PROMPT_TEST5 = "Java 后端学习计划：12 周 / 每周 40 小时，W1-W4 Spring Boot，W5-W8 MySQL 与 Redis，W9-W12 微服务与分布式，每周学习 6 天，周日休息，最终目标是能独立开发电商后端。"
PROMPT_TEST6 = "向量数据库选型指南：Milvus、Qdrant、Weaviate 都支持 ANN 检索，选型时重点看数据规模、召回率要求和部署成本，大模型应用接入前建议先用基准数据集测召回。"
PROMPT_TEST7 = "微信小程序开发教程：先用 wx.request 封装统一网络层，再写云函数处理业务逻辑，上线前记得在微信公众平台配置合法域名和体验版权限。"
PROMPT_TEST8 = "男方孕前检查记录与结论：在朝阳医院常营院区完成精液分析和传染病筛查，各项指标正常，医生建议提前三个月补充叶酸、戒烟酒，复查周期为半年一次。"
PROMPT_TEST9 = "iPhone 代理客户端配置索引：Karing 与 sing-box 在 iOS 上的配置要点，包括节点订阅导入、分流规则设置和 MTU 调优，整理用于 iPhone 端代理客户端的日常运维。"

SYSTEM_PROMPT = """1.角色与边界，是一个知识库入库的分类员，只管理分类情况，不评论内容的质量的好坏。 
2.任务定义：输入标题和摘要，输出主题+置信度+理由。
3.主题词表：
fde：腾讯云 FDE 认证备考相关内容，包括周计划、日程总表、评测标准、交付物。
    不属于：其他任何技术栈的学习计划、培训课程（即使结构相似，含周数/小时数/日程表）。
rag：RAG 与知识库技术领域：本人项目（Dify + Qdrant + FastMCP、MCP 接入、摄入管道）
     及该领域通用技术内容（向量库、检索、Embedding、选型、架构对比）。
     不属于：与检索/知识库无关的纯大模型 API 使用技巧、与语义检索无关的通用数据库文章。
hulu：Hululu Garden 品牌业务内容：官网与产品展示、hululugarden.com 域名、
      烙画与针刺葫芦手作工艺、摆摊与售卖、为品牌服务的谷圈/IP 市场调研。
   不属于：与品牌业务无关的泛手工艺科普、非葫芦类工艺品、与品牌无关的泛娱乐内容。
live：本人生活事务的记录与决策：健康与孕前检查、备孕、家当采购决策（电视、
      陶瓷寄送等）、家庭事务、娱乐与游戏记录。
   不属于：泛生活科普、通用菜谱、任何技术内容（技术文档按所属主题归类，
      即使与生活场景相关）。
public：跨项目通用技术与基建：VPS 与代理配置、网络与系统运维、开发工具链、
        与任何单一主题无关但值得沉淀的技术内容。
   不属于：已有专属主题的内容优先归专属主题——RAG/检索/向量库技术文章归 rag
      （即使通用）；围绕具体项目写的技术文档归该项目主题；非技术内容若属本人生活事务归 live，其余归 unknown。。
wechat：威哥家宴小助手小程序项目：云函数（hostApi / guestApi）、体验版发布与部署、
        微信云开发环境、私有仓库同步、家宴相关的功能设计与记录。
     不属于：通用小程序开发教程、微信公众平台/开放平台的通用文档与配置说明
        （即使出现 wx.request、云函数、小程序、合法域名等关键词）。
zfc：ZFC 足球队相关内容，包括财务报表、运营手册等。
    不属于：与 ZFC 足球队无关的其他内容。

4.事实来源：只能依据给定标题和摘要判断，理由必须引用摘要原句。证据不足就返回 unknown，reason 说明无法匹配的原因。
5.约束与红线：主题必须从fde/hulu/live/public/rag/wechat/zfc这七个主题中选取，当无法满足这七个主题的时候，可放置于 unknown。不能编造主题名。
6.输出格式：明确 JSON 字段，为topic/confidence/reason。confidence的取值范围是 0-1，topic 的合法集合是七个主题（fde/hulu/live/public/rag/wechat/zfc）和 unknown。reason 要求引用摘要。只输出 JSON 本体，不加任何代码块围栏或解释文字。
7.示例：
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
}
再次强调：输出的第一个字符必须是 {，最后一个字符必须是 }，不允许任何代码块围栏或解释文字。
"""
