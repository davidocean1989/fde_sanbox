from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class TopicResult(BaseModel):
    model_config = ConfigDict(strict=True)
    topic: Literal[
        "fde", "hulu", "live", "public", "rag", "wechat", "zfc", "unknown"
    ] = Field(
        description="主题名称，必须从给定的七个主题中选取，无法匹配时为 unknown作为兜底"
    )
    confidence: float = Field(ge=0, le=1, description="置信度，取值范围为 0-1")
    reason: str = Field(
        min_length=1, description="理由，必须引用摘要原句，证据不足时说明无法匹配的原因"
    )


# print(TopicResult.model_validate_json('{"topic":"live","confidence":0.3,"reason":"this is a true reason for the mage"}'))
print(TopicResult.model_json_schema())  # Df. df
