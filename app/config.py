import os
import math
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# app/config.py -> app/ -> 项目根；不依赖本地目录名称。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# M9：等级阈值默认值（A / B / C 三档分数线，低于最后一档为 D）
DEFAULT_GRADE_CUTOFFS: tuple[float, float, float] = (8.0, 6.0, 4.0)

# .env 必须在 Settings() 实例化之前加载：
# pydantic-settings 只读取它自己配置的 env_file 与进程环境变量，
# .env 里的 key（例如 TAVILY_API_KEY）如果不先注入 os.environ，
# Settings 就取不到。load_dotenv 与 env_file 同时保留，互不影响。
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""

    tavily_api_key: str = ""
    tavily_max_results: int = 5

    tesseract_cmd: str = "tesseract"
    ocr_languages: str = "chi_sim+eng"
    ocr_tessdata_dir: str = ""  # Auto-select DATA_DIR/ocr/tessdata when installed.
    ocr_timeout_seconds: float = Field(default=30, gt=0, le=120)
    ocr_pdf_dpi: int = Field(default=300, ge=72, le=600)
    ocr_psm: int = Field(default=6, ge=1, le=13)  # Single-column resume/JD blocks.
    ocr_adaptive_languages: bool = True
    ocr_pdf_renderer: Literal["pdfium", "poppler"] = "pdfium"
    poppler_path: str = ""

    temp_parse: float = 0.2
    temp_plan: float = 0.5
    temp_ask: float = 0.7
    temp_assess: float = 0.2
    temp_fact_check: float = 0.1
    temp_evaluate: float = 0.3
    temp_follow_up: float = 0.4

    context_budget: int = Field(default=16000, ge=512)
    # Local practice data must not be exported merely because tracing env vars exist.
    enable_external_tracing: bool = False
    compress_threshold: int = 12000
    keep_recent_messages: int = 12

    # M25：题量上下限由**代码强制**（不足自动补题、超额自动截断），
    # 且 12-18 之间的题量原样保留（不再只是提示词文案）。
    target_questions: int = 15
    min_questions: int = 12
    max_questions: int = 18
    max_follow_ups: int = 3
    # P0/B2：**会话级**追问总预算——单场面试所有追问次数之和不得超此值。
    # 次数边界不依赖某供应商的人民币单价，不等同于费用硬上限。
    max_total_follow_ups: int = 12
    max_tool_rounds: int = 2
    max_output_tokens: int = Field(default=2048, ge=1)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_output_limit_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    llm_send_temperature: bool = True
    session_token_budget: int = Field(default=500000, ge=0)
    final_report_token_reserve: int = Field(default=50000, ge=0)

    # No implicit provider/currency/rates. Only matched, dated snapshots are priced.
    cost_input_per_1m: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    cost_output_per_1m: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    cost_model: str = ""
    cost_currency: str = ""
    cost_price_source: str = ""
    cost_price_effective_date: str = ""

    @field_validator("cost_input_per_1m", "cost_output_per_1m", mode="before")
    @classmethod
    def optional_price(cls, value):
        return None if value is None or (isinstance(value, str) and not value.strip()) else value

    @model_validator(mode="after")
    def validate_session_budget(self):
        if self.session_token_budget and self.final_report_token_reserve >= self.session_token_budget:
            raise ValueError("final report reserve must be smaller than session token budget")
        return self

    # M9：等级阈值（降序的三档分数线，依次对应 A / B / C，低于最后一档为 D）。
    # 格式为逗号分隔的数字，例如 "8,6,4"。解析失败时回退到 DEFAULT_GRADE_CUTOFFS。
    # 默认分界线与 1–5 档 ×2 的程序计分口径对应，不代表专家校准结果。
    grade_thresholds: str = "8,6,4"

    data_dir: str = "data"
    session_db: str = "data/sessions.db"
    checkpoint_db: str = "data/checkpoints.db"
    # 0 表示会话永久保留；>0 时按天数清理未更新的会话
    session_stale_days: int = 0

    # M21：是否让题目难度参与总分加权（easy 0.8 / medium 1.0 / hard 1.2）
    difficulty_weighting: bool = True

    @property
    def data_root(self) -> Path:
        return PROJECT_ROOT / self.data_dir

    @property
    def ocr_model_path(self) -> Path:
        """The installer and runtime share one configured language directory."""
        if self.ocr_tessdata_dir:
            path = Path(self.ocr_tessdata_dir).expanduser()
            return path if path.is_absolute() else PROJECT_ROOT / path
        return self.data_root / "ocr" / "tessdata"

    @property
    def grade_cutoffs(self) -> tuple[float, float, float]:
        """M9：解析等级阈值；不足三档或格式非法时回退到默认值。"""

        try:
            values = tuple(float(chunk.strip()) for chunk in self.grade_thresholds.split(","))
        except (ValueError, AttributeError):
            return DEFAULT_GRADE_CUTOFFS
        if (len(values) != 3 or any(not math.isfinite(value) or not 0 <= value <= 10 for value in values)
                or not values[0] > values[1] > values[2]):
            return DEFAULT_GRADE_CUTOFFS
        return values

    @property
    def session_db_path(self) -> Path:
        return PROJECT_ROOT / self.session_db

    @property
    def checkpoint_db_path(self) -> Path:
        return PROJECT_ROOT / self.checkpoint_db


settings = Settings()
if not settings.enable_external_tracing:
    for tracing_flag in ('LANGSMITH_TRACING','LANGCHAIN_TRACING_V2','LANGCHAIN_TRACING'):
        os.environ[tracing_flag] = 'false'


def self_check() -> dict[str, object]:
    """运行期配置自检（排障用，不参与任何面试逻辑）。

    api_key_set 反映的是"实际可用的 key"：settings.api_key 为空时，
    会按 app/llm/client.py 的回退顺序检查 OPENAI_API_KEY / DEEPSEEK_API_KEY。
    """
    from app.parsers.ocr import dependency_status
    api_key = (
        settings.api_key
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or ""
    )
    return {
        "project_root": str(PROJECT_ROOT),
        "env_file": str(ENV_FILE),
        "env_file_exists": ENV_FILE.exists(),
        "data_dir": str(settings.data_root),
        "data_dir_exists": settings.data_root.exists(),
        "api_key_set": bool(api_key),
        "tavily_key_set": bool(settings.tavily_api_key),
        "model": settings.model,
        "base_url": settings.base_url,
        "ocr": dependency_status(),
    }
