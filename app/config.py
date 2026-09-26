"""全局配置。

与原始设计稿的两处差异：
1. Pydantic v2 用 `model_config = SettingsConfigDict(...)`，原来的 `class Config`
   在 v2 里已被弃用，字段会静默地不按预期加载。
2. 去掉了 LEARNING_RATE / KL_COEF —— 本实现不做梯度更新，保留这两个字段
   会误导阅读者以为存在反向传播。替换为真实存在的边界参数。
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- LLM ----
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_JUDGE_MODEL: str = "deepseek-chat"

    # ---- 服务 ----
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ---- 采样 ----
    ACT_TEMPERATURE: float = 0.7
    GROUP_TEMPERATURE: float = 1.0
    MAX_CONCURRENCY: int = 8

    # ---- 训练超参 ----
    GRPO_GROUP_SIZE: int = 4
    MAX_STEPS: int = 3
    MAX_ROLLOUTS: int = 24
    ADVANTAGE_EPS: float = 1e-4

    # ---- 奖励构成 ----
    VERIFY_WEIGHT: float = 0.7
    JUDGE_WEIGHT: float = 0.3

    # ---- 策略资产 θ 的边界 ----
    MAX_RULES: int = 24
    MIN_RULE_GAIN: float = 0.05

    # ---- 环境 ----
    ASSET_DIR: str = "assets"
    EXEC_TIMEOUT: float = 6.0

    @property
    def llm_configured(self) -> bool:
        return bool(self.DEEPSEEK_API_KEY.strip())


settings = Settings()
