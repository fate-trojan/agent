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
    #: 单次提交代码的字符上限。超限直接拒绝，不执行（工具参数校验）
    MAX_CODE_CHARS: int = 20000

    # ---- 安全红线 ----
    #: True = 命中红线直接判失败（reward 归零），而不是仅扣分
    SAFETY_ENFORCE: bool = True

    # ---- 可审计轨迹 ----
    TRACE_DIR: str = "runs"
    TRACE_ENABLED: bool = True

    # ---- 自治边界（人工介入触发条件）----
    #: 训练 token 预算，0 表示不限。超出即停机并升级人工
    TOKEN_BUDGET: int = 0
    #: 连续多少个 group 组内奖励无方差（无学习信号）就升级人工
    MAX_FLAT_GROUPS: int = 4
    #: eval 变差时是否自动回滚 θ 到本轮基线
    ROLLBACK_ON_REGRESSION: bool = True

    @property
    def llm_configured(self) -> bool:
        return bool(self.DEEPSEEK_API_KEY.strip())


settings = Settings()
