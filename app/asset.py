"""策略资产 θ 的存储与版本管理。

本范式下"可训练的参数"就是这个对象：一段可编辑的 base_prompt + 规则库。
对 θ 的更新 = 改写规则库文本再落盘，因此：

  - θ 是人类可读、可审查、可手工编辑的（这是本方案相对梯度 RL 的主要优势）
  - θ 有硬上限 MAX_RULES，且靠 gain 剪枝，这是本范式真正的"正则化"
  - θ 的容量受上下文窗口限制，规则写多了会稀释注意力，不是越多越好
"""

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from app.config import settings
from app.models import PolicyAsset, Rule

DEFAULT_BASE_PROMPT = """你是代码生成 Agent。你的产出由一批隐藏测试用例判定，你看不到这些用例。

可用工具：
  execute_code(code) —— 在独立进程中执行你的代码并返回测试结果。

输出协议（严格遵守，每次回复只输出一个块，不要输出多余解释）：

<attempt>
```python
# 你的代码
```
</attempt>
→ 提交一版代码试执行，我会把测试结果反馈给你。

<final>
```python
# 你认为正确的最终代码
```
</final>
→ 提交最终答案，任务结束。

硬性要求：
1. 必须定义名为 solution 的函数，签名与题目要求一致。
2. 只能使用 Python 标准库，禁止 input()、网络请求、文件读写。
3. 先考虑边界情况（空输入、重复元素、负数、单元素），再写代码。
4. 收到失败反馈后必须针对具体失败用例修正，不要重复提交同一版代码。"""


def _norm(text: str) -> str:
    """规则去重用的归一化键，中英文都保留。"""
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text).lower()


class AssetStore:
    """一个 task_type 对应一份 θ。"""

    def __init__(self, task_type: str = "codegen"):
        self.task_type = task_type
        self.dir = Path(settings.ASSET_DIR)
        self.path = self.dir / f"{task_type}.json"
        self.asset = self._load()

    # ---------- 持久化 ----------

    def _load(self) -> PolicyAsset:
        if self.path.exists():
            try:
                return PolicyAsset.model_validate_json(self.path.read_text("utf-8"))
            except Exception:
                # 资产文件损坏时不要让训练直接崩，重新起一份并在文件里留痕
                pass
        return PolicyAsset(task_type=self.task_type, base_prompt=DEFAULT_BASE_PROMPT)

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = self.asset.model_dump_json(indent=2)
        fd, tmp = tempfile.mkstemp(dir=str(self.dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, self.path)  # 原子替换，避免训练中断留下半截 JSON
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def reset(self) -> None:
        """清空规则库，回到未训练的 θ₀。用于复现基线。"""
        self.asset = PolicyAsset(task_type=self.task_type, base_prompt=DEFAULT_BASE_PROMPT)
        self.save()

    # ---------- 读写 ----------

    @property
    def version(self) -> int:
        return self.asset.version

    def render(self) -> str:
        return self.asset.render()

    def add_rules(self, draft: List[Rule]) -> Tuple[List[Rule], List[Rule]]:
        """写回 θ。返回 (新增的规则, 被剪掉的规则)。

        三道闸门：非空/长度校验 → 去重 → 超限按 gain 剪枝。
        没有这三道闸门，规则库会在一轮训练内膨胀到把上下文吃光。
        """
        existing = {_norm(r.text) for r in self.asset.rules}
        added: List[Rule] = []
        for r in draft:
            text = (r.text or "").strip()
            if not text or len(text) > 240:
                continue
            key = _norm(text)
            if not key or key in existing:
                continue
            existing.add(key)
            r.text = text
            r.asset_version = self.asset.version + 1
            self.asset.rules.append(r)
            added.append(r)

        pruned: List[Rule] = []
        if len(self.asset.rules) > settings.MAX_RULES:
            # gain 相同的（多为 0）保留更早写入的
            keep = sorted(
                self.asset.rules, key=lambda r: (-r.gain, r.asset_version)
            )[: settings.MAX_RULES]
            keep_ids = {id(r) for r in keep}
            pruned = [r for r in self.asset.rules if id(r) not in keep_ids]
            self.asset.rules = [r for r in self.asset.rules if id(r) in keep_ids]

        if added or pruned:
            self.asset.version += 1
            self.asset.updated_at = datetime.now().isoformat(timespec="seconds")
            self.save()
        return added, pruned
