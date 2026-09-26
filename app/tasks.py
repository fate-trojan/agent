"""任务集：可验证奖励的来源。

tests 是 (表达式, 期望值) 形态，因此奖励是「通过率」而不是 0/1 —— GRPO 的组内方差靠它。
split="eval" 的任务不参与训练，只在训练前后各评一次。eval 与 train 刻意不重叠，
否则无法区分「泛化」和「背题」。
"""

from typing import Dict, List

from app.models import TaskSpec

_RAW = [
    # ---------------- 训练集 ----------------
    dict(
        id="two_sum",
        split="train",
        difficulty="easy",
        prompt=(
            "实现 two_sum(nums, target)：在整数列表 nums 中找出两个元素，使它们的和等于 target，"
            "返回这两个元素的下标，按升序返回（小下标在前）。假设恰好存在一组解。"
        ),
        tests=[
            ("solution([2,7,11,15], 9)", [0, 1]),
            ("solution([3,2,4], 6)", [1, 2]),
            ("solution([3,3], 6)", [0, 1]),
            ("solution([-1,-2,-3,-4,5], 1)", [3, 4]),
            ("solution([0,4,3,0], 0)", [0, 3]),
        ],
    ),
    dict(
        id="valid_parentheses",
        split="train",
        difficulty="easy",
        prompt=(
            "实现 solution(s)：判断只由 '()[]{}' 组成的字符串 s 是否为合法的括号序列。"
            "合法指每个左括号都被正确顺序的右括号闭合。空串视为合法。返回 bool。"
        ),
        tests=[
            ("solution('()')", True),
            ("solution('()[]{}')", True),
            ("solution('(]')", False),
            ("solution('([)]')", False),
            ("solution('{[]}')", True),
            ("solution('')", True),
            ("solution('(')", False),
        ],
    ),
    dict(
        id="merge_intervals",
        split="train",
        difficulty="medium",
        prompt=(
            "实现 solution(intervals)：合并所有重叠或相邻的区间。"
            "intervals 是 [[start, end], ...]，可能未排序、可能为空，单个区间视为 [x, x]。"
            "返回按 start 升序排列的合并结果。"
        ),
        tests=[
            ("solution([[1,3],[2,6],[8,10],[15,18]])", [[1, 6], [8, 10], [15, 18]]),
            ("solution([[1,4],[4,5]])", [[1, 5]]),
            ("solution([[1,4],[0,4]])", [[0, 4]]),
            ("solution([[1,4],[2,3]])", [[1, 4]]),
            ("solution([])", []),
            ("solution([[1,1]])", [[1, 1]]),
        ],
    ),
    dict(
        id="top_k_frequent",
        split="train",
        difficulty="medium",
        prompt=(
            "实现 solution(nums, k)：返回 nums 中出现频率最高的 k 个元素。"
            "返回顺序必须确定：按出现次数降序，次数相同时按元素值升序。返回列表。"
        ),
        tests=[
            ("solution([1,1,1,2,2,3], 2)", [1, 2]),
            ("solution([1], 1)", [1]),
            ("solution([4,4,4,4], 1)", [4]),
            ("solution([5,5,6,6,7], 2)", [5, 6]),
            ("solution([9,8,7], 3)", [7, 8, 9]),
            ("solution([1,2,1,2], 2)", [1, 2]),
        ],
    ),
    dict(
        id="max_subarray",
        split="train",
        difficulty="medium",
        prompt=(
            "实现 solution(nums)：返回 nums 中连续子数组的最大和。"
            "子数组至少包含一个元素，nums 可能全为负数，此时答案是其中最大的那个负数。"
        ),
        tests=[
            ("solution([-2,1,-3,4,-1,2,1,-5,4])", 6),
            ("solution([1])", 1),
            ("solution([5,4,-1,7,8])", 23),
            ("solution([-1])", -1),
            ("solution([-2,-1])", -1),
            ("solution([0,0])", 0),
        ],
    ),
    dict(
        id="roman_to_int",
        split="train",
        difficulty="medium",
        prompt=(
            "实现 solution(s)：把罗马数字字符串 s 转成整数。"
            "字符表：I=1, V=5, X=10, L=50, C=100, D=500, M=1000。"
            "小值在大值左侧表示减去（如 IV=4, IX=9, MCMXCIV=1994）。返回 int。"
        ),
        tests=[
            ("solution('III')", 3),
            ("solution('IV')", 4),
            ("solution('IX')", 9),
            ("solution('LVIII')", 58),
            ("solution('MCMXCIV')", 1994),
            ("solution('MMXXV')", 2025),
        ],
    ),
    # ---------------- 评测集（不参与训练） ----------------
    dict(
        id="longest_common_prefix",
        split="eval",
        difficulty="easy",
        prompt=(
            "实现 solution(strs)：返回字符串列表中所有字符串的最长公共前缀。"
            "若不存在公共前缀则返回空字符串 ''。列表可能包含空字符串。"
        ),
        tests=[
            ("solution(['flower','flow','flight'])", "flow"),
            ("solution(['dog','racecar','car'])", ""),
            ("solution(['a'])", "a"),
            ("solution(['','b'])", ""),
            ("solution(['ab','ab'])", "ab"),
            ("solution(['abc','abd'])", "ab"),
        ],
    ),
    dict(
        id="rotate_right",
        split="eval",
        difficulty="medium",
        prompt=(
            "实现 solution(nums, k)：把列表 nums 向右轮转 k 个位置并返回新列表。"
            "不要原地修改入参。k 可能为 0、可能等于或大于列表长度。"
        ),
        tests=[
            ("solution([1,2,3,4,5,6,7], 3)", [5, 6, 7, 1, 2, 3, 4]),
            ("solution([1], 0)", [1]),
            ("solution([-1,-100,3,99], 2)", [3, 99, -1, -100]),
            ("solution([1,2,3,4], 4)", [1, 2, 3, 4]),
            ("solution([1,2,3], 5)", [2, 3, 1]),
        ],
    ),
    dict(
        id="count_primes",
        split="eval",
        difficulty="medium",
        prompt=(
            "实现 solution(n)：返回严格小于 n 的质数个数。"
            "n 可能为 0 或 1，此时返回 0；n=2 时也返回 0（2 不小于 2）。返回 int。"
        ),
        tests=[
            ("solution(10)", 4),
            ("solution(1)", 0),
            ("solution(2)", 0),
            ("solution(0)", 0),
            ("solution(20)", 8),
            ("solution(30)", 10),
        ],
    ),
]

TASKS: List[TaskSpec] = [TaskSpec(**r) for r in _RAW]
_BY_ID: Dict[str, TaskSpec] = {t.id: t for t in TASKS}


def train_tasks() -> List[TaskSpec]:
    return [t for t in TASKS if t.split == "train"]


def eval_tasks() -> List[TaskSpec]:
    return [t for t in TASKS if t.split == "eval"]


def get_task(task_id: str) -> TaskSpec:
    if task_id not in _BY_ID:
        raise KeyError(f"未知 task_id: {task_id}（可用：{', '.join(_BY_ID)}）")
    return _BY_ID[task_id]
