"""安全红线的规则化检查（不依赖模型判断）。

为什么必须是 AST 而不是正则：正则会被 `__import__("o"+"s")`、`getattr(os, "system")`
这类拼接绕过，而 AST 看到的是真实的 import 与调用节点。

这里的定位很清楚 —— 它是一道「禁止操作清单」闸门，不是沙箱。真正能防住恶意代码的
只有进程/网络/文件系统级隔离，而本实现没有（见 README 风险清单）。
"""

import ast
from typing import Dict, List, Tuple

from app.models import SafetyViolation

#: 禁止导入的模块 -> 违规规则名
FORBIDDEN_MODULES: Dict[str, str] = {
    "os": "module:os",
    "sys": "module:sys",
    "socket": "module:socket",
    "subprocess": "module:subprocess",
    "shutil": "module:shutil",
    "pathlib": "module:pathlib",
    "glob": "module:glob",
    "tempfile": "module:tempfile",
    "urllib": "module:urllib",
    "http": "module:http",
    "requests": "module:requests",
    "ftplib": "module:ftplib",
    "smtplib": "module:smtplib",
    "telnetlib": "module:telnetlib",
    "importlib": "module:importlib",
    "ctypes": "module:ctypes",
    "pickle": "module:pickle",
    "multiprocessing": "module:multiprocessing",
    "pty": "module:pty",
    "resource": "module:resource",
    "signal": "module:signal",
    "platform": "module:platform",
}

#: 禁止调用的内建函数 -> 违规规则名
FORBIDDEN_CALLS: Dict[str, str] = {
    "open": "call:open",
    "input": "call:input",
    "eval": "call:eval",
    "exec": "call:exec",
    "compile": "call:compile",
    "__import__": "call:__import__",
    "globals": "call:globals",
    "vars": "call:vars",
    "breakpoint": "call:breakpoint",
    "memoryview": "call:memoryview",
}

#: 禁止访问的魔术属性 -> 违规规则名（沙箱逃逸的典型入口）
FORBIDDEN_ATTRS: Dict[str, str] = {
    "__globals__": "attr:__globals__",
    "__subclasses__": "attr:__subclasses__",
    "__builtins__": "attr:__builtins__",
    "__bases__": "attr:__bases__",
    "__mro__": "attr:__mro__",
    "__code__": "attr:__code__",
    "__closure__": "attr:__closure__",
    "__dict__": "attr:__dict__",
    "__loader__": "attr:__loader__",
    "__getattribute__": "attr:__getattribute__",
}

#: 允许通过环境变量传给被测进程的变量名（其余一律剥离）
_PASSTHROUGH_ENV: Tuple[str, ...] = ("PATH", "LANG", "LC_ALL", "TZ", "HOME")


def scrubbed_env(base: Dict[str, str]) -> Dict[str, str]:
    """构造子进程环境变量：只保留白名单。

    默认继承的 os.environ 里带着 DEEPSEEK_API_KEY，被测代码一句
    `os.environ["DEEPSEEK_API_KEY"]` 就能把它读走，所以连同其他变量一起剥掉。
    """
    out = {k: base[k] for k in _PASSTHROUGH_ENV if k in base}
    out["PYTHONIOENCODING"] = "utf-8"
    out["PYTHONDONTWRITEBYTECODE"] = "1"
    return out


def scan(code: str) -> List[SafetyViolation]:
    """扫描代码里的越界操作，返回空列表 = 未命中任何红线。

    语法错误不在这里报（交给执行器给出编译错误更准确）。
    """
    src = (code or "").strip()
    if not src:
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # 语法都不合法，根本执行不了，交给执行器给出编译错误更准确
        return []

    found: List[SafetyViolation] = []
    seen: set = set()

    def add(rule: str, detail: str, line: int) -> None:
        key = (rule, line)
        if key in seen:
            return
        seen.add(key)
        found.append(SafetyViolation(rule=rule, detail=detail, line=line))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_MODULES:
                    add(FORBIDDEN_MODULES[root], f"禁止导入 {alias.name}", node.lineno)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in FORBIDDEN_MODULES:
                names = ", ".join(a.name for a in node.names)
                add(FORBIDDEN_MODULES[root], f"禁止从 {root} 导入 {names}", node.lineno)
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in FORBIDDEN_CALLS:
                add(FORBIDDEN_CALLS[name], f"禁止调用 {name}()", node.lineno)
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                add(
                    FORBIDDEN_ATTRS[node.attr],
                    f"禁止访问 {node.attr}",
                    node.lineno,
                )
        elif isinstance(node, ast.Name):
            # 覆盖 `from os import environ` 之后直接用 environ 的写法
            if node.id in ("environ", "getenv", "system", "popen"):
                add(f"name:{node.id}", f"禁止使用 {node.id}", node.lineno)

    return found


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""
