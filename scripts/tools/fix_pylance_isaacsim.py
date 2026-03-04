#!/usr/bin/env python3
"""
Isaac Sim Pylance 符号链接修复工具

问题根因：
  Isaac Sim 的 Python 包分散在 exts/extscache/extsDeprecated/extsPhysics 下的
  数百个扩展目录里，运行时由 Carbonite 框架动态加入 sys.path。
  Pylance 是静态分析器，看不到这些动态路径，导致无法跳转。

解决方案：
  在 site-packages 的主包目录（isaacsim/、omni/ 等）下创建符号链接，
  将所有扩展的 Python 包"聚合"到 Pylance 能找到的位置。

用法：
  python scripts/tools/fix_pylance_isaacsim.py          # 执行修复
  python scripts/tools/fix_pylance_isaacsim.py --dry-run # 预览，不实际操作
  python scripts/tools/fix_pylance_isaacsim.py --clean   # 删除所有链接（回滚）
"""
import os
import sys
import argparse

# ============================================================
# 配置：根据实际环境修改
# ============================================================
SITE_PACKAGES = "/home/xiao/miniconda3/envs/lab/lib/python3.11/site-packages"
ISAACSIM_PATH = os.path.join(SITE_PACKAGES, "isaacsim")

# 需要处理的顶层命名空间（必须在 site-packages 中存在）
TARGET_NAMESPACES = {"isaacsim", "omni", "carb"}

# 扩展目录（按优先级排列，前面的优先）
EXT_BASES = [
    os.path.join(ISAACSIM_PATH, "exts"),
    os.path.join(ISAACSIM_PATH, "extsDeprecated"),
    os.path.join(ISAACSIM_PATH, "extsPhysics"),
    os.path.join(ISAACSIM_PATH, "extscache"),
]
# ============================================================


def find_real_packages(ext_path: str, namespace: str) -> list[tuple[str, str]]:
    """
    在扩展目录中找到所有"真实包"（有 __init__.py 的最浅层目录）。

    例如扩展 isaacsim.core.api 的结构：
        isaacsim/         <- 命名空间包（无 __init__.py），跳过
            core/         <- 命名空间包（无 __init__.py），跳过
                api/      <- 真实包（有 __init__.py），返回此项
                    world.py
                    ...

    Returns:
        list of (relative_path_from_namespace, absolute_source_path)
        e.g., [("core/api", "/path/to/ext/isaacsim/core/api")]
    """
    ns_root = os.path.join(ext_path, namespace)
    if not os.path.isdir(ns_root):
        return []

    results = []

    def walk(current: str, rel_parts: list[str]):
        try:
            entries = os.listdir(current)
        except PermissionError:
            return

        if "__init__.py" in entries:
            # 这是真实包，记录并停止向下递归
            rel = os.path.join(*rel_parts) if rel_parts else "."
            results.append((rel, current))
            return

        # 命名空间目录，继续向下找
        for entry in sorted(entries):
            if entry.startswith((".", "__")):
                continue
            subpath = os.path.join(current, entry)
            if os.path.isdir(subpath):
                walk(subpath, rel_parts + [entry])

    walk(ns_root, [])
    return results


def get_created_links(ns_site_path: str) -> set[str]:
    """收集已存在的符号链接路径（用于 --clean 模式）"""
    links = set()

    def walk(path, rel=""):
        try:
            for entry in os.listdir(path):
                full = os.path.join(path, entry)
                rel_entry = os.path.join(rel, entry) if rel else entry
                if os.path.islink(full):
                    links.add(full)
                elif os.path.isdir(full):
                    walk(full, rel_entry)
        except PermissionError:
            pass

    walk(ns_site_path)
    return links


def fix(dry_run: bool = False):
    stats = {"created": 0, "skipped": 0, "error": 0}
    new_links = []

    # 记录已处理的目标路径，避免重复（多个扩展贡献同一包时取第一个）
    processed_targets: set[str] = set()

    for ns in TARGET_NAMESPACES:
        ns_site = os.path.join(SITE_PACKAGES, ns)
        if not os.path.exists(ns_site):
            print(f"跳过：{ns}（site-packages 中不存在）")
            continue

        for ext_base in EXT_BASES:
            if not os.path.exists(ext_base):
                continue

            for ext_name in sorted(os.listdir(ext_base)):
                ext_path = os.path.join(ext_base, ext_name)
                if not os.path.isdir(ext_path):
                    continue

                packages = find_real_packages(ext_path, ns)

                for rel_path, pkg_source in packages:
                    parts = rel_path.split(os.sep)

                    # 确保中间命名空间目录存在（创建为真实目录 + 空 __init__.py）
                    current = ns_site
                    ok = True
                    for part in parts[:-1]:
                        current = os.path.join(current, part)
                        if not os.path.exists(current):
                            if not dry_run:
                                os.makedirs(current, exist_ok=True)
                                # 空 __init__.py 让 Pylance 识别为包
                                open(os.path.join(current, "__init__.py"), "w").close()
                        elif os.path.islink(current):
                            # 已经是符号链接（指向整个包），跳过子路径
                            ok = False
                            break

                    if not ok:
                        stats["skipped"] += 1
                        continue

                    target = os.path.join(current, parts[-1])

                    if target in processed_targets:
                        stats["skipped"] += 1
                        continue

                    processed_targets.add(target)

                    if os.path.exists(target) or os.path.islink(target):
                        stats["skipped"] += 1
                        continue

                    rel_display = f"{ns}/{rel_path}"
                    ext_display = os.path.relpath(ext_path, ISAACSIM_PATH)

                    if dry_run:
                        print(f"  [预览] {rel_display}  ←  {ext_display}")
                        stats["created"] += 1
                    else:
                        try:
                            os.symlink(pkg_source, target)
                            new_links.append(rel_display)
                            stats["created"] += 1
                        except Exception as e:
                            print(f"  [错误] {rel_display}: {e}")
                            stats["error"] += 1

    # 输出统计
    mode = "[预览]" if dry_run else "[完成]"
    print(f"\n{mode} 创建: {stats['created']}  跳过: {stats['skipped']}  错误: {stats['error']}")

    if new_links and not dry_run:
        print("\n新创建的链接:")
        for link in sorted(new_links):
            print(f"  ✓ {link}")

    if not dry_run:
        print("\n请在 VSCode 中执行：Ctrl+Shift+P → Python: Restart Language Server")


def clean():
    """删除所有由本脚本创建的符号链接，回滚到初始状态"""
    removed = 0
    empty_dirs = []

    for ns in TARGET_NAMESPACES:
        ns_site = os.path.join(SITE_PACKAGES, ns)
        if not os.path.exists(ns_site):
            continue

        links = get_created_links(ns_site)
        for link in links:
            # 只删除指向 Isaac Sim 扩展目录的链接
            if ISAACSIM_PATH in os.path.realpath(link):
                print(f"  删除: {os.path.relpath(link, SITE_PACKAGES)}")
                os.unlink(link)
                removed += 1
                # 记录父目录，稍后清理空目录
                parent = os.path.dirname(link)
                if parent not in empty_dirs:
                    empty_dirs.append(parent)

    # 清理空的中间目录（只含 __init__.py 的目录）
    for d in sorted(empty_dirs, key=len, reverse=True):
        try:
            entries = set(os.listdir(d)) - {"__init__.py", "__pycache__"}
            if not entries and d != os.path.join(SITE_PACKAGES, "isaacsim"):
                init = os.path.join(d, "__init__.py")
                if os.path.exists(init):
                    os.unlink(init)
                os.rmdir(d)
                print(f"  清理目录: {os.path.relpath(d, SITE_PACKAGES)}")
        except OSError:
            pass

    print(f"\n[完成] 删除 {removed} 个符号链接")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isaac Sim Pylance 符号链接修复工具")
    parser.add_argument("--dry-run", action="store_true", help="预览操作，不实际执行")
    parser.add_argument("--clean", action="store_true", help="删除所有链接（回滚）")
    args = parser.parse_args()

    if args.clean:
        confirm = input("确认删除所有 Isaac Sim 符号链接？[y/N] ")
        if confirm.lower() == "y":
            clean()
        else:
            print("已取消")
    else:
        fix(dry_run=args.dry_run)
