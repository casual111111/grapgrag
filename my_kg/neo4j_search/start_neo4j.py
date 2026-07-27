"""启动 Neo4j Docker 容器。

用法：
    python start_neo4j.py

容器信息：
    - 名称: graphrag-neo4j
    - HTTP Browser: http://localhost:7474
    - Bolt: bolt://localhost:7687
    - 用户名: neo4j
    - 密码: neo4j_test
"""

import subprocess
import sys
import time

CONTAINER_NAME = "graphrag-neo4j"
NEO4J_IMAGE = "neo4j:5.26"
NEO4J_PASSWORD = "neo4j_test"
HTTP_PORT = 7474
BOLT_PORT = 7687


def is_container_running() -> bool:
    """检查容器是否已经在运行。"""
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME],
        capture_output=True, text=True
    )
    return result.returncode == 0 and "true" in result.stdout.strip().lower()


def container_exists() -> bool:
    """检查容器是否存在（无论是否运行）。"""
    result = subprocess.run(
        ["docker", "inspect", CONTAINER_NAME],
        capture_output=True, text=True
    )
    return result.returncode == 0


def start_neo4j():
    """启动 Neo4j 容器。"""
    print("=" * 60)
    print("Neo4j Docker 容器启动脚本")
    print("=" * 60)

    # 检查 Docker 是否可用
    try:
        subprocess.run(["docker", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] Docker 未安装或未启动，请先安装 Docker Desktop")
        sys.exit(1)

    # 检查容器状态
    if is_container_running():
        print(f"[OK] 容器 '{CONTAINER_NAME}' 已在运行中")
        print(f"  Browser: http://localhost:{HTTP_PORT}")
        print(f"  Bolt:    bolt://localhost:{BOLT_PORT}")
        print(f"  用户名:  neo4j")
        print(f"  密码:    {NEO4J_PASSWORD}")
        return

    if container_exists():
        print(f"[INFO] 容器 '{CONTAINER_NAME}' 已存在但未运行，正在启动...")
        subprocess.run(["docker", "start", CONTAINER_NAME], check=True)
        print("[OK] 容器已启动")
    else:
        print(f"[INFO] 创建并启动新容器 '{CONTAINER_NAME}'...")
        cmd = [
            "docker", "run",
            "-d",
            "--name", CONTAINER_NAME,
            "-p", f"{HTTP_PORT}:{HTTP_PORT}",
            "-p", f"{BOLT_PORT}:{BOLT_PORT}",
            "-e", f"NEO4J_AUTH=neo4j/{NEO4J_PASSWORD}",
            "-e", "NEO4J_PLUGINS=[\"apoc\"]",
            "-e", "NEO4J_dbms_security_procedures_unrestricted=apoc.*",
            "-e", "NEO4J_dbms_security_procedures_allowlist=apoc.*",
            "-v", "graphrag-neo4j-data:/data",
            NEO4J_IMAGE,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] 启动失败:\n{result.stderr}")
            sys.exit(1)
        print(f"[OK] 容器已创建并启动: {result.stdout.strip()[:12]}")

    # 等待 Neo4j 就绪
    print("\n[INFO] 等待 Neo4j 就绪...")
    wait_for_neo4j(max_wait=60)

    print("\n" + "=" * 60)
    print("Neo4j 已就绪！")
    print(f"  Browser: http://localhost:{HTTP_PORT}")
    print(f"  Bolt:    bolt://localhost:{BOLT_PORT}")
    print(f"  用户名:  neo4j")
    print(f"  密码:    {NEO4J_PASSWORD}")
    print("=" * 60)


def wait_for_neo4j(max_wait: int = 60):
    """等待 Neo4j 服务就绪。"""
    import urllib.request
    import urllib.error

    url = f"http://localhost:{HTTP_PORT}"
    start = time.time()

    while time.time() - start < max_wait:
        try:
            req = urllib.request.urlopen(url, timeout=3)
            if req.status == 200:
                print(f"[OK] Neo4j 已就绪 (耗时 {int(time.time() - start)}s)")
                return
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        time.sleep(2)
        elapsed = int(time.time() - start)
        print(f"  等待中... ({elapsed}s)")

    print(f"[WARN] 等待超时 ({max_wait}s)，Neo4j 可能还未完全就绪")
    print("  请稍后手动检查 http://localhost:7474")


if __name__ == "__main__":
    start_neo4j()
