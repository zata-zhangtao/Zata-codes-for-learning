# mcp_math_server.py
import sys
import signal
import logging
import traceback
from mcp.server.fastmcp import FastMCP

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MCP-Server")

# 初始化 FastMCP 服务器实例，并给它一个名字
# 这个名字可以帮助客户端识别不同的 MCP 服务器
mcp_server = FastMCP(name="MathToolsServer")

@mcp_server.tool()
def add(a: int, b: int) -> int:
    """
    计算两个整数的和。
    例如: add(a=5, b=3) 将返回 8。
    """
    logger.info(f"Received add request: a={a}, b={b}")
    try:
        # 确保输入是整数
        a_int = int(a)
        b_int = int(b)
        result = a_int + b_int
        logger.info(f"Returning sum: {result}")
        return result
    except ValueError:
        logger.error(f"Invalid input types: a={type(a)}, b={type(b)}")
        raise ValueError(f"Both inputs must be integers, got a={type(a)}, b={type(b)}")
    except Exception as e:
        logger.error(f"Error in add: {e}")
        logger.error(traceback.format_exc())
        raise

@mcp_server.tool()
def multiply(a: int, b: int) -> int:
    """
    计算两个整数的乘积。
    例如: multiply(a=5, b=3) 将返回 15。
    """
    logger.info(f"Received multiply request: a={a}, b={b}")
    try:
        # 确保输入是整数
        a_int = int(a)
        b_int = int(b)
        result = a_int * b_int
        logger.info(f"Returning product: {result}")
        return result
    except ValueError:
        logger.error(f"Invalid input types: a={type(a)}, b={type(b)}")
        raise ValueError(f"Both inputs must be integers, got a={type(a)}, b={type(b)}")
    except Exception as e:
        logger.error(f"Error in multiply: {e}")
        logger.error(traceback.format_exc())
        raise

@mcp_server.tool()
def subtract(a: int, b: int) -> int:
    """
    计算两个整数的差 (a - b)。
    例如: subtract(a=5, b=3) 将返回 2。
    """
    logger.info(f"Received subtract request: a={a}, b={b}")
    try:
        # 确保输入是整数
        a_int = int(a)
        b_int = int(b)
        result = a_int - b_int
        logger.info(f"Returning difference: {result}")
        return result
    except ValueError:
        logger.error(f"Invalid input types: a={type(a)}, b={type(b)}")
        raise ValueError(f"Both inputs must be integers, got a={type(a)}, b={type(b)}")
    except Exception as e:
        logger.error(f"Error in subtract: {e}")
        logger.error(traceback.format_exc())
        raise

def handle_signals(signum, frame):
    """信号处理函数，用于优雅地关闭服务器"""
    logger.info(f"接收到信号 {signum}，准备关闭服务器...")
    # 此时可以执行任何清理操作
    sys.exit(0)

if __name__ == "__main__":
    # 注册信号处理程序
    signal.signal(signal.SIGINT, handle_signals)  # Ctrl+C
    signal.signal(signal.SIGTERM, handle_signals)  # 终止信号
    
    try:
        logger.info("Starting MathToolsServer on stdio...")
        # 显示工具信息
        tools = [func for func in [add, multiply, subtract]]
        logger.info(f"Registered tools: {[tool.__name__ for tool in tools]}")
        for tool in tools:
            logger.info(f"Tool '{tool.__name__}': {tool.__doc__.strip() if tool.__doc__ else 'No description'}")
        
        # 运行 MCP 服务器，使用 stdio (标准输入/输出) 进行通信。
        # 这对于在本地作为子进程运行服务器非常方便。
        mcp_server.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("服务器被用户终止")
    except Exception as e:
        logger.error(f"服务器运行时出错: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("MathToolsServer stopped.")