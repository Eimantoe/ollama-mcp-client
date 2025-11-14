import asyncio
import sys
from src.clients.ollama_client import OllamaMCPClient

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script> [--uv]")
        print("  --uv: Use 'uv run' to run the server (for projects with their own dependencies)")
        sys.exit(1)

    server_path = sys.argv[1]
    use_uv = "--uv" in sys.argv

    client = OllamaMCPClient()
    print("client initiated")
    try:
        if use_uv:
            # Use uv run for servers with their own dependencies
            print(f"Connecting to server using 'uv run {server_path}'")
            await client.connect_to_server(
                server_script_path=server_path,
                command="uv",
                args=["run", server_path]
            )
        else:
            # Use default python/node detection
            await client.connect_to_server(server_path)

        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
