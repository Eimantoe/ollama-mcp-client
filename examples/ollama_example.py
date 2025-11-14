import asyncio
import sys
from src.clients.ollama_client import OllamaMCPClient

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    server_path = sys.argv[1]

    client = OllamaMCPClient()
    
    print("client initiated")
    try:
        await client.connect_to_server(server_path)
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
