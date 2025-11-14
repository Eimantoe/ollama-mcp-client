from typing import Optional
from anyio import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ollama import Client
from src.abstract.base_client import AbstractMCPClient

class OllamaMCPClient(AbstractMCPClient):
    def __init__(self):
        # Initialize session and client objects
        super().__init__()

        self.client = Client()
        self.tools = []

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        if not is_python:
            raise ValueError("Server script must be a .py file")

        # CRITICAL FIX: Detect if server has its own UV environment
        server_dir = Path(server_script_path).parent.parent.parent  # Go up to project root
        server_venv = server_dir / ".venv" / "bin" / "python"
        
        if await server_venv.exists():
            # Use server's own Python environment
            command = str(server_venv)
            print(f"Using server's venv: {command}")
        else:
            # Fallback to system Python
            command = "python"
            print("Warning: Using system Python, server dependencies may not be available")

        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize() # type: ignore

        # List available tools
        response = await self.session.list_tools() # type: ignore
        self.tools = [{
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    },
                } for tool in response.tools]
        print("\nConnected to server with tools:", [tool["function"]["name"] for tool in self.tools])


    async def process_query(self, query: str) -> str:
        """Process a query using LLM and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = self.client.chat(
            model="llama3.1:8b",
            messages=messages
        )

        # Process response and handle tool calls
        tool_results = []
        final_text = []

        if response.message.content:
            final_text.append(response.message.content)
        elif response.message.tool_calls:
            for tool in response.message.tool_calls:
                tool_name = tool.function.name
                tool_args = tool.function.arguments

                # Execute tool call
                result = await self.session.call_tool(tool_name, dict(tool_args)) # type: ignore
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # Continue conversation with tool results
                messages.append({
                    "role": "user",
                    "content": result.content[0].text # type: ignore
                })

                response = self.client.chat(
                    model="llama3.1:8b",
                    messages=messages,
                )

                final_text.append(response.message.content)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
