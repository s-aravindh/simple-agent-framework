"""
Example showing streaming capabilities of the Simple Agent Framework.
"""

import asyncio
from simple_agent import (
    Agent, 
    ContentChunkEvent, 
    ToolCallEvent, 
    ToolResultEvent, 
    ErrorEvent, 
    DoneEvent,
    function_tool
)


# Define a simple tool that takes a bit of time to run
@function_tool
async def search_database(query: str) -> str:
    """Search a database for information.
    
    Args:
        query: The search query
        
    Returns:
        The search results
    """
    # Simulate a slow database search
    await asyncio.sleep(1)
    
    databases = {
        "weather": "The weather is sunny with a high of 75°F.",
        "sports": "The home team won 4-2 in yesterday's game.",
        "news": "Latest headlines: New technology breakthrough announced.",
        "stocks": "The market is up 2% today with tech stocks leading gains."
    }
    
    # Find relevant info based on keywords in the query
    results = []
    for topic, info in databases.items():
        if topic.lower() in query.lower():
            results.append(info)
    
    if not results:
        return "No relevant information found in the database."
    
    return "\n".join(results)


# Create an agent with streaming capabilities
agent = Agent(
    name="StreamingAssistant",
    instructions="""You are a helpful assistant that provides information from a database.
When users ask questions, search the database for relevant information and provide helpful answers.
If you don't find specific information, be honest about what you don't know.""",
    tools=[search_database]
)


# Example 1: Synchronous streaming with event collection
def run_sync_example():
    print("\n=== Synchronous Streaming Example ===")
    
    # Get all events at once
    events = agent.stream("What's the latest weather and stock market news?")
    
    # Process the collected events
    content_chunks = []
    tool_calls = []
    
    for event in events:
        if isinstance(event, ContentChunkEvent):
            content_chunks.append(event.content)
            print(f"Content: {event.content}", end="", flush=True)
        elif isinstance(event, ToolCallEvent):
            tool_calls.append(event)
            print(f"\nTool Call: {event.tool_name}({event.arguments})")
        elif isinstance(event, ToolResultEvent):
            print(f"Tool Result: {event.result}")
        elif isinstance(event, ErrorEvent):
            print(f"Error: {event.message}")
        elif isinstance(event, DoneEvent):
            print(f"\nDone! Final content length: {len(event.final_content or '')}")
    
    # Print final statistics
    print(f"\nReceived {len(content_chunks)} content chunks and {len(tool_calls)} tool calls")


# Example 2: Asynchronous streaming with real-time processing
async def run_async_example():
    print("\n=== Asynchronous Streaming Example ===")
    
    # Process events as they arrive
    async for event in agent.astream("Tell me about the latest sports news and weather forecast."):
        if isinstance(event, ContentChunkEvent):
            print(f"{event.content}", end="", flush=True)
        elif isinstance(event, ToolCallEvent):
            print(f"\n[Calling tool: {event.tool_name}...]")
        elif isinstance(event, ToolResultEvent):
            print(f"\n[Tool result received]")
        elif isinstance(event, ErrorEvent):
            print(f"\nError: {event.message}")
        elif isinstance(event, DoneEvent):
            print("\n[Conversation complete]")


# Example 3: Interactive chat with streaming
async def interactive_chat():
    print("\n=== Interactive Chat with Streaming ===")
    print("Type your messages (or 'exit' to quit):")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ("exit", "quit"):
            break
            
        print("Assistant: ", end="", flush=True)
        async for event in agent.astream(user_input):
            if isinstance(event, ContentChunkEvent):
                print(f"{event.content}", end="", flush=True)
            elif isinstance(event, ToolCallEvent):
                print(f"\n[Searching database...]", end="", flush=True)
            elif isinstance(event, ToolResultEvent):
                pass  # Just silently process tool results
            elif isinstance(event, ErrorEvent):
                print(f"\nError: {event.message}", end="", flush=True)
            elif isinstance(event, DoneEvent):
                print("")  # Add a newline at the end


async def main():
    # Run the synchronous example
    run_sync_example()
    
    # Run the asynchronous example
    await run_async_example()
    
    # Run the interactive chat
    await interactive_chat()


if __name__ == "__main__":
    asyncio.run(main())