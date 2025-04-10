"""
Basic example of using the Simple Agent Framework with OpenAI.
"""

import asyncio
from simple_agent import Agent, function_tool

# Define a simple tool
@function_tool
def get_weather(location: str) -> str:
    """Get the current weather for a location.
    
    Args:
        location: The city or location to get weather for.
        
    Returns:
        A description of the weather.
    """
    # In a real implementation, this would call a weather API
    weather_data = {
        "San Francisco": "Foggy, 60°F",
        "New York": "Partly cloudy, 72°F",
        "London": "Rainy, 55°F",
        "Tokyo": "Sunny, 80°F",
    }
    
    return weather_data.get(location, f"Weather data not available for {location}")

# Define another tool for currency conversion
@function_tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert between currencies.
    
    Args:
        amount: The amount to convert.
        from_currency: The source currency code.
        to_currency: The target currency code.
        
    Returns:
        The converted amount.
    """
    # Simplified conversion rates
    rates = {
        "USD": 1.0,
        "EUR": 0.92,
        "JPY": 153.2,
        "GBP": 0.79,
    }
    
    if from_currency not in rates or to_currency not in rates:
        return f"Currency not supported: {from_currency} or {to_currency}"
    
    # Calculate conversion
    in_usd = amount / rates[from_currency]
    result = in_usd * rates[to_currency]
    
    return f"{amount} {from_currency} = {result:.2f} {to_currency}"

# Create an agent with OpenAI
agent = Agent(
    name="HelperAssistant",
    instructions="""You are a helpful assistant that can provide information and use tools.
When asked about weather or currency conversion, use the appropriate tools.
Be friendly and concise in your responses.""",
    tools=[get_weather, convert_currency]
)

# Run the agent
async def main():
    # Example 1: Weather
    result1 = await agent.arun("What's the weather like in Tokyo?")
    print(f"Example 1 Response: {result1}")
    
    # Example 2: Currency conversion
    result2 = await agent.arun("I need to convert 100 USD to EUR, can you help?")
    print(f"Example 2 Response: {result2}")
    
    # Example 3: General knowledge (no tool use)
    result3 = await agent.arun("What are some famous landmarks in Paris?")
    print(f"Example 3 Response: {result3}")

if __name__ == "__main__":
    asyncio.run(main())