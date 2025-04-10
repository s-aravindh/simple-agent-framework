"""
Example of using the Simple Agent Framework with AWS Bedrock and structured output.
"""

import asyncio
from typing import List, Optional
from pydantic import BaseModel

from simple_agent import Agent, AgentConfig, BedrockModel, function_tool, pydantic_tool


# Define a Pydantic model for structured output
class MovieRecommendation(BaseModel):
    title: str
    year: int
    genre: str
    director: str
    description: str
    rating: Optional[float] = None


class MovieRecommendations(BaseModel):
    recommendations: List[MovieRecommendation]
    search_query: str


# Define a tool for movie information
@function_tool
def get_movie_info(title: str) -> dict:
    """Get information about a specific movie.
    
    Args:
        title: The title of the movie to look up.
        
    Returns:
        Dictionary containing movie information.
    """
    # Simplified movie database
    movies = {
        "The Shawshank Redemption": {
            "title": "The Shawshank Redemption",
            "year": 1994,
            "genre": "Drama",
            "director": "Frank Darabont",
            "description": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.",
            "rating": 9.3
        },
        "The Godfather": {
            "title": "The Godfather",
            "year": 1972,
            "genre": "Crime, Drama",
            "director": "Francis Ford Coppola",
            "description": "The aging patriarch of an organized crime dynasty transfers control to his reluctant son.",
            "rating": 9.2
        },
        "Inception": {
            "title": "Inception",
            "year": 2010,
            "genre": "Action, Sci-Fi",
            "director": "Christopher Nolan",
            "description": "A thief who steals corporate secrets through dream-sharing technology is given the task of planting an idea into the mind of a C.E.O.",
            "rating": 8.8
        }
    }
    
    return movies.get(title, {"error": f"Movie '{title}' not found"})


# Create an agent with Bedrock model
agent = Agent(
    name="MovieRecommendationAgent",
    instructions="""You are a movie recommendation agent. When asked for movie recommendations, 
provide a list of suitable movies in the structured output format.
Use the get_movie_info tool to retrieve details about specific movies when needed.
Your responses should focus on providing useful movie recommendations.""",
    model=BedrockModel(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-west-2"
    ),
    tools=[get_movie_info],
    config=AgentConfig(
        temperature=0.2,  # Lower temperature for more deterministic responses
        max_iterations=5
    ),
    output_type=MovieRecommendations  # Use Pydantic model for structured output
)


async def main():
    # Example: Get movie recommendations with structured output
    query = "Can you recommend some classic movies from the 1990s? Include The Shawshank Redemption in your list."
    result = await agent.arun(query)
    
    # Since we specified output_type, result should be a MovieRecommendations object
    print(f"Search Query: {result.search_query}")
    print(f"Number of Recommendations: {len(result.recommendations)}")
    
    print("\nMovie Recommendations:")
    for i, movie in enumerate(result.recommendations, 1):
        print(f"\n{i}. {movie.title} ({movie.year}) - {movie.genre}")
        print(f"   Director: {movie.director}")
        print(f"   Rating: {movie.rating if movie.rating else 'Not rated'}")
        print(f"   Description: {movie.description}")


if __name__ == "__main__":
    # Note: For this example to work, you need AWS credentials configured
    # with permissions to access Amazon Bedrock
    asyncio.run(main())