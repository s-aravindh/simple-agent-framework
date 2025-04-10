"""
Advanced example showing Pydantic models for both input and output validation.
"""

import asyncio
from datetime import datetime
from enum import Enum
from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator

from simple_agent import Agent, OpenAIModel, AgentConfig, pydantic_tool


# --- Define Pydantic models for structured input and output ---

class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Task(BaseModel):
    """A task in the task management system."""
    id: Optional[int] = None
    title: str
    description: str
    priority: TaskPriority
    due_date: Optional[datetime] = None
    status: TaskStatus = TaskStatus.TODO
    tags: List[str] = Field(default_factory=list)
    
    @validator('due_date')
    def due_date_must_be_future(cls, v):
        if v and v < datetime.now():
            raise ValueError('Due date must be in the future')
        return v


class CreateTaskInput(BaseModel):
    """Input model for creating a task."""
    title: str
    description: str
    priority: TaskPriority
    due_date: Optional[str] = None  # ISO format date string
    tags: List[str] = Field(default_factory=list)


class TaskResponse(BaseModel):
    """Response model for task operations."""
    success: bool
    message: str
    task: Optional[Task] = None


class TaskList(BaseModel):
    """List of tasks with metadata."""
    tasks: List[Task]
    total_count: int
    filter_applied: Optional[str] = None


# --- Simulated task database ---
task_db = {
    1: Task(
        id=1,
        title="Implement agent framework",
        description="Create a simple agent framework combining OpenAI and Pydantic",
        priority=TaskPriority.HIGH,
        due_date=datetime.fromisoformat("2025-04-15T00:00:00"),
        status=TaskStatus.COMPLETED,
        tags=["coding", "ai"]
    ),
    2: Task(
        id=2,
        title="Write documentation",
        description="Document the agent framework with examples",
        priority=TaskPriority.MEDIUM,
        due_date=datetime.fromisoformat("2025-04-20T00:00:00"),
        status=TaskStatus.IN_PROGRESS,
        tags=["documentation", "writing"]
    ),
    3: Task(
        id=3,
        title="Add Bedrock support",
        description="Implement AWS Bedrock model integration",
        priority=TaskPriority.MEDIUM,
        due_date=datetime.fromisoformat("2025-04-25T00:00:00"),
        status=TaskStatus.TODO,
        tags=["coding", "aws"]
    )
}
task_id_counter = 4


# --- Define tools using pydantic_tool decorator ---

@pydantic_tool(input_model=CreateTaskInput)
def create_task(title: str, description: str, priority: TaskPriority, 
               due_date: Optional[str] = None, tags: List[str] = None) -> TaskResponse:
    """Create a new task in the task management system.
    
    Args:
        title: The title of the task
        description: Detailed description of the task
        priority: Task priority level
        due_date: Optional due date in ISO format
        tags: Optional list of tags for categorization
        
    Returns:
        TaskResponse with the created task details
    """
    global task_id_counter
    
    # Process the due date if provided
    parsed_due_date = None
    if due_date:
        try:
            parsed_due_date = datetime.fromisoformat(due_date)
            # Validate that due date is in the future
            if parsed_due_date < datetime.now():
                return TaskResponse(
                    success=False,
                    message="Due date must be in the future"
                )
        except ValueError:
            return TaskResponse(
                success=False, 
                message=f"Invalid date format: {due_date}. Use ISO format (YYYY-MM-DDTHH:MM:SS)"
            )
    
    # Create the task
    new_task = Task(
        id=task_id_counter,
        title=title,
        description=description,
        priority=priority,
        due_date=parsed_due_date,
        tags=tags or []
    )
    
    # Save to "database"
    task_db[task_id_counter] = new_task
    task_id_counter += 1
    
    return TaskResponse(
        success=True,
        message="Task created successfully",
        task=new_task
    )


@pydantic_tool
def get_tasks(status: Optional[TaskStatus] = None, 
             priority: Optional[TaskPriority] = None,
             tag: Optional[str] = None) -> TaskList:
    """Get tasks from the task management system with optional filtering.
    
    Args:
        status: Optional filter by task status
        priority: Optional filter by task priority
        tag: Optional filter by tag
        
    Returns:
        List of tasks matching the filters
    """
    filtered_tasks = task_db.values()
    filter_description = []
    
    # Apply filters if provided
    if status:
        filtered_tasks = [t for t in filtered_tasks if t.status == status]
        filter_description.append(f"status={status}")
    
    if priority:
        filtered_tasks = [t for t in filtered_tasks if t.priority == priority]
        filter_description.append(f"priority={priority}")
    
    if tag:
        filtered_tasks = [t for t in filtered_tasks if tag in t.tags]
        filter_description.append(f"tag={tag}")
    
    filter_applied = " AND ".join(filter_description) if filter_description else "none"
    
    return TaskList(
        tasks=list(filtered_tasks),
        total_count=len(filtered_tasks),
        filter_applied=filter_applied
    )


@pydantic_tool
def update_task_status(task_id: int, new_status: TaskStatus) -> TaskResponse:
    """Update the status of an existing task.
    
    Args:
        task_id: The ID of the task to update
        new_status: The new status to set
        
    Returns:
        TaskResponse with the updated task details
    """
    if task_id not in task_db:
        return TaskResponse(
            success=False,
            message=f"Task with ID {task_id} not found"
        )
    
    # Update the task status
    task = task_db[task_id]
    old_status = task.status
    task.status = new_status
    
    return TaskResponse(
        success=True,
        message=f"Task status updated from {old_status} to {new_status}",
        task=task
    )


# --- Define output model for the agent ---

class TaskAnalysis(BaseModel):
    """Structured analysis of tasks."""
    total_tasks: int
    tasks_by_status: dict
    tasks_by_priority: dict
    upcoming_deadlines: List[Task]
    recommendations: List[str]
    summary: str


# --- Create the agent ---

agent = Agent(
    name="TaskManagerAgent",
    instructions="""You are a task management assistant that helps users manage their tasks.
You can create tasks, list tasks, and update task status.

When responding to the user, provide a detailed analysis of the current task state,
along with helpful recommendations based on priorities and deadlines.

Always use the structured output format to present your analysis.

Think step by step to ensure you're providing accurate recommendations.
""",
    model=OpenAIModel(model="gpt-4o"),
    tools=[create_task, get_tasks, update_task_status],
    config=AgentConfig(temperature=0.2),
    output_type=TaskAnalysis
)


async def main():
    # Example: Interact with the task manager agent
    query = "Can you show me all my current tasks and help me prioritize what I should work on next?"
    result = await agent.arun(query)
    
    # Pretty print the structured response
    print("\n=== TASK MANAGEMENT REPORT ===")
    print(f"\nSummary: {result.summary}")
    
    print(f"\nTotal Tasks: {result.total_tasks}")
    
    print("\nTask Status Breakdown:")
    for status, count in result.tasks_by_status.items():
        print(f"  {status}: {count}")
    
    print("\nPriority Breakdown:")
    for priority, count in result.tasks_by_priority.items():
        print(f"  {priority}: {count}")
    
    print("\nUpcoming Deadlines:")
    for task in result.upcoming_deadlines:
        due_date = task.due_date.strftime("%Y-%m-%d") if task.due_date else "No deadline"
        print(f"  • {task.title} ({task.priority}) - Due: {due_date}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"  {i}. {rec}")


if __name__ == "__main__":
    asyncio.run(main())