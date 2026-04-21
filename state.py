from typing import TypedDict

class AgentState(TypedDict):
    target_url: str
    user_request: str
    scraped_data: str
    final_code: str
    feedback: str
    is_approved: bool
    iteration: int
