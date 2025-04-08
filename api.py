from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
from BaseAgent import AgentCoordinator

app = FastAPI(title="Multi-Agent Chatbot API")
coordinator = AgentCoordinator()

class Query(BaseModel):
    text: str

class Feedback(BaseModel):
    score: int

class ChatResponse(BaseModel):
    response: str
    agent_type: str
    #metrics: Optional[Dict] = None

@app.get("/")
async def root():
    return {"status": "online", "message": "Multi-Agent Chatbot System"}

@app.post("/chat", response_model=ChatResponse)
async def chat(query: Query):
    try:
        # Get response from coordinator
        response = coordinator.route_query(query.text)
        
        # Get agent type from last interaction
        agent_type = coordinator.last_interaction['agent_type']
        
        return ChatResponse(
            response=response,
            agent_type=agent_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def add_feedback(feedback: Feedback):
    try:
        coordinator.add_user_feedback(feedback.score)
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    try:
        return coordinator.metrics.get_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))