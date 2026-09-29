from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

class ObserverData(BaseModel):
    ''' The observer data structure (Data communicated during departure from an activity)'''
    simulationIteration: int
    agentID:str
    subpopulation: Optional[str] = "default"
    possibleModeSet: list[str]
    endOfDayFlag: Optional[bool] = False

    features: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ArrivalData(BaseModel):
    ''' The reward update data structure'''
    agentID: str
    reward: float
    matsimScore: float
    isTerminal: Optional[bool] = False
    accumulativeScore: Optional[float] = None
    accumulativeReward: Optional[float] = None
    
    features: Optional[Dict[str, Any]] = Field(default_factory=dict)