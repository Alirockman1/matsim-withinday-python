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
    ''' The observer data structure (Data communicated during arrival at an activity)'''
    agentID: str
    travelTimeSeconds: float
    numberOfTransfers: int
    distance: float
    reward: float
    matsimScore: float
    isTerminal: Optional[bool] = False
    accumulativeScore: Optional[float] = None
    accumulativeReward: Optional[float] = None
    
    features: Optional[Dict[str, Any]] = Field(default_factory=dict)