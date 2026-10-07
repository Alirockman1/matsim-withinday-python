import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.responses import PlainTextResponse

from python_matsim_bridge import BaseSimulationBridgeService, load_bridge_service
from Models import ObserverData, ArrivalData

# Configure module-level logger
logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("matsim_bridge")

# Safely load the bridge service (can be None in baseline mode)
active_service_instance: Optional[BaseSimulationBridgeService] = load_bridge_service()

def get_bridge_service_optional() -> Optional[BaseSimulationBridgeService]:
    """Dependency provider that returns the optional active bridge service instance."""

    return active_service_instance

def get_bridge_service() -> BaseSimulationBridgeService:
    """Dependency provider that enforces an active bridge service, raising 503 if in baseline mode."""

    if active_service_instance is None:
        raise HTTPException(
            status_code=503, 
            detail="Bridge service is disabled or not loaded (Baseline Mode)."
        )
    return active_service_instance

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown lifecycle events, 
    logging initialization modes and persisting state checkpoints upon shutdown."""

    logger.info("Starting MATSim Generic Simulation Bridge API...")
    if active_service_instance:
        logger.info("Bridge service active.")
    else:
        logger.info("Running in Baseline Mode: No bridge service initialized.")
    
    yield
    
    if active_service_instance:
        logger.info("Persisting session checkpoints before shutdown...")
        active_service_instance.checkpoint_state()


app = FastAPI(
    title="MATSim Simulation Bridge API",
    lifespan=lifespan)

# ==========================================
# Global Network & Session Endpoints
# ==========================================

@app.get("/global/healthz", status_code=status.HTTP_200_OK)
def health_check():
    """Performs a health check on the bridge service and reports operational status."""

    return {
        "status": "ok", 
        "service": "matsim_simulation_bridge",
        "baseline_mode": active_service_instance is None
    }

@app.post("/global/session/configure", status_code=status.HTTP_200_OK)
def configure_session(config_data: dict, 
    service: Optional[BaseSimulationBridgeService] = Depends(get_bridge_service_optional)):
    """Configures active simulation session parameters."""

    if service is None:
        logger.info("Baseline mode: skipping session configuration.")
        return {"status": "skipped_baseline"}
    try:
        return service.configure_session(config_data)
    except Exception as e:
        logger.error(f"Configuration error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/global/session/checkpoint/{iteration}", status_code=status.HTTP_200_OK)
async def trigger_checkpoint(iteration: int,
    service: Optional[BaseSimulationBridgeService] = Depends(get_bridge_service_optional)):
    """Persists model checkpoints for the specified simulation iteration."""

    if service is None:
        return {"status": "skipped_baseline"}
    try:
        print(iteration)

        if iteration is None:
            raise HTTPException(status_code=400, detail="Missing 'iteration' in payload.")
            
        # Pass the extracted integer to your checkpoint service
        if service.checkpoint_state(iteration):
            return {"status": "checkpoint_saved", "iteration": iteration}
            
        raise HTTPException(status_code=500, detail="Failed to save checkpoint.")
    except Exception as e:
        logger.error(f"Checkpoint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/global/session/metrics", status_code=status.HTTP_200_OK)
def get_metrics(service: Optional[BaseSimulationBridgeService] = Depends(get_bridge_service_optional)):
    """Retrieves current session and performance metrics."""

    if service is None:
        return {"metrics": "unavailable_in_baseline_mode"}
    return service.get_service_metrics()

@app.post("/global/reset/iteration_memory", status_code=status.HTTP_200_OK)
def reset_temporary_memory_route(service: Optional[BaseSimulationBridgeService] = Depends(get_bridge_service_optional)):
    """Resets temporary episode or iteration memory buffers."""

    if service is None:
            return {"status": "skipped_baseline"}
    try:
        return service.reset_episode()
    except Exception as e:
        logger.error(f"Memory reset failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# Agent-Specific Endpoints
# ==========================================   

@app.post("/agent/{agent_tag}/action/mode_choice", status_code=status.HTTP_200_OK)
def request_decision(agent_tag: int,
    observation: ObserverData, 
    service: Optional[BaseSimulationBridgeService] = Depends(get_bridge_service_optional)):
    """Requests a transport mode choice decision for a specific agent based on current observations."""

    if service is None:
        return {"response": "pedestrian"}
    try:
        decision = service.request_decision(agent_tag, observation)
        return {"response": decision}
    except Exception as e:
        logger.error(f"Decision resolution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/{agent_tag}/feedback/score", status_code=status.HTTP_200_OK)
def process_feedback(agent_tag: int,
    feedback: ArrivalData, 
    service: Optional[BaseSimulationBridgeService] = Depends(get_bridge_service_optional)):
    """Processes arrival feedback and reward scoring for a specific agent."""
    
    if service is None:
        return {"status": "ignored_baseline"}
    try:
        service.process_feedback(agent_tag, feedback)
        return {"status": "update successful"}
    except Exception as e:
        logger.error(f"Feedback execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/{agent_tag}/policy/update", status_code=status.HTTP_200_OK)
def update_policy_route(agent_tag: int,
    service: Optional[BaseSimulationBridgeService] = Depends(get_bridge_service_optional)):
    """Triggers policy updates for a specific agent."""

    if service is None:
        return {"status": "skipped_baseline"}
    try:
        return service.call_update_policy(agent_tag)
    except Exception as e:
        logger.error(f"Policy update failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

