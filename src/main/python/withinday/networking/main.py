import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import PlainTextResponse

from python_matsim_bridge import BaseSimulationBridgeService, load_bridge_service
from Models import ObserverData, ArrivalData

logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("matsim_bridge")

# Safely load the bridge service (can be None in baseline mode)
active_service_instance: Optional[BaseSimulationBridgeService] = load_bridge_service()

def get_bridge_service_optional() -> Optional[BaseSimulationBridgeService]:
    return active_service_instance

def get_bridge_service() -> BaseSimulationBridgeService:
    if active_service_instance is None:
        raise HTTPException(
            status_code=503, 
            detail="Bridge service is disabled or not loaded (Baseline Mode)."
        )
    return active_service_instance

@asynccontextmanager
async def lifespan(app: FastAPI):
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
    lifespan=lifespan
)

@app.get("/healthz", status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "ok", 
        "service": "matsim_simulation_bridge",
        "baseline_mode": active_service_instance is None
    }

@app.post("/session/configure", status_code=status.HTTP_200_OK)
def configure_session(
    config_data: dict, 
    service: Optional[BaseSimulationBridgeService] = Depends(get_bridge_service_optional)
):
    if service is None:
        logger.info("Baseline mode: skipping session configuration.")
        return {"status": "skipped_baseline"}
    try:
        return service.configure_session(config_data)
    except Exception as e:
        logger.error(f"Configuration error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/decision/mode_choice")
def request_decision(
    observation: ObserverData, 
    service: Optional[BaseSimulationBridgeService] = Depends(get_bridge_service_optional)
):
    if service is None:
        # Fallback default choice for baseline mode
        return {"response": "car"}
    try:
        decision = service.request_decision(observation)
        return {"response": decision}
    except Exception as e:
        logger.error(f"Decision resolution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback/score", status_code=status.HTTP_200_OK)
def process_feedback(
    feedback: ArrivalData, 
    service: Optional[BaseSimulationBridgeService] = Depends(get_bridge_service_optional)
):
    if service is None:
        return {"status": "ignored_baseline"}
    try:
        service.process_feedback(feedback)
        return {"status": "update successful"}
    except Exception as e:
        logger.error(f"Feedback execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/policy/update", status_code=status.HTTP_200_OK)
def update_policy_route(
    payload: dict, # or a custom Pydantic model
    service: Optional[BaseSimulationBridgeService] = Depends(get_bridge_service_optional)
):
    if service is None:
        return {"status": "skipped_baseline"}
    try:
        agent_id = payload.get("agentID")
        return service.call_update_policy(agent_id)
    except Exception as e:
        logger.error(f"Policy update failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/checkpoint", status_code=status.HTTP_200_OK)
def trigger_checkpoint(service: Optional[BaseSimulationBridgeService] = Depends(get_bridge_service_optional)):
    if service is None:
        return {"status": "skipped_baseline"}
    if service.checkpoint_state():
        return {"status": "checkpoint_saved"}
    raise HTTPException(status_code=500, detail="Failed to save checkpoint.")

@app.get("/session/metrics", status_code=status.HTTP_200_OK)
def get_metrics(service: Optional[BaseSimulationBridgeService] = Depends(get_bridge_service_optional)):
    if service is None:
        return {"metrics": "unavailable_in_baseline_mode"}
    return service.get_service_metrics()

@app.get("/reset/iteration_memory", status_code=status.HTTP_200_OK)
def reset_temporary_memory_route(service: Optional[BaseSimulationBridgeService] = Depends(get_bridge_service_optional)):
    if service is None:
            return {"status": "skipped_baseline"}
    try:
        return service.reset_memory_history() # or service.reset_iteration_memory() depending on your method name
    except Exception as e:
        logger.error(f"Memory reset failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))