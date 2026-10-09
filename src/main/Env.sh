#!/usr/bin/env bash
# Environment for the MATSim Within-Day Python Bridge (Linux / macOS).
#
# Usage (in every new terminal, from anywhere):
#     source env.sh
#
# Edit the "USER SETTINGS" block below, then leave the rest alone.
# NOTE: this file must be *sourced*, not executed, otherwise the variables
# are lost when the script ends.
 
# ===========================================================================
# USER SETTINGS - edit these
# ===========================================================================
 
# Path to your JDK 25. Leave empty to auto-detect from `javac` on your PATH.
# Example: "/usr/lib/jvm/java-25-openjdk-amd64"
JAVA_HOME_OVERRIDE=""
 
# Scenario folder name under scenarios/   (run `ls scenarios` to see them)
SCENARIO="<scenario>"
 
# Python bridge service Java should start:
#   <project_name>.python.networking.bridge_service.<Service>
# (project_name = a folder under src/projects)
SERVICE_CLASS_VALUE="<project_name>.python.networking.bridge_service.<Service>"
 
# ===========================================================================
# DO NOT EDIT BELOW (unless you know what you are doing)
# ===========================================================================
 
# Repo root = folder containing this file (works in bash and zsh)
if [ -n "${BASH_SOURCE[0]:-}" ]; then
    _env_src="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    eval '_env_src="${(%):-%x}"'
else
    _env_src="$0"
fi
REPO_ROOT="$(cd "$(dirname "${_env_src}")" && pwd)"
unset _env_src
 
# --- Python virtual environment -------------------------------------------
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
else
    echo "WARNING: ${REPO_ROOT}/.venv not found. Run ./setup.sh first." >&2
fi
 
# --- Java -------------------------------------------------------------------
if [ -n "${JAVA_HOME_OVERRIDE}" ]; then
    export JAVA_HOME="${JAVA_HOME_OVERRIDE}"
elif [ -z "${JAVA_HOME:-}" ] && command -v javac >/dev/null 2>&1; then
    JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")"
    export JAVA_HOME
fi
 
if [ -n "${JAVA_HOME:-}" ]; then
    export PATH="${JAVA_HOME}/bin:${PATH}"
else
    echo "WARNING: JAVA_HOME is not set and javac was not found. Install a JDK 25 or set JAVA_HOME_OVERRIDE." >&2
fi
 
# --- Project variables --------------------------------------------------------
export PYTHONPATH="${REPO_ROOT}/src/main/python:${REPO_ROOT}/src/projects"
export MATSIM_OUTPUT_BASE="${REPO_ROOT}/scenarios/${SCENARIO}/output/"
export SERVICE_CLASS="${SERVICE_CLASS_VALUE}"
 
# --- Sanity checks ------------------------------------------------------------
case "${SCENARIO}${SERVICE_CLASS_VALUE}" in
    *"<"*) echo "WARNING: placeholders like <scenario> are still in env.sh - edit the USER SETTINGS block." >&2 ;;
esac
 
echo "Environment ready:"
echo "  JAVA_HOME          = ${JAVA_HOME:-<not set>}"
echo "  python             = $(command -v python3 || command -v python)"
echo "  MATSIM_OUTPUT_BASE = ${MATSIM_OUTPUT_BASE}"
echo "  SERVICE_CLASS      = ${SERVICE_CLASS}"