package org.matsim.withinday.networking;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.google.inject.Inject;
import com.google.inject.Singleton;

import org.matsim.core.controler.events.ShutdownEvent;
import org.matsim.core.controler.events.StartupEvent;
import org.matsim.core.controler.listener.ShutdownListener;
import org.matsim.core.controler.listener.StartupListener;
import org.matsim.withinday.core.AgentSelector;

import java.io.File;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.URI;
import java.net.URL;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;

/**
 * CommunicationManager acts as a singleton MATSim controler listener responsible for 
 * launching, managing health checks for, and gracefully shutting down the backend 
 * Python Uvicorn service. It also provides robust HTTP client methods (GET/POST) 
 * to handle agent-level and global communication between the Java simulation and Python models.
 */
@Singleton
public class CommunicationManager implements StartupListener, ShutdownListener {
    private static final Logger log = LogManager.getLogger(CommunicationManager.class);

    private static final int HEALTH_CHECK_SECONDS = 30;
    private static final String HOST = "127.0.0.1";

    // --- JVM-wide state: shared by every instance, constant for the lifetime of the JVM ---
    private static final Object LOCK = new Object();
    private static int sharedPort = -1;
    private static Process sharedProcess;

    private final AgentSelector agentSelector;
    private final HttpClient client;

    /**
     * Constructs a new CommunicationManager instance. The port is resolved once per JVM:
     * from the INTERNAL_PORT environment variable if set, otherwise a free port from the OS.
     *
     * @param agentSelector The agent selector utility used for mapping agent IDs to integer tags.
     */
    @Inject
    public CommunicationManager(AgentSelector agentSelector) {
        this.agentSelector = agentSelector;

        synchronized (LOCK) {
            if (sharedPort < 0) {
                String portEnvironment = System.getenv("INTERNAL_PORT");
                if (portEnvironment != null && !portEnvironment.isBlank()) {
                    sharedPort = Integer.parseInt(portEnvironment.trim());
                    log.info("COMMUNICATION NET: Port set from INTERNAL_PORT: " + sharedPort);
                } else {
                    sharedPort = findFreePort();
                    log.info("COMMUNICATION NET: Port dynamically assigned (once per JVM): " + sharedPort);
                }
            }
        }

        log.info("COMMUNICATION NET: Instance " + System.identityHashCode(this)
                + " (" + getClass().getSimpleName() + ") uses port " + sharedPort);

        this.client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .followRedirects(HttpClient.Redirect.NORMAL)
                .version(HttpClient.Version.HTTP_1_1)
                .build();
    }

    /**
     * @return The base URL of the Python service, always derived from the current shared port.
     */
    private static String baseUrl() {
        return "http://" + HOST + ":" + sharedPort;
    }

    /**
     * Asks the OS for a currently free port on the loopback interface.
     * There is a small window between closing the probe socket and the Python server
     * binding the port; if it is lost, startup fails with a clear error (no port switching).
     *
     * @return A free port number.
     */
    private static int findFreePort() {
        try (ServerSocket socket = new ServerSocket()) {
            socket.setReuseAddress(true);
            socket.bind(new InetSocketAddress(InetAddress.getByName(HOST), 0));
            return socket.getLocalPort();
        } catch (IOException e) {
            throw new RuntimeException("COMMUNICATION NET: No free port available", e);
        }
    }

    /**
     * Defines the execution priority for this listener during startup and shutdown events.
     *
     * @return The priority value (100.0).
     */
    @Override
    public double priority() {
        return 100.0;
    }

    /**
     * Specifies the HTTP decision endpoint on the Python server.
     * Subclasses override this method to route requests to custom serving models.
     *
     * @return Endpoint URL path string (default is "/action/mode_choice").
     */
    public String getDecisionEndpoint() {return "/action/mode_choice";}

    /**
     * Specifies the HTTP update score endpoint on the Python server.
     * Subclasses override this method to route requests to custom serving models.
     *
     * @return Endpoint URL path string (default is "/feedback/score").
     */
    public String getRewardFeedbackEndpoint() {return "/feedback/score";}

    /**
     * Specifies the HTTP checkpoint endpoint to save the model during training.
     * Subclasses can override this method to route requests to custom serving models.
     *
     * @param iteration The current simulation iteration number.
     * @return Endpoint URL path string.
     */
    public String getSaveEndPoint(int iteration) {
        return "/session/checkpoint/" + iteration;
    }

    /**
     * Specifies the HTTP reset endpoint on the Python server.
     * Subclasses override this method to route requests to custom serving models.
     *
     * @return Endpoint URL path string (default is "/reset/iteration_memory").
     */
    public String getIterationResetEndPoint() {return "/reset/iteration_memory";}

    /**
     * Triggered at simulation startup to initiate the backend Python service.
     * If the service is already running in this JVM, nothing is launched.
     *
     * @param event The MATSim startup event.
     */
    @Override
    public void notifyStartup(StartupEvent event) {
        log.info("COMMUNICATION NET: Starting Python Service...");
        launchPythonService();
    }

    /**
     * Triggered at simulation shutdown to safely terminate the backend Python process and its descendants.
     *
     * @param event The MATSim shutdown event.
     */
    @Override
    public void notifyShutdown(ShutdownEvent event) {
        log.info("COMMUNICATION NET: Shutting down Python Service...");
        synchronized (LOCK) {
            stopPythonProcess();
        }
    }

    /**
     * Terminates the shared Python process and all of its descendants, if running.
     * Must be called while holding LOCK.
     */
    private static void stopPythonProcess() {
        if (sharedProcess != null) {
            sharedProcess.descendants().forEach(ProcessHandle::destroy);
            sharedProcess.destroy();
            sharedProcess = null;
        }
    }

    /**
     * Launches the Python Uvicorn server process once per JVM and verifies readiness via health checks.
     * The port is fixed for the lifetime of the JVM: it is never changed after being chosen.
     * If startup fails (e.g. the port was taken), this throws instead of switching ports.
     */
    private void launchPythonService() {
        synchronized (LOCK) {
            if (sharedProcess != null && sharedProcess.isAlive()) {
                log.info("COMMUNICATION NET: Python Service already running on port " + sharedPort + ", reusing it.");
                return;
            }

            try {
                File serverPath = new File(System.getProperty("user.dir"), "src/main/python/withinday/networking");
                log.info("COMMUNICATION NET: Launching Python Service on port " + sharedPort + " (dir=" + serverPath + ")");

                ProcessBuilder pb = new ProcessBuilder(
                        "python3", "-m", "uvicorn",
                        "main:app",
                        "--host", HOST,
                        "--port", String.valueOf(sharedPort),
                        "--log-level", "info", // "warning" hides the bind address; switch back if too noisy
                        "--timeout-keep-alive", "60"
                );

                pb.inheritIO();
                pb.directory(serverPath);
                sharedProcess = pb.start();

                if (!waitForPython()) {
                    stopPythonProcess();
                    throw new RuntimeException("COMMUNICATION NET: Python Service failed to start or respond to health check on port " + sharedPort);
                }

                log.info("COMMUNICATION NET: Python Service is LIVE on " + baseUrl());
            } catch (IOException e) {
                stopPythonProcess();
                throw new RuntimeException("COMMUNICATION NET: Failed to launch Python Service", e);
            }
        }
    }

    /**
     * Polls the Python server health endpoint until it responds successfully, the process dies, or it times out.
     *
     * @return {@code true} if the service responded with HTTP 200, {@code false} otherwise.
     */
    private boolean waitForPython() {
        for (int i = 0; i < HEALTH_CHECK_SECONDS; i++) {
            // Fail fast if uvicorn already exited (bad import, port in use, ...)
            if (sharedProcess != null && !sharedProcess.isAlive()) {
                log.error("COMMUNICATION NET: Python process exited early with code " + sharedProcess.exitValue());
                return false;
            }

            try {
                HttpURLConnection con = (HttpURLConnection) new URL(baseUrl() + "/global/healthz").openConnection();
                con.setRequestMethod("GET");
                con.setConnectTimeout(1000);
                con.setReadTimeout(1000);
                int code = con.getResponseCode();
                con.disconnect();
                if (code == 200) return true;
                log.warn("COMMUNICATION NET: Health check returned HTTP " + code);
            } catch (Exception e) {
                log.debug("COMMUNICATION NET: Health check attempt " + (i + 1) + " failed: " + e.getMessage());
            }

            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return false;
            }
            log.info("Waiting for Python Service... " + (i + 1) + "s");
        }
        return false;
    }

    /**
     * Formats and constructs the target URI for agent-specific or global endpoint paths.
     *
     * @param requestName   The endpoint path string.
     * @param agentIdString The optional agent ID string for agent-scoped requests.
     * @param arg           If non-null (and agent-scoped), routes to the agent's "/test" variant.
     * @return The fully formed request URI.
     */
    protected URI formatUrl(String requestName, String agentIdString, Object arg) {
        String path = requestName.startsWith("/") ? requestName.substring(1) : requestName;

        if (agentIdString != null) {
            int agentTag = agentSelector.getAgentTag(agentIdString);

            if (arg != null) {
                return URI.create(baseUrl() + "/agent/" + agentTag + "/" + path + "/test");
            } else {
                return URI.create(baseUrl() + "/agent/" + agentTag + "/" + path);
            }

        } else {
            return URI.create(baseUrl() + "/global/" + path);
        }
    }

    // --- Communication Network --- //

    /**
     * Sends a global HTTP POST request with a JSON payload.
     *
     * @param json        The JSON payload string.
     * @param requestName The target endpoint path.
     * @param timeout     The request timeout in seconds.
     * @return The response body string, or null if the request failed.
     */
    public String httpPost(String json, String requestName, long timeout) {
        return httpPost(json, requestName, timeout, null, null);
    }

    /**
     * Sends an HTTP POST request with a JSON payload, for specific agent for training.
     *
     * @param json          The JSON payload string.
     * @param requestName   The target endpoint path.
     * @param timeout       The request timeout in seconds.
     * @param agentIdString The agent ID string.
     * @return The response body string, or null if the request failed.
     */
    public String httpPost(String json, String requestName, long timeout, String agentIdString) {
        return httpPost(json, requestName, timeout, agentIdString, null);
    }

    /**
     * Sends an HTTP POST request with a JSON payload, for a specific agent for testing.
     *
     * @param json          The JSON payload string.
     * @param requestName   The target endpoint path.
     * @param timeout       The request timeout in seconds.
     * @param agentIdString The optional agent ID string.
     * @param arg           If non-null, routes to the agent's "/test" endpoint.
     * @return The response body string, or null if the request failed.
     */
    public String httpPost(String json, String requestName, long timeout, String agentIdString, Object arg) {
        try {
            URI fullUrl = formatUrl(requestName, agentIdString, arg);

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(fullUrl)
                    .timeout(Duration.ofSeconds(timeout))
                    .version(HttpClient.Version.HTTP_1_1)
                    .header("Content-Type", "application/json; charset=UTF-8")
                    .POST(HttpRequest.BodyPublishers.ofString(json, StandardCharsets.UTF_8))
                    .build();

            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() == 200) {
                return response.body().trim();
            } else {
                log.warn("POST " + fullUrl + " returned HTTP " + response.statusCode());
                return null;
            }

        } catch (Exception e) {
            log.error("Network transactional failure in POST: " + e.getMessage());
            return null;
        }
    }

    /**
     * Sends a global HTTP GET request.
     *
     * @param requestName The target endpoint path.
     * @param timeout     The request timeout in seconds.
     * @return The response body string, or null if the request failed.
     */
    public String httpGet(String requestName, long timeout) {
        return httpGet(requestName, timeout, null, null);
    }

    /**
     * Sends an HTTP GET request scoped to a specific agent.
     *
     * @param requestName   The target endpoint path.
     * @param timeout       The request timeout in seconds.
     * @param agentIdString The agent ID string.
     * @return The response body string, or null if the request failed.
     */
    public String httpGet(String requestName, long timeout, String agentIdString) {
        return httpGet(requestName, timeout, agentIdString, null);
    }

    /**
     * Sends an HTTP GET request, scoped globally or to a specific agent.
     *
     * @param requestName   The target endpoint path.
     * @param timeout       The request timeout in seconds.
     * @param agentIdString The optional agent ID string.
     * @param datasetType   If non-null (and agent-scoped), routes to the agent's "/test" endpoint.
     * @return The response body string, or null if the request failed.
     */
    public String httpGet(String requestName, long timeout, String agentIdString, String datasetType) {
        try {
            URI fullUrl = formatUrl(requestName, agentIdString, datasetType);

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(fullUrl)
                    .timeout(Duration.ofSeconds(timeout))
                    .GET()
                    .build();

            return client.send(request, HttpResponse.BodyHandlers.ofString()).body();
        } catch (Exception e) {
            log.error("Error in Get: " + e.getMessage());
            return null;
        }
    }
}