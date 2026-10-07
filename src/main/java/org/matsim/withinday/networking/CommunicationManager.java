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
import java.net.HttpURLConnection;
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
    private final AgentSelector agentSelector;
    private String host = "127.0.0.1";
    private final int internalPort = 5000;
    private final int externalPort = 5064;
    private final HttpClient client;
    private final String baseUrl;
    private Process pythonProcess;

    /**
     * Constructs a new CommunicationManager instance, configuring the host, port, 
     * and HTTP client settings based on environment variables and defaults.
     *
     * @param agentSelector The agent selector utility used for mapping agent IDs to integer tags.
     */
    @Inject
    public CommunicationManager(AgentSelector agentSelector){
        this.agentSelector = agentSelector;
        this.baseUrl = "http://" + this.host + ":" + this.internalPort;

        this.client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .followRedirects(HttpClient.Redirect.NORMAL)
                .version(HttpClient.Version.HTTP_1_1)
                .build();
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
     * @return Endpoint URL path string (default is "/decision/mode-choice").
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
        String endPoint = "/session/checkpoint/" + iteration;
        return endPoint;}

    /**
     * Specifies the HTTP reset endpoint on the Python server.
     * Subclasses override this method to route requests to custom serving models.
     *
     * @return Endpoint URL path string (default is "/reset/iteration_memory").
     */
    public String getIterationResetEndPoint() {return "/reset/iteration_memory";}

    /**
     * Triggered at simulation startup to initiate the backend Python service.
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
        if (pythonProcess != null) {
            // Add a log statement
            pythonProcess.descendants().forEach(ProcessHandle::destroy);
            pythonProcess.destroy();
        }
    }

    /**
     * Launches the Python Uvicorn server process via ProcessBuilder and verifies readiness via health checks.
     */
    private void launchPythonService() {
        try {
            String projectRoot = System.getProperty("user.dir");
            File serverPath = new File(projectRoot, "src/main/python/withinday/networking");

            ProcessBuilder pb = new ProcessBuilder(
                "python3", "-m", "uvicorn", 
                "main:app", 
                "--host", this.host, 
                "--port", String.valueOf(this.internalPort),
                "--log-level", "warning",
                "--timeout-keep-alive", "60"
            );
            
            pb.inheritIO();
            pb.directory(serverPath);
            this.pythonProcess = pb.start();

            if (!waitForPython()) {
                throw new RuntimeException("COMMUNICATION NET: Python Service failed to start or respond to health check.");
            }

            log.info("COMMUNICATION NET: Python Service is LIVE.");
        } catch (Exception e) {
            throw new RuntimeException("COMMUNICATION NET: Failed to launch Python Service", e);
        }
    }

    /**
     * Polls the Python server health endpoint until it responds successfully or times out.
     *
     * @return {@code True} if the service responded with HTTP 200, {@code false} otherwise.
     */
    private boolean waitForPython() {
        for (int i = 0; i < 30; i++) {
            try {
                HttpURLConnection con = (HttpURLConnection) new URL(baseUrl + "/global/healthz").openConnection();
                con.setRequestMethod("GET");
                con.setConnectTimeout(1000);
                if (con.getResponseCode() == 200) return true;
            } catch (Exception ignored) {}
            
            try { Thread.sleep(1000); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            log.info("Waiting for Python Service... " + (i + 1) + "s");
        }
        return false;
    }

    /**
     * Formats and constructs the target URI for agent-specific or global endpoint paths.
     *
     * @param requestName   The endpoint path string.
     * @param agentIdString The optional agent ID string for agent-scoped requests.
     * @return The fully formed request URI.
     */
    protected URI formatUrl(String requestName, String agentIdString) {
        String path = requestName.startsWith("/") ? requestName.substring(1) : requestName;

        if (agentIdString != null){ 
            int agentTag = agentSelector.getAgentTag(agentIdString);
            return URI.create(baseUrl + "/agent/" + agentTag + "/" + path);
        }else{
            return URI.create(baseUrl + "/global/" + path);
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
    public String httpPost(String json, String requestName, long timeout){
        return httpPost(json, requestName, timeout, null);
    }

    /**
     * Sends an HTTP POST request with a JSON payload, scoped globally or to a specific agent.
     *
     * @param json          The JSON payload string.
     * @param requestName   The target endpoint path.
     * @param timeout       The request timeout in seconds.
     * @param agentIdString The optional agent ID string.
     * @return The response body string, or null if the request failed.
     */
    public String httpPost(String json, String requestName, long timeout, String agentIdString) {
        try {
            URI fullUrl = formatUrl(requestName, agentIdString);

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(fullUrl)
                    .timeout(Duration.ofSeconds(timeout))
                    .version(HttpClient.Version.HTTP_1_1)
                    .header("Content-Type", "application/json; charset=UTF-8")
                    .POST(HttpRequest.BodyPublishers.ofString(json, StandardCharsets.UTF_8))
                    .build();

            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() == 200) {

                String responseBody = response.body().trim();

                return responseBody;
            } else {
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
    public String httpGet(String requestName, long timeout){
        return httpGet(requestName, timeout, null);
    }

    /**
     * Sends an HTTP GET request, scoped globally or to a specific agent.
     *
     * @param requestName   The target endpoint path.
     * @param timeout       The request timeout in seconds.
     * @param agentIdString The optional agent ID string.
     * @return The response body string, or null if the request failed.
     */
    public String httpGet(String requestName, long timeout, String agentIdString) {
        try {
            URI fullUrl = formatUrl(requestName, agentIdString);

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(fullUrl)
                    .timeout(Duration.ofSeconds(timeout))
                    .GET()
                    .build();

            String response = client.send(request, HttpResponse.BodyHandlers.ofString()).body();
            //JsonNode node = mapper.readTree(response);
            return response;
        } catch (Exception e) {
            log.error("Error in Get: " + e.getMessage());
            return null;
        }
    }
}