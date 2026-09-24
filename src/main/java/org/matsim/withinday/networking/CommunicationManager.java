package org.matsim.withinday.networking;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.google.inject.Singleton;

import org.matsim.core.controler.events.ShutdownEvent;
import org.matsim.core.controler.events.StartupEvent;
import org.matsim.core.controler.listener.ShutdownListener;
import org.matsim.core.controler.listener.StartupListener;

import java.io.File;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;


@Singleton
public class CommunicationManager implements StartupListener, ShutdownListener {
    
    // Get the base model without having to manually code it in
    private static final Logger log = LogManager.getLogger(CommunicationManager.class);

    private String host = System.getenv("HPC_PYTHON_IP");
    private final int port;
    private final HttpClient client;
    private final String baseUrl;
    private Process pythonProcess;

    public CommunicationManager(){
        
        if (host == null) this.host = "127.0.0.1";

        String portString = System.getenv("DYNAMIC_PORT");

        if (portString != null){this.port = Integer.parseInt(portString);}
        else{this.port = 5000;}
        
        this.baseUrl = "http://" + host + ":" + port;

        this.client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .followRedirects(HttpClient.Redirect.NORMAL)
                .version(HttpClient.Version.HTTP_1_1)
                .build();
    }

    @Override
    public double priority() {
        return 100.0; 
    }

    @Override
    public void notifyStartup(StartupEvent event) {
        log.info("COMMUNICATION NET: Starting Python Service...");
        launchPythonService();
    }

    @Override
    public void notifyShutdown(ShutdownEvent event) {
        log.info("COMMUNICATION NET: Shutting down Python Service...");
        if (pythonProcess != null) {
            // Add a log statement
            pythonProcess.descendants().forEach(ProcessHandle::destroy);
            pythonProcess.destroy();
        }
    }

    // --- Service Launcher Logic ---

    private void launchPythonService() {
        try {
            String projectRoot = System.getProperty("user.dir");
            File serverPath = new File(projectRoot, "src/main/python/withinday/networking");

            ProcessBuilder pb = new ProcessBuilder(
                "python3", "-m", "uvicorn", 
                "main:app", 
                "--host", this.host, 
                "--port", String.valueOf(this.port),
                "--log-level", "warning",
                "--timeout-keep-alive", "60"
            );
            
            pb.inheritIO();
            pb.directory(serverPath);

            this.pythonProcess = pb.start();

            // Ping loop to wait for readiness
            if (!waitForPython()) {
                throw new RuntimeException("COMMUNICATION NET: Python Service failed to start or respond to health check.");
            }

            log.info("COMMUNICATION NET: Python Service is LIVE.");
        } catch (Exception e) {
            throw new RuntimeException("COMMUNICATION NET: Failed to launch Python Service", e);
        }
    }

    private boolean waitForPython() {
        for (int i = 0; i < 30; i++) {
            try {
                HttpURLConnection con = (HttpURLConnection) new URL(baseUrl + "/healthz").openConnection();
                con.setRequestMethod("GET");
                con.setConnectTimeout(1000);
                if (con.getResponseCode() == 200) return true;
            } catch (Exception ignored) {}
            
            try { Thread.sleep(1000); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            log.info("Waiting for Python Service... " + (i + 1) + "s");
        }
        return false;
    }

    // Helper method to fix endpoint paths safely
    private URI formatUrl(String requestName) {
        String path = requestName.startsWith("/") ? requestName.substring(1) : requestName;
        return URI.create(baseUrl + "/" + path);
    }

    // --- Communication Network ---

    public String httpPost(String json, String requestName, long timeout) {
        try {
            URI fullUrl = formatUrl(requestName);

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

    public String httpGet(String requestName, long timeout) {
        try {
            URI fullUrl = formatUrl(requestName);

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