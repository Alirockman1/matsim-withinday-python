package org.matsim.withinday.networking;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.google.inject.Inject;
import com.google.inject.Singleton;

import org.matsim.core.controler.events.ShutdownEvent;
import org.matsim.core.controler.events.StartupEvent;
import org.matsim.withinday.core.AgentSelector;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.net.StandardProtocolFamily;
import java.net.URI;
import java.net.UnixDomainSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.SocketChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * High-performance Unix Domain Socket Communication Manager.
 * <p>
 * Maintains a persistent, long-lived {@link SocketChannel} across simulation steps 
 * and utilizes strict HTTP Content-Length framing to eliminate connection 
 * open/close OS kernel overhead.
 */
@Singleton
public class UnixSocketCommunicationManager extends CommunicationManager {

    private static final Logger log = LogManager.getLogger(UnixSocketCommunicationManager.class);
    private final AgentSelector agentSelector;
    private final Path socketPath;
    private Process pythonProcess;
    
    private SocketChannel persistentChannel;

    private ByteBuffer writeBuffer = ByteBuffer.allocateDirect(32768);
    private final ByteBuffer readBuffer = ByteBuffer.allocateDirect(8192);

    /**
     * Initializes the Unix Domain Socket communication manager and guarantees 
     * the creation of the target directory for the socket file.
     */
    @Inject
    public UnixSocketCommunicationManager(AgentSelector agentSelector) {
        super(agentSelector);
        this.agentSelector = agentSelector;
        
        this.socketPath = Path.of("/app/temp", "matsim_rl.sock");

        try {
            if (this.socketPath.getParent() != null) {
                Files.createDirectories(this.socketPath.getParent());
            }
        } catch (IOException e) {
            log.error("COMMUNICATION NET: Could not create socket directory: " + e.getMessage());
        }
    }

    /**
     * MATSim startup listener callback. Automatically launches the Python Uvicorn 
     * process over the Unix Domain Socket when the controller starts.
     *
     * @param event The MATSim controller startup event object.
     */
    @Override
    public void notifyStartup(StartupEvent event) {
        log.info("COMMUNICATION NET: Starting Python Service over Unix Domain Socket...");
        launchPythonService();
    }

    /**
     * MATSim shutdown listener callback. Safely terminates the persistent socket channel 
     * and stops the underlying Python background process upon simulation completion.
     *
     * @param event The MATSim controller shutdown event object.
     */
    @Override
    public void notifyShutdown(ShutdownEvent event) {
        log.info("COMMUNICATION NET: Closing persistent socket and stopping Python Service...");
        closeChannel();
        if (pythonProcess != null) {
            pythonProcess.descendants().forEach(ProcessHandle::destroy);
            pythonProcess.destroy();
        }
        try {
            Files.deleteIfExists(socketPath);
        } catch (IOException ignored) {}
    }

    /**
     * Retrieves the active persistent socket channel, automatically re-establishing 
     * the connection or launching the Python service if the socket file is missing.
     *
     * @return The active, connected Unix Domain {@link SocketChannel}.
     * @throws IOException If the socket channel fails to open or connect.
     */
    private synchronized SocketChannel getChannel() throws IOException {
        if (!Files.exists(socketPath)) {
            log.warn("COMMUNICATION NET: Socket file missing! Auto-launching Python service...");
            launchPythonService();
        }

        if (persistentChannel == null || !persistentChannel.isOpen() || !persistentChannel.isConnected()) {
            UnixDomainSocketAddress address = UnixDomainSocketAddress.of(socketPath);
            persistentChannel = SocketChannel.open(StandardProtocolFamily.UNIX);
            persistentChannel.connect(address);
        }
        return persistentChannel;
    }

    /**
     * Safely closes and resets the persistent socket channel upon stream errors.
     */
    private synchronized void closeChannel() {
        if (persistentChannel != null) {
            try {
                persistentChannel.close();
            } catch (IOException ignored) {}
            persistentChannel = null;
        }
    }

    /**
     * Spawns the external Python process running Uvicorn listening on the Unix socket path.
     */
    private void launchPythonService() {
        try {
            Files.deleteIfExists(socketPath);

            String projectRoot = System.getProperty("user.dir");
            File serverPath = new File(projectRoot, "src/main/python/withinday/networking");

            ProcessBuilder pb = new ProcessBuilder(
                "python3", "-m", "uvicorn", 
                "main:app", 
                "--uds", socketPath.toString(),
                "--log-level", "warning",
                "--timeout-keep-alive", "60"
            );
            
            pb.inheritIO();
            pb.directory(serverPath);

            this.pythonProcess = pb.start();

            if (!waitForPython()) {
                throw new RuntimeException("COMMUNICATION NET: Python Service failed to respond on Unix Socket.");
            }

            log.info("COMMUNICATION NET: Python Service is LIVE on Unix Socket: " + socketPath);
        } catch (Exception e) {
            throw new RuntimeException("COMMUNICATION NET: Failed to launch Python Service", e);
        }
    }

    /**
     * Repeatedly pings the Python health endpoint until the service becomes active.
     *
     * @return {@code true} if the Python service responded successfully; {@code false} otherwise.
     */
    private boolean waitForPython() {
        for (int i = 0; i < 30; i++) {
            if (Files.exists(socketPath)) {
                String response = httpGet("/healthz", 1, null);
                if (response != null && !response.isEmpty()) {
                    return true;
                }
            }
            try { Thread.sleep(1000); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            log.info("Waiting for Python Service socket... " + (i + 1) + "s");
        }
        return false;
    }

    /**
     * Transmits a JSON text payload over the persistent socket channel using HTTP POST.
     *
     * @param json          The JSON formatted string representation of the payload.
     * @param requestName   The target endpoint URI path (e.g., "/observe").
     * @param timeout       Maximum socket operation timeout in milliseconds.
     * @param agentIdString The ID of the agent requesting the communication end point
     * @return Raw string response body returned by the Python endpoint, or null on failure.
     */
    @Override
    public synchronized String httpPost(String json, String requestName, long timeout, String agentIdString) {
        return sendPayload(json.getBytes(StandardCharsets.UTF_8), "application/json; charset=UTF-8", requestName, agentIdString);
    }

    /**
     * Transmits a binary MessagePack payload over the persistent socket channel using HTTP POST.
     *
     * @param binaryPayload Raw MessagePack encoded byte array.
     * @param requestName   The target endpoint URI path (e.g., "/observe").
     * @return Raw string response body returned by the Python endpoint, or null on failure.
     */
    public synchronized String httpPostBytes(byte[] binaryPayload, String requestName, String agentIdString) {
        return sendPayload(binaryPayload, "application/x-msgpack", requestName, agentIdString);
    }

    /**
     * Performs an HTTP GET request over the persistent socket channel.
     *
     * @param requestName The target endpoint URI path (e.g., "/healthz").
     * @param timeout     Maximum socket operation timeout in milliseconds.
     * @return Raw string response body returned by the Python endpoint, or null on failure.
     */
    @Override
    public synchronized String httpGet(String requestName, long timeout, String agentIdString) {
        URI URI = formatUrl(requestName, agentIdString);

        try {
            SocketChannel channel = getChannel();

            String rawHttpRequest = "GET " + URI.toString() + " HTTP/1.1\r\n" +
                    "Host: localhost\r\n" +
                    "Connection: keep-alive\r\n\r\n";

            writeBuffer.clear();
            writeBuffer.put(rawHttpRequest.getBytes(StandardCharsets.UTF_8));
            writeBuffer.flip();

            while (writeBuffer.hasRemaining()) {
                channel.write(writeBuffer);
            }

            return readHttpResponse(channel);

        } catch (Exception e) {
            log.warn("Error in persistent GET: " + e.getMessage() + ". Resetting channel...");
            closeChannel();
            return null;
        }
    }

    /**
     * Internal helper to transmit byte payloads over the persistent channel using HTTP keep-alive framing.
     *
     * @param bodyBytes   Raw request body byte array.
     * @param contentType The MIME content-type header string.
     * @param requestName Target endpoint path.
     * @return Raw decoded string response body.
     */
    private String sendPayload(byte[] bodyBytes, String contentType, String requestName, String agentIdString) {
        System.out.println("INFO UnixSocketCommunicationManager: The agent id is {" + agentIdString + "}.");
        URI URI = formatUrl(requestName, agentIdString);

        try {
            SocketChannel channel = getChannel();

            System.out.println("INFO UnixSocketCommunicationManager: The URI is {" + URI + "}.");

            String rawHttpRequest = "POST " + URI.toString() + " HTTP/1.1\r\n" +
                    "Host: localhost\r\n" +
                    "Content-Type: " + contentType + "\r\n" +
                    "Content-Length: " + bodyBytes.length + "\r\n" +
                    "Connection: keep-alive\r\n\r\n";

            byte[] headerBytes = rawHttpRequest.getBytes(StandardCharsets.UTF_8);
            int totalLength = headerBytes.length + bodyBytes.length;

            if (writeBuffer.capacity() < totalLength) {
                writeBuffer = ByteBuffer.allocateDirect(totalLength + 4096);
            }

            writeBuffer.clear();
            writeBuffer.put(headerBytes);
            writeBuffer.put(bodyBytes);
            writeBuffer.flip();

            while (writeBuffer.hasRemaining()) {
                channel.write(writeBuffer);
            }

            return readHttpResponse(channel);

        } catch (Exception e) {
            log.warn("Socket persistent error during POST: " + e.getMessage() + ". Resetting channel...");
            closeChannel();
            return null;
        }
    }

    /**
     * Parses exactly one complete HTTP response frame from the persistent stream using 
     * the HTTP Content-Length header to prevent buffer pollution across agent steps.
     *
     * @param channel The active persistent {@link SocketChannel} stream.
     * @return Decoded UTF-8 HTTP body content.
     * @throws IOException If the stream is closed prematurely or Content-Length is missing.
     */
    private String readHttpResponse(SocketChannel channel) throws IOException {
        ByteArrayOutputStream headerStream = new ByteArrayOutputStream();
        int contentLength = -1;
        boolean headerEnded = false;
        
        while (!headerEnded) {
            readBuffer.clear();
            int bytesRead = channel.read(readBuffer);
            if (bytesRead <= 0) {
                throw new IOException("Socket closed or unreadable by peer.");
            }
            
            readBuffer.flip();
            while (readBuffer.hasRemaining()) {
                byte b = readBuffer.get();
                headerStream.write(b);
                
                byte[] currentBytes = headerStream.toByteArray();
                int len = currentBytes.length;
                if (len >= 4 && 
                    currentBytes[len - 4] == '\r' && currentBytes[len - 3] == '\n' &&
                    currentBytes[len - 2] == '\r' && currentBytes[len - 1] == '\n') {
                    
                    headerEnded = true;
                    
                    String headersStr = new String(currentBytes, StandardCharsets.UTF_8);
                    for (String line : headersStr.split("\r\n")) {
                        if (line.toLowerCase().startsWith("content-length:")) {
                            contentLength = Integer.parseInt(line.split(":")[1].trim());
                            break;
                        }
                    }
                    break;
                }
            }
        }

        if (contentLength < 0) {
            throw new IOException("No Content-Length header present in HTTP response.");
        }

        ByteArrayOutputStream bodyStream = new ByteArrayOutputStream();
        
        while (readBuffer.hasRemaining() && bodyStream.size() < contentLength) {
            bodyStream.write(readBuffer.get());
        }

        while (bodyStream.size() < contentLength) {
            readBuffer.clear();
            int bytesRead = channel.read(readBuffer);
            if (bytesRead <= 0) break;
            
            readBuffer.flip();
            while (readBuffer.hasRemaining() && bodyStream.size() < contentLength) {
                bodyStream.write(readBuffer.get());
            }
        }

        return bodyStream.toString(StandardCharsets.UTF_8);
    }
}