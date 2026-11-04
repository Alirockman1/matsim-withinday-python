package berlin.java.prepare;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * <h2>PrepareMiniBerlin</h2>
 * <p>
 * One-shot preparation of a small Open Berlin scenario for within-day development runs. The
 * supporting files (network, vehicles, transit, facilities, counts) are mirrored locally, and the
 * population is reduced to a handful of agents.
 * </p>
 * <p>
 * The population is filtered on the raw XML, person block by person block, instead of being read
 * into a {@code Population} and written back out. That keeps every person attribute Open Berlin
 * relies on - {@code modeTasteVariations}, the {@code vehicles} map, {@code income} - byte
 * identical, without needing an attribute converter for each of them.
 * </p>
 * <p>
 * Usage (from the repository root):
 * </p>
 * <pre>
 * java -cp a7-rl-testbed-0.0.1-SNAPSHOT.jar berlin.java.prepare.PrepareMiniBerlin \
 *      --num-agents 200 --keep-ids bb_00007dd9
 * </pre>
 */
public final class PrepareMiniBerlin {

    private static final String DEFAULT_SOURCE =
            "https://svn.vsp.tu-berlin.de/repos/public-svn/matsim/scenarios/countries/de/berlin/berlin-v7.1/calibrated-1pct/output";
    private static final String DEFAULT_PREFIX = "berlin-v7.1-1pct.";
    private static final String DEFAULT_OUTPUT_DIR = "scenarios/berlin-v7.1.1/input";
    private static final int DEFAULT_NUM_AGENTS = 200;

    /** Supporting files that are mirrored as they are. */
    private static final List<String> SUPPORTING_FILES = List.of(
            "output_network.xml.gz",
            "output_vehicles.xml.gz",
            "output_transitSchedule.xml.gz",
            "output_transitVehicles.xml.gz",
            "output_facilities.xml.gz",
            "output_counts.xml.gz");

    private static final Pattern PERSON_ID = Pattern.compile("<person[^>]*\\sid=\"([^\"]+)\"");

    private PrepareMiniBerlin() {
    }

    public static void main(String[] args) throws IOException {
        String source = argument(args, "--source", DEFAULT_SOURCE);
        String prefix = argument(args, "--prefix", DEFAULT_PREFIX);
        Path outputDir = Paths.get(argument(args, "--output-dir", DEFAULT_OUTPUT_DIR));
        int numAgents = Integer.parseInt(argument(args, "--num-agents", String.valueOf(DEFAULT_NUM_AGENTS)));
        Set<String> keepIds = new LinkedHashSet<>();
        String keepIdsArgument = argument(args, "--keep-ids", "");
        if (!keepIdsArgument.isBlank()) {
            keepIds.addAll(Arrays.asList(keepIdsArgument.split("\\s*,\\s*")));
        }
        boolean skipDownload = Arrays.asList(args).contains("--skip-download");

        Path rawDir = outputDir.resolve("raw");
        Files.createDirectories(rawDir);

        if (!skipDownload) {
            for (String fileName : SUPPORTING_FILES) {
                mirror(source + "/" + prefix + fileName, rawDir.resolve(prefix + fileName));
            }
        }

        Path plansFile = outputDir.resolve("berlin-mini.plans.xml.gz");
        List<String> keptIds = filterPopulation(source + "/" + prefix + "output_plans.xml.gz", plansFile, numAgents, keepIds);

        System.out.println();
        System.out.println("Mini Berlin written to " + outputDir.toAbsolutePath());
        System.out.println("  population: " + plansFile.getFileName() + " with " + keptIds.size() + " persons");
        System.out.println("  supporting files: " + rawDir.getFileName() + "/");
        System.out.println();
        System.out.println("Candidates for withinday.trainAgentIDList:");
        System.out.println("  " + String.join(",", keptIds.subList(0, Math.min(10, keptIds.size()))));
    }

    /**
     * Downloads a file unless it is already present.
     *
     * @param url    Source URL or local path.
     * @param target Local target path.
     */
    private static void mirror(String url, Path target) throws IOException {
        if (Files.exists(target)) {
            System.out.println("Already present, skipping: " + target);
            return;
        }

        System.out.println("Downloading " + url);
        try (InputStream in = open(url)) {
            Files.copy(in, target, StandardCopyOption.REPLACE_EXISTING);
        }
        System.out.println("  -> " + target + " (" + Files.size(target) / 1024 / 1024 + " MB)");
    }

    /**
     * Copies the first {@code numAgents} person blocks plus all explicitly requested ones into a
     * new plans file.
     *
     * @param plansUrl Source plans file, gzipped MATSim XML.
     * @param target   Target plans file, gzipped.
     * @param numAgents Number of persons to keep in file order.
     * @param keepIds  Person ids that are always kept, no matter where they appear.
     * @return The ids that ended up in the target file.
     */
    private static List<String> filterPopulation(String plansUrl, Path target, int numAgents, Set<String> keepIds)
            throws IOException {
        System.out.println("Filtering population from " + plansUrl);

        List<String> keptIds = new ArrayList<>();
        Set<String> wantedIds = new HashSet<>(keepIds);

        try (BufferedReader reader = new BufferedReader(
                     new InputStreamReader(new GZIPInputStream(open(plansUrl)), StandardCharsets.UTF_8));
             BufferedWriter writer = new BufferedWriter(
                     new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(target)), StandardCharsets.UTF_8))) {

            List<String> block = new ArrayList<>();
            boolean insidePerson = false;
            boolean headerDone = false;
            String currentId = null;

            String line;
            while ((line = reader.readLine()) != null) {
                String trimmed = line.stripLeading();

                if (!insidePerson && trimmed.startsWith("<person ")) {
                    headerDone = true;
                    insidePerson = true;
                    block.clear();
                    currentId = personId(line);
                }

                if (!insidePerson) {
                    // Header before the first person, or the trailing </population>.
                    if (!headerDone || trimmed.startsWith("</population>")) {
                        writer.write(line);
                        writer.newLine();
                    }
                    continue;
                }

                block.add(line);
                boolean personDone = trimmed.startsWith("</person>")
                        || (block.size() == 1 && trimmed.endsWith("/>"));
                if (!personDone) {
                    continue;
                }

                insidePerson = false;
                boolean keep = keptIds.size() < numAgents || (currentId != null && wantedIds.contains(currentId));
                if (keep) {
                    for (String blockLine : block) {
                        writer.write(blockLine);
                        writer.newLine();
                    }
                    keptIds.add(currentId);
                    wantedIds.remove(currentId);
                }

                if (keptIds.size() >= numAgents && wantedIds.isEmpty()) {
                    writer.write("</population>");
                    writer.newLine();
                    return keptIds;
                }
            }
        }

        return keptIds;
    }

    private static String personId(String line) {
        Matcher matcher = PERSON_ID.matcher(line);
        return matcher.find() ? matcher.group(1) : null;
    }

    private static InputStream open(String url) throws IOException {
        if (url.startsWith("http://") || url.startsWith("https://")) {
            return URI.create(url).toURL().openStream();
        }
        return Files.newInputStream(Paths.get(url));
    }

    private static String argument(String[] args, String name, String defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (name.equals(args[i])) {
                return args[i + 1];
            }
        }
        return defaultValue;
    }
}
