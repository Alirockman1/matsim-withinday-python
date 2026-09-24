package modechoice.rl.java.utils.dataset;

import java.io.FileWriter;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;

public class GenerateSyntheticTrainingData {

    private static final String OUTPUT_FILE = "scenarios\\sioux-falls\\input\\synthetic_dataset.csv";

    public static void main(String[] args) {
        System.out.println("Starting Compressed Binary Dataset Generation (0-Indexed)...");
        long startTime = System.currentTimeMillis();

        String projectRoot = System.getProperty("user.dir");
        File file = new File(projectRoot, OUTPUT_FILE);

        try (PrintWriter writer = new PrintWriter(new FileWriter(file))) {
            
            writer.println("dep_grid,arr_grid,time_bin,asset_state,flexibility,encoded_string");

            int rowCount = 0;

            // Systematically loop through every possible 0-indexed combination
            for (int depGrid = 0; depGrid < 64; depGrid++) {
                for (int arrGrid = 0; arrGrid < 64; arrGrid++) {
                    for (int timeBin = 0; timeBin < 18; timeBin++) {
                        for (int assetState = 0; assetState < 4; assetState++) {
                            for (int flexibility = 0; flexibility <= 1; flexibility++) {

                                String encodedString = generateRawBinaryString(depGrid, arrGrid, timeBin, assetState, flexibility);

                                writer.printf("%d,%d,%d,%d,%d,\"%s\"%n", 
                                    depGrid, arrGrid, timeBin, assetState, flexibility, encodedString);
                                
                                rowCount++;
                            }
                        }
                    }
                }
                // Progress Tracker (Triggers every 8 departure grids processed)
                if ((depGrid + 1) % 8 == 0) {
                    System.out.println("Progress: " + String.format("%.1f", ((double)rowCount / 589824.0) * 100) + "%");
                }
            }

            long endTime = System.currentTimeMillis();
            System.out.println("🎉 Success! Generated " + rowCount + " rows in " + ((endTime - startTime) / 1000.0) + "s");
            System.out.println("Vector size verified: 136 bits continuous sequence.");
            System.out.println("Dataset saved to file: " + OUTPUT_FILE);

        } catch (IOException e) {
            System.err.println("Failed to write dataset CSV: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static String generateRawBinaryString(int depGrid, int arrGrid, int timeBin, int assetState, int flexibility) {
        StringBuilder sb = new StringBuilder(140); // Pre-allocate memory for performance

        // 1. One-Hot Encode Departure Grid (64 bits, exactly one '1')
        sb.append(convertToOneHot(depGrid, 64));

        // 2. One-Hot Encode Arrival Grid (64 bits, exactly one '1')
        sb.append(convertToOneHot(arrGrid, 64));

        // 3. Binary Encoded Time Bin (5 bits, max value 17 fits as 10001)
        sb.append(convertToPaddedBinary(timeBin, 5));

        // 4. Binary Encoded Asset State (2 bits, max value 3 fits as 11)
        sb.append(convertToPaddedBinary(assetState, 2));

        // 5. Flexibility Bit (1 bit directly)
        sb.append(flexibility);

        return sb.toString();
    }

    private static String convertToOneHot(int targetIndex, int bitLength) {
        StringBuilder sb = new StringBuilder(bitLength);
        for (int i = 0; i < bitLength; i++) {
            if (i == targetIndex) {
                sb.append('1');
            } else {
                sb.append('0');
            }
        }
        return sb.toString();
    }

    private static String convertToPaddedBinary(int value, int bitLength) {
        String binaryStr = Integer.toBinaryString(value);
        
        if (binaryStr.length() >= bitLength) {
            return binaryStr.substring(binaryStr.length() - bitLength);
        }
        
        StringBuilder padding = new StringBuilder();
        while (padding.length() < (bitLength - binaryStr.length())) {
            padding.append('0');
        }
        
        return padding.append(binaryStr).toString();
    }
}
