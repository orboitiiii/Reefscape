package org.slsh.frc9427.lib.data;

import edu.wpi.first.networktables.DoublePublisher;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.wpilibj.Notifier;
import edu.wpi.first.wpilibj.smartdashboard.SmartDashboard;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Usage: - Access via CustomDataLogger.getInstance(). - Add fields with addField(name, supplier,
 * type) where type is FieldType.CAN or NON_CAN. - Start updaters in robotInit() with
 * startUpdaters(slowPeriodMs, fastPeriodMs). - Call logData() in main loop (e.g.,
 * teleopPeriodic()).
 */
public class IDearLog {
  private static IDearLog instance;

  public static IDearLog getInstance() {
    if (instance == null) {
      instance = new IDearLog();
    }
    return instance;
  }

  public enum FieldType {
    CAN,
    NON_CAN
  }

  private final NetworkTable table;
  private final Map<String, DoublePublisher> publishers = new HashMap<>();
  private final Map<String, Supplier<Double>> canSuppliers = new HashMap<>();
  private final Map<String, Supplier<Double>> nonCanSuppliers = new HashMap<>();
  private final Map<String, Double> cache = new HashMap<>();
  private final Object lock = new Object(); // For thread safety
  private Notifier slowNotifier;
  private Notifier fastNotifier;

  private IDearLog() {
    table = NetworkTableInstance.getDefault().getTable("CustomLogger");
  }

  /**
   * Add a field to log.
   *
   * @param name Field name for NetworkTables.
   * @param supplier Lambda to fetch data (e.g., motor::getPosition).
   * @param type CAN for slow updates, NON_CAN for fast.
   */
  public void addField(String name, Supplier<Double> supplier, FieldType type) {
    if (publishers.containsKey(name)) {
      return; // Avoid duplicates
    }
    publishers.put(name, table.getDoubleTopic(name).publish());
    cache.put(name, 0.0);
    if (type == FieldType.CAN) {
      canSuppliers.put(name, supplier);
    } else {
      nonCanSuppliers.put(name, supplier);
    }
  }

  /**
   * Start background updaters.
   *
   * @param slowPeriodMs For CAN fields (100ms).
   * @param fastPeriodMs For non-CAN fields (20ms).
   */
  public void startUpdaters(int slowPeriodMs, int fastPeriodMs) {
    stopUpdaters(); // Reset if running
    if (!canSuppliers.isEmpty()) {
      slowNotifier = new Notifier(new UpdaterRunnable(canSuppliers));
      slowNotifier.startPeriodic(slowPeriodMs / 1000.0); // ms transform to seconds
    }
    if (!nonCanSuppliers.isEmpty()) {
      fastNotifier = new Notifier(new UpdaterRunnable(nonCanSuppliers));
      fastNotifier.startPeriodic(fastPeriodMs / 1000.0); // ms transform to seconds
    }
  }

  /** Stop updaters (e.g., in disabledInit()). */
  public void stopUpdaters() {
    if (slowNotifier != null) {
      slowNotifier.stop();
      slowNotifier.close(); // release resources
      slowNotifier = null;
    }
    if (fastNotifier != null) {
      fastNotifier.stop();
      fastNotifier.close(); // release resources
      fastNotifier = null;
    }
  }

  /** Publish cached data to NetworkTables. Call in main loop—fast and non-blocking. */
  public void logData() {
    synchronized (lock) {
      for (Map.Entry<String, Double> entry : cache.entrySet()) {
        publishers.get(entry.getKey()).set(entry.getValue());
      }
    }
    SmartDashboard.putNumber("Logger/NumFields", publishers.size());
  }

  private class UpdaterRunnable implements Runnable {
    private final Map<String, Supplier<Double>> suppliers;

    public UpdaterRunnable(Map<String, Supplier<Double>> suppliers) {
      this.suppliers = suppliers;
    }

    @Override
    public void run() {
      synchronized (lock) {
        for (Map.Entry<String, Supplier<Double>> entry : suppliers.entrySet()) {
          try {
            double value = entry.getValue().get();
            cache.put(entry.getKey(), value);
          } catch (Exception e) {
            System.err.println("Error updating " + entry.getKey() + ": " + e.getMessage());
            // Optional: Cache last good value or set to NaN
            cache.put(entry.getKey(), Double.NaN);
          }
        }
      }
    }
  }
}
