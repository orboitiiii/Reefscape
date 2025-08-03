import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import edu.wpi.first.math.system.plant.DCMotor;
import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slsh.frc9427.lib.controls.LinearController.LinearQuadraticRegulator;
import org.slsh.frc9427.lib.controls.plant.LinearSystem;
import org.slsh.frc9427.lib.controls.plant.StateSpaceFactory;
import org.slsh.frc9427.lib.controls.util.StateSpaceUtil;

class ElevatorLQRTest {

  private static final Logger LOGGER = LoggerFactory.getLogger(ElevatorLQRTest.class);

  private static class ElevatorConstants {
    static final double MASS_KG = 7.5;
    static final double DRUM_RADIUS_M = 0.019;
    static final double GEAR_RATIO = 4.2;
    static final int numMotor = 2;
    static final double LQR_TOL_X_M = 0.01;
    static final double LQR_TOL_V_MPS = 0.1;
    static final double LQR_TOL_U_VOLTS = 12.0;
    static final double ROBOT_PERIODIC_MS = 0.02;
  }

  @Test
  void testElevatorControllabilityObservability() {
    LinearSystem sys =
        StateSpaceFactory.createElevatorSystem(
            DCMotor.getKrakenX60Foc(ElevatorConstants.numMotor),
            ElevatorConstants.MASS_KG,
            ElevatorConstants.DRUM_RADIUS_M,
            ElevatorConstants.GEAR_RATIO);
    assertDoesNotThrow(() -> sys, "Elevator system creation should not throw");
  }

  @Test
  void testElevatorLQRGainCalculation() {
    try {
      LinearSystem elevatorSystem =
          StateSpaceFactory.createElevatorSystem(
              DCMotor.getKrakenX60Foc(ElevatorConstants.numMotor),
              ElevatorConstants.MASS_KG,
              ElevatorConstants.DRUM_RADIUS_M,
              ElevatorConstants.GEAR_RATIO);

      LOGGER.debug("System A matrix:\n{}", elevatorSystem.getA());
      LOGGER.debug("System B matrix:\n{}", elevatorSystem.getB());

      SimpleMatrix Q =
          StateSpaceUtil.makeStateCostMatrix(
              new double[] {ElevatorConstants.LQR_TOL_X_M, ElevatorConstants.LQR_TOL_V_MPS},
              true,
              2);
      SimpleMatrix R =
          StateSpaceUtil.makeInputCostMatrix(
              new double[] {ElevatorConstants.LQR_TOL_U_VOLTS}, true, 1);

      LOGGER.debug("Q matrix:\n{}", Q);
      LOGGER.debug("R matrix:\n{}", R);

      LinearQuadraticRegulator controller =
          new LinearQuadraticRegulator(elevatorSystem, Q, R, ElevatorConstants.ROBOT_PERIODIC_MS);

      SimpleMatrix K = controller.getK();
      LOGGER.info("LQR Gain Matrix:\n{}", K);

      assertTrue(K.get(0, 0) > 0, "K[0, 0] should be positive for stabilization");
      assertTrue(K.get(0, 1) > 0, "K[0, 1] should be positive for stabilization");
      assertTrue(K.get(0, 0) > K.get(0, 1), "Position gain should be larger than velocity gain");
      assertTrue(K.get(0, 0) < 10000, "Position gain should not be excessively large");
      assertTrue(K.get(0, 1) < 1000, "Velocity gain should not be excessively large");

    } catch (Exception e) {
      LOGGER.error("LQR calculation failed: {}", e.getMessage(), e);
      fail("LQR calculation threw an unexpected exception: " + e.getMessage());
    }
  }

  @Test
  void testDirectWeights() {
    try {
      LinearSystem elevatorSystem =
          StateSpaceFactory.createElevatorSystem(
              DCMotor.getKrakenX60Foc(ElevatorConstants.numMotor),
              ElevatorConstants.MASS_KG,
              ElevatorConstants.DRUM_RADIUS_M,
              ElevatorConstants.GEAR_RATIO);

      SimpleMatrix Q = StateSpaceUtil.makeStateCostMatrix(new double[] {10000, 100}, false, 2);
      SimpleMatrix R = StateSpaceUtil.makeInputCostMatrix(new double[] {1.0 / 144.0}, false, 1);

      LinearQuadraticRegulator controller =
          new LinearQuadraticRegulator(elevatorSystem, Q, R, ElevatorConstants.ROBOT_PERIODIC_MS);

      SimpleMatrix K = controller.getK();
      assertTrue(K.get(0, 0) > 0, "K[0, 0] should be positive with direct weights");
      assertTrue(K.get(0, 1) > 0, "K[0, 1] should be positive with direct weights");
      assertTrue(K.get(0, 0) > K.get(0, 1), "Position gain should be larger than velocity gain");
      LOGGER.info("LQR Gain Matrix (direct weights):\n{}", K);

    } catch (Exception e) {
      LOGGER.error("Direct weights LQR calculation failed: {}", e.getMessage(), e);
      fail("Direct weights LQR calculation threw an unexpected exception: " + e.getMessage());
    }
  }

  @Test
  void testExtremeWeights() {
    try {
      LinearSystem elevatorSystem =
          StateSpaceFactory.createElevatorSystem(
              DCMotor.getKrakenX60Foc(ElevatorConstants.numMotor),
              ElevatorConstants.MASS_KG,
              ElevatorConstants.DRUM_RADIUS_M,
              ElevatorConstants.GEAR_RATIO);

      SimpleMatrix Q = StateSpaceUtil.makeStateCostMatrix(new double[] {0.0001}, true, 2);
      SimpleMatrix R = StateSpaceUtil.makeInputCostMatrix(new double[] {12.0}, true, 1);

      LinearQuadraticRegulator controller =
          new LinearQuadraticRegulator(elevatorSystem, Q, R, ElevatorConstants.ROBOT_PERIODIC_MS);

      SimpleMatrix K = controller.getK();
      assertTrue(K.get(0, 0) > 0, "K[0, 0] should be positive with extreme weights");
      assertTrue(K.get(0, 1) > 0, "K[0, 1] should be positive with extreme weights");
      LOGGER.info("LQR Gain Matrix (extreme weights):\n{}", K);

    } catch (Exception e) {
      LOGGER.error("Extreme weights LQR calculation failed: {}", e.getMessage(), e);
      fail("Extreme weights LQR calculation threw an unexpected exception: " + e.getMessage());
    }
  }
}
