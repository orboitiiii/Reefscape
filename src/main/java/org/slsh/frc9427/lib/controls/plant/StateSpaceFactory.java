package org.slsh.frc9427.lib.controls.plant;

import edu.wpi.first.math.system.plant.DCMotor;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.MatrixFeatures_DDRM;
import org.ejml.simple.SimpleMatrix;

/**
 * A factory class for creating state-space models for common FRC mechanisms
 *
 * <p>This utility class uses first-principles modeling to derive linear systems from physical
 * parameters. All systems it produces use {@link LinearSystem} and {@link SimpleMatrix}, making
 * them fully compatible with garbage-free control loops.
 */
public final class StateSpaceFactory {
  private StateSpaceFactory() {
    // Utility class
  }

  // =================================================================================
  //
  // Single-DOF Systems
  //
  // =================================================================================

  /**
   * Creates a state-space model for a generic single-degree-of-freedom system driven by a DC motor.
   *
   * <p>States: [position, velocity]ᵀ, Inputs: [voltage], Outputs: [position, velocity]ᵀ.
   *
   * @param motor The motor or gearbox
   * @param J_eff_KgMetersSquared The system's effective moment of inertia at the output
   * @param gearing The gear ratio (output / input)
   * @return A LinearSystem representing the given physical constants
   */
  private static LinearSystem createSingleDOFSystem(
      DCMotor motor, double J_eff_KgMetersSquared, double gearing) {
    if (J_eff_KgMetersSquared <= 0.0) {
      throw new IllegalArgumentException("J_eff_KgMetersSquared must be greater than zero.");
    }
    if (gearing <= 0.0) {
      throw new IllegalArgumentException("Gearing must be greater than zero.");
    }
    if (motor.KtNMPerAmp <= 0 || motor.KvRadPerSecPerVolt <= 0 || motor.rOhms <= 0) {
      throw new IllegalArgumentException("Invalid motor parameters");
    }

    SimpleMatrix A = new SimpleMatrix(2, 2);
    A.set(0, 1, 1.0);
    A.set(
        1,
        1,
        -Math.pow(gearing, 2)
            * motor.KtNMPerAmp
            / (motor.KvRadPerSecPerVolt * motor.rOhms * J_eff_KgMetersSquared));

    if (!Double.isFinite(A.get(1, 1))) {
      throw new RuntimeException("Unstable parameters leading to Inf/NaN in A");
    }

    SimpleMatrix B = new SimpleMatrix(2, 1);
    B.set(1, 0, gearing * motor.KtNMPerAmp / (motor.rOhms * J_eff_KgMetersSquared));

    SimpleMatrix C = SimpleMatrix.identity(2);
    SimpleMatrix D = new SimpleMatrix(2, 1);

    LinearSystem sys = new LinearSystem(A, B, C, D);

    // Check controllability and observability
    SimpleMatrix ctrb = controllabilityMatrix(A, B);
    if (MatrixFeatures_DDRM.rank(ctrb.getDDRM()) != 2) {
      throw new RuntimeException("System not controllable");
    }
    SimpleMatrix obsv = observabilityMatrix(A, C);
    if (MatrixFeatures_DDRM.rank(obsv.getDDRM()) != 2) {
      throw new RuntimeException("System not observable");
    }

    return sys;
  }

  /** Creates a state-space model for an elevator system. */
  public static LinearSystem createElevatorSystem(
      DCMotor motor, double massKg, double radiusMeters, double gearing) {
    double effectiveInertia = massKg * radiusMeters * radiusMeters;
    return createSingleDOFSystem(motor, effectiveInertia, gearing);
  }

  /** Creates a state-space model for a single-jointed arm. */
  public static LinearSystem createSingleJointedArmSystem(
      DCMotor motor, double JKgMetersSquared, double gearing) {
    return createSingleDOFSystem(motor, JKgMetersSquared, gearing);
  }

  /**
   * Creates a state-space model for a flywheel system. States: [angular velocity], Inputs:
   * [voltage], Outputs: [angular velocity].
   */
  public static LinearSystem createFlywheelSystem(
      DCMotor motor, double JKgMetersSquared, double gearing) {
    if (JKgMetersSquared <= 0.0) {
      throw new IllegalArgumentException("JKgMetersSquared must be greater than zero.");
    }
    if (gearing <= 0.0) {
      throw new IllegalArgumentException("Gearing must be greater than zero.");
    }
    if (motor.KtNMPerAmp <= 0 || motor.KvRadPerSecPerVolt <= 0 || motor.rOhms <= 0) {
      throw new IllegalArgumentException("Invalid motor parameters");
    }

    SimpleMatrix A = new SimpleMatrix(1, 1);
    A.set(
        0,
        0,
        -Math.pow(gearing, 2)
            * motor.KtNMPerAmp
            / (motor.KvRadPerSecPerVolt * motor.rOhms * JKgMetersSquared));

    if (!Double.isFinite(A.get(0, 0))) {
      throw new RuntimeException("Unstable parameters leading to Inf/NaN in A");
    }

    SimpleMatrix B = new SimpleMatrix(1, 1);
    B.set(0, 0, gearing * motor.KtNMPerAmp / (motor.rOhms * JKgMetersSquared));

    SimpleMatrix C = SimpleMatrix.identity(1);
    SimpleMatrix D = new SimpleMatrix(1, 1);

    LinearSystem sys = new LinearSystem(A, B, C, D);

    // Check controllability and observability
    SimpleMatrix ctrb = controllabilityMatrix(A, B);
    if (MatrixFeatures_DDRM.rank(ctrb.getDDRM()) != 1) {
      throw new RuntimeException("System not controllable");
    }
    SimpleMatrix obsv = observabilityMatrix(A, C);
    if (MatrixFeatures_DDRM.rank(obsv.getDDRM()) != 1) {
      throw new RuntimeException("System not observable");
    }

    return sys;
  }

  // =================================================================================
  //
  // System Augmentation
  //
  // =================================================================================

  /** Augments a 2-state, 1-input, 2-output system to estimate input error. */
  public static LinearSystem augmentWithInputError(LinearSystem system) {
    int states = system.getNumStates();
    int inputs = system.getNumInputs();
    int outputs = system.getNumOutputs();

    if (states != 2 || inputs != 1 || outputs != 2) {
      throw new IllegalArgumentException("Only for 2-state, 1-input, 2-output system");
    }

    SimpleMatrix oldA = system.getA();
    SimpleMatrix oldB = system.getB();
    SimpleMatrix oldC = system.getC();
    SimpleMatrix oldD = system.getD();

    int newStates = states + inputs;

    SimpleMatrix newA = new SimpleMatrix(newStates, newStates);
    SimpleMatrix newB = new SimpleMatrix(newStates, inputs);
    SimpleMatrix newC = new SimpleMatrix(outputs, newStates);
    SimpleMatrix newD = new SimpleMatrix(outputs, inputs);

    // A_aug = [A, B]
    //         [0, 0]
    CommonOps_DDRM.insert(oldA.getDDRM(), newA.getDDRM(), 0, 0);
    CommonOps_DDRM.insert(oldB.getDDRM(), newA.getDDRM(), 0, states);

    // B_aug = [B]
    //         [0]
    CommonOps_DDRM.insert(oldB.getDDRM(), newB.getDDRM(), 0, 0);

    // C_aug = [C, D]
    CommonOps_DDRM.insert(oldC.getDDRM(), newC.getDDRM(), 0, 0);
    CommonOps_DDRM.insert(oldD.getDDRM(), newC.getDDRM(), 0, states);

    // D_aug = [D]
    CommonOps_DDRM.insert(oldD.getDDRM(), newD.getDDRM(), 0, 0);

    LinearSystem sys = new LinearSystem(newA, newB, newC, newD);

    // Check controllability and observability
    SimpleMatrix ctrb = controllabilityMatrix(newA, newB);
    if (MatrixFeatures_DDRM.rank(ctrb.getDDRM()) != newStates) {
      throw new RuntimeException("Augmented system not controllable");
    }
    SimpleMatrix obsv = observabilityMatrix(newA, newC);
    if (MatrixFeatures_DDRM.rank(obsv.getDDRM()) != newStates) {
      throw new RuntimeException("Augmented system not observable");
    }

    return sys;
  }

  // =================================================================================
  //
  // Nonlinear Systems - Two-Jointed Arm
  //
  // =================================================================================

  /** A record that holds all the physical parameters of a two-jointed arm. */
  public record TwoJointedArmPlant_NoGC(
      DCMotor motor,
      double m1,
      double l1,
      double r1,
      double I1,
      double m2,
      double l2,
      double r2,
      double I2,
      double G1,
      double G2,
      int numMotors1,
      int numMotors2,
      double g) {

    /**
     * Linearizes the arm's dynamics around a static operating point (zero velocity).
     *
     * @param operatingAngles A [theta1, theta2] vector (rad) representing the center point for
     *     linearization.
     * @return A LinearSystem representing the arm's dynamics near that point.
     */
    public LinearSystem linearize(SimpleMatrix operatingAngles) {
      final double t1 = operatingAngles.get(0, 0);
      final double t2 = operatingAngles.get(1, 0);

      final double s1 = Math.sin(t1);
      // final double c1 = Math.cos(t1);
      // final double s2 = Math.sin(t2);
      final double c2 = Math.cos(t2);
      final double s12 = Math.sin(t1 + t2);
      // final double c12 = Math.cos(t1 + t2);

      // --- Recalculate matrices at the operating point ---
      final double M11 = m1 * r1 * r1 + m2 * (l1 * l1 + r2 * r2) + I1 + I2 + 2 * m2 * l1 * r2 * c2;
      final double M12 = m2 * r2 * r2 + I2 + m2 * l1 * r2 * c2;
      final double M22 = m2 * r2 * r2 + I2;
      SimpleMatrix M = new SimpleMatrix(new double[][] {{M11, M12}, {M12, M22}});
      SimpleMatrix M_inv = M.invert();

      final double R = motor.rOhms;
      final double Kt = motor.KtNMPerAmp;
      final double Kv = motor.KvRadPerSecPerVolt;
      SimpleMatrix K_b =
          new SimpleMatrix(
              new double[][] {
                {G1 * G1 * numMotors1 * Kt / (Kv * R), 0},
                {0, G2 * G2 * numMotors2 * Kt / (Kv * R)}
              });

      SimpleMatrix B_m =
          new SimpleMatrix(
              new double[][] {
                {G1 * numMotors1 * Kt / R, 0},
                {0, G2 * numMotors2 * Kt / R}
              });

      final double d_tau_g1_d_t1 = -(m1 * r1 + m2 * l1) * g * s1 - m2 * r2 * g * s12;
      final double d_tau_g1_d_t2 = -m2 * r2 * g * s12;
      final double d_tau_g2_d_t1 = -m2 * r2 * g * s12;
      final double d_tau_g2_d_t2 = -m2 * r2 * g * s12;
      SimpleMatrix d_tau_g_d_theta =
          new SimpleMatrix(
              new double[][] {
                {d_tau_g1_d_t1, d_tau_g1_d_t2},
                {d_tau_g2_d_t1, d_tau_g2_d_t2}
              });

      // --- Assemble the linearized state-space matrices A and B ---
      SimpleMatrix A21 = M_inv.mult(d_tau_g_d_theta).scale(-1.0);
      SimpleMatrix A22 = M_inv.mult(K_b).scale(-1.0);

      SimpleMatrix A = new SimpleMatrix(4, 4);
      CommonOps_DDRM.insert(SimpleMatrix.identity(2).getDDRM(), A.getDDRM(), 0, 2);
      CommonOps_DDRM.insert(A21.getDDRM(), A.getDDRM(), 2, 0);
      CommonOps_DDRM.insert(A22.getDDRM(), A.getDDRM(), 2, 2);

      SimpleMatrix B2 = M_inv.mult(B_m);
      SimpleMatrix B = new SimpleMatrix(4, 2);
      CommonOps_DDRM.insert(B2.getDDRM(), B.getDDRM(), 2, 0);

      SimpleMatrix C = SimpleMatrix.identity(4);
      SimpleMatrix D = new SimpleMatrix(4, 2);

      LinearSystem sys = new LinearSystem(A, B, C, D);

      // Check controllability and observability
      SimpleMatrix ctrb = controllabilityMatrix(A, B);
      if (MatrixFeatures_DDRM.rank(ctrb.getDDRM()) != 4) {
        throw new RuntimeException("System not controllable");
      }
      SimpleMatrix obsv = observabilityMatrix(A, C);
      if (MatrixFeatures_DDRM.rank(obsv.getDDRM()) != 4) {
        throw new RuntimeException("System not observable");
      }

      return sys;
    }
  }

  public static TwoJointedArmPlant_NoGC createTwoJointedArmPlant(
      DCMotor motor,
      double m1,
      double l1,
      double r1,
      double I1,
      double m2,
      double l2,
      double r2,
      double I2,
      double G1,
      double G2,
      int numMotors1,
      int numMotors2,
      double g) {
    return new TwoJointedArmPlant_NoGC(
        motor, m1, l1, r1, I1, m2, l2, r2, I2, G1, G2, numMotors1, numMotors2, g);
  }

  public static TwoJointedArmPlant_NoGC createTwoJointedArmPlant(
      DCMotor motor,
      double m1,
      double l1,
      double r1,
      double I1,
      double m2,
      double l2,
      double r2,
      double I2,
      double G1,
      double G2,
      int numMotors1,
      int numMotors2) {
    return new TwoJointedArmPlant_NoGC(
        motor, m1, l1, r1, I1, m2, l2, r2, I2, G1, G2, numMotors1, numMotors2, 9.80665);
  }

  private static SimpleMatrix controllabilityMatrix(SimpleMatrix A, SimpleMatrix B) {
    int n = A.getNumRows();
    int m = B.getNumCols();
    SimpleMatrix ctrb = new SimpleMatrix(n, n * m);
    SimpleMatrix temp = new SimpleMatrix(n, m);
    SimpleMatrix nextTemp = new SimpleMatrix(n, m);
    CommonOps_DDRM.insert(B.getDDRM(), ctrb.getDDRM(), 0, 0);
    System.arraycopy(B.getDDRM().getData(), 0, temp.getDDRM().getData(), 0, B.getNumElements());
    for (int i = 1; i < n; i++) {
      CommonOps_DDRM.mult(A.getDDRM(), temp.getDDRM(), nextTemp.getDDRM());
      CommonOps_DDRM.insert(nextTemp.getDDRM(), ctrb.getDDRM(), 0, i * m);
      System.arraycopy(
          nextTemp.getDDRM().getData(), 0, temp.getDDRM().getData(), 0, nextTemp.getNumElements());
    }
    return ctrb;
  }

  private static SimpleMatrix observabilityMatrix(SimpleMatrix A, SimpleMatrix C) {
    int n = A.getNumRows();
    int p = C.getNumRows();
    SimpleMatrix obsv = new SimpleMatrix(n * p, n);
    SimpleMatrix temp = new SimpleMatrix(p, n);
    SimpleMatrix nextTemp = new SimpleMatrix(p, n);
    CommonOps_DDRM.insert(C.getDDRM(), obsv.getDDRM(), 0, 0);
    System.arraycopy(C.getDDRM().getData(), 0, temp.getDDRM().getData(), 0, C.getNumElements());
    for (int i = 1; i < n; i++) {
      CommonOps_DDRM.mult(temp.getDDRM(), A.getDDRM(), nextTemp.getDDRM());
      CommonOps_DDRM.insert(nextTemp.getDDRM(), obsv.getDDRM(), i * p, 0);
      System.arraycopy(
          nextTemp.getDDRM().getData(), 0, temp.getDDRM().getData(), 0, nextTemp.getNumElements());
    }
    return obsv;
  }
}
