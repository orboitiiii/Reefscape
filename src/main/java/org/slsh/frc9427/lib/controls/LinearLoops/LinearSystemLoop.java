package org.slsh.frc9427.lib.controls.LinearLoops;

import java.util.function.Consumer;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.simple.SimpleMatrix;
import org.slsh.frc9427.lib.controls.LinearController.LinearPlantInversionFeedforward;
import org.slsh.frc9427.lib.controls.LinearController.LinearQuadraticRegulator;
import org.slsh.frc9427.lib.controls.filter.KalmanFilter;

/**
 * A complete state-space control loop that integrates an LQR, feedforward, and a Kalman filter.
 *
 * <p>This is the top-level object used directly in the robot code.
 */
public class LinearSystemLoop {
  private final LinearQuadraticRegulator m_controller;
  private final LinearPlantInversionFeedforward m_feedforward;
  private final KalmanFilter m_observer;

  // Target reference state vector
  private final SimpleMatrix m_nextR;
  // Final voltage vector u to be sent to the motors
  private final SimpleMatrix m_u;

  // Function to clamp the voltage output
  private final Consumer<SimpleMatrix> m_clampFunction;

  /**
   * Constructs a state-space loop.
   *
   * @param controller The LQR controller (NoGC version)
   * @param feedforward The feedforward controller (NoGC version)
   * @param observer The Kalman filter (NoGC version)
   * @param clampFunction A Consumer<SimpleMatrix> that clamps the voltage magnitude in-place
   */
  public LinearSystemLoop(
      LinearQuadraticRegulator controller,
      LinearPlantInversionFeedforward feedforward,
      KalmanFilter observer,
      Consumer<SimpleMatrix> clampFunction) {
    this.m_controller = controller;
    this.m_feedforward = feedforward;
    this.m_observer = observer;
    this.m_clampFunction = clampFunction;

    int numInputs = feedforward.getUff().getNumRows();
    int numStates = feedforward.getR().getNumRows();

    m_nextR = new SimpleMatrix(numStates, 1);
    m_u = new SimpleMatrix(numInputs, 1);
  }

  public SimpleMatrix getXHat() {
    return m_observer.getXhat();
  }

  public double getXHat(int row) {
    return m_observer.getXhat(row);
  }

  public SimpleMatrix getNextR() {
    return m_nextR;
  }

  public void setNextR(SimpleMatrix nextR) {
    System.arraycopy(
        nextR.getDDRM().getData(), 0, this.m_nextR.getDDRM().getData(), 0, nextR.getNumElements());
  }

  public SimpleMatrix getU() {
    return m_u;
  }

  public void reset(SimpleMatrix initialState) {
    m_nextR.zero();
    m_controller.reset();
    m_feedforward.reset(initialState);
    m_observer.setXhat(initialState);
  }

  /**
   * Corrects the state estimate using the measurement y.
   *
   * @param y The sensor measurement vector
   */
  public void correct(SimpleMatrix y) {
    m_observer.correct(m_u, y);
  }

  /**
   * Predicts the next state and calculates the next control output.
   *
   * @param dtSeconds The time step
   */
  public void predict(double dtSeconds) {
    // Set the target reference for the LQR
    m_controller.setR(m_nextR);

    // Calculate the LQR control output
    m_controller.calculate(m_observer.getXhat());

    // Calculate the feedforward control output
    m_feedforward.calculate(m_feedforward.getR(), m_nextR);

    // Combine LQR and feedforward outputs
    // u = u_lqr + u_ff
    // FIX: Use System.arraycopy to copy first, then use CommonOps_DDRM.add for garbage-free
    // addition
    System.arraycopy(
        m_controller.getU().getDDRM().getData(),
        0,
        this.m_u.getDDRM().getData(),
        0,
        m_controller.getU().getNumElements());

    CommonOps_DDRM.add(m_u.getDDRM(), m_feedforward.getUff().getDDRM(), m_u.getDDRM());

    // Pre-clamp u for anti-windup calculation
    SimpleMatrix temp_u = new SimpleMatrix(m_u); // Temporary copy for pre-clamp u

    // Clamp the voltage output
    m_clampFunction.accept(m_u);

    // Anti-windup compensation: Adjust observer state x += B * (u_sat - u) / dt
    // This helps prevent integrator windup in the presence of saturation
    DMatrixRMaj du = new DMatrixRMaj(m_u.getNumRows(), 1);
    CommonOps_DDRM.subtract(m_u.getDDRM(), temp_u.getDDRM(), du);
    CommonOps_DDRM.scale(1.0 / dtSeconds, du); // Scale by 1/dt for compensation
    CommonOps_DDRM.multAdd(
        m_observer.m_system.getB().getDDRM(), du, m_observer.getXhat().getDDRM());

    // Use the final (clamped) voltage to predict the observer's next state
    m_observer.predict(m_u, dtSeconds);
  }

  /**
   * An example voltage clamping function that can modify the matrix in-place.
   *
   * @param maxVoltage The maximum voltage
   * @return A Consumer that can be passed to the LinearSystemLoop constructor
   */
  public static Consumer<SimpleMatrix> createVoltageClamp(double maxVoltage) {
    return u -> {
      for (int i = 0; i < u.getNumElements(); i++) {
        double val = u.get(i);
        if (val > maxVoltage) {
          u.set(i, 0, maxVoltage);
        } else if (val < -maxVoltage) {
          u.set(i, 0, -maxVoltage);
        }
      }
    };
  }
}
