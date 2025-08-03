package org.slsh.frc9427.lib.controls.LinearController;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.simple.SimpleMatrix;
import org.slsh.frc9427.lib.controls.plant.LinearSystem;
import org.slsh.frc9427.lib.controls.util.StateSpaceUtil;

/**
 * Linear Quadratic Regulator (LQR).
 *
 * <p>The controller law is u = K(r - x).
 *
 * <p>This class calculates the controller gain K once at construction and performs operations in
 * the `calculate` method in a memory-allocation-free manner.
 */
public class LinearQuadraticRegulator {
  // Controller gain matrix
  private final SimpleMatrix m_K;
  // Target reference state vector r
  private final SimpleMatrix m_r;
  // Controller output vector u
  private final SimpleMatrix m_u;
  // Pre-allocated temporary matrix for storing the error (r - x) during calculation
  private final SimpleMatrix m_error;

  /**
   * Constructs an LQR controller.
   *
   * @param system The linear system to control
   * @param Q The state cost matrix
   * @param R The input cost matrix
   * @param dtSeconds The discretization time step
   */
  public LinearQuadraticRegulator(
      LinearSystem system, SimpleMatrix Q, SimpleMatrix R, double dtSeconds) {

    int states = system.getNumStates();
    int inputs = system.getNumInputs();

    // --- Pre-allocate all necessary matrices for the initialization process ---
    // m_K is the final result
    m_K = new SimpleMatrix(inputs, states);

    // Required for discretization
    SimpleMatrix discA = new SimpleMatrix(states, states);
    SimpleMatrix discB = new SimpleMatrix(states, inputs);
    SimpleMatrix M_temp = new SimpleMatrix(states + inputs, states + inputs);
    SimpleMatrix phi_temp = new SimpleMatrix(states + inputs, states + inputs);

    // Required for K calculation
    SimpleMatrix temp_S = new SimpleMatrix(states, states);
    SimpleMatrix temp_S_B = new SimpleMatrix(states, inputs);
    SimpleMatrix temp_B_S_B = new SimpleMatrix(inputs, inputs);
    SimpleMatrix temp_B_S_B_plus_R = new SimpleMatrix(inputs, inputs);
    SimpleMatrix temp_B_S_A = new SimpleMatrix(inputs, states);
    SimpleMatrix temp_BTS = new SimpleMatrix(inputs, states); // temp_BTS for B^T S

    // Discretize A and B
    StateSpaceUtil.discretizeAB(
        system.getA(), system.getB(), dtSeconds, discA, discB, M_temp, phi_temp);

    // Calculate LQR gain K
    StateSpaceUtil.calculateK(
        discA,
        discB,
        Q,
        R,
        m_K, // output result
        temp_S,
        temp_S_B,
        temp_B_S_B,
        temp_B_S_B_plus_R,
        temp_B_S_A,
        temp_BTS);

    // --- Pre-allocate matrices for the control loop ---
    m_r = new SimpleMatrix(states, 1);
    m_u = new SimpleMatrix(inputs, 1);
    m_error = new SimpleMatrix(states, 1);

    reset();
  }

  /** Resets the reference and output. */
  public void reset() {
    m_r.zero();
    m_u.zero();
  }

  /**
   * Calculates the controller output in a memory-allocation-free manner. u = K * (r - x)
   *
   * @param x The current state vector
   */
  public void calculate(SimpleMatrix x) {
    // Get the underlying DMatrixRMaj of the matrices to use CommonOps_DDRM
    DMatrixRMaj r_ddrm = m_r.getDDRM();
    DMatrixRMaj x_ddrm = x.getDDRM();
    DMatrixRMaj error_ddrm = m_error.getDDRM();
    DMatrixRMaj K_ddrm = m_K.getDDRM();
    DMatrixRMaj u_ddrm = m_u.getDDRM();

    // Calculate the error: error = r - x
    CommonOps_DDRM.subtract(r_ddrm, x_ddrm, error_ddrm);

    // Calculate the output: u = K * error
    CommonOps_DDRM.mult(K_ddrm, error_ddrm, u_ddrm);
  }

  public SimpleMatrix getK() {
    return m_K;
  }

  public SimpleMatrix getU() {
    return m_u;
  }

  public SimpleMatrix getR() {
    return m_r;
  }

  public void setR(SimpleMatrix r) {
    System.arraycopy(r.getDDRM().getData(), 0, this.m_r.getDDRM().getData(), 0, r.getNumElements());
  }
}
