package org.slsh.frc9427.lib.controls.filter;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.decomposition.chol.CholeskyDecompositionCommon_DDRM;
import org.ejml.dense.row.factory.DecompositionFactory_DDRM;
import org.ejml.dense.row.linsol.chol.LinearSolverChol_DDRM;
import org.ejml.interfaces.decomposition.CholeskyDecomposition_F64;
import org.ejml.simple.SimpleMatrix;
import org.slsh.frc9427.lib.controls.plant.LinearSystem;
import org.slsh.frc9427.lib.controls.util.StateSpaceUtil;

/** Kalman Filter for state estimation. */
public class KalmanFilter {
  public final LinearSystem m_system;
  private final SimpleMatrix m_Q; // Process noise
  private final SimpleMatrix m_R; // Measurement noise

  private final SimpleMatrix m_xhat;
  private final SimpleMatrix m_P;

  private final SimpleMatrix m_discA;
  private final SimpleMatrix m_discB;
  private final SimpleMatrix m_M_temp;
  private final SimpleMatrix m_phi_temp;
  private double m_lastDt = -1.0;

  private final SimpleMatrix m_temp_predict_x1;
  private final SimpleMatrix m_temp_predict_x2;
  private final SimpleMatrix m_temp_predict_p1;

  private final SimpleMatrix m_C;
  private final SimpleMatrix m_temp_correct_p1;
  private final SimpleMatrix m_temp_correct_p2;
  private final SimpleMatrix m_S; // C P C^T + R
  private final SimpleMatrix m_S_inv;
  private final SimpleMatrix m_K;
  private final SimpleMatrix m_temp_correct_y;
  private final SimpleMatrix m_I;
  private final SimpleMatrix m_P_backup;

  public KalmanFilter(LinearSystem system, SimpleMatrix Q, SimpleMatrix R) {
    this.m_system = system;
    this.m_Q = Q;
    this.m_R = R;

    int states = system.getNumStates();
    int outputs = system.getNumOutputs();
    int M_size = states + system.getNumInputs();

    m_xhat = new SimpleMatrix(states, 1);
    m_P = SimpleMatrix.identity(states);
    m_C = system.getC();

    m_discA = new SimpleMatrix(states, states);
    m_discB = new SimpleMatrix(states, system.getNumInputs());
    m_M_temp = new SimpleMatrix(M_size, M_size);
    m_phi_temp = new SimpleMatrix(M_size, M_size);

    m_temp_predict_x1 = new SimpleMatrix(states, 1);
    m_temp_predict_x2 = new SimpleMatrix(states, 1);
    m_temp_predict_p1 = new SimpleMatrix(states, states);

    m_temp_correct_p1 = new SimpleMatrix(states, outputs);
    m_temp_correct_p2 = new SimpleMatrix(outputs, states);
    m_S = new SimpleMatrix(outputs, outputs);
    m_S_inv = new SimpleMatrix(outputs, outputs);
    m_K = new SimpleMatrix(states, outputs);
    m_temp_correct_y = new SimpleMatrix(outputs, 1);
    m_I = SimpleMatrix.identity(states);
    m_P_backup = new SimpleMatrix(states, states);
  }

  public void reset() {
    m_xhat.zero();
    CommonOps_DDRM.setIdentity(m_P.getDDRM());
  }

  /** Predicts next state: x' = A x + B u, P' = A P A^T + Q */
  public void predict(SimpleMatrix u, double dtSeconds) {
    if (dtSeconds != m_lastDt) {
      StateSpaceUtil.discretizeAB(
          m_system.getA(), m_system.getB(), dtSeconds, m_discA, m_discB, m_M_temp, m_phi_temp);
      m_lastDt = dtSeconds;
    }

    DMatrixRMaj A = m_discA.getDDRM();
    DMatrixRMaj B = m_discB.getDDRM();
    DMatrixRMaj xhat = m_xhat.getDDRM();
    DMatrixRMaj u_ddrm = u.getDDRM();
    DMatrixRMaj P = m_P.getDDRM();
    DMatrixRMaj Q = m_Q.getDDRM();

    // x' = A x + B u
    CommonOps_DDRM.mult(A, xhat, m_temp_predict_x1.getDDRM());
    CommonOps_DDRM.mult(B, u_ddrm, m_temp_predict_x2.getDDRM());
    CommonOps_DDRM.add(m_temp_predict_x1.getDDRM(), m_temp_predict_x2.getDDRM(), xhat);

    // P' = A P A^T + Q
    CommonOps_DDRM.mult(A, P, m_temp_predict_p1.getDDRM());
    System.arraycopy(Q.getData(), 0, P.getData(), 0, Q.getNumElements());
    CommonOps_DDRM.multAddTransB(m_temp_predict_p1.getDDRM(), A, P);
  }

  /** Corrects state: K = P C^T (C P C^T + R)^{-1}, x = x' + K (y - C x'), P = (I - K C) P' */
  public void correct(SimpleMatrix u, SimpleMatrix y) {
    DMatrixRMaj C = m_C.getDDRM();
    DMatrixRMaj P = m_P.getDDRM();
    DMatrixRMaj R = m_R.getDDRM();
    DMatrixRMaj xhat = m_xhat.getDDRM();
    DMatrixRMaj y_ddrm = y.getDDRM();

    DMatrixRMaj K = m_K.getDDRM();
    DMatrixRMaj S = m_S.getDDRM();
    DMatrixRMaj S_inv = m_S_inv.getDDRM();
    DMatrixRMaj temp_p1 = m_temp_correct_p1.getDDRM();
    DMatrixRMaj temp_p2 = m_temp_correct_p2.getDDRM();
    DMatrixRMaj temp_y = m_temp_correct_y.getDDRM();

    // S = C P C^T + R
    CommonOps_DDRM.multTransB(P, C, temp_p1);
    CommonOps_DDRM.mult(C, P, temp_p2);
    System.arraycopy(R.getData(), 0, S.getData(), 0, R.getNumElements());
    CommonOps_DDRM.multAddTransB(temp_p2, C, S);

    // S_inv = S^{-1} using Cholesky
    CholeskyDecomposition_F64<DMatrixRMaj> chol = DecompositionFactory_DDRM.chol(S.numRows, true);
    if (CommonOps_DDRM.det(S) <= 0) {
      throw new RuntimeException("S not positive definite");
    }
    LinearSolverChol_DDRM cholSolver =
        new LinearSolverChol_DDRM((CholeskyDecompositionCommon_DDRM) chol);
    cholSolver.invert(S_inv);

    // K = (P C^T) S_inv
    CommonOps_DDRM.mult(temp_p1, S_inv, K);

    // x = x' + K (y - C x')
    CommonOps_DDRM.mult(C, xhat, temp_y);
    CommonOps_DDRM.subtract(y_ddrm, temp_y, temp_y);
    CommonOps_DDRM.multAdd(K, temp_y, xhat);

    // P = (I - K C) P'
    CommonOps_DDRM.mult(K, C, m_temp_predict_p1.getDDRM());
    CommonOps_DDRM.subtract(
        m_I.getDDRM(), m_temp_predict_p1.getDDRM(), m_temp_predict_p1.getDDRM());
    System.arraycopy(P.getData(), 0, m_P_backup.getDDRM().getData(), 0, P.getNumElements());
    CommonOps_DDRM.mult(m_temp_predict_p1.getDDRM(), m_P_backup.getDDRM(), P);
  }

  public SimpleMatrix getXhat() {
    return m_xhat;
  }

  public double getXhat(int row) {
    return m_xhat.get(row, 0);
  }

  public void setXhat(SimpleMatrix xhat) {
    System.arraycopy(
        xhat.getDDRM().getData(), 0, this.m_xhat.getDDRM().getData(), 0, xhat.getNumElements());
  }

  public void setXhat(int row, double val) {
    this.m_xhat.set(row, 0, val);
  }
}
