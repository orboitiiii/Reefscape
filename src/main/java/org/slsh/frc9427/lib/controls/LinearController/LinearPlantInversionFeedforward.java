package org.slsh.frc9427.lib.controls.LinearController;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.factory.DecompositionFactory_DDRM;
import org.ejml.dense.row.factory.LinearSolverFactory_DDRM;
import org.ejml.interfaces.decomposition.SingularValueDecomposition_F64;
import org.ejml.interfaces.linsol.LinearSolverDense;
import org.ejml.simple.SimpleMatrix;
import org.slsh.frc9427.lib.controls.plant.LinearSystem;
import org.slsh.frc9427.lib.controls.util.StateSpaceUtil;

/**
 * Plant-inversion feedforward.
 *
 * <p>The feedforward law is u_ff = B⁺ (r_k+1 - A * r_k), where B⁺ is the pseudoinverse of B.
 *
 * <p>The constructor and `calculate` method of this class execute with almost no memory allocation.
 */
public class LinearPlantInversionFeedforward {
  // Discretized A and B matrices
  private final SimpleMatrix m_A;
  private final SimpleMatrix m_B;

  // Reference state vector
  private SimpleMatrix m_r;
  // Feedforward output vector
  private final SimpleMatrix m_uff;

  // Pre-allocated temporary matrix
  private final SimpleMatrix m_temp_state_term;

  // Pre-allocated linear solver
  private final LinearSolverDense<DMatrixRMaj> m_solver;

  /**
   * Constructs a feedforward controller.
   *
   * @param system The linear system to control
   * @param dtSeconds The discretization time step
   */
  public LinearPlantInversionFeedforward(LinearSystem system, double dtSeconds) {
    int states = system.getNumStates();
    int inputs = system.getNumInputs();

    // --- Pre-allocate all necessary matrices for initialization and the control loop ---

    // Required for initialization
    m_A = new SimpleMatrix(states, states);
    m_B = new SimpleMatrix(states, inputs);
    SimpleMatrix M_temp = new SimpleMatrix(states + inputs, states + inputs);
    SimpleMatrix phi_temp = new SimpleMatrix(states + inputs, states + inputs);

    // Required for the control loop
    m_r = new SimpleMatrix(states, 1);
    m_uff = new SimpleMatrix(inputs, 1);
    m_temp_state_term = new SimpleMatrix(states, 1);

    // Discretize A and B
    StateSpaceUtil.discretizeAB(
        system.getA(),
        system.getB(),
        dtSeconds,
        m_A,
        m_B, // output result
        M_temp,
        phi_temp // temporary
        );

    // --- Initialize other components ---
    // Pre-configure the linear solver to solve for u_ff
    m_solver = LinearSolverFactory_DDRM.pseudoInverse(true);

    reset();
  }

  /** Resets the reference and output. */
  public void reset() {
    this.m_r.zero();
    this.m_uff.zero();
  }

  public void reset(SimpleMatrix initialState) {
    System.arraycopy(
        initialState.getDDRM().getData(),
        0,
        this.m_r.getDDRM().getData(),
        0,
        initialState.getNumElements());
    this.m_uff.zero();
  }

  /**
   * Calculates the feedforward output in a memory-allocation-free manner. u_ff = B⁺(r_k+1 − Ar_k)
   *
   * @param r The current reference state r_k
   * @param nextR The next reference state r_k+1
   */
  public void calculate(SimpleMatrix r, SimpleMatrix nextR) {
    // Get the underlying DMatrixRMaj
    DMatrixRMaj A_ddrm = m_A.getDDRM();
    DMatrixRMaj r_ddrm = r.getDDRM();
    DMatrixRMaj nextR_ddrm = nextR.getDDRM();
    DMatrixRMaj temp_ddrm = m_temp_state_term.getDDRM();
    DMatrixRMaj uff_ddrm = m_uff.getDDRM();
    DMatrixRMaj B_ddrm = m_B.getDDRM();

    // Check B condition number using SVD
    int states = m_A.getNumRows();
    int inputs = m_B.getNumCols();
    SingularValueDecomposition_F64<DMatrixRMaj> svd =
        DecompositionFactory_DDRM.svd(states, inputs, false, false, true);
    svd.decompose(B_ddrm.copy());
    double[] sv = svd.getSingularValues();
    if (sv[sv.length - 1] < 1e-12) {
      throw new RuntimeException("B singular");
    }
    double cond = sv[0] / sv[sv.length - 1];
    if (cond > 1e10) {
      throw new RuntimeException("B ill-conditioned");
    }

    // Calculate A * r_k, store the result in temp
    CommonOps_DDRM.mult(A_ddrm, r_ddrm, temp_ddrm);

    // Calculate r_k+1 - (A * r_k), store the result back into temp
    CommonOps_DDRM.subtract(nextR_ddrm, temp_ddrm, temp_ddrm);

    // Solve for u_ff: B * u_ff = temp
    if (!m_solver.setA(B_ddrm)) {
      throw new RuntimeException("Unable to set solver's A matrix");
    }
    m_solver.solve(temp_ddrm, uff_ddrm);

    // Update the current reference
    System.arraycopy(
        nextR.getDDRM().getData(), 0, this.m_r.getDDRM().getData(), 0, nextR.getNumElements());
  }

  public SimpleMatrix getUff() {
    return m_uff;
  }

  public SimpleMatrix getR() {
    return m_r;
  }
}
