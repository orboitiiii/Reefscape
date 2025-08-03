package org.slsh.frc9427.lib.controls.util;

import edu.wpi.first.math.jni.DAREJNI;
import edu.wpi.first.math.jni.EigenJNI;
import org.ejml.data.Complex_F64;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.factory.DecompositionFactory_DDRM;
import org.ejml.dense.row.factory.LinearSolverFactory_DDRM;
import org.ejml.interfaces.decomposition.CholeskyDecomposition_F64;
import org.ejml.interfaces.decomposition.EigenDecomposition_F64;
import org.ejml.interfaces.decomposition.SingularValueDecomposition_F64;
import org.ejml.interfaces.linsol.LinearSolverDense;
import org.ejml.simple.SimpleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A memory-allocation-free utility class for state-space computations. */
public final class StateSpaceUtil {

  private static final Logger LOGGER = LoggerFactory.getLogger(StateSpaceUtil.class);

  private static final int MAX_SIZE = 1000; // Limit to prevent OOM

  // ThreadLocal pools for temporary matrices
  private static final ThreadLocal<DMatrixRMaj> TEMP_P_COPY =
      ThreadLocal.withInitial(() -> new DMatrixRMaj(20, 20));
  private static final ThreadLocal<DMatrixRMaj> TEMP_DIAG =
      ThreadLocal.withInitial(() -> new DMatrixRMaj(20, 1));
  private static final ThreadLocal<DMatrixRMaj> TEMP_A_CL =
      ThreadLocal.withInitial(() -> new DMatrixRMaj(20, 20));
  private static final ThreadLocal<DMatrixRMaj> TEMP_Q_COPY =
      ThreadLocal.withInitial(() -> new DMatrixRMaj(20, 20));
  private static final ThreadLocal<DMatrixRMaj> TEMP_R_COPY =
      ThreadLocal.withInitial(() -> new DMatrixRMaj(4, 4));
  private static final ThreadLocal<DMatrixRMaj> TEMP_BSBR_COPY =
      ThreadLocal.withInitial(() -> new DMatrixRMaj(4, 4));

  // Static block to pre-init pools and catch OOM
  static {
    try {
      TEMP_P_COPY.get();
      TEMP_DIAG.get();
      TEMP_A_CL.get();
      TEMP_Q_COPY.get();
      TEMP_R_COPY.get();
      TEMP_BSBR_COPY.get();
    } catch (OutOfMemoryError e) {
      LOGGER.error("Failed to initialize temp pools due to OOM: {}", e.getMessage());
      throw new RuntimeException("Critical OOM in temp pool init", e);
    }
  }

  private StateSpaceUtil() {
    // Utility class
  }

  /**
   * Discretizes the continuous system matrices A and B in a memory-allocation-free manner.
   *
   * @param contA Continuous A (states x states)
   * @param contB Continuous B (states x inputs)
   * @param dtSeconds Time step
   * @param discA Discretized A
   * @param discB Discretized B
   * @param M_temp Temp matrix ((states+inputs) x (states+inputs))
   * @param phi_temp Temp for exp result
   */
  public static void discretizeAB(
      SimpleMatrix contA,
      SimpleMatrix contB,
      double dtSeconds,
      SimpleMatrix discA,
      SimpleMatrix discB,
      SimpleMatrix M_temp,
      SimpleMatrix phi_temp) {

    int states = contA.getNumRows();
    int inputs = contB.getNumCols();
    int M_size = states + inputs;

    // Runtime size check
    if (contA.getNumRows() != states
        || contA.getNumCols() != states
        || contB.getNumRows() != states
        || contB.getNumCols() != inputs
        || discA.getNumRows() != states
        || discA.getNumCols() != states
        || discB.getNumRows() != states
        || discB.getNumCols() != inputs
        || M_temp.getNumRows() != M_size
        || M_temp.getNumCols() != M_size
        || phi_temp.getNumRows() != M_size
        || phi_temp.getNumCols() != M_size) {
      throw new IllegalArgumentException("Dimensions mismatch in discretizeAB_NoGC");
    }

    if (dtSeconds > 1.0) {
      LOGGER.warn("Large dtSeconds ({}) may cause numerical instability.", dtSeconds);
    }

    // Check if A is singular
    double detA = CommonOps_DDRM.det(contA.getDDRM());
    if (Math.abs(detA) < 1e-10) {
      LOGGER.debug("Matrix A is singular (expected for double-integrator systems).");
    }

    // M = [A  B]
    //     [0  0]
    M_temp.zero();
    CommonOps_DDRM.insert(contA.getDDRM(), M_temp.getDDRM(), 0, 0);
    CommonOps_DDRM.insert(contB.getDDRM(), M_temp.getDDRM(), 0, states);

    // M = M * dt (in-place scaling)
    CommonOps_DDRM.scale(dtSeconds, M_temp.getDDRM());

    // phi = exp(M)
    EigenJNI.exp(M_temp.getDDRM().getData(), M_size, phi_temp.getDDRM().getData());

    // Extract A_d and B_d from phi
    CommonOps_DDRM.extract(phi_temp.getDDRM(), 0, states, 0, states, discA.getDDRM(), 0, 0);
    CommonOps_DDRM.extract(phi_temp.getDDRM(), 0, states, states, M_size, discB.getDDRM(), 0, 0);
  }

  /**
   * @param discA Discretized A
   * @param discB Discretized B
   * @param Q State cost
   * @param R Input cost
   * @param K_out Result K
   * @param temp_S Temp S
   * @param temp_S_B Temp S*B
   * @param temp_B_S_B Temp B^T*S*B
   * @param temp_B_S_B_plus_R Temp (B^T*S*B + R)
   * @param temp_B_S_A Temp B^T*S*A
   * @param temp_BTS Temp B^T*S (inputs x states)
   */
  public static void calculateK(
      SimpleMatrix discA,
      SimpleMatrix discB,
      SimpleMatrix Q,
      SimpleMatrix R,
      SimpleMatrix K_out,
      SimpleMatrix temp_S,
      SimpleMatrix temp_S_B,
      SimpleMatrix temp_B_S_B,
      SimpleMatrix temp_B_S_B_plus_R,
      SimpleMatrix temp_B_S_A,
      SimpleMatrix temp_BTS) {

    // Runtime size check
    int states = discA.getNumRows();
    int inputs = discB.getNumCols();
    if (discA.getNumCols() != states
        || discB.getNumRows() != states
        || Q.getNumRows() != states
        || Q.getNumCols() != states
        || R.getNumRows() != inputs
        || R.getNumCols() != inputs
        || K_out.getNumRows() != inputs
        || K_out.getNumCols() != states
        || temp_S.getNumRows() != states
        || temp_S.getNumCols() != states
        || temp_S_B.getNumRows() != states
        || temp_S_B.getNumCols() != inputs
        || temp_B_S_B.getNumRows() != inputs
        || temp_B_S_B.getNumCols() != inputs
        || temp_B_S_B_plus_R.getNumRows() != inputs
        || temp_B_S_B_plus_R.getNumCols() != inputs
        || temp_B_S_A.getNumRows() != inputs
        || temp_B_S_A.getNumCols() != states
        || temp_BTS.getNumRows() != inputs
        || temp_BTS.getNumCols() != states) {
      throw new IllegalArgumentException("Dimensions mismatch in calculateK_NoGC");
    }

    resizeAndCopy(Q.getDDRM(), TEMP_Q_COPY.get(), states, states, "Q");
    SingularValueDecomposition_F64<DMatrixRMaj> svdQ =
        DecompositionFactory_DDRM.svd(states, states, false, false, true);
    svdQ.decompose(TEMP_Q_COPY.get());
    double[] svQ = svdQ.getSingularValues();
    if (svQ[svQ.length - 1] < 0) {
      throw new IllegalArgumentException(
          "Q is not positive semi-definite (min singular value < 0)");
    }

    resizeAndCopy(R.getDDRM(), TEMP_R_COPY.get(), inputs, inputs, "R");
    SingularValueDecomposition_F64<DMatrixRMaj> svdR =
        DecompositionFactory_DDRM.svd(inputs, inputs, false, false, true);
    svdR.decompose(TEMP_R_COPY.get());
    double[] svR = svdR.getSingularValues();
    if (svR[svR.length - 1] <= 0) {
      throw new IllegalArgumentException("R is not positive definite (min singular value <= 0)");
    }

    // Solve DARE for S
    DAREJNI.dareABQR(
        discA.getDDRM().getData(),
        discB.getDDRM().getData(),
        Q.getDDRM().getData(),
        R.getDDRM().getData(),
        states,
        inputs,
        temp_S.getDDRM().getData());

    DMatrixRMaj K_ddrm = K_out.getDDRM();
    DMatrixRMaj S_ddrm = temp_S.getDDRM();
    DMatrixRMaj A_ddrm = discA.getDDRM();
    DMatrixRMaj B_ddrm = discB.getDDRM();
    DMatrixRMaj R_ddrm = R.getDDRM();

    DMatrixRMaj S_B_ddrm = temp_S_B.getDDRM();
    DMatrixRMaj B_S_B_ddrm = temp_B_S_B.getDDRM();
    DMatrixRMaj B_S_B_plus_R_ddrm = temp_B_S_B_plus_R.getDDRM();
    DMatrixRMaj B_S_A_ddrm = temp_B_S_A.getDDRM();
    DMatrixRMaj BTS_ddrm = temp_BTS.getDDRM();

    resizeAndCopy(S_ddrm, TEMP_P_COPY.get(), states, states, "P");
    CholeskyDecomposition_F64<DMatrixRMaj> cholesky = DecompositionFactory_DDRM.chol(states, true);
    if (!cholesky.decompose(TEMP_P_COPY.get())) {
      throw new RuntimeException("P is not positive definite (Cholesky decomposition failed)");
    }
    DMatrixRMaj L = cholesky.getT(null);
    DMatrixRMaj diag = TEMP_DIAG.get();
    diag.reshape(states, 1);
    if (diag.data.length < states) {
      if (states > MAX_SIZE)
        throw new IllegalArgumentException("diag size exceeds max_size " + MAX_SIZE);
      diag.data = new double[states * 2];
      LOGGER.warn("Temp diag grew to size {}", diag.data.length);
    }
    CommonOps_DDRM.extractDiag(L, diag);

    for (int i = 0; i < states; i++) {
      double d = diag.get(i);
      if (d <= -1e-10) {
        throw new RuntimeException(
            "P is not positive definite (Cholesky diagonal <= 0 at index " + i + ")");
      } else if (d < 1e-10) {
        LOGGER.warn("P near singular (Cholesky diagonal ~0 at index {})", i);
      }
    }

    //  B^T S A caculate：use (B^T S) A
    // B^T S -> temp_BTS (inputs × states)
    CommonOps_DDRM.multTransA(B_ddrm, S_ddrm, BTS_ddrm);
    // (B^T S) A -> B_S_A (inputs × states)
    CommonOps_DDRM.mult(BTS_ddrm, A_ddrm, B_S_A_ddrm);

    // B^T S B + R = B^T * (S * B) + R
    CommonOps_DDRM.mult(S_ddrm, B_ddrm, S_B_ddrm);
    CommonOps_DDRM.multTransA(B_ddrm, S_B_ddrm, B_S_B_ddrm);
    CommonOps_DDRM.add(B_S_B_ddrm, R_ddrm, B_S_B_plus_R_ddrm);

    // Check condition number using SVD
    SingularValueDecomposition_F64<DMatrixRMaj> svd =
        DecompositionFactory_DDRM.svd(inputs, inputs, false, false, true);

    resizeAndCopy(B_S_B_plus_R_ddrm, TEMP_BSBR_COPY.get(), inputs, inputs, "BSBR");
    svd.decompose(TEMP_BSBR_COPY.get());
    double[] singularValues = svd.getSingularValues();
    if (singularValues[singularValues.length - 1] < 1e-12) {
      throw new RuntimeException("Matrix singular; condition number infinite");
    }
    double cond = singularValues[0] / singularValues[singularValues.length - 1];
    if (cond > 1e10 || Double.isInfinite(cond) || Double.isNaN(cond)) {
      throw new RuntimeException("Matrix ill-conditioned in K gain calculation");
    }

    // Solve K = (B^T S B + R)^{-1} (B^T S A)
    LinearSolverDense<DMatrixRMaj> solver = LinearSolverFactory_DDRM.pseudoInverse(true);
    if (!solver.setA(B_S_B_plus_R_ddrm)) {
      throw new RuntimeException("Unable to set solver's A matrix (K gain calculation)");
    }
    solver.solve(B_S_A_ddrm, K_ddrm);

    DMatrixRMaj A_cl = TEMP_A_CL.get();
    A_cl.reshape(states, states);
    int numElements = states * states;
    if (A_cl.data.length < numElements) {
      if (numElements > MAX_SIZE)
        throw new IllegalArgumentException("A_cl numElements exceeds max_size " + MAX_SIZE);
      A_cl.data = new double[numElements * 2];
      LOGGER.warn("Temp A_cl grew to size {}", A_cl.data.length);
    }
    CommonOps_DDRM.mult(B_ddrm, K_ddrm, temp_B_S_B.getDDRM()); // Temp as B K
    CommonOps_DDRM.subtract(A_ddrm, temp_B_S_B.getDDRM(), A_cl);

    Complex_F64[] eigenvalues = new Complex_F64[states];
    EigenDecomposition_F64<DMatrixRMaj> eigDecomp = DecompositionFactory_DDRM.eig(states, false);
    eigDecomp.decompose(A_cl);
    for (int i = 0; i < states; i++) {
      eigenvalues[i] = eigDecomp.getEigenvalue(i);
      double mag = eigenvalues[i].getMagnitude();
      if (mag >= 1.0) {
        throw new RuntimeException(
            "Closed-loop not stable (eigenvalue magnitude >= 1 at index " + i + ")");
      }
    }
  }

  // Utility to resize and copy matrix data safely
  private static void resizeAndCopy(
      DMatrixRMaj src, DMatrixRMaj dst, int rows, int cols, String name) {
    dst.reshape(rows, cols);
    int numElements = rows * cols;
    if (dst.data.length < numElements) {
      if (numElements > MAX_SIZE) {
        throw new IllegalArgumentException(name + " numElements exceeds max_size " + MAX_SIZE);
      }
      dst.data = new double[numElements * 2];
      LOGGER.warn("Temp {} copy grew to size {}", name, dst.data.length);
    }
    if (src.data.length != numElements) {
      throw new IllegalStateException(
          name
              + " source data length mismatch (expected "
              + numElements
              + ", got "
              + src.data.length
              + ")");
    }
    System.arraycopy(src.data, 0, dst.data, 0, numElements);
  }

  /**
   * Creates diagonal state cost matrix Q based on Bryson's rule or direct weights. Ensures Q is
   * positive semi-definite.
   *
   * @param weights Either maximum allowable state deviations (Bryson's rule: Q_ii = 1 /
   *     weights[i]^2) or direct diagonal weights (Q_ii = weights[i]). If a single value is
   *     provided, it is applied to all states.
   * @param useBrysonRule If true, apply Bryson's rule; if false, use weights directly.
   * @param size Number of states (required if weights is a single value).
   * @return A diagonal Q matrix (size x size).
   */
  public static SimpleMatrix makeStateCostMatrix(
      double[] weights, boolean useBrysonRule, int size) {
    if (weights == null || weights.length == 0) {
      throw new IllegalArgumentException("Weights array cannot be null or empty");
    }
    if (size <= 0) {
      throw new IllegalArgumentException("Size must be positive");
    }
    int actualSize = weights.length == 1 ? size : weights.length;
    if (weights.length != 1 && weights.length != size) {
      throw new IllegalArgumentException(
          "Weights length (" + weights.length + ") must be 1 or equal to size (" + size + ")");
    }

    SimpleMatrix Q = new SimpleMatrix(actualSize, actualSize);
    for (int i = 0; i < actualSize; i++) {
      double val = weights.length == 1 ? weights[0] : weights[i];
      if (val <= 0) {
        throw new IllegalArgumentException("Weight at index " + i + " must be positive");
      }
      double qValue = useBrysonRule ? 1.0 / (val * val) : val;
      if (qValue > 1e6) {
        LOGGER.warn(
            "Q[{}] = {} is large; consider adjusting weight to avoid numerical instability",
            i,
            qValue);
      }
      Q.set(i, i, qValue);
    }

    return Q;
  }

  /**
   * Creates diagonal input cost matrix R based on Bryson's rule or direct weights. Ensures R is
   * positive definite.
   *
   * @param weights Either maximum allowable input values (Bryson's rule: R_jj = 1 / weights[j]^2)
   *     or direct diagonal weights (R_jj = weights[j]). If a single value is provided, it is
   *     applied to all inputs.
   * @param useBrysonRule If true, apply Bryson's rule; if false, use weights directly.
   * @param size Number of inputs (required if weights is a single value).
   * @return A diagonal R matrix (size x size).
   */
  public static SimpleMatrix makeInputCostMatrix(
      double[] weights, boolean useBrysonRule, int size) {
    if (weights == null || weights.length == 0) {
      throw new IllegalArgumentException("Weights array cannot be null or empty");
    }
    if (size <= 0) {
      throw new IllegalArgumentException("Size must be positive");
    }
    int actualSize = weights.length == 1 ? size : weights.length;
    if (weights.length != 1 && weights.length != size) {
      throw new IllegalArgumentException(
          "Weights length (" + weights.length + ") must be 1 or equal to size (" + size + ")");
    }

    SimpleMatrix R = new SimpleMatrix(actualSize, actualSize);
    for (int i = 0; i < actualSize; i++) {
      double val = weights.length == 1 ? weights[0] : weights[i];
      if (val <= 0) {
        throw new IllegalArgumentException("Weight at index " + i + " must be positive");
      }
      double rValue = useBrysonRule ? 1.0 / (val * val) : val;
      if (rValue > 1e6 || (!useBrysonRule && rValue <= 1e-6)) {
        LOGGER.warn(
            "R[{}] = {} is extreme; consider adjusting weight to avoid numerical instability",
            i,
            rValue);
      }
      R.set(i, i, rValue);
    }

    return R;
  }
}
