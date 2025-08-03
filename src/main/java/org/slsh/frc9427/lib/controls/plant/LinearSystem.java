package org.slsh.frc9427.lib.controls.plant;

import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.simple.SimpleMatrix;

/**
 * Represents a Linear Time-Invariant (LTI) system.
 *
 * <p>This class directly uses EJML's SimpleMatrix and does not include generics for maximum
 * performance. It serves only as a data container, holding the system's A, B, C, and D matrices.
 */
public class LinearSystem {
  private final SimpleMatrix m_A;
  private final SimpleMatrix m_B;
  private final SimpleMatrix m_C;
  private final SimpleMatrix m_D;

  private final int m_states;
  private final int m_inputs;
  private final int m_outputs;

  // Pre-allocated temp for calculateX/Y to maintain NoGC
  private final SimpleMatrix m_tempX;
  private final SimpleMatrix m_tempY;

  /**
   * Constructs a new LinearSystem.
   *
   * @param A The system matrix A
   * @param B The input matrix B
   * @param C The output matrix C
   * @param D The feedforward matrix D
   */
  public LinearSystem(SimpleMatrix A, SimpleMatrix B, SimpleMatrix C, SimpleMatrix D) {
    this.m_A = A;
    this.m_B = B;
    this.m_C = C;
    this.m_D = D;

    this.m_states = A.getNumRows();
    this.m_inputs = B.getNumCols();
    this.m_outputs = C.getNumRows();

    // Runtime size check to simulate generics (official uses Nat<Num>)
    if (A.getNumRows() != A.getNumCols()
        || A.getNumRows() != B.getNumRows()
        || A.getNumRows() != C.getNumCols()
        || B.getNumCols() != D.getNumCols()
        || C.getNumRows() != D.getNumRows()) {
      throw new IllegalArgumentException("Matrix dimensions do not match!");
    }

    // Pre-allocate temps for NoGC calculations
    m_tempX = new SimpleMatrix(m_states, 1);
    m_tempY = new SimpleMatrix(m_outputs, 1);
  }

  public SimpleMatrix getA() {
    return m_A;
  }

  public SimpleMatrix getB() {
    return m_B;
  }

  public SimpleMatrix getC() {
    return m_C;
  }

  public SimpleMatrix getD() {
    return m_D;
  }

  public int getNumStates() {
    return m_states;
  }

  public int getNumInputs() {
    return m_inputs;
  }

  public int getNumOutputs() {
    return m_outputs;
  }

  // Added: calculateX from official, NoGC using pre-allocated temp
  public void calculateX(
      SimpleMatrix x,
      SimpleMatrix clampedU,
      double dtSeconds,
      SimpleMatrix discA,
      SimpleMatrix discB,
      SimpleMatrix result) {
    // Runtime check
    if (x.getNumRows() != m_states
        || clampedU.getNumRows() != m_inputs
        || discA.getNumRows() != m_states
        || discB.getNumRows() != m_states
        || result.getNumRows() != m_states) {
      throw new IllegalArgumentException("Dimensions mismatch in calculateX");
    }

    // temp = discA * x
    CommonOps_DDRM.mult(discA.getDDRM(), x.getDDRM(), m_tempX.getDDRM());
    // temp += discB * clampedU
    CommonOps_DDRM.multAdd(discB.getDDRM(), clampedU.getDDRM(), m_tempX.getDDRM());
    // Copy to result
    System.arraycopy(m_tempX.getDDRM().getData(), 0, result.getDDRM().getData(), 0, m_states);
  }

  // Added: calculateY from official, NoGC using pre-allocated temp
  public void calculateY(SimpleMatrix x, SimpleMatrix clampedU, SimpleMatrix result) {
    // Runtime check
    if (x.getNumRows() != m_states
        || clampedU.getNumRows() != m_inputs
        || result.getNumRows() != m_outputs) {
      throw new IllegalArgumentException("Dimensions mismatch in calculateY");
    }

    // temp = C * x
    CommonOps_DDRM.mult(m_C.getDDRM(), x.getDDRM(), m_tempY.getDDRM());
    // temp += D * clampedU
    CommonOps_DDRM.multAdd(m_D.getDDRM(), clampedU.getDDRM(), m_tempY.getDDRM());
    // Copy to result
    System.arraycopy(m_tempY.getDDRM().getData(), 0, result.getDDRM().getData(), 0, m_outputs);
  }

  // Added: slice from official, using arraycopy for NoGC sub-matrix extraction
  public LinearSystem slice(int... outputIndices) {
    // Runtime checks (simulate generics)
    for (int index : outputIndices) {
      if (index < 0 || index >= m_outputs) {
        throw new IllegalArgumentException("Output index out of range: " + index);
      }
    }
    if (outputIndices.length >= m_outputs || outputIndices.length == 0) {
      throw new IllegalArgumentException("Invalid number of output indices");
    }

    // Check for duplicates
    java.util.Set<Integer> set = new java.util.HashSet<>();
    for (int index : outputIndices) {
      if (!set.add(index)) {
        throw new IllegalArgumentException("Duplicate output indices");
      }
    }

    int newOutputs = outputIndices.length;
    SimpleMatrix newC = new SimpleMatrix(newOutputs, m_states);
    SimpleMatrix newD = new SimpleMatrix(newOutputs, m_inputs);

    // Extract rows using arraycopy for NoGC
    for (int i = 0; i < newOutputs; i++) {
      int srcRow = outputIndices[i];
      System.arraycopy(
          m_C.getDDRM().getData(),
          srcRow * m_states,
          newC.getDDRM().getData(),
          i * m_states,
          m_states);
      System.arraycopy(
          m_D.getDDRM().getData(),
          srcRow * m_inputs,
          newD.getDDRM().getData(),
          i * m_inputs,
          m_inputs);
    }

    return new LinearSystem(m_A, m_B, newC, newD);
  }
}
