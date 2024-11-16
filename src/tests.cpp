#include <iostream>
#include <vector>
#include <stdexcept>
#include "matrix.hpp"

void testMatrixClass() {
    try {
        // Test 1: Constructor and basic accessors
        Matrix mat1(3, 3); // 3x3 matrix initialized to 0
        if (mat1.rowSize() != 3 || mat1.colSize() != 3 || mat1.at(0, 0) != 0) {
            std::cerr << "Test 1 Failed: Constructor or size accessors incorrect." << std::endl;
            return;
        }
        std::cout << "Test 1 Passed." << std::endl;

        // Test 2: Setting and getting elements
        mat1.set(0, 0, 5.0);
        if (mat1.at(0, 0) != 5.0) {
            std::cerr << "Test 2 Failed: Element access or modification incorrect." << std::endl;
            return;
        }
        std::cout << "Test 2 Passed." << std::endl;

        // Test 3: getRow
        mat1.set(0, 1, 3.0);
        std::vector<float> row = mat1.getRow(0);
        if (row[0] != 5.0 || row[1] != 3.0) {
            std::cerr << "Test 3 Failed: getRow function incorrect." << std::endl;
            return;
        }
        std::cout << "Test 3 Passed." << std::endl;

        // Test 4: Addition operator
        Matrix mat2(3, 3);
        mat2.set(0, 0, 1.0);
        Matrix mat3 = mat1 + mat2;
        if (mat3.at(0, 0) != 6.0 || mat3.at(0, 1) != 3.0) {
            std::cerr << "Test 4 Failed: Addition operator incorrect." << std::endl;
            return;
        }
        std::cout << "Test 4 Passed." << std::endl;

        // Test 5: Subtraction operator
        mat3 = mat1 - mat2;
        if (mat3.at(0, 0) != 4.0) {
            std::cerr << "Test 5 Failed: Subtraction operator incorrect." << std::endl;
            return;
        }
        std::cout << "Test 5 Passed." << std::endl;

        // Test 6: Scalar multiplication
        Matrix mat4 = mat1 * 2.0;
        if (mat4.at(0, 0) != 10.0 || mat4.at(0, 1) != 6.0) {
            std::cerr << "Test 6 Failed: Scalar multiplication incorrect." << std::endl;
            return;
        }
        std::cout << "Test 6 Passed." << std::endl;

        // Test 7: Matrix multiplication
        Matrix mat5(3, 2);
        mat5.set(0, 0, 2.0);
        mat5.set(1, 1, 3.0);
        Matrix mat6 = mat1 * mat5; // Should throw an error due to dimension mismatch
        std::cerr << "Test 7 Failed: Dimension mismatch not detected." << std::endl;
        return;
    } catch (const std::invalid_argument& e) {
        std::cout << "Test 7 Passed: " << e.what() << std::endl;
    }

    // Test 8: Matrix multiplication
    Matrix mat7(2, 3);
    mat7.set(0, 0, 1.0);
    mat7.set(0, 1, 2.0);
    mat7.set(0, 2, 3.0);
    mat7.set(1, 0, 4.0);
    mat7.set(1, 1, 5.0);
    mat7.set(1, 2, 6.0);

    Matrix mat8(3, 2);
    mat8.set(0, 0, 7.0);
    mat8.set(0, 1, 8.0);
    mat8.set(1, 0, 9.0);
    mat8.set(1, 1, 10.0);
    mat8.set(2, 0, 11.0);
    mat8.set(2, 1, 12.0);

    Matrix result = mat7 * mat8;
    if (result.rowSize() != 2 || result.colSize() != 2) {
        std::cerr << "Test 8 Failed: Incorrect dimensions of result matrix." << std::endl;
        return;
    }

    if (result.at(0, 0) != 58.0 || result.at(0, 1) != 64.0 ||
        result.at(1, 0) != 139.0 || result.at(1, 1) != 154.0) {
        std::cerr << "Test 8 Failed: Incorrect values in result matrix." << std::endl;
        return;
    }

    std::cout << "Test 8 Passed." << std::endl;

    // Test 9: Transpose
    Matrix mat9(2, 3);
    mat9.set(0, 0, 1.0);
    mat9.set(0, 1, 2.0);
    mat9.set(0, 2, 3.0);
    mat9.set(1, 0, 4.0);
    mat9.set(1, 1, 5.0);
    mat9.set(1, 2, 6.0);

    Matrix transposed = mat9.transpose();
    if (transposed.rowSize() != 3 || transposed.colSize() != 2 || transposed.at(0, 1) != 4.0) {
        std::cerr << "Test 9 Failed: Transpose function incorrect." << std::endl;
        return;
    }
    std::cout << "Test 9 Passed." << std::endl;

    // Test 10: SumRowsToOne
    Matrix mat10(2, 3);
    mat10.set(0, 0, 1.0);
    mat10.set(0, 1, 2.0);
    mat10.set(0, 2, 3.0);
    mat10.set(1, 0, 4.0);
    mat10.set(1, 1, 5.0);
    mat10.set(1, 2, 6.0);

    Matrix sumRows = mat10.SumRowsToOne();
    if (sumRows.rowSize() != 1 || sumRows.colSize() != 3 || sumRows.at(0, 2) != 9.0) {
        std::cerr << "Test 10 Failed: SumRowsToOne function incorrect." << std::endl;
        return;
    }
    std::cout << "Test 10 Passed." << std::endl;

    std::cout << "All Tests Passed." << std::endl;
}

int main() {
    testMatrixClass();
    return 0;
}
