#include <vector>

class Matrix
{
private:
    int rows;
    int cols;
    std::vector<std::vector<float>> data;

public:
    Matrix(int rows, int cols);
    Matrix(std::vector<std::vector<float>> data);

    Matrix &operator+=(const Matrix &other);
    Matrix operator+(const Matrix &other) const;
    Matrix &operator-=(const Matrix &other);
    Matrix operator-(const Matrix &other) const;
    Matrix operator*(const Matrix &other) const;
    Matrix operator*(float value) const;
    Matrix transpose() const;
    Matrix SumRowsToOne() const;

    float at(int row, int col) const;
    void set(int row, int col, float value);
    std::vector<float> getRow(int row) const;
    int rowSize() const;
    int colSize() const;
};
