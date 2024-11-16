#include <vector>
#include <stdexcept>
#include "matrix.hpp"

Matrix::Matrix(int rows, int cols)
{
    this->rows = rows;
    this->cols = cols;
    // Initialize a 2D vector with all elements set to 0
    this->data = std::vector<std::vector<float>>(rows, std::vector<float>(cols, 0.0));
}

Matrix::Matrix(std::vector<std::vector<float>> data)
{
    this->rows = data.size();
    this->cols = data[0].size();
    this->data = data;
}

float Matrix::at(int row, int col) const
{
    return data[row][col];
}

void Matrix::set(int row, int col, float value)
{
    data[row][col] = value;
}

std::vector<float> Matrix::getRow(int row) const
{
    return data[row];
}

int Matrix::rowSize() const
{
    return this->rows;
}

int Matrix::colSize() const
{
    return this->cols;
}

Matrix &Matrix::operator+=(const Matrix &other)
{
    if (rows != other.rows || cols != other.cols)
    {
        throw std::invalid_argument("Matrix dimensions must match for addition.");
    }

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            data[i][j] += other.data[i][j];
        }
    }
    return *this;
}

Matrix Matrix::operator+(const Matrix &other) const
{
    if (rows != other.rows || cols != other.cols)
    {
        throw std::invalid_argument("Matrix dimensions must match for addition.");
    }

    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            result.data[i][j] = data[i][j] + other.data[i][j];
        }
    }
    return result;
}

Matrix &Matrix::operator-=(const Matrix &other)
{
    if (rows != other.rows || cols != other.cols)
    {
        throw std::invalid_argument("Matrix dimensions must match for subtraction.");
    }

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            data[i][j] -= other.data[i][j];
        }
    }
    return *this;
}

Matrix Matrix::operator-(const Matrix &other) const
{
    if (rows != other.rows || cols != other.cols)
    {
        throw std::invalid_argument("Matrix dimensions must match for subtraction.");
    }

    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            result.data[i][j] = data[i][j] - other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator*(float value) const
{
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            result.data[i][j] = data[i][j] * value;
        }
    }
    return result;
}

// 'other' matrix is left side of operation
Matrix Matrix::operator*(const Matrix &other) const
{
    if (other.cols != rows)
    {
        throw std::invalid_argument("Matrix dimensions do not allow multiplication.");
    }

    Matrix result(rows, other.cols);

    for (int i = 0; i < result.rows; i++)
    {
        for (int j = 0; j < result.cols; j++)
        {
            float value = 0;
            for (int k = 0; k < cols; k++)
            {
                value += data[i][k] * other.data[k][j];
            }
            result.data[i][j] = value;
        }
    }

    return result;
}

Matrix Matrix::transpose() const
{
    Matrix result(cols, rows);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result.data[j][i] = data[i][j];
        }
    }
    return result;
}

Matrix Matrix::SumRowsToOne() const
{
    Matrix result(1, cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result.data[0][j] += data[i][j];
        }
    }
    return result;
}