#include <vector>

class Matrix
{
private:
    int rows;
    int cols;
    std::vector<std::vector<float>> data;

public:
    Matrix(int rows, int cols = 0);
    Matrix(std::vector<std::vector<float>> data);

    Matrix &operator+=(const Matrix &other);
    Matrix operator-(const Matrix &other);
    Matrix operator+(const Matrix &other);
    Matrix operator*(const Matrix &other);
    Matrix operator*(float value);
    Matrix transpose() const;

    float at(int row, int col) const;
    void set(int row, int col, float value);
    std::vector<float> getRow(int row);
};

Matrix::Matrix(int rows, int cols)
{
    this->rows = rows;
    this->cols = cols;
    std::vector<std::vector<float>> d;
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            d[i][j] = 0;
        }
    }

    this->data = d;
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

std::vector<float> Matrix::getRow(int row)
{
    return data[row];
}

Matrix& Matrix::operator+=(const Matrix &other){
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] += other.data[i][j];
        }
    }
    return *this;
}

Matrix Matrix::operator+(const Matrix &other){
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] += other.data[i][j];
        }
    }
    return *this;
}

Matrix Matrix::operator-(const Matrix &other){
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] -= other.data[i][j];
        }
    }
    return *this;
}

Matrix Matrix::operator*(float value){
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] *= value;
        }
    }
    return *this;
}

Matrix Matrix::operator*(const Matrix &other){
Matrix m(this->rows, other.cols);

  for (int i = 0; i < m.rows; i++) {
    for (int j = 0; j < m.cols; j++) {
      float value = 0;
      for (int k = 0; k < cols; k++) {
        value += this->at(i, k) * other.at(k, j);
      }
      m.set(i, j, value);
    }
  }

  return m;
}

Matrix Matrix::transpose() const{
  Matrix m(cols, rows);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      m.set(j, i, at(i, j));
    }
  }
  return m;
}
