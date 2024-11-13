#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <sstream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

double all_values = 0;
double all_values_sq = 0;

vector<int> read_labels(string file_path)
{
    std::ifstream file;
    file.open(file_path);
    std::string line;
    vector<int> labels;
    if (file.is_open())
    {
        int label;
        while (getline(file, line))
        {
            string tempString = "";
            stringstream inputString(line);
            getline(inputString, tempString);
            label = atoi(tempString.c_str());

            labels.push_back(label);

            line = "";
        }
    }
    return labels;
}

vector<vector<float>> read_vectors(string file_path, bool count_values)
{
    std::ifstream file;
    file.open(file_path);
    std::string line;
    vector<vector<float>> vectors;
    if (file.is_open())
    {

        while (getline(file, line))
        {
            vector<float> vtr = {};
            stringstream inputString(line);
            int values = 0;    // store pixel values for mean comutation
            int values_sq = 0; // store squared values for std
            while (inputString.good())
            {
                string substr;
                getline(inputString, substr, ',');
                int pixel = atoi(substr.c_str());
                if (count_values)
                {
                    values += pixel;
                    values_sq += pow(pixel, 2);
                }

                vtr.push_back(pixel);
            }

            vectors.push_back(vtr);
            if (count_values)
            {
                all_values += values / 784; // add mean values for one image to the sum of all means
                all_values_sq += values_sq / 784;
            }

            line = "";
        }
    }
    return vectors;
}

void normalize(vector<vector<float>> &vectors, float mean, float sd)
{
    for (size_t j = 0; j < vectors.size(); j++)
    {
        for (size_t i = 0; i < vectors[j].size(); i++)
        {
            vectors[j][i] = (vectors[j][i] - mean) / sd;
        }
    }
}

int main()
{
    auto start = high_resolution_clock::now();

    // load data

    vector<int> train_labels = read_labels("../data/fashion_mnist_train_labels.csv");
    vector<int> test_labels = read_labels("../data/fashion_mnist_test_labels.csv");
    vector<vector<float>> train_vectors = read_vectors("../data/fashion_mnist_train_vectors.csv", true);
    vector<vector<float>> test_vectors = read_vectors("../data/fashion_mnist_test_vectors.csv", false);

    // normalize data

    int num_of_images = (train_vectors.size());
    float mean = all_values / num_of_images;
    float sd = sqrt(all_values_sq / num_of_images - pow(mean, 2));
    normalize(train_vectors, mean, sd);
    normalize(test_vectors, mean, sd);
    auto stop = high_resolution_clock::now();

    // train a model

    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Time taken by function: "
         << duration.count() << " microseconds" << endl;
}