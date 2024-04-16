#include <iostream>
#include <vector>

// Function to calculate the factorial of a non-negative integer
unsigned long long unknown1(int n) {
    unsigned long long factorial = 1;
    for (int i = 1; i <= n; ++i) {
        factorial *= i;
    }
    return factorial;
}

// Function to check if a number is prime
bool unknown2(int num) {
    if (num <= 1) {
        return false;
    }
    for (int i = 2; i * i <= num; ++i) {
        if (num % i == 0) {
            return false;
        }
    }
    return true;
}

// Function to find the maximum element in a vector
int unknown3(const std::vector<int>& vec) {
    if (vec.empty()) {
        std::cerr << "Vector is empty!" << std::endl;
        return -1; // Error code for empty vector
    }
    int max = vec[0];
    for (int num : vec) {
        if (num > max) {
            max = num;
        }
    }
    return max;
}