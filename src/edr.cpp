
#include <iostream>
#include <vector>
#include <cmath> // For std::abs
#include <algorithm> // For std::min

double calculateEDR(const std::vector<double>& seq1, const std::vector<double>& seq2, double threshold) {
    size_t len1 = seq1.size();
    size_t len2 = seq2.size();

    // Create a 2D vector to store distances, initialize with zeros
    std::vector<std::vector<double>> dp(len1 + 1, std::vector<double>(len2 + 1, 0));

    // Initialize the first column and first row of the matrix
    for (size_t i = 0; i <= len1; ++i) {
        dp[i][0] = i; // Cost of deleting elements from seq1
    }
    for (size_t j = 0; j <= len2; ++j) {
        dp[0][j] = j; // Cost of inserting elements into seq1
    }

    // Fill the dp matrix
    for (size_t i = 1; i <= len1; ++i) {
        for (size_t j = 1; j <= len2; ++j) {
            // Check if the current elements are within the threshold
            double cost = (std::abs(seq1[i - 1] - seq2[j - 1]) <= threshold) ? 0 : 1;
            dp[i][j] = std::min({dp[i - 1][j] + 1, // Deletion
                                 dp[i][j - 1] + 1, // Insertion
                                 dp[i - 1][j - 1] + cost}); // Substitution or match
        }
    }

    return dp[len1][len2];
}