#include <iostream>
#include <vector>
#include <fstream>
#include "type_def.h"

void saveToCSV(const std::vector<std::vector<idx_t>>& results, const std::string& filename) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
        return;
    }

    for (const auto& row : results) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
}