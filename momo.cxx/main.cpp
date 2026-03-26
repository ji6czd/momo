#include <iostream>
#include <string>
#include "func.h"

int main(int argc, char* argv[]) {
    std::cout << "Hello, World!" << std::endl;
    int result = add(3, 4);
    std::cout << "3 + 4 = " << result << std::endl;
    return 0;
}