Language: Julia version 1.6.

- This projects contains 3 files:
1. MIMSSVMClassifier.jl: The implementation of Multi-Instance Multi-Shape Support Vector Machine (MIMSSVM). 
2. mimssvmclassifier_test.jl: The test code for MIMMSVM. This tests (1) whether we update each variable where the derivative is zero, and (2) whether the objective is decreasing at each update.
3. Project.toml: This specifies the required packages to run the code.

- How to run:
1. Creates a Julia session.
2. Creates a project and install the required packages.
3. Run 'include("path-to-mimssvmclassifier_test.jl")'.