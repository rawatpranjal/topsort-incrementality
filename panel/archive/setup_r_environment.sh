#!/bin/bash

# Setup script for R environment on macOS
# Fixes libRblas.dylib and other R-related issues

echo "R Environment Setup Script"
echo "=========================="

# Check if R is installed
if ! command -v R &> /dev/null; then
    echo "❌ R is not installed. Please install it from: https://cran.r-project.org/"
    exit 1
fi

echo "✅ R is installed at: $(which R)"
R_VERSION=$(R --version | head -n 1)
echo "   Version: $R_VERSION"

# Set environment variables
export R_HOME="/Library/Frameworks/R.framework/Resources"
export DYLD_LIBRARY_PATH="$R_HOME/lib:$DYLD_LIBRARY_PATH"

echo ""
echo "Setting environment variables..."
echo "R_HOME=$R_HOME"
echo "DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH"

# Check for BLAS library
BLAS_PATH="/Library/Frameworks/R.framework/Versions/Current/Resources/lib/libRblas.dylib"
if [ -f "$BLAS_PATH" ]; then
    echo "✅ BLAS library found at: $BLAS_PATH"
else
    echo "❌ BLAS library not found at expected location"
    echo "   Checking alternative locations..."
    
    # Try to find libRblas.dylib
    FOUND_BLAS=$(find /Library/Frameworks/R.framework -name "libRblas.dylib" 2>/dev/null | head -n 1)
    
    if [ -n "$FOUND_BLAS" ]; then
        echo "✅ Found BLAS at: $FOUND_BLAS"
        echo "   You may need to create a symbolic link:"
        echo "   sudo ln -s $FOUND_BLAS $BLAS_PATH"
    else
        echo "❌ Could not find libRblas.dylib"
        echo "   Please reinstall R from: https://cran.r-project.org/"
    fi
fi

echo ""
echo "Checking R packages..."

# Function to check if R package is installed
check_r_package() {
    package=$1
    R -q -e "if (!require('$package', quietly=TRUE)) quit(status=1)" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ R package '$package' is installed"
        return 0
    else
        echo "❌ R package '$package' is NOT installed"
        return 1
    fi
}

# Check required R packages
REQUIRED_PACKAGES=("arrow" "fixest")
MISSING_PACKAGES=()

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! check_r_package $pkg; then
        MISSING_PACKAGES+=($pkg)
    fi
done

# Install missing packages
if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo ""
    echo "Installing missing R packages..."
    for pkg in "${MISSING_PACKAGES[@]}"; do
        echo "Installing $pkg..."
        R -q -e "install.packages('$pkg', repos='https://cran.rstudio.com/')"
    done
fi

echo ""
echo "Checking Python packages..."

# Check Python packages
pip list | grep -E "rpy2|pandas|numpy" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Python packages detected"
    pip list | grep -E "rpy2|pandas|numpy"
else
    echo "❌ Required Python packages may be missing"
    echo "   Install with: pip install rpy2 pandas numpy"
fi

echo ""
echo "Creating .env file for environment variables..."
cat > .env << EOF
# R Environment Variables
export R_HOME="/Library/Frameworks/R.framework/Resources"
export DYLD_LIBRARY_PATH="\$R_HOME/lib:\$DYLD_LIBRARY_PATH"
EOF

echo "✅ Created .env file"
echo ""
echo "Setup complete!"
echo ""
echo "To use the environment variables in your current session, run:"
echo "  source .env"
echo ""
echo "To test the R integration, run:"
echo "  python fixed_r_integration.py"
echo ""
echo "Or use the pure Python alternative (no R required):"
echo "  python python_fe_logistic.py"