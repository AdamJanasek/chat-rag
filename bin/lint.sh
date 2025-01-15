RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Starting quality checks...${NC}\n"

error=0

run_check() {
    local cmd=$1
    local name=$2

    echo -e "${YELLOW}Running $name...${NC}"
    if $cmd; then
        echo -e "${GREEN}✓ $name passed${NC}\n"
    else
        echo -e "${RED}✗ $name failed${NC}\n"
        error=1
    fi
}

run_check "isort . --check-only --diff" "isort"

run_check "flake8 ." "flake8"

run_check "mypy ." "mypy"

if [ $error -eq 1 ]; then
    echo -e "${RED}✗ Quality checks failed${NC}"
    exit 1
else
    echo -e "${GREEN}✓ All quality checks passed${NC}"
fi
