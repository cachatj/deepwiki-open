#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="/Users/jcachat/code_sandbox/deepwiki-open"

# Use AppleScript to open two new Terminal windows and run commands
osascript <<EOF
tell application "Terminal"
    activate
    
    # Run backend in the first new window after activating conda env
    do script "cd '$SCRIPT_DIR' && conda activate deepwiki && python -m api.main"
    
    # Run frontend in the second new window after activating conda env
    do script "cd '$SCRIPT_DIR' && conda activate deepwiki && npm run dev"
end tell
EOF

echo "DeepWiki backend and frontend launched in separate Terminal windows (with conda env activated)."