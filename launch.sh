#!/bin/bash

# Prepare virtual environment
if [ ! -d "venv" ]; then

    # 1. Create a virtual environment
    python3 -m venv venv
    
    # 2. Activate the virtual environment
    source venv/bin/activate
    
    # 3. Install the required packages
    pip install -r requirements.txt

    # 4.1 optional Flash attention support (easy to install on Linux but much harder on Windows)
    pip install flash-attn==2.7.2.post1

    # 4.2 optional Sage attention support (30% faster, easy to install on Linux but much harder on Windows)
    pip install sageattention==1.0.6 
else
    # Activate the virtual environment
    source venv/bin/activate
fi

# Start the server
python gradio_server.py --profile ${HUNYUAN_PROFILE}
