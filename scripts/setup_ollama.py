"""
Ollama Setup Script

Automated setup and configuration of Ollama with Llama 3.1 model
for enhanced explainable AI capabilities.
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path


class OllamaSetup:
    """Automated Ollama setup and configuration"""
    
    def __init__(self, model_name="llama3.1:8b", host="localhost", port=11434):
        self.model_name = model_name
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        
    def check_ollama_installed(self) -> bool:
        """Check if Ollama is installed"""
        try:
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def install_ollama(self) -> bool:
        """Install Ollama using official installation script"""
        print("Installing Ollama...")
        
        try:
            # Download and run installation script
            install_cmd = "curl -fsSL https://ollama.ai/install.sh | sh"
            result = subprocess.run(install_cmd, shell=True, timeout=300)
            
            if result.returncode == 0:
                print("Ollama installed successfully")
                return True
            else:
                print(f"Ollama installation failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("Ollama installation timed out")
            return False
        except Exception as e:
            print(f"Error installing Ollama: {e}")
            return False
    
    def check_ollama_service(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def start_ollama_service(self) -> bool:
        """Start Ollama service"""
        print("Starting Ollama service...")
        
        try:
            # Start Ollama in background
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            # Wait for service to be ready
            for i in range(30):  # Wait up to 30 seconds
                time.sleep(1)
                if self.check_ollama_service():
                    print("Ollama service started successfully")
                    return True
                print(f"Waiting for Ollama service... ({i+1}/30)")
            
            print("Ollama service failed to start within timeout")
            return False
            
        except Exception as e:
            print(f"Error starting Ollama service: {e}")
            return False
    
    def check_model_available(self) -> bool:
        """Check if the required model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                return self.model_name in model_names
            return False
        except requests.exceptions.RequestException:
            return False
    
    def pull_model(self) -> bool:
        """Download the required model"""
        print(f"Downloading model: {self.model_name}")
        
        try:
            # Use subprocess to pull model (provides progress feedback)
            process = subprocess.Popen(['ollama', 'pull', self.model_name],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,
                                     universal_newlines=True)
            
            # Print progress
            for line in process.stdout:
                print(line.rstrip())
            
            process.wait()
            
            if process.returncode == 0:
                print(f"Model {self.model_name} downloaded successfully")
                return True
            else:
                print(f"Model download failed")
                return False
                
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False
    
    def test_ollama_connection(self) -> bool:
        """Test Ollama connection with a simple request"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": "Hello, this is a test.",
                "stream": False
            }
            
            response = requests.post(f"{self.base_url}/api/generate", 
                                   json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'response' in result:
                    print("Ollama connection test successful")
                    return True
            
            print(f"Ollama connection test failed: {response.text}")
            return False
            
        except Exception as e:
            print(f"Ollama connection test error: {e}")
            return False
    
    def setup_complete(self) -> bool:
        """Run complete Ollama setup process"""
        print("=" * 60)
        print("OLLAMA SETUP FOR ENHANCED XAI")
        print("=" * 60)
        
        # Step 1: Check if Ollama is installed
        if not self.check_ollama_installed():
            print("Ollama is not installed. Installing...")
            if not self.install_ollama():
                print("Failed to install Ollama")
                return False
        else:
            print("Ollama is already installed")
        
        # Step 2: Check if service is running
        if not self.check_ollama_service():
            print("Ollama service is not running. Starting...")
            if not self.start_ollama_service():
                print("Failed to start Ollama service")
                return False
        else:
            print("Ollama service is already running")
        
        # Step 3: Check if model is available
        if not self.check_model_available():
            print(f"Model {self.model_name} is not available. Downloading...")
            if not self.pull_model():
                print("Failed to download model")
                return False
        else:
            print(f"Model {self.model_name} is already available")
        
        # Step 4: Test connection
        if not self.test_ollama_connection():
            print("Ollama connection test failed")
            return False
        
        print("=" * 60)
        print("OLLAMA SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Service URL: {self.base_url}")
        print(f"Model: {self.model_name}")
        print("Enhanced XAI is now ready to use!")
        
        return True
    
    def create_service_file(self) -> bool:
        """Create systemd service file for Ollama (Linux)"""
        if os.name != 'nt':  # Not Windows
            service_content = f"""[Unit]
Description=Ollama Service
After=network.target

[Service]
Type=simple
User=ollama
ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
"""
            
            service_path = "/etc/systemd/system/ollama.service"
            
            try:
                with open(service_path, 'w') as f:
                    f.write(service_content)
                
                # Enable and start service
                subprocess.run(['systemctl', 'daemon-reload'], check=True)
                subprocess.run(['systemctl', 'enable', 'ollama'], check=True)
                subprocess.run(['systemctl', 'start', 'ollama'], check=True)
                
                print("Systemd service created and started")
                return True
                
            except Exception as e:
                print(f"Failed to create systemd service: {e}")
                return False
        else:
            print("Systemd service creation skipped (not Linux)")
            return True
    
    def get_status(self) -> dict:
        """Get current Ollama status"""
        status = {
            'installed': self.check_ollama_installed(),
            'service_running': self.check_ollama_service(),
            'model_available': self.check_model_available(),
            'connection_test': False
        }
        
        if status['service_running']:
            status['connection_test'] = self.test_ollama_connection()
        
        return status


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup Ollama for Enhanced XAI')
    parser.add_argument('--model', default='llama3.1:8b', help='Model to setup')
    parser.add_argument('--host', default='localhost', help='Ollama host')
    parser.add_argument('--port', type=int, default=11434, help='Ollama port')
    parser.add_argument('--status', action='store_true', help='Check current status')
    parser.add_argument('--service', action='store_true', help='Create systemd service')
    
    args = parser.parse_args()
    
    setup = OllamaSetup(args.model, args.host, args.port)
    
    if args.status:
        print("Checking Ollama status...")
        status = setup.get_status()
        
        print("\nCurrent Status:")
        print(f"  Installed: {'Yes' if status['installed'] else 'No'}")
        print(f"  Service Running: {'Yes' if status['service_running'] else 'No'}")
        print(f"  Model Available: {'Yes' if status['model_available'] else 'No'}")
        print(f"  Connection Test: {'Yes' if status['connection_test'] else 'No'}")
        
        overall_status = all(status.values())
        print(f"\nOverall Status: {'Ready' if overall_status else 'Not Ready'}")
        
    elif args.service:
        setup.create_service_file()
    
    else:
        success = setup.setup_complete()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
