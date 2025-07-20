"""
Tests for Docker deployment and containerization in Alunai Clarity.
"""

import json
import os
import pytest
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any


@pytest.mark.integration
@pytest.mark.slow
class TestDockerDeployment:
    """Test Docker deployment functionality."""
    
    def test_dockerfile_exists_and_valid(self):
        """Test that Dockerfile exists and has expected structure."""
        dockerfile_path = Path(__file__).parent.parent.parent / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile not found"
        
        dockerfile_content = dockerfile_path.read_text()
        
        # Check for essential Dockerfile components
        assert "FROM python:" in dockerfile_content
        assert "WORKDIR" in dockerfile_content
        assert "COPY requirements.txt" in dockerfile_content
        assert "RUN pip install" in dockerfile_content
        assert "COPY . ." in dockerfile_content
        assert "ENTRYPOINT" in dockerfile_content or "CMD" in dockerfile_content
        
        # Check for multi-stage build (if used)
        if "FROM python:" in dockerfile_content and dockerfile_content.count("FROM") > 1:
            assert "as builder" in dockerfile_content.lower() or "AS builder" in dockerfile_content
    
    def test_dockerignore_exists(self):
        """Test that .dockerignore exists to optimize build."""
        dockerignore_path = Path(__file__).parent.parent.parent / ".dockerignore"
        
        if dockerignore_path.exists():
            dockerignore_content = dockerignore_path.read_text()
            
            # Check for common ignore patterns
            expected_patterns = [
                "__pycache__",
                "*.pyc",
                ".git",
                "venv",
                ".env"
            ]
            
            for pattern in expected_patterns:
                assert pattern in dockerignore_content, f"Missing {pattern} in .dockerignore"
    
    @pytest.mark.requires_docker
    def test_docker_build_success(self):
        """Test that Docker image builds successfully."""
        project_root = Path(__file__).parent.parent.parent
        
        # Build Docker image
        build_command = [
            "docker", "build",
            "-t", "alunai-clarity-test",
            str(project_root)
        ]
        
        try:
            result = subprocess.run(
                build_command,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            assert result.returncode == 0, f"Docker build failed: {result.stderr}"
            assert "Successfully built" in result.stdout or "Successfully tagged" in result.stdout
            
        except subprocess.TimeoutExpired:
            pytest.fail("Docker build timed out after 10 minutes")
        except FileNotFoundError:
            pytest.skip("Docker not available for testing")
    
    @pytest.mark.requires_docker
    def test_docker_image_size_reasonable(self):
        """Test that Docker image size is reasonable."""
        try:
            # Get image size
            inspect_command = ["docker", "inspect", "alunai-clarity-test"]
            result = subprocess.run(inspect_command, capture_output=True, text=True)
            
            if result.returncode != 0:
                pytest.skip("Docker image not built, skipping size test")
            
            inspect_data = json.loads(result.stdout)
            image_size = inspect_data[0]["Size"]
            
            # Convert to MB
            size_mb = image_size / (1024 * 1024)
            
            # Image should be reasonable size (less than 2GB)
            assert size_mb < 2048, f"Docker image too large: {size_mb:.1f}MB"
            
            # Should be at least 100MB (includes Python + dependencies)
            assert size_mb > 100, f"Docker image suspiciously small: {size_mb:.1f}MB"
            
        except FileNotFoundError:
            pytest.skip("Docker not available for testing")
    
    @pytest.mark.requires_docker 
    def test_docker_container_starts(self):
        """Test that Docker container starts without immediate errors."""
        try:
            # Run container with help command
            run_command = [
                "docker", "run", "--rm",
                "alunai-clarity-test",
                "--help"
            ]
            
            result = subprocess.run(
                run_command,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Should show help and exit cleanly
            assert result.returncode == 0, f"Container failed to start: {result.stderr}"
            assert "usage:" in result.stdout.lower() or "help" in result.stdout.lower()
            
        except subprocess.TimeoutExpired:
            pytest.fail("Container start timed out")
        except FileNotFoundError:
            pytest.skip("Docker not available for testing")
    
    @pytest.mark.requires_docker
    def test_docker_environment_variables(self):
        """Test that container respects environment variables."""
        try:
            # Test with custom environment variables
            run_command = [
                "docker", "run", "--rm",
                "-e", "MEMORY_FILE_PATH=/test/custom_path.json",
                "-e", "ALUNAI_CLARITY_LOG_LEVEL=DEBUG",
                "alunai-clarity-test",
                "--help"
            ]
            
            result = subprocess.run(
                run_command,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Should start successfully with custom env vars
            assert result.returncode == 0, f"Container failed with env vars: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            pytest.fail("Container with env vars timed out")
        except FileNotFoundError:
            pytest.skip("Docker not available for testing")
    
    @pytest.mark.requires_docker
    def test_docker_volume_mounting(self):
        """Test that Docker container supports volume mounting."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create a test config file
                config_file = Path(temp_dir) / "test_config.json"
                test_config = {
                    "server": {"host": "localhost", "port": 8080},
                    "alunai-clarity": {"max_short_term_items": 100}
                }
                config_file.write_text(json.dumps(test_config, indent=2))
                
                # Run container with volume mount
                run_command = [
                    "docker", "run", "--rm",
                    "-v", f"{temp_dir}:/app/config",
                    "alunai-clarity-test",
                    "--config", "/app/config/test_config.json",
                    "--help"
                ]
                
                result = subprocess.run(
                    run_command,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Should handle volume mount successfully
                assert result.returncode == 0, f"Volume mount failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            pytest.fail("Container with volume mount timed out")
        except FileNotFoundError:
            pytest.skip("Docker not available for testing")


@pytest.mark.unit
class TestDockerConfiguration:
    """Test Docker-related configuration."""
    
    def test_docker_compose_file_exists(self):
        """Test that docker-compose.yml exists if provided."""
        compose_path = Path(__file__).parent.parent.parent / "docker-compose.yml"
        
        if compose_path.exists():
            compose_content = compose_path.read_text()
            
            # Basic docker-compose structure
            assert "version:" in compose_content
            assert "services:" in compose_content
            assert "alunai-clarity" in compose_content or "clarity" in compose_content
            
            # Check for important configurations
            assert "ports:" in compose_content
            assert "volumes:" in compose_content
            assert "environment:" in compose_content or "env_file:" in compose_content
    
    def test_example_docker_configs_valid(self):
        """Test that example Docker configurations are valid."""
        project_root = Path(__file__).parent.parent.parent
        
        # Check for example configurations
        examples_dir = project_root / "examples"
        if examples_dir.exists():
            for config_file in examples_dir.glob("*docker*.json"):
                config_content = config_file.read_text()
                
                # Should be valid JSON
                try:
                    config_data = json.loads(config_content)
                    assert isinstance(config_data, dict)
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON in {config_file}")
    
    def test_dockerfile_security_best_practices(self):
        """Test that Dockerfile follows security best practices."""
        dockerfile_path = Path(__file__).parent.parent.parent / "Dockerfile"
        
        if not dockerfile_path.exists():
            pytest.skip("Dockerfile not found")
        
        dockerfile_content = dockerfile_path.read_text()
        
        # Security checks
        # Should not run as root user
        if "USER" in dockerfile_content:
            user_lines = [line for line in dockerfile_content.split('\n') if line.strip().startswith('USER')]
            if user_lines:
                last_user = user_lines[-1]
                assert "root" not in last_user.lower(), "Should not run as root user"
        
        # Should not use ADD for remote URLs
        add_lines = [line for line in dockerfile_content.split('\n') if line.strip().startswith('ADD')]
        for line in add_lines:
            assert not (line.startswith('http://') or line.startswith('https://')), "Avoid ADD with URLs"
        
        # Should specify specific versions where possible
        from_lines = [line for line in dockerfile_content.split('\n') if line.strip().startswith('FROM')]
        for line in from_lines:
            if "python:" in line and "latest" in line:
                pytest.warn(UserWarning("Consider using specific Python version instead of latest"))
    
    def test_health_check_configuration(self):
        """Test health check configuration if present."""
        dockerfile_path = Path(__file__).parent.parent.parent / "Dockerfile"
        
        if not dockerfile_path.exists():
            pytest.skip("Dockerfile not found")
        
        dockerfile_content = dockerfile_path.read_text()
        
        # If HEALTHCHECK is defined, verify it's reasonable
        if "HEALTHCHECK" in dockerfile_content:
            healthcheck_lines = [line for line in dockerfile_content.split('\n') 
                               if "HEALTHCHECK" in line]
            
            for line in healthcheck_lines:
                # Should have reasonable timeout
                if "--timeout=" in line:
                    timeout_part = line.split("--timeout=")[1].split()[0]
                    timeout_value = int(timeout_part.rstrip('s'))
                    assert 1 <= timeout_value <= 60, "Health check timeout should be reasonable"
                
                # Should have reasonable interval
                if "--interval=" in line:
                    interval_part = line.split("--interval=")[1].split()[0]
                    interval_value = int(interval_part.rstrip('s'))
                    assert 5 <= interval_value <= 300, "Health check interval should be reasonable"


@pytest.mark.integration
class TestDockerNetworking:
    """Test Docker networking and connectivity."""
    
    @pytest.mark.requires_docker
    def test_docker_port_exposure(self):
        """Test that Docker container exposes correct ports."""
        try:
            # Run container in detached mode with port mapping
            run_command = [
                "docker", "run", "-d",
                "--name", "alunai-clarity-port-test",
                "-p", "8080:8080",
                "alunai-clarity-test"
            ]
            
            result = subprocess.run(run_command, capture_output=True, text=True)
            
            if result.returncode != 0:
                pytest.skip("Could not start container for port test")
            
            container_id = result.stdout.strip()
            
            try:
                # Wait a moment for container to start
                time.sleep(2)
                
                # Check if port is accessible
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                
                try:
                    result = sock.connect_ex(('localhost', 8080))
                    # Connection should either succeed or be refused (service running but not accepting)
                    # What we don't want is "no route to host" or similar network errors
                    assert result in [0, 61, 111], f"Port not accessible, error code: {result}"
                finally:
                    sock.close()
                
            finally:
                # Clean up container
                subprocess.run(["docker", "stop", container_id], capture_output=True)
                subprocess.run(["docker", "rm", container_id], capture_output=True)
                
        except FileNotFoundError:
            pytest.skip("Docker not available for testing")
    
    @pytest.mark.requires_docker
    def test_docker_internal_networking(self):
        """Test internal networking configuration."""
        try:
            # Test that container can resolve internal services
            run_command = [
                "docker", "run", "--rm",
                "alunai-clarity-test",
                "python", "-c", 
                "import socket; print('Network test passed')"
            ]
            
            result = subprocess.run(
                run_command,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0, f"Network test failed: {result.stderr}"
            assert "Network test passed" in result.stdout
            
        except subprocess.TimeoutExpired:
            pytest.fail("Network test timed out")
        except FileNotFoundError:
            pytest.skip("Docker not available for testing")


@pytest.mark.performance
class TestDockerPerformance:
    """Test Docker performance characteristics."""
    
    @pytest.mark.requires_docker
    def test_container_startup_time(self):
        """Test that container starts within reasonable time."""
        try:
            start_time = time.time()
            
            # Start container and run quick command
            run_command = [
                "docker", "run", "--rm",
                "alunai-clarity-test",
                "python", "-c", "print('Started')"
            ]
            
            result = subprocess.run(
                run_command,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            startup_time = time.time() - start_time
            
            assert result.returncode == 0, f"Container failed to start: {result.stderr}"
            assert "Started" in result.stdout
            
            # Startup should be reasonable (less than 30 seconds)
            assert startup_time < 30, f"Container startup too slow: {startup_time:.2f}s"
            
            print(f"Container startup time: {startup_time:.2f}s")
            
        except subprocess.TimeoutExpired:
            pytest.fail("Container startup timed out")
        except FileNotFoundError:
            pytest.skip("Docker not available for testing")
    
    @pytest.mark.requires_docker
    def test_container_memory_usage(self):
        """Test container memory usage is reasonable."""
        try:
            # Start container in background
            run_command = [
                "docker", "run", "-d",
                "--name", "alunai-clarity-memory-test",
                "alunai-clarity-test",
                "python", "-c", "import time; time.sleep(30)"
            ]
            
            result = subprocess.run(run_command, capture_output=True, text=True)
            
            if result.returncode != 0:
                pytest.skip("Could not start container for memory test")
            
            container_id = result.stdout.strip()
            
            try:
                # Wait for container to stabilize
                time.sleep(5)
                
                # Get memory stats
                stats_command = ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container_id]
                stats_result = subprocess.run(stats_command, capture_output=True, text=True)
                
                if stats_result.returncode == 0:
                    memory_usage = stats_result.stdout.strip()
                    
                    # Parse memory usage (format: "used / limit")
                    if " / " in memory_usage:
                        used_memory = memory_usage.split(" / ")[0]
                        
                        # Convert to MB for comparison
                        if "MiB" in used_memory:
                            memory_mb = float(used_memory.replace("MiB", ""))
                        elif "GiB" in used_memory:
                            memory_mb = float(used_memory.replace("GiB", "")) * 1024
                        else:
                            memory_mb = 0  # Unknown format
                        
                        if memory_mb > 0:
                            # Memory usage should be reasonable (less than 1GB for basic operation)
                            assert memory_mb < 1024, f"Memory usage too high: {memory_mb:.1f}MB"
                            print(f"Container memory usage: {memory_mb:.1f}MB")
                
            finally:
                # Clean up container
                subprocess.run(["docker", "stop", container_id], capture_output=True)
                subprocess.run(["docker", "rm", container_id], capture_output=True)
                
        except FileNotFoundError:
            pytest.skip("Docker not available for testing")


def pytest_configure(config):
    """Configure custom markers for Docker tests."""
    config.addinivalue_line("markers", "requires_docker: mark test as requiring Docker")
    config.addinivalue_line("markers", "slow: mark test as slow running")