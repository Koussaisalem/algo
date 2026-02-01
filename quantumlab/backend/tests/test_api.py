"""
Tests for the FastAPI inference server endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from inference_server import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health and basic endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns API info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestModelEndpoints:
    """Test model-related endpoints"""
    
    def test_list_models(self):
        """Test listing available models"""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
    
    def test_get_elements(self):
        """Test getting element list"""
        response = client.get("/elements")
        assert response.status_code == 200
        data = response.json()
        assert "elements" in data
        assert len(data["elements"]) > 0


class TestLibraryEndpoints:
    """Test molecule library endpoints"""
    
    def test_search_library_empty(self):
        """Test searching empty library"""
        response = client.post("/library/search", json={
            "query": "",
            "page": 1,
            "limit": 10
        })
        assert response.status_code == 200
        data = response.json()
        assert "molecules" in data
        assert "total" in data
    
    def test_library_stats(self):
        """Test getting library statistics"""
        response = client.get("/library/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_molecules" in data


class TestSystemEndpoints:
    """Test system monitoring endpoints"""
    
    def test_get_system_specs(self):
        """Test getting system specifications"""
        response = client.get("/system/specs")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        if data["success"]:
            assert "specs" in data
            assert "runtime" in data["specs"]
            assert "cpu" in data["specs"]
            assert "memory" in data["specs"]
    
    def test_get_gpus(self):
        """Test getting GPU information"""
        response = client.get("/system/gpus")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "gpus" in data
        assert isinstance(data["gpus"], list)


class TestCloudEndpoints:
    """Test cloud training endpoints"""
    
    def test_list_vms(self):
        """Test listing VMs"""
        response = client.get("/cloud/vms")
        assert response.status_code == 200
        data = response.json()
        assert "vms" in data
        assert isinstance(data["vms"], list)
    
    def test_list_credentials(self):
        """Test listing SSH credentials"""
        response = client.get("/cloud/credentials/ssh")
        assert response.status_code == 200
        data = response.json()
        assert "credentials" in data
        assert isinstance(data["credentials"], list)


class TestValidation:
    """Test input validation and error handling"""
    
    def test_invalid_molecule_search(self):
        """Test invalid search request"""
        response = client.post("/library/search", json={
            "limit": -1  # Invalid limit
        })
        assert response.status_code in [400, 422]
    
    def test_nonexistent_molecule(self):
        """Test getting nonexistent molecule"""
        response = client.get("/library/molecule/999999")
        assert response.status_code == 404


@pytest.mark.asyncio
class TestGenerationEndpoints:
    """Test molecule generation endpoints"""
    
    def test_generate_molecule_endpoint_exists(self):
        """Test that generation endpoint exists"""
        response = client.post("/generate", json={
            "num_atoms": 10,
            "elements": ["C", "H", "O"]
        })
        # Should return 200 or 400, but not 404 or 500
        assert response.status_code in [200, 400, 422]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
